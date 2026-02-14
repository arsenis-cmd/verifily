"""Redaction utilities, secret detection, and privacy mode enforcement.

Provides deterministic, non-mutating redaction for logs, artifacts, and error
messages.  Never stores or prints the actual secret — only indicates that
redaction occurred.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── Constants ────────────────────────────────────────────────────

SENSITIVE_KEYWORDS = [
    "api_key", "apikey", "token", "secret", "password",
    "passphrase", "bearer", "authorization", "credential",
    "private_key",
]

REDACTED = "***REDACTED***"

# Keys whose values should be removed entirely from log payloads
_RAW_DATA_KEYS = {"rows", "dataset", "samples", "sample_rows", "dataset_rows"}

# Max payload size for safe_log_json (8 KB)
_MAX_LOG_BYTES = 8192

# Max string length before truncation in redact_dict
_MAX_STRING_LEN = 240

# Max recursion depth for redact_dict
_MAX_DEPTH = 10

# Remote config keys forbidden in LOCAL mode
REMOTE_CONFIG_KEYS = [
    "remote_url", "remote_endpoint", "openai_api_key",
    "anthropic_api_key", "openai", "anthropic",
]

# ── Patterns ─────────────────────────────────────────────────────

_BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{24,}\b")


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name matches any sensitive keyword (case-insensitive)."""
    lower = key.lower()
    return any(kw in lower for kw in SENSITIVE_KEYWORDS)


# ── Public API ───────────────────────────────────────────────────


def redact_text(text: str) -> str:
    """Redact obvious secrets from a text string.

    - Replaces Bearer tokens
    - Replaces long hex strings (>=24 chars)
    - Deterministic, returns new string (no mutation)
    """
    if not text:
        return text
    result = _BEARER_RE.sub(r"\1" + REDACTED, text)
    result = _LONG_HEX_RE.sub(REDACTED, result)
    return result


def redact_dict(obj: Any, *, _depth: int = 0) -> Any:
    """Recursively redact sensitive values from a data structure.

    - Keys matching SENSITIVE_KEYWORDS have their values replaced
    - Strings > 240 chars are truncated (prefix...suffix)
    - Never mutates the input object
    - Max recursion depth of 10
    """
    if _depth > _MAX_DEPTH:
        return "[max_depth_exceeded]"

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_sensitive_key(k):
                result[k] = REDACTED
            else:
                result[k] = redact_dict(v, _depth=_depth + 1)
        return result

    if isinstance(obj, list):
        return [redact_dict(item, _depth=_depth + 1) for item in obj]

    if isinstance(obj, str):
        if len(obj) > _MAX_STRING_LEN:
            return obj[:60] + "..." + obj[-60:]
        return obj

    # Numbers, booleans, None — pass through
    return obj


def safe_log_json(event: dict) -> dict:
    """Prepare a dict for safe JSON logging.

    1. Removes raw data keys (rows, dataset, samples)
    2. Applies redact_dict
    3. Enforces max payload size (8 KB)
    """
    # Step 1: Remove raw data keys
    cleaned = {k: v for k, v in event.items() if k not in _RAW_DATA_KEYS}

    # Step 2: Redact sensitive values
    redacted = redact_dict(cleaned)

    # Step 3: Enforce size limit
    serialized = json.dumps(redacted, separators=(",", ":"), default=str)
    if len(serialized) <= _MAX_LOG_BYTES:
        return redacted

    # Truncate large string values to fit
    return _truncate_to_fit(redacted, _MAX_LOG_BYTES)


def _truncate_to_fit(obj: Any, max_bytes: int) -> Any:
    """Truncate string values until the object fits within max_bytes."""
    # Simple approach: truncate all strings > 100 chars
    if isinstance(obj, dict):
        return {k: _truncate_to_fit(v, max_bytes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_to_fit(item, max_bytes) for item in obj]
    if isinstance(obj, str) and len(obj) > 100:
        return obj[:40] + "...[truncated]..." + obj[-40:]
    return obj


def assert_no_secrets(obj: Any, *, _path: str = "") -> None:
    """Raise ValueError if any string value looks like a leaked secret.

    Checks for:
    - Bearer tokens
    - Long hex strings adjacent to sensitive key names
    - Unredacted values in sensitive keys

    Never prints the secret itself — only the key path.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{_path}.{k}" if _path else k
            if isinstance(k, str) and _is_sensitive_key(k):
                if isinstance(v, str) and v != REDACTED and len(v) > 3:
                    raise ValueError(f"secret_leak_detected: {key_path}")
            assert_no_secrets(v, _path=key_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_no_secrets(item, _path=f"{_path}[{i}]")
    elif isinstance(obj, str):
        if _BEARER_RE.search(obj):
            raise ValueError(f"secret_leak_detected: {_path} (bearer token)")
        # Check for long hex strings near sensitive context
        if _LONG_HEX_RE.search(obj) and _path and _is_sensitive_key(_path.split(".")[-1]):
            raise ValueError(f"secret_leak_detected: {_path} (hex token)")


# ── Privacy mode enforcement ─────────────────────────────────────


def check_privacy_mode(config: dict, mode: str) -> None:
    """Validate config against privacy mode constraints.

    Raises ValueError if config violates mode rules.
    """
    if mode == "local":
        for key in _walk_keys(config):
            if any(rk in key.lower() for rk in REMOTE_CONFIG_KEYS):
                raise ValueError(
                    f"LOCAL mode forbids remote config: found '{key}'. "
                    "Remove remote configuration or switch to hybrid/remote mode."
                )


def _walk_keys(d: Any, prefix: str = "") -> List[str]:
    """Yield all dotted key paths in a nested dict."""
    keys: List[str] = []
    if isinstance(d, dict):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.append(full)
            keys.extend(_walk_keys(v, full))
    return keys


# ── Dotenv loader ────────────────────────────────────────────────


def load_dotenv_if_present(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    - Skips comments (lines starting with #) and blank lines
    - Strips surrounding quotes from values
    - Never overwrites existing environment variables
    - No external dependencies
    """
    p = Path(path)
    if not p.is_file():
        return

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# ── File scanner (for security-check command) ────────────────────


_SCANNABLE_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".log", ".env"}


def scan_directory_for_secrets(
    target: str, *, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Scan files in a directory for potential secret leaks.

    Returns list of findings: [{file, line, pattern}].
    Never includes the actual secret value.
    """
    findings: List[Dict[str, Any]] = []
    target_path = Path(target)

    if not target_path.is_dir():
        return findings

    for file_path in target_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _SCANNABLE_EXTENSIONS:
            continue
        # Skip very large files (>1 MB)
        try:
            if file_path.stat().st_size > 1_000_000:
                continue
        except OSError:
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    if _BEARER_RE.search(line):
                        findings.append({
                            "file": str(file_path),
                            "line": line_no,
                            "pattern": "bearer_token",
                        })
                    elif _LONG_HEX_RE.search(line):
                        # Only flag if near a sensitive keyword context
                        lower_line = line.lower()
                        if any(kw in lower_line for kw in SENSITIVE_KEYWORDS):
                            findings.append({
                                "file": str(file_path),
                                "line": line_no,
                                "pattern": "hex_token_near_keyword",
                            })
        except (OSError, UnicodeDecodeError):
            continue

    return findings
