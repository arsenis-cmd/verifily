"""Verifily Transform utilities: hashing, timestamps, file I/O."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sha256_str(s: str) -> str:
    """SHA-256 of a string."""
    return f"sha256:{hashlib.sha256(s.encode()).hexdigest()}"


def sha256_dict(d: Dict[str, Any]) -> str:
    """SHA-256 of a JSON-serializable dict (sorted, compact)."""
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def utcnow_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_transform_id() -> str:
    """Generate a unique transform run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(3).hex()
    return f"transform_{ts}_{rand}"


def ensure_dir(path: str) -> Path:
    """Create directory and parents if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# PII patterns (regex-based)
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """Detect PII in text. Returns list of {type, match, start, end}."""
    findings = []
    for pii_type, pattern in _PII_PATTERNS.items():
        for m in pattern.finditer(text):
            findings.append({
                "type": pii_type,
                "match": m.group(),
                "start": m.start(),
                "end": m.end(),
            })
    return findings


def redact_pii(text: str) -> str:
    """Replace detected PII with redaction placeholders."""
    for pii_type, pattern in _PII_PATTERNS.items():
        placeholder = f"[{pii_type.upper()}_REDACTED]"
        text = pattern.sub(placeholder, text)
    return text
