"""Config precedence resolution for Verifily.

Merges project config + env overrides + request overrides into a single
deterministic, JSON-serializable config dict.  Never includes secrets.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from verifily_cli_v1.core.secrets import SENSITIVE_KEYWORDS, _is_sensitive_key

# ── Env var → config key mapping ─────────────────────────────────
# Format: env_var → (dotted_config_key, type, allowed_values_or_None)

_ENV_OVERRIDES: Dict[str, Tuple[str, type, Optional[List[str]]]] = {
    "VERIFILY_PRIVACY_MODE": ("privacy_mode", str, ["local", "remote", "hybrid"]),
    "VERIFILY_RATE_LIMIT_RPM": ("rate_limit_rpm", int, None),
    "VERIFILY_CONTAM_EXACT_MAX": ("contamination.exact_threshold", float, None),
    "VERIFILY_CONTAM_NGRAM_MAX": ("contamination.near_threshold", float, None),
    "VERIFILY_DECISION_MIN_DELTA": ("ship_if.min_delta", float, None),
    "VERIFILY_DECISION_MAX_DROP": ("ship_if.max_f1_regression", float, None),
    "VERIFILY_LOG_FORMAT": ("log_format", str, ["json", "text"]),
    "VERIFILY_USAGE_PERSIST": ("usage_persist", str, ["0", "1"]),
    "VERIFILY_JOBS_PERSIST": ("jobs_persist", str, ["0", "1"]),
}

# Env vars that must NEVER appear in resolved output
_SECRET_ENV_VARS = {"VERIFILY_API_KEY"}


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path.

    Example: _set_nested(d, "contamination.exact_threshold", 0.1)
    sets d["contamination"]["exact_threshold"] = 0.1
    """
    parts = dotted_key.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _parse_value(raw: str, target_type: type, allowed: Optional[List[str]]) -> Any:
    """Parse and validate an env var string value."""
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    # str
    value = raw.strip()
    if allowed and value not in allowed:
        raise ValueError(f"Invalid value '{value}'. Allowed: {allowed}")
    return value


def _strip_secrets(d: Any) -> Any:
    """Remove keys matching sensitive keywords from a dict (recursive)."""
    if isinstance(d, dict):
        return {
            k: _strip_secrets(v)
            for k, v in d.items()
            if not _is_sensitive_key(k)
        }
    if isinstance(d, list):
        return [_strip_secrets(item) for item in d]
    return d


def resolve_config(
    project_config: dict,
    env: Dict[str, str],
    request_config: Optional[dict] = None,
    *,
    allow_env_override: bool = True,
) -> dict:
    """Resolve effective config with precedence: project < env < request.

    Args:
        project_config: Base config from verifily.yaml
        env: Environment variables (typically os.environ)
        request_config: Optional request-level overrides
        allow_env_override: Whether to apply env var overrides

    Returns:
        Resolved config dict (deep-copied, secrets stripped, JSON-serializable)
    """
    result = copy.deepcopy(project_config)

    # Apply env overrides
    if allow_env_override:
        for env_var, (config_key, target_type, allowed) in _ENV_OVERRIDES.items():
            raw = env.get(env_var)
            if raw is None:
                continue
            try:
                value = _parse_value(raw, target_type, allowed)
                _set_nested(result, config_key, value)
            except (ValueError, TypeError):
                # Skip invalid env values silently
                pass

    # Apply request overrides (shallow merge of known keys)
    if request_config:
        for k, v in request_config.items():
            if not _is_sensitive_key(k):
                result[k] = v

    # Strip any sensitive keys from output
    result = _strip_secrets(result)

    # Ensure secrets from env are never included
    for secret_var in _SECRET_ENV_VARS:
        result.pop(secret_var, None)
        result.pop(secret_var.lower(), None)

    return result
