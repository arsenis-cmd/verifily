# Secrets & Redaction

Centralized redaction, secret detection, and config hardening for Verifily.

## Redaction Rules

### What Is Redacted

| Pattern | Action |
|---------|--------|
| Keys matching `api_key`, `token`, `secret`, `password`, `bearer`, `authorization`, `credential`, `private_key` | Value replaced with `***REDACTED***` |
| `Bearer <token>` in text | Token portion replaced |
| Hex strings >= 24 chars near sensitive keywords | Replaced with `***REDACTED***` |
| Strings > 240 chars in dicts | Truncated to `first_60...last_60` |

### Where Redaction Is Applied

- **Error messages**: `redact_text()` applied before returning error envelopes
- **Middleware JSON logs**: `safe_log_json()` wraps all structured log entries
- **Audit log**: `redact_dict()` applied before writing each event
- **Effective config endpoint**: All config values run through `redact_dict()`

### What Is Never Stored

- Raw API keys (hashed via SHA256[:12] in usage store)
- Dataset row content in audit logs
- PII text in report output (only counts + row indices)
- Bearer tokens in any log or artifact

## Effective Config Endpoint

```
GET /v1/config/effective
```

Returns the current server configuration with all secrets stripped:

```json
{
  "config": {
    "privacy_mode": "local",
    "log_format": "text",
    "rate_limit_rpm": null,
    "usage_persist": "0",
    "jobs_persist": "0",
    "auth_enabled": false
  }
}
```

### SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(base_url="http://localhost:8080")
config = client.effective_config()
print(config.config["privacy_mode"])
```

## `security-check` Command

Scan a directory for leaked secrets:

```bash
# Scan current directory
verifily security-check

# Scan a specific directory
verifily security-check /path/to/artifacts

# Verbose mode
verifily security-check --verbose
```

Exit codes: 0 = clean, 1 = suspected leaks found.

Scans `.json`, `.jsonl`, `.yaml`, `.yml`, `.txt`, `.log`, `.env` files for:
- Bearer tokens
- Long hex strings near sensitive keywords

Never prints the actual secret — only file, line number, and pattern type.

## Environment Variable Overrides

Config values can be overridden via environment variables. Precedence: project config < env vars < request overrides.

| Env Var | Config Key | Type | Allowed |
|---------|-----------|------|---------|
| `VERIFILY_PRIVACY_MODE` | `privacy_mode` | str | `local`, `remote`, `hybrid` |
| `VERIFILY_RATE_LIMIT_RPM` | `rate_limit_rpm` | int | any |
| `VERIFILY_CONTAM_EXACT_MAX` | `contamination.exact_threshold` | float | any |
| `VERIFILY_CONTAM_NGRAM_MAX` | `contamination.near_threshold` | float | any |
| `VERIFILY_DECISION_MIN_DELTA` | `ship_if.min_delta` | float | any |
| `VERIFILY_DECISION_MAX_DROP` | `ship_if.max_f1_regression` | float | any |
| `VERIFILY_LOG_FORMAT` | `log_format` | str | `json`, `text` |
| `VERIFILY_USAGE_PERSIST` | `usage_persist` | str | `0`, `1` |
| `VERIFILY_JOBS_PERSIST` | `jobs_persist` | str | `0`, `1` |

`VERIFILY_API_KEY` is never included in resolved config output.

## Privacy Modes

| Mode | Behavior |
|------|----------|
| `local` | Forbids any remote config keys (`remote_url`, `openai_api_key`, etc.). Raises `ValueError` if found. |
| `hybrid` | Allows remote config but enforces redaction at runtime. |
| `remote` | No restrictions on remote config. |

## `.env` File Support

Verifily loads `.env` files automatically via `load_dotenv_if_present()`:
- Skips comments (`#`) and blank lines
- Parses `KEY=VALUE` (strips surrounding quotes)
- Never overwrites existing environment variables
- No external dependencies

## Doctor Security Checks

The `verifily doctor` command includes three security checks:

| Check | PASS | WARN |
|-------|------|------|
| Log format | `json` (structured) | `text` (use json for production) |
| Privacy mode | Any valid mode | — |
| API auth | `VERIFILY_API_KEY` set | Not configured (open access) |
