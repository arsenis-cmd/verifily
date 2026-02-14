# API Production Hardening

Opt-in features that make the Verifily API server feel enterprise-ready.
All features are gated behind environment variables.
Default behavior is unchanged — unset = feature OFF.

## Enabling Usage Persistence

Persist usage accounting events across server restarts.

```bash
export VERIFILY_USAGE_PERSIST=1
export VERIFILY_USAGE_LOG_PATH=/tmp/verifily_usage_events.jsonl  # optional, this is the default

verifily serve --port 8080
```

Events are appended to an append-only JSONL file. On startup, the file is
replayed to rebuild in-memory aggregates. Only metadata is stored — never raw
payloads, PII, or full API keys.

Verify persisted counts survive a restart:

```bash
# Make some requests
curl -s http://localhost:8080/v1/usage?group_by=total
# Restart the server, then query again — counts are preserved
```

## Enabling Rate Limiting

Per-API-key fixed-window rate limiter (requests per minute).

```bash
export VERIFILY_RATE_LIMIT_RPM=60   # 60 requests/minute per api_key_id

verifily serve --port 8080
```

Only `/v1/*` endpoints are limited.
`/health`, `/ready`, `/metrics`, `/docs` are always exempt.

Trigger a 429:

```bash
export VERIFILY_RATE_LIMIT_RPM=2

# First two pass
curl -s http://localhost:8080/v1/usage | jq .
curl -s http://localhost:8080/v1/usage | jq .

# Third triggers 429
curl -s -w "\nHTTP %{http_code}\n" http://localhost:8080/v1/usage
```

Expected 429 response:

```json
{
  "error": {
    "type": "RATE_LIMITED",
    "message": "Rate limit exceeded (2 requests/minute). Try again later.",
    "request_id": "abc123def456"
  }
}
```

The response includes a `Retry-After` header (seconds until the window resets).

## Enabling Structured JSON Logging

Machine-readable JSON log lines for observability pipelines.

```bash
export VERIFILY_LOG_FORMAT=json   # default: text

verifily serve --port 8080
```

Each request emits one JSON line:

```json
{"ts":"2025-01-15T10:30:00Z","level":"INFO","request_id":"abc123","api_key_id":"anonymous","project_id":null,"method":"GET","path":"/health","status":200,"elapsed_ms":2}
```

Fields: `ts`, `level`, `request_id`, `api_key_id`, `project_id`, `method`,
`path`, `status`, `elapsed_ms`.

No request bodies or PII are ever logged.

## Error Envelope (always-on)

All API errors follow a consistent envelope — no opt-in needed:

```json
{
  "error": {
    "type": "VALIDATION_ERROR",
    "message": "Either config_path or project_path must be provided.",
    "request_id": "abc123def456"
  }
}
```

Stable error types:

| Type | HTTP Status | When |
|---|---|---|
| `AUTH_ERROR` | 401 | Missing or invalid API key |
| `VALIDATION_ERROR` | 422 | Invalid request parameters |
| `NOT_FOUND` | 404 | Config or data file missing |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Unhandled server error |

The `request_id` in the envelope body matches the `X-Request-ID` response
header, enabling end-to-end request tracing.

## Running Tests

```bash
# Hardening tests only
python3 -m pytest verifily_cli_v1/tests/test_api_hardening.py -v

# Full suite (must stay < 2s)
python3 -m pytest verifily_cli_v1/tests/ -v
```

## All Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VERIFILY_USAGE_PERSIST` | unset (off) | Set to `1` to enable usage persistence |
| `VERIFILY_USAGE_LOG_PATH` | `/tmp/verifily_usage_events.jsonl` | Path to JSONL events file |
| `VERIFILY_RATE_LIMIT_RPM` | unset (off) | Max requests per minute per api_key_id |
| `VERIFILY_LOG_FORMAT` | `text` | Log format: `text` or `json` |
| `VERIFILY_API_KEY` | unset (off) | API key for auth (existing feature) |

## API Version Stability

The `/v1` API namespace is **frozen** within the current MAJOR version (v1.x).

**What is stable:**
- The error envelope format: `{"error": {"type", "message", "request_id"}}`
- All existing endpoint paths and their response shapes
- Error types: `AUTH_ERROR`, `VALIDATION_ERROR`, `NOT_FOUND`, `RATE_LIMITED`, `INTERNAL_ERROR`
- HTTP status code semantics for each error type

**What may change in MINOR releases:**
- New optional fields may be added to success responses
- New endpoints may be added under `/v1`
- New error types may be added for new features

**What requires a MAJOR version bump:**
- Removing or renaming existing response fields
- Changing the error envelope structure
- Altering HTTP status code semantics for existing error types
- Removing or restructuring existing endpoints

See [VERSIONING.md](../VERSIONING.md) for the full stability and deprecation policy.
