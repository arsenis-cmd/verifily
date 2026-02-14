# Enterprise Readiness v1

Opt-in enterprise security layer for Verifily API. Activated via `VERIFILY_ENTERPRISE_SECURITY=1`.

## RBAC Roles & Permissions

| Permission | OWNER | ADMIN | MEMBER | VIEWER |
|---|---|---|---|---|
| `run_pipeline` | Y | Y | Y | - |
| `run_report` | Y | Y | Y | - |
| `run_contamination` | Y | Y | Y | - |
| `submit_jobs` | Y | Y | Y | - |
| `view_usage` | Y | Y | - | Y |
| `export_audit` | Y | Y | - | - |

## Scoped Tokens

Tokens use HMAC-SHA256 signatures with the format `vft_<base64url(JSON)>.<hex_sig>`.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `VERIFILY_ENTERPRISE_SECURITY` | Yes | Set to `1` to enable |
| `VERIFILY_TOKEN_SECRET` | Yes | HMAC signing secret (min 16 chars recommended) |

### Creating a Token

```python
from verifily_cli_v1.core.security.tokens import create_token
import time

token = create_token("your-secret", {
    "token_id": "tok_user1",
    "role": "member",
    "project_id": "proj_abc",
    "exp": time.time() + 86400,  # 24h
})
# Returns: vft_eyJ0b2tlbl9pZCI6....<hex_sig>
```

### Using a Token

```bash
curl -H "Authorization: Bearer vft_..." http://localhost:8080/v1/usage
```

## Policy Engine

Enterprise policies can force `DONT_SHIP` on pipeline results.

| Variable | Type | Description |
|---|---|---|
| `VERIFILY_POLICY_REQUIRE_CONTAM_PASS` | bool | Require contamination status = PASS |
| `VERIFILY_POLICY_REQUIRE_REPRO` | bool | Require contract validity |
| `VERIFILY_POLICY_MAX_PII_HITS` | int | Maximum PII hits allowed |
| `VERIFILY_POLICY_MIN_F1` | float | Minimum F1 score threshold |

When a policy violation occurs, the pipeline response includes:
```json
{
  "decision": {
    "recommendation": "DONT_SHIP",
    "exit_code": 1
  },
  "policy_violations": [
    "Policy violation: contamination status is FAIL, but require_contamination_pass=True"
  ]
}
```

## Audit Export API

```
GET /v1/audit/export?project_id=proj1&from_ts=2026-01-01T00:00:00Z&to_ts=2026-02-01T00:00:00Z&format=json
```

**Parameters:**

| Param | Default | Description |
|---|---|---|
| `project_id` | all | Filter by project |
| `from_ts` | none | ISO start timestamp |
| `to_ts` | none | ISO end timestamp |
| `format` | `json` | `json` or `jsonl` |

**Requires:** `export_audit` permission (OWNER or ADMIN role).

**JSON response:**
```json
{
  "events": [...],
  "total": 42
}
```

**JSONL response:** one JSON object per line, `Content-Type: application/x-ndjson`.

## Security Notes

- Enterprise tokens (`vft_`) are detected before any other auth mode — they work regardless of `auth_mode` setting.
- Token secrets should be at least 16 characters.
- HMAC signatures use constant-time comparison to prevent timing attacks.
- Audit events are redacted (secrets masked) before export.
- When `enterprise_security=False` (default), all enterprise features are invisible — zero behavioral change.
