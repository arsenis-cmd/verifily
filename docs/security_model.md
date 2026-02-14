# Verifily Security Model

This document describes the security architecture and guarantees of Verifily.

## Security Principles

1. **Defense in Depth**: Multiple layers of protection
2. **Least Privilege**: Minimal permissions required
3. **Zero Trust**: Verify every request
4. **Observability**: All actions logged and auditable

## Authentication

### API Key Authentication

Verifily uses API keys for authentication:

```
Authorization: Bearer vf_<32-char-hex>
```

Key features:
- **Scoped Keys**: Keys can be limited to specific projects
- **Rotation**: Support for key rotation without downtime
- **Revocation**: Instant key revocation
- **No Storage**: Keys are hashed, never stored in plain text

### Authentication Flow

```
Request + API Key → Hash Key → Lookup Hash → Validate Scope → Process
```

## Authorization

### Role-Based Access Control (RBAC)

| Role | Description | Permissions |
|------|-------------|-------------|
| `ADMIN` | Full access | All operations |
| `DEV` | Developer access | Run validations, view results |
| `VIEWER` | Read-only access | View reports, metrics |

### Project Isolation

- Projects are completely isolated
- Keys can be scoped to single or multiple projects
- Cross-project access requires explicit permissions

### Permission Enforcement

Every API request:
1. Authenticates the key
2. Validates project scope
3. Checks role permissions
4. Logs access attempt

## Rate Limiting

### Global Rate Limits

- Default: 120 requests per minute per key
- Configurable per-project
- Burst allowance for legitimate spikes

### Rate Limit Response

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640995200
```

## Hash Integrity

### Artifact Signing

All decision artifacts can be cryptographically signed:

```
decision.json → Canonical JSON → SHA-256 → HMAC Signature → decision.sig
```

Verification:
```bash
verifily verify-run --run ./runs/run_001
```

### Tamper Detection

Hash chain verification detects:
- Modified datasets
- Altered decisions
- Missing files
- Unexpected additions

## Redaction Guarantees

### PII Detection

Automatic scanning for:
- Email addresses
- Phone numbers
- API keys
- Passwords
- Private keys

### Audit Report

```bash
verifily verify-run --run ./runs/run_001 --audit
```

Output: `redaction_audit.json`
```json
{
  "status": "PASS",
  "findings_count": 0,
  "findings_by_type": {}
}
```

### Compliance

- **GDPR**: Data processing audit trail
- **SOC 2**: Access logging
- **HIPAA**: PHI detection (configurable)

## Workspace Security

### Path Traversal Protection

All file paths are validated:

```python
# Blocked: ../etc/passwd
# Blocked: /etc/passwd (in production)
# Allowed: ./data/train.jsonl
```

### Production Mode

Set `VERIFILY_PROD=1` to enable:
- Absolute path blocking
- Stricter validation
- Audit logging

## Network Security

### Transport Security

- **TLS 1.3**: All communications encrypted
- **Certificate Pinning**: Optional client-side
- **HSTS**: HTTP Strict Transport Security

### Deployment Patterns

Recommended: Reverse proxy (Caddy/Nginx) → Verifily

```
Internet → HTTPS → Caddy → HTTP → Verifily
                    (TLS termination)
```

## Audit Logging

### Logged Events

- All API requests (metadata only, never body)
- Authentication attempts (success/failure)
- Permission denials
- Configuration changes

### Log Format

```json
{
  "ts": "2025-01-20T12:00:00Z",
  "level": "INFO",
  "request_id": "abc123",
  "api_key_id": "key_xxx",
  "method": "POST",
  "path": "/v1/pipeline",
  "status": 200,
  "elapsed_ms": 45
}
```

### Log Security

- Structured JSON format
- No sensitive data in logs
- Tamper-resistant (append-only)
- Retention configurable

## Secret Management

### Environment Variables

Sensitive configuration via environment:

```bash
VERIFILY_API_KEY=vf_...
VERIFILY_KEY_SALT=...
VERIFILY_BOOTSTRAP_TOKEN=...
```

### Key Generation

```bash
# API Key
openssl rand -hex 16

# Salt
openssl rand -hex 32

# Bootstrap Token
openssl rand -hex 16
```

## Threat Model

### Assets Protected

1. **Datasets**: Customer data
2. **Models**: Trained model artifacts
3. **Decision Records**: Quality gate results
4. **API Keys**: Authentication credentials

### Threats Mitigated

| Threat | Mitigation |
|--------|------------|
| Unauthorized Access | API key auth, RBAC |
| Data Exfiltration | Path traversal protection |
| Tampering | Hash chains, signatures |
| Replay Attacks | Request IDs, timestamps |
| DoS | Rate limiting |
| Privilege Escalation | Scope validation |

### Out of Scope

- Physical security (hosting provider)
- Network infrastructure
- Client-side security
- Social engineering

## Compliance Certifications

Target certifications:
- SOC 2 Type II
- ISO 27001
- GDPR compliance
- HIPAA (optional)

## Security Contact

Security issues: security@verifily.dev

PGP Key: [Download](https://verifily.dev/security.asc)

## See Also

- [Deployment Hardening](./deployment_hardening.md)
- [Enterprise Deployment](./enterprise_deployment.md)
