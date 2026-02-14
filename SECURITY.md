# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Email: security@verifily.dev

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Expectations

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 5 business days
- **Fix timeline**: critical issues within 14 days, others within 30 days
- **Disclosure**: coordinated disclosure after fix is released

## Security Model

### Scope

Verifily is a **local-first** tool. By default:

- All processing happens on the local machine
- No data is sent to external services
- No telemetry is collected
- No raw dataset content is logged or stored in artifacts (only hashes and statistical summaries)
- API server binds to localhost by default

### Data Handling

- **Fingerprints** contain only hashes and statistical sketches — no raw text
- **Audit logs** record operations and metadata — never raw payloads or PII
- **Decision artifacts** contain scores and reasons — no dataset content
- **API key comparison** uses constant-time comparison to prevent timing attacks

### Authentication

- API authentication is opt-in via `VERIFILY_API_KEY` environment variable
- When enabled, all `/v1/*` endpoints require a valid Bearer token
- Health, ready, and docs endpoints are always unauthenticated

### Rate Limiting

- Opt-in per-key rate limiting via `VERIFILY_RATE_LIMIT_RPM`
- Fixed-window algorithm, per API key

## Explicit Non-Goals

Verifily is **not**:

- A data vault or encrypted storage system
- An encryption-at-rest solution
- A network security tool
- A replacement for filesystem permissions or OS-level access control

Verifily assumes the local filesystem is trusted. Access control to datasets, artifacts, and configuration files is the responsibility of the operating system and deployment environment.

## Contact

security@verifily.dev
