# Threat Model

This document describes the assets Verifily protects, trust boundaries, known attack surfaces, and mitigations.

## Assets

| Asset | Sensitivity | Where |
|-------|-------------|-------|
| Raw datasets | High — may contain PII or proprietary content | User filesystem |
| Model artifacts | Medium — trained weights and configs | Run directories |
| API keys | High — controls API access | Environment variables |
| Decision outputs | Medium — gate verdicts with metrics | Artifact directories, stdout |
| Audit logs | Low — operation metadata only | Artifact directories |

## Trust Boundaries

```
┌──────────────────────────────────────────┐
│              User Machine                │
│                                          │
│  ┌──────────┐    ┌──────────────────┐    │
│  │  CLI     │───→│  Filesystem      │    │
│  │          │    │  (datasets,      │    │
│  └──────────┘    │   artifacts,     │    │
│       │          │   configs)       │    │
│       ▼          └──────────────────┘    │
│  ┌──────────┐                            │
│  │  API     │◄──── localhost only        │
│  │  Server  │                            │
│  └──────────┘                            │
│       ▲                                  │
│       │                                  │
│  ┌──────────┐                            │
│  │  SDK     │                            │
│  │  Client  │                            │
│  └──────────┘                            │
└──────────────────────────────────────────┘
```

**Boundary 1: CLI ↔ Filesystem**
- CLI reads datasets and configs from local paths
- Writes artifacts (hashes, reports, decisions) to local paths
- Trust assumption: filesystem permissions are correctly configured

**Boundary 2: API Server ↔ Network**
- Server binds to localhost by default
- If exposed to a network, authentication should be enabled via `VERIFILY_API_KEY`
- All `/v1/*` endpoints are behind optional auth middleware

**Boundary 3: SDK Client ↔ API Server**
- SDK communicates over HTTP to the API server
- API key is passed via Bearer token in Authorization header

## Attack Surfaces

### HTTP API

| Vector | Risk | Mitigation |
|--------|------|------------|
| Unauthenticated access | Medium | Opt-in API key auth; localhost-only by default |
| Brute-force API key | Low | Constant-time comparison; opt-in rate limiting |
| Request injection | Low | Pydantic validation on all request bodies |
| Path traversal via config_path | Medium | Config paths resolved relative to config directory; no shell execution |
| Denial of service | Low | Rate limiting; no unbounded computation |

### Filesystem Artifacts

| Vector | Risk | Mitigation |
|--------|------|------------|
| Raw data exfiltration via artifacts | High | No raw text in fingerprints, audit logs, or decision outputs |
| PII in training data | High | PII scan with configurable thresholds; blocker in decision gate |
| Artifact tampering | Medium | SHA-256 chain hashes in run contracts for integrity verification |

### Logs

| Vector | Risk | Mitigation |
|--------|------|------------|
| PII in logs | Medium | Structured logging records only metadata (method, path, status, timing) |
| API key in logs | High | API keys are never logged; only key ID (first 8 chars) appears |
| Raw data in logs | High | No raw payloads logged; audit events contain only operation summaries |

## Mitigations Already Present

1. **No raw data in outputs**: Fingerprints contain only hashes and statistical sketches. Audit logs record operations, not payloads. Decision artifacts contain scores, not examples.

2. **Constant-time API key comparison**: Uses `hmac.compare_digest()` to prevent timing side-channel attacks.

3. **Opt-in authentication**: API key auth is disabled by default (local-only use case). When enabled, all `/v1/*` endpoints require valid credentials.

4. **PII detection and blocking**: Regex-based PII scan in the report step. Configurable `max_pii_hits` threshold in the decision gate — exceeding it blocks the SHIP decision.

5. **Structured error envelope**: All API errors use a consistent `{error: {type, message, request_id}}` format. No stack traces or internal paths in production error responses.

6. **Request ID tracing**: Every API request gets a unique ID for end-to-end tracing without exposing internal state.

7. **Input validation**: All API request bodies are validated via Pydantic models before processing.

## Residual Risks

| Risk | Impact | Notes |
|------|--------|-------|
| Local filesystem access | High | Verifily trusts the local filesystem. No sandboxing. |
| No encryption at rest | Medium | Artifacts are written as plain JSON. Use OS-level encryption if needed. |
| API server on public network | High | If exposed without auth, all endpoints are accessible. Always enable `VERIFILY_API_KEY` for non-localhost deployments. |
| Log file permissions | Low | Audit and usage logs are written with default umask. Restrict if needed. |

## Future Hardening

- TLS support for API server (currently HTTP-only)
- API key rotation support
- Audit log signing for tamper detection
- Optional artifact encryption at rest
