# Docker Production Deployment

Minimal guidance for running Verifily API server in Docker.

## Quick Start

```bash
docker compose up -d
curl http://localhost:8080/health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFILY_API_KEY` | (empty) | API key for authentication. When set, all `/v1/*` endpoints require `Authorization: Bearer <key>`. |
| `VERIFILY_RATE_LIMIT_RPM` | (empty) | Per-key rate limit in requests per minute. |
| `VERIFILY_USAGE_PERSIST` | `0` | Set to `1` to persist usage data to disk. |
| `VERIFILY_USAGE_LOG_PATH` | `/workspace/verifily_usage.jsonl` | Path for usage event log (when persistence enabled). |
| `VERIFILY_JOBS_PERSIST` | `0` | Set to `1` to persist job history to disk. |
| `VERIFILY_JOBS_LOG_PATH` | `/workspace/verifily_jobs.jsonl` | Path for job event log (when persistence enabled). |

## Volume Mount

The compose file mounts `./workspace` to `/workspace` in the container. Place your configs and data here:

```bash
mkdir -p workspace
cp verifily.yaml workspace/
cp data/*.jsonl workspace/
```

All API paths should reference `/workspace/`:

```bash
curl -X POST http://localhost:8080/v1/pipeline \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VERIFILY_API_KEY" \
  -d '{"config_path": "/workspace/verifily.yaml"}'
```

## Authentication

Enable authentication by setting `VERIFILY_API_KEY`:

```bash
VERIFILY_API_KEY=my-secret-key docker compose up -d
```

Unauthenticated endpoints (always accessible):
- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /docs`

Authenticated endpoints (require Bearer token):
- All `/v1/*` endpoints

## Health Checks

The Docker image includes a built-in health check (every 30s, 5s timeout):

```bash
docker inspect --format='{{.State.Health.Status}}' <container_id>
```

Manual check:

```bash
curl http://localhost:8080/health
# {"status":"ok","version":"1.0.0","time":"...","mode":"local"}

curl http://localhost:8080/ready
# {"status":"ready","checks":{"python":"ok","temp_write":"ok","imports":"ok",...}}
```

## Example: curl

```bash
# Pipeline gate
curl -X POST http://localhost:8080/v1/pipeline \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VERIFILY_API_KEY" \
  -d '{"config_path": "/workspace/verifily.yaml", "ci": true}'

# Dataset report
curl -X POST http://localhost:8080/v1/report \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VERIFILY_API_KEY" \
  -d '{"dataset_path": "/workspace/train.jsonl", "schema": "sft"}'
```

## Example: Python SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(
    base_url="http://localhost:8080",
    api_key="my-secret-key",
)

health = client.health()
print(health.status, health.version)

result = client.pipeline(config_path="/workspace/verifily.yaml", ci=True)
print(result.decision)
```

## Reverse Proxy

For non-localhost deployments, place Verifily behind a reverse proxy (nginx, Caddy, Traefik) for:
- TLS termination (Verifily serves HTTP only)
- Additional rate limiting
- Access logging

Verifily does not provide TLS natively. Do not expose the container port directly to the internet without a proxy.

## Logs

```bash
docker compose logs -f verifily
```

Structured log format: `timestamp logger_name message`. No raw data or API keys appear in logs.

## Resource Usage

Verifily is lightweight:
- Memory: ~100MB baseline
- CPU: minimal at idle, proportional to dataset size during pipeline runs
- Disk: only for persisted usage/job logs (if enabled)
