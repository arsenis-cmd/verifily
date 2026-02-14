# Deploy Verifily on Fly.io

Deploy Verifily API to Fly.io with persistent storage in under 30 minutes.

## Prerequisites

- [flyctl](https://fly.io/docs/getting-started/installing-flyctl/) installed
- Fly.io account (`fly auth login`)
- Verifily repo cloned locally

## 1. Create the App

From the repo root:

```bash
fly launch --no-deploy \
  --name verifily-api \
  --region iad \
  --internal-port 8080
```

## 2. Create a Volume for Persistence

```bash
fly volumes create verifily_data --size 1 --region iad
```

This creates a 1 GB persistent volume for usage and job logs.

## 3. Configure `fly.toml`

Create or update `fly.toml` in the repo root:

```toml
app = "verifily-api"
primary_region = "iad"

[build]
  dockerfile = "docker/Dockerfile"

[env]
  VERIFILY_ENV = "prod"
  VERIFILY_ALLOW_NONLOCAL = "1"
  VERIFILY_USAGE_PERSIST = "1"
  VERIFILY_JOBS_PERSIST = "1"
  VERIFILY_DATA_DIR = "/data"
  VERIFILY_LOG_FORMAT = "json"
  VERIFILY_RATE_LIMIT_RPM = "60"

[mounts]
  source = "verifily_data"
  destination = "/data"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [[services.http_checks]]
    interval = 15000
    timeout = 5000
    path = "/ready"
    method = "GET"
```

## 4. Set the API Key Secret

```bash
fly secrets set VERIFILY_API_KEY="sk-your-secret-key-here"
```

This enables authentication. All `/v1/*` requests must include `Authorization: Bearer sk-your-secret-key-here`.

## 5. Deploy

```bash
fly deploy
```

First deploy takes ~2 minutes (Docker build). Subsequent deploys are faster.

## 6. Validate

### Check health

```bash
curl https://verifily-api.fly.dev/ready
```

Expected:

```json
{"status": "ready", "checks": {"python": "PASS", ...}}
```

### Run a pipeline request

```bash
curl -X POST https://verifily-api.fly.dev/v1/pipeline \
  -H "Authorization: Bearer sk-your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/verifily.yaml", "plan": true, "ci": true}'
```

### Check effective config

```bash
curl https://verifily-api.fly.dev/v1/config/effective \
  -H "Authorization: Bearer sk-your-secret-key-here"
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFILY_ENV` | `dev` | `dev` or `prod` |
| `VERIFILY_API_KEY` | — | API key (set via `fly secrets`) |
| `VERIFILY_RATE_LIMIT_RPM` | `0` | Requests per minute per key (0=disabled) |
| `VERIFILY_USAGE_PERSIST` | `0` | Enable usage logging (`1`) |
| `VERIFILY_JOBS_PERSIST` | `0` | Enable job history (`1`) |
| `VERIFILY_DATA_DIR` | `/data` | Persistence directory (mount volume here) |
| `VERIFILY_LOG_FORMAT` | `text` | `text` or `json` |
| `VERIFILY_ENABLE_DOCS` | `0` (prod) | Enable `/docs` endpoint |

## Common Pitfalls

**Port mismatch**: The Dockerfile exposes 8080. Make sure `internal_port` in `fly.toml` matches.

**No volume**: Without a mounted volume at `/data`, persistence data is lost on redeploy. Always create a volume.

**Forgot API key**: In prod mode, Verifily prints a startup warning but still runs without auth. Set `VERIFILY_API_KEY` via `fly secrets set`.

**Health check path**: Use `/ready` (not `/health`) for deployment health checks — it verifies all subsystems, not just "server is up".

**Region alignment**: The volume and app must be in the same region. If you change regions, create a new volume.

## Scaling

Verifily uses in-memory state with JSONL persistence. It's designed for single-instance deployments. For multi-instance, use separate volumes per instance and don't share state.

```bash
# Scale memory (default 256 MB is usually enough)
fly scale memory 512

# View logs
fly logs
```

## Updating

```bash
git pull
fly deploy
```

The Docker build caches pip packages, so updates are fast unless dependencies change.
