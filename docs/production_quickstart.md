# Verifily Production Quickstart

Deploy Verifily in production in 15 minutes with Docker and sensible defaults.

## Overview

This guide covers single-node production deployment with:
- Docker containerization
- Workspace-based file security
- JSON structured logging
- Health checks and monitoring
- Reverse proxy (Caddy/Nginx) for TLS

## Prerequisites

- Docker & Docker Compose
- A server with at least 2GB RAM
- (Optional) A domain name for TLS

## Quick Start (5 minutes)

### 1. Clone and Prepare

```bash
git clone <verifily-repo>
cd verifily

# Create workspace directory
mkdir -p workspace logs

# Copy environment file
cp .env.example .env.prod
```

### 2. Configure Environment

Edit `.env.prod`:

```bash
# Minimum required settings
VERIFILY_API_KEY=$(openssl rand -hex 32)  # Generate random key
VERIFILY_PROD=1
VERIFILY_LOG_FORMAT=json
```

### 3. Start Server

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8080/health

# Configuration (safe, no secrets)
curl http://localhost:8080/v1/config

# Ready check
curl http://localhost:8080/ready
```

Expected response from `/v1/config`:
```json
{
  "config": {
    "bind_host": "0.0.0.0",
    "port": 8080,
    "prod_mode": true,
    "workspace_root": "/workspace",
    "auth_enabled": true,
    ...
  },
  "request_id": "abc123"
}
```

## Workspace Security Model

Verifily enforces that all file operations stay within the workspace:

```
workspace/
├── datasets/        # Input datasets
├── runs/           # Pipeline outputs
└── temp/           # Temporary files
```

### Rules

1. **Relative paths only** - All API paths must be relative (e.g., `datasets/v1/train.jsonl`)
2. **No path traversal** - `../etc/passwd` is blocked
3. **No absolute paths** - `/etc/passwd` is blocked in production
4. **Auto-creation** - Directories created on first use

### Example API Call

```bash
# Upload dataset to workspace/datasets/v1/
cp my-data.jsonl workspace/datasets/v1/data.jsonl

# Run pipeline with relative path
curl -X POST http://localhost:8080/v1/pipeline \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "config_path": "configs/default.yaml",
    "project_path": "datasets/v1"
  }'
```

## Org Mode (Multi-tenant)

For multi-tenant deployments, use org mode instead of a single API key:

```bash
# In .env.prod
VERIFILY_ORG_MODE=1
VERIFILY_API_KEY=  # Leave empty
```

Create first org (bootstrap):
```bash
curl -X POST http://localhost:8080/v1/orgs \
  -d '{"name": "Acme Corp"}'
```

Create project and keys:
```bash
# Create project
curl -X POST http://localhost:8080/v1/projects \
  -d '{"org_id": "org_xxx", "name": "prod-llm"}'

# Create API key
curl -X POST http://localhost:8080/v1/keys \
  -d '{"project_id": "proj_xxx", "role": "dev"}'
# Returns: {"secret": "vf_xxx...", "key_id": "..."}
```

## Reverse Proxy with TLS

### Option 1: Caddy (Automatic TLS)

```bash
# Edit deploy/Caddyfile - set your domain
# example.com {
#     reverse_proxy verifily:8080
# }

# Start with Caddy
docker-compose -f docker-compose.prod.yml --profile with-caddy up -d
```

Caddy automatically obtains Let's Encrypt certificates.

### Option 2: Nginx (Manual TLS)

```bash
# Place certificates in deploy/ssl/
cp cert.pem key.pem deploy/ssl/

# Edit deploy/nginx.conf - uncomment HTTPS server

# Start with Nginx
docker-compose -f docker-compose.prod.yml --profile with-nginx up -d
```

## Monitoring

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Basic liveness |
| `/ready` | Readiness (checks all subsystems) |
| `/metrics` | Prometheus metrics |
| `/v1/config` | Configuration (safe) |

### Logs

```bash
# View JSON logs
docker-compose -f docker-compose.prod.yml logs -f verifily

# Structured log format:
# {"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "..."}
```

### Log Aggregation Example (jq)

```bash
# Find all errors
docker-compose logs | jq 'select(.level == "ERROR")'

# Request latency histogram
docker-compose logs | jq -r '.request_time' | sort -n | uniq -c
```

## Security Checklist

- [ ] Changed default API key or enabled org mode
- [ ] Set `VERIFILY_PROD=1`
- [ ] Disabled docs in production (`VERIFILY_ENABLE_DOCS=0`)
- [ ] Enabled persistence for audit trails
- [ ] Using workspace-only paths (no absolute paths)
- [ ] Bound to localhost on host (127.0.0.1:8080)
- [ ] Using reverse proxy for external access
- [ ] TLS enabled (HTTPS)
- [ ] Rate limiting configured
- [ ] Log rotation configured

## Troubleshooting

### "Configuration errors"

```
Configuration errors:
  1. PROD=1 requires workspace: /workspace
     Create: mkdir -p /workspace
```

Fix: Create workspace directory
```bash
mkdir -p workspace
```

### "Path outside workspace"

API request used absolute path or path traversal. Use relative paths:
```json
{"config_path": "configs/default.yaml"}  ✓
{"config_path": "/etc/passwd"}            ✗
{"config_path": "../../../etc/passwd"}    ✗
```

### "Bind host not localhost"

In production, binding to 0.0.0.0 without allow_nonlocal triggers a warning.
Use a reverse proxy instead of exposing directly.

## Advanced Configuration

### Custom Workspace Path

```bash
# Host: /data/verifily/workspace
# Container: /workspace

docker-compose -f docker-compose.prod.yml up -d \
  -v /data/verifily/workspace:/workspace
```

### External Logging

```bash
# Forward to syslog
docker-compose -f docker-compose.prod.yml up -d \
  --log-driver=syslog \
  --log-opt syslog-address=udp://logs.example.com:514
```

### Database-less HA (simple)

For HA without complex infrastructure:
1. Run multiple instances behind a load balancer
2. Use shared NFS/S3 for workspace
3. Each instance has independent persistence
4. Reconcile usage logs periodically

## Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFILY_BIND` | 127.0.0.1 | Bind address |
| `VERIFILY_PORT` | 8080 | Server port |
| `VERIFILY_PROD` | 0 | Production mode |
| `VERIFILY_WORKSPACE_ROOT` | ./workspace | File workspace |
| `VERIFILY_API_KEY` | - | Legacy auth key |
| `VERIFILY_ORG_MODE` | 0 | Multi-tenant mode |
| `VERIFILY_LOG_FORMAT` | text | json or text |
| `VERIFILY_RATE_LIMIT_RPM` | - | Rate limit |

### Docker Compose Commands

```bash
# Start
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Restart
docker-compose -f docker-compose.prod.yml restart

# Stop
docker-compose -f docker-compose.prod.yml down

# Update
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```
