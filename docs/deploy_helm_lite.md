# Verifily Helm-Lite Deployment Guide

Deploy Verifily in production without Kubernetes - a "Helm-lite" approach using Docker Compose with conventions that map to K8s later.

## Quick Start (10 minutes)

### 1. Installation

```bash
# Clone repository
git clone <verifily-repo>
cd verifily

# Create required directories
mkdir -p /opt/verifily/{store,logs,runs}
```

### 2. Configuration

Create `/opt/verifily/verifily.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  log_format: "json"
  enable_docs: false

auth:
  enabled: true
  workspaces_enabled: true
  key_salt: "${VERIFILY_KEY_SALT}"  # Set via env var

persistence:
  usage: true
  jobs: true
  monitor: true
  workspaces: true

limits:
  rate_limit_rpm: 120
```

Create `/opt/verifily/.env`:

```bash
VERIFILY_HOME=/opt/verifily
VERIFILY_CONFIG_PATH=/opt/verifily/verifily.yaml
VERIFILY_KEY_SALT=$(openssl rand -hex 32)
```

### 3. Run with Docker

```bash
docker run -d \
  --name verifily \
  --env-file /opt/verifily/.env \
  -v /opt/verifily:/opt/verifily \
  -p 127.0.0.1:8080:8080 \
  verifily:prod
```

### 4. Verify

```bash
# Check readiness
curl http://localhost:8080/ready

# Expected: {"status": "ready", ...}

# View config (safe, no secrets)
curl http://localhost:8080/v1/config
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFILY_HOME` | `/tmp/verifily` | Base directory for all data |
| `VERIFILY_CONFIG_PATH` | - | Path to YAML config file |
| `VERIFILY_KEY_SALT` | - | Salt for key hashing (workspaces) |
| `VERIFILY_BIND` | `127.0.0.1` | Server bind address |
| `VERIFILY_PORT` | `8080` | Server port |
| `VERIFILY_LOG_FORMAT` | `text` | `text` or `json` |
| `VERIFILY_RATE_LIMIT_RPM` | `0` | Rate limit (0 = unlimited) |

## Directory Structure

```
VERIFILY_HOME/
├── store/              # Persistent JSONL files
│   ├── usage_events.jsonl
│   ├── jobs_events.jsonl
│   ├── monitor_events.jsonl
│   └── workspaces.jsonl
├── logs/               # Application logs
├── runs/               # Pipeline run artifacts
└── temp/               # Temporary files
```

## Backup and Restore

### Create Backup

```bash
# Create backup
verifily backup --out /backups/verifily_$(date +%Y%m%d).tar.gz

# Output:
# ✓ Backup created
#   Path: /backups/verifily_20240115.tar.gz
#   Files: 3
#   Size: 45,230 bytes (compressed: 12,456)
```

### Verify Backup

```bash
# List contents
tar -tzf /backups/verifily_20240115.tar.gz

# Contains:
# - store/usage_events.jsonl
# - store/jobs_events.jsonl
# - store/workspaces.jsonl
# - backup_manifest.json
```

### Restore

```bash
# Restore (requires --force if files exist)
verifily restore --file /backups/verifily_20240115.tar.gz --force

# Verify restoration
curl http://localhost:8080/v1/usage
```

## Deployment Checklist

- [ ] Set `VERIFILY_HOME` to persistent location (not /tmp)
- [ ] Generate and set `VERIFILY_KEY_SALT`
- [ ] Enable required persistence flags
- [ ] Set rate limits appropriate for your load
- [ ] Configure log rotation for JSON logs
- [ ] Set up backup schedule
- [ ] Configure reverse proxy (nginx/Caddy) for TLS

## Docker Compose Example

```yaml
version: "3.8"

services:
  verifily:
    image: verifily:prod
    container_name: verifily
    restart: unless-stopped
    env_file: .env
    volumes:
      - /opt/verifily:/opt/verifily
    ports:
      - "127.0.0.1:8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/ready"]
      interval: 30s
      timeout: 10s

  # Optional: Caddy for TLS
  caddy:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
    depends_on:
      - verifily

volumes:
  caddy_data:
```

## Migration to Kubernetes

When ready for K8s:

1. **ConfigMap**: Convert `verifily.yaml` to ConfigMap
2. **Secrets**: Move `VERIFILY_KEY_SALT` to K8s Secret
3. **PVC**: Map `VERIFILY_HOME` to PersistentVolumeClaim
4. **Deployment**: Use Docker image with health checks
5. **Service**: Expose port 8080
6. **Ingress**: Replace Caddy with Ingress + cert-manager

The Helm-lite conventions map directly:
- `VERIFILY_HOME` → PVC mount path
- Config file → ConfigMap
- Env vars → Secret + Deployment env

## Troubleshooting

### "Configuration errors" on startup

```
Configuration errors:
  1. auth.key_salt required for workspaces
```

Fix: Generate salt: `export VERIFILY_KEY_SALT=$(openssl rand -hex 32)`

### "/ready returns 503"

Check: `docker logs verifily`

Common causes:
- Persistence path not writable
- Port already in use
- Invalid config file syntax

### "Path outside workspace" errors

All file operations must use relative paths within `VERIFILY_HOME`.

## Security Considerations

- Never commit `VERIFILY_KEY_SALT` to version control
- Use secrets management (Vault, AWS Secrets Manager, etc.)
- Enable TLS via reverse proxy
- Use network policies to restrict access
- Regular backups tested via restore

## Support

- Health: `GET /health`
- Ready: `GET /ready`
- Config: `GET /v1/config`
- Doctor: `verifily doctor --deploy`
