# Enterprise Deployment Guide

This guide covers deploying Verifily in enterprise/production environments with standardized paths, unified configuration, backup/restore, and deployment validation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Runtime Paths](#runtime-paths)
3. [Deployment Configuration](#deployment-configuration)
4. [Backup and Restore](#backup-and-restore)
5. [Deployment Validation](#deployment-validation)
6. [Production Checklist](#production-checklist)

---

## Quick Start

```bash
# Set deployment home (default: /tmp/verifily)
export VERIFILY_HOME=/var/lib/verifily

# Load config from file
export VERIFILY_CONFIG=/etc/verifily/config.yaml

# Start server
verifily server --prod
```

---

## Runtime Paths

Verifily uses standardized runtime directories under `VERIFILY_HOME`:

| Directory | Purpose | Override Env Var |
|-----------|---------|------------------|
| `store/` | Operational metadata (usage, jobs, workspaces) | - |
| `logs/` | Application logs | `VERIFILY_LOGS_DIR` |
| `runs/` | Job run outputs | `VERIFILY_RUNS_DIR` |
| `temp/` | Temporary files | `VERIFILY_TEMP_DIR` |

### Usage

```python
from verifily_cli_v1.core.runtime_paths import get_runtime_paths

paths = get_runtime_paths()
paths.ensure_directories()  # Creates all dirs

# Get specific paths
usage_log = paths.get_usage_log()
jobs_log = paths.get_jobs_log()
```

### Python API

```python
from verifily_cli_v1.core.runtime_paths import (
    get_runtime_paths,
    get_verifily_home,
    get_store_dir,
    get_logs_dir,
)

# Reset singleton (useful in tests)
from verifily_cli_v1.core.runtime_paths import reset_runtime_paths
reset_runtime_paths()
```

---

## Deployment Configuration

Verifily supports a unified YAML configuration file with environment variable overrides (Helm-lite pattern).

### Configuration File

Create `/etc/verifily/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  log_format: "json"        # or "text"
  enable_docs: false        # disable Swagger UI in prod
  allow_nonlocal: true      # required for 0.0.0.0

auth:
  enabled: true
  api_key: null             # use VERIFILY_API_KEY env var
  workspaces_enabled: true
  bootstrap_token: null     # use VERIFILY_BOOTSTRAP_TOKEN env var
  key_salt: null            # use VERIFILY_KEY_SALT env var

persistence:
  usage: true               # persist usage events
  jobs: true                # persist job events
  workspaces: true          # persist workspace metadata

limits:
  rate_limit_rpm: 120
```

### Environment Variables

Environment variables override file configuration:

| Variable | Config Path |
|----------|-------------|
| `VERIFILY_BIND` | `server.host` |
| `VERIFILY_PORT` | `server.port` |
| `VERIFILY_LOG_FORMAT` | `server.log_format` |
| `VERIFILY_API_KEY` | `auth.api_key` |
| `VERIFILY_KEY_SALT` | `auth.key_salt` |
| `VERIFILY_BOOTSTRAP_TOKEN` | `auth.bootstrap_token` |
| `VERIFILY_RATE_LIMIT_RPM` | `limits.rate_limit_rpm` |

### Python API

```python
from verifily_cli_v1.core.deploy_config import (
    DeployConfig,
    load_deploy_config,
    validate_deploy_config,
)

# Load from file or defaults
config = load_deploy_config("/etc/verifily/config.yaml")

# Validate
is_valid, errors = validate_deploy_config(config)
if not is_valid:
    for error in errors:
        print(f"Config error: {error}")

# Access settings
print(f"Server: {config.server.host}:{config.server.port}")
print(f"Auth: {config.auth.enabled}")

# Get redacted version for logging
safe_config = config.as_safe_dict()  # Secrets masked as "***"
```

---

## Backup and Restore

Backups include operational metadata (usage logs, job events, workspace metadata) but **exclude raw datasets/runs** by default to keep backup sizes manageable.

### CLI Commands

```bash
# Create backup (metadata only)
verifily backup --out backup-$(date +%Y%m%d).tar.gz

# Include datasets/runs (WARNING: may be large)
verifily backup --out full-backup.tar.gz --include-data

# Verify backup
verifily backup --verify full-backup.tar.gz

# Restore (requires --force if files exist)
verifily restore --file backup.tar.gz --force
```

### Python API

```python
from pathlib import Path
from verifily_cli_v1.core.backup_restore import (
    create_backup,
    restore_backup,
    verify_backup,
)

# Create backup
result = create_backup(Path("/backups/verifily.tar.gz"))
print(f"Backed up {result['files_backed_up']} files")

# Verify
is_valid, manifest = verify_backup(Path("/backups/verifily.tar.gz"))

# Restore
result = restore_backup(Path("/backups/verifily.tar.gz"), force=True)
```

### Backup Manifest Format

```json
{
  "version": "1.0",
  "created_at": "2025-01-20T12:00:00Z",
  "verifily_version": "1.0.0",
  "files": [
    {
      "path": "store/usage_events.jsonl",
      "sha256": "abc123...",
      "size": 1024
    }
  ]
}
```

---

## Deployment Validation

### Doctor Command

```bash
# Standard checks
verifily doctor

# Deployment-specific checks
verifily doctor --deploy

# With custom config
verifily doctor --deploy --config /etc/verifily/config.yaml
```

### Readiness Endpoint

The server exposes a `/ready` endpoint that returns:

- **200 OK**: All checks passed
- **503 Service Unavailable**: Persistence enabled but directories not writable

```bash
curl http://localhost:8080/ready
```

Response:
```json
{
  "status": "ready",
  "checks": {
    "paths_writable": true,
    "persistence_ready": true
  }
}
```

---

## Production Checklist

### Configuration

- [ ] Set `VERIFILY_HOME` outside `/tmp` (persistent storage)
- [ ] Disable Swagger UI (`server.enable_docs: false`)
- [ ] Enable authentication (`auth.enabled: true`)
- [ ] Set API key or enable workspaces
- [ ] Configure `key_salt` for workspaces (generate with `openssl rand -hex 32`)
- [ ] Set log format to `json` for structured logging
- [ ] Configure rate limits appropriate for your workload

### Paths

- [ ] Verify `VERIFILY_HOME` is writable
- [ ] Ensure adequate disk space for logs and persistence
- [ ] Consider separate volumes for `store/` and `logs/`

### Backup

- [ ] Schedule regular backups (operational metadata only)
- [ ] Test restore procedure
- [ ] Store backups off-site
- [ ] Document recovery procedures

### Security

- [ ] Run behind reverse proxy (Caddy/Nginx)
- [ ] Enable TLS termination at proxy
- [ ] Rotate API keys regularly
- [ ] Monitor authentication logs

### Example Production Config

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  log_format: "json"
  enable_docs: false
  allow_nonlocal: true

auth:
  enabled: true
  workspaces_enabled: true
  key_salt: "${VERIFILY_KEY_SALT}"
  bootstrap_token: "${VERIFILY_BOOTSTRAP_TOKEN}"

persistence:
  usage: true
  jobs: true
  workspaces: true

limits:
  rate_limit_rpm: 120
```

Environment:
```bash
export VERIFILY_HOME=/var/lib/verifily
export VERIFILY_CONFIG=/etc/verifily/config.yaml
export VERIFILY_KEY_SALT=$(openssl rand -hex 32)
export VERIFILY_BOOTSTRAP_TOKEN=$(openssl rand -hex 16)
```

---

## See Also

- `scripts/enterprise_deploy_demo.sh` - Interactive demo
- `verifily_cli_v1/tests/test_deploy_enterprise.py` - Test examples
