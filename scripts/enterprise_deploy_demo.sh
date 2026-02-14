#!/bin/bash
# Enterprise Deployment Demo for Verifily
# Shows runtime paths, backup/restore, and deployment checks

set -e

echo "=== Verifily Enterprise Deployment Demo ==="
echo

# Use temporary directory for demo
export VERIFILY_HOME=${VERIFILY_HOME:-/tmp/verifily-demo}
mkdir -p "$VERIFILY_HOME"

echo "VERIFILY_HOME: $VERIFILY_HOME"
echo

# --- 1. Runtime Paths Demo ---
echo "=== 1. Runtime Paths ==="
python3 -c "
from verifily_cli_v1.core.runtime_paths import get_runtime_paths
paths = get_runtime_paths()
for name, path in paths.describe_paths().items():
    print(f'  {name}: {path}')
"
echo

# --- 2. Create some mock data ---
echo "=== 2. Creating Mock Data ==="
python3 -c "
from verifily_cli_v1.core.runtime_paths import get_runtime_paths
paths = get_runtime_paths()
paths.ensure_directories()

# Usage events
usage = paths.get_usage_log()
with open(usage, 'w') as f:
    f.write('{\"timestamp\": \"2025-01-20T10:00:00Z\", \"user\": \"admin\", \"action\": \"login\"}\n')
    f.write('{\"timestamp\": \"2025-01-20T10:01:00Z\", \"user\": \"dev1\", \"action\": \"run_validation\"}\n')

# Jobs
jobs = paths.get_jobs_log()
with open(jobs, 'w') as f:
    f.write('{\"job_id\": \"job-001\", \"status\": \"completed\"}\n')

print(f'Created: {usage}')
print(f'Created: {jobs}')
"
echo

# --- 3. Backup Demo ---
echo "=== 3. Creating Backup ==="
BACKUP_FILE="$VERIFILY_HOME/backup-$(date +%Y%m%d-%H%M%S).tar.gz"
python3 -c "
from verifily_cli_v1.core.backup_restore import create_backup
from pathlib import Path

result = create_backup(Path('$BACKUP_FILE'))
print(f\"Files backed up: {result['files_backed_up']}\")
print(f\"Total bytes: {result['total_bytes']}\")
print(f\"Backup file: {result['backup_path']}\")
"
echo

# --- 4. Verify Backup ---
echo "=== 4. Verifying Backup ==="
python3 -c "
from verifily_cli_v1.core.backup_restore import verify_backup
from pathlib import Path

is_valid, manifest = verify_backup(Path('$BACKUP_FILE'))
print(f\"Valid: {is_valid}\")
print(f\"Manifest version: {manifest['version']}\")
print(f\"Files in backup: {len(manifest['files'])}\")
for f in manifest['files']:
    print(f\"  - {f['path']} ({f['size']} bytes)\")
"
echo

# --- 5. Restore Demo (force mode) ---
echo "=== 5. Simulating Restore ==="
python3 -c "
from verifily_cli_v1.core.runtime_paths import get_runtime_paths
from verifily_cli_v1.core.backup_restore import create_backup, restore_backup
from pathlib import Path

paths = get_runtime_paths()

# Add new data that would be overwritten
usage = paths.get_usage_log()
with open(usage, 'a') as f:
    f.write('{\"timestamp\": \"2025-01-20T11:00:00Z\", \"user\": \"newuser\", \"action\": \"new_event\"}\n')

print('Before restore:')
with open(usage) as f:
    lines = f.readlines()
    print(f'  Lines: {len(lines)}')

# Restore
result = restore_backup(Path('$BACKUP_FILE'), force=True)
print(f\"\nRestored: {result['files_restored']} files\")

print('After restore:')
with open(usage) as f:
    lines = f.readlines()
    print(f'  Lines: {len(lines)}')
"
echo

# --- 6. Deploy Config Demo ---
echo "=== 6. Deploy Config Loading ==="
python3 -c "
from verifily_cli_v1.core.deploy_config import DeployConfig, load_deploy_config

# Default config
config = DeployConfig()
print('Default config:')
print(f\"  Server: {config.server.host}:{config.server.port}\")
print(f\"  Auth enabled: {config.auth.enabled}\")
print(f\"  Log format: {config.server.log_format}\")
print()

# Production-like config
prod_config = DeployConfig(
    server=DeployConfig().server.__class__(host='0.0.0.0', port=8080, log_format='json', enable_docs=False),
    auth=DeployConfig().auth.__class__(enabled=True, api_key='test-api-key'),
    persistence=DeployConfig().persistence.__class__(usage=True, jobs=True)
)
print('Production-like config:')
print(f\"  Server: {prod_config.server.host}:{prod_config.server.port}\")
print(f\"  Auth enabled: {prod_config.auth.enabled}\")
print(f\"  Is production-like: {prod_config.is_production_like()}\")
"
echo

# --- 7. Validation Demo ---
echo "=== 7. Config Validation ==="
python3 -c "
from verifily_cli_v1.core.deploy_config import DeployConfig, ServerConfig, validate_deploy_config

# Valid config
config = DeployConfig()
is_valid, errors = validate_deploy_config(config)
print(f\"Valid config check: {is_valid}\")

# Invalid config (bad port)
bad_config = DeployConfig(server=ServerConfig(port=99999))
is_valid, errors = validate_deploy_config(bad_config)
print(f\"Invalid port check: {is_valid}\")
print(f\"Errors: {errors}\")
"
echo

# --- 8. Cleanup ---
echo "=== 8. Cleanup ==="
rm -rf "$VERIFILY_HOME"
echo "Removed: $VERIFILY_HOME"
echo

echo "=== Demo Complete ==="
echo
echo "Key Takeaways:"
echo "  - Runtime paths use VERIFILY_HOME (default: /tmp/verifily)"
echo "  - Backups include operational metadata (usage, jobs, workspaces)"
echo "  - Restore requires --force if files exist"
echo "  - Deploy config can be loaded from YAML or env vars"
echo "  - Config validation catches common errors"
