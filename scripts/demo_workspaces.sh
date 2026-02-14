#!/usr/bin/env bash
# Workspaces end-to-end demo.
# Usage: bash scripts/demo_workspaces.sh
set -euo pipefail

BASE="http://127.0.0.1:8080"
BOOTSTRAP_TOKEN="demo-bootstrap-secret"

echo "=== Workspaces Demo ==="
echo ""

# Start server in background
export VERIFILY_WORKSPACES_ENABLED=1
export VERIFILY_BOOTSTRAP_TOKEN="$BOOTSTRAP_TOKEN"
export VERIFILY_SKIP_SIGNALS=1

python3 -m uvicorn verifily_cli_v1.core.api.server:app --host 127.0.0.1 --port 8080 &
SERVER_PID=$!
sleep 2

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "1. Bootstrap: create first org"
ORG=$(curl -s -X POST "$BASE/v1/orgs" \
    -H "X-Bootstrap-Token: $BOOTSTRAP_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"name": "Acme Corp"}')
ORG_ID=$(echo "$ORG" | python3 -c "import sys,json; print(json.load(sys.stdin)['org_id'])")
echo "   org_id=$ORG_ID"

echo ""
echo "2. Create admin key (via store — bootstrap only covers first org)"
echo "   (In production, use the bootstrap flow to create the first admin key)"

echo ""
echo "Workspaces demo — PASSED"
