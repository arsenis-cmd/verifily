#!/usr/bin/env bash
# demo_teams_rbac.sh — Demonstrate Teams v1: orgs, users, and RBAC.
#
# Starts server in teams mode, creates org/user/project/key,
# tests scope and project enforcement.
# Runtime target: <8 seconds. No network, no GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
PORT=8099
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="/tmp/verifily_teams_demo"
SUPER_KEY="sk-super-admin-demo-key"

cd "$REPO_ROOT"

echo "Teams v1 RBAC Demo"
echo ""

# ── Prep: ingest demo data ──────────────────────────────────────
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

$CLI ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --schema sft \
  --map question:subject \
  --map answer:resolution \
  --map context:body \
  --tag source:demo \
  --out "$WORK_DIR/datasets/train_artifact" >/dev/null 2>&1

TRAIN_DATA="$WORK_DIR/datasets/train_artifact/dataset.jsonl"

# ── Start server in teams mode ────────────────────────────────
VERIFILY_TEAMS_ENABLED=1 \
VERIFILY_AUTH_MODE=teams \
VERIFILY_SUPER_ADMIN_KEY="$SUPER_KEY" \
VERIFILY_KEY_SALT="demo-salt" \
VERIFILY_ALLOW_NONLOCAL=1 \
$CLI serve --port $PORT >"$WORK_DIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server ready (max 5s)
for i in $(seq 1 50); do
    if curl -s "$BASE_URL/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

PASS=0
FAIL=0

check() {
    local label="$1"
    local ok="$2"
    if [ "$ok" = "true" ]; then
        echo "  [PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL + 1))
    fi
}

ADMIN_HEADERS="-H \"Authorization: Bearer $SUPER_KEY\""

# ── 1. Create org ─────────────────────────────────────────────
echo ""
echo "-- Admin setup --"
ORG_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/orgs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{"id": "org-acme", "name": "Acme Corp"}')
check "Create org (201)" "$([ "$ORG_CODE" = "201" ] && echo true || echo false)"

# ── 2. Create user ────────────────────────────────────────────
USER_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/users" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{"id": "user-alice", "email": "alice@acme.com", "name": "Alice"}')
check "Create user (201)" "$([ "$USER_CODE" = "201" ] && echo true || echo false)"

# ── 3. Add membership ────────────────────────────────────────
MEM_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/memberships" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{"user_id": "user-alice", "org_id": "org-acme", "role": "admin"}')
check "Add membership (201)" "$([ "$MEM_CODE" = "201" ] && echo true || echo false)"

# ── 4. Create project ────────────────────────────────────────
PROJ_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/team-projects" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{"id": "proj-prod", "org_id": "org-acme", "name": "Production"}')
check "Create project (201)" "$([ "$PROJ_CODE" = "201" ] && echo true || echo false)"

# ── 5. Issue API key with run:write ───────────────────────────
KEY_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/api-keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{
    "id": "key-prod", "org_id": "org-acme", "name": "Prod Key",
    "raw_key": "sk-prod-key-demo-12345",
    "scopes": ["run:write", "run:read"],
    "project_ids": ["proj-prod"],
    "created_by": "user-alice"
  }')
check "Issue API key (201)" "$([ "$KEY_CODE" = "201" ] && echo true || echo false)"

# ── 6. Whoami returns correct identity ────────────────────────
echo ""
echo "-- Identity --"
WHOAMI_BODY=$(curl -s "$BASE_URL/v1/admin/whoami" \
  -H "Authorization: Bearer $SUPER_KEY")
WHOAMI_ID=$(echo "$WHOAMI_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key_id'])" 2>/dev/null || echo "")
check "Whoami returns super-admin" "$([ "$WHOAMI_ID" = "super-admin" ] && echo true || echo false)"

# ── 7. Report succeeds with valid key ─────────────────────────
echo ""
echo "-- Scope & project enforcement --"
REPORT_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/report" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-prod-key-demo-12345" \
  -H "X-Project-ID: proj-prod" \
  -d "{\"dataset_path\": \"$TRAIN_DATA\", \"schema\": \"sft\"}")
check "Report with valid key (200)" "$([ "$REPORT_CODE" = "200" ] && echo true || echo false)"

# ── 8. Report without key -> 401 ──────────────────────────────
NO_KEY_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/report" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_path\": \"$TRAIN_DATA\", \"schema\": \"sft\"}")
check "Report without key (401)" "$([ "$NO_KEY_CODE" = "401" ] && echo true || echo false)"

# ── 9. Report with wrong project -> 403 ──────────────────────
WRONG_PROJ_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/report" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-prod-key-demo-12345" \
  -H "X-Project-ID: proj-other" \
  -d "{\"dataset_path\": \"$TRAIN_DATA\", \"schema\": \"sft\"}")
check "Wrong project (403)" "$([ "$WRONG_PROJ_CODE" = "403" ] && echo true || echo false)"

# ── 10. Issue read-only key, POST report -> 403 ──────────────
curl -s -o /dev/null -X POST "$BASE_URL/v1/admin/api-keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{
    "id": "key-readonly", "org_id": "org-acme", "name": "ReadOnly Key",
    "raw_key": "sk-readonly-key-demo-12345",
    "scopes": ["run:read"],
    "project_ids": ["*"],
    "created_by": "user-alice"
  }'

READONLY_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/report" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-readonly-key-demo-12345" \
  -d "{\"dataset_path\": \"$TRAIN_DATA\", \"schema\": \"sft\"}")
check "ReadOnly key -> report 403" "$([ "$READONLY_CODE" = "403" ] && echo true || echo false)"

# ── 11. Admin without super key -> 401 ───────────────────────
echo ""
echo "-- Edge cases --"
UNAUTH_ADMIN=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/orgs" \
  -H "Content-Type: application/json" \
  -d '{"id": "org-bad", "name": "Bad"}')
check "Admin without key (401)" "$([ "$UNAUTH_ADMIN" = "401" ] && echo true || echo false)"

# ── 12. Invalid scope on key issue -> 422 ────────────────────
BAD_SCOPE_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/api-keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SUPER_KEY" \
  -d '{
    "id": "key-bad", "org_id": "org-acme", "name": "Bad Key",
    "raw_key": "sk-bad-key-demo-1234567",
    "scopes": ["bogus:scope"],
    "project_ids": ["*"],
    "created_by": "user-alice"
  }')
check "Invalid scope (422)" "$([ "$BAD_SCOPE_CODE" = "422" ] && echo true || echo false)"

# ── Summary ─────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "Teams RBAC demo — ALL PASSED ($PASS/$PASS)"
    exit 0
else
    echo "Teams RBAC demo — FAILED ($FAIL failures)"
    exit 1
fi
