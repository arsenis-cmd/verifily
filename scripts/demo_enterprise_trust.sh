#!/usr/bin/env bash
# demo_enterprise_trust.sh — Demonstrate Enterprise Trust v1 features.
#
# Starts server in advanced auth mode, creates projects and scoped keys,
# tests scope/project enforcement and quota limits.
# Runtime target: <8 seconds. No network, no GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
PORT=8099
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="/tmp/verifily_enterprise_trust"

cd "$REPO_ROOT"

echo "Enterprise Trust v1 Demo"
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

# ── Start server in advanced auth mode ──────────────────────────
VERIFILY_AUTH_MODE=advanced \
VERIFILY_KEY_SALT=demo-salt-2025 \
VERIFILY_ENABLE_ADMIN=1 \
VERIFILY_ALLOW_NONLOCAL=1 \
VERIFILY_QUOTA_REQ_PER_DAY=5 \
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

# ── 1. Bootstrap: create project (no auth needed) ──────────────
echo ""
echo "-- Bootstrap mode (no admin keys yet) --"
PROJ_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/projects" \
  -H "Content-Type: application/json" \
  -d '{"id": "demo", "name": "Demo Project"}')
check "Create project (bootstrap)" "$([ "$PROJ_CODE" = "201" ] && echo true || echo false)"

# ── 2. Create admin key ────────────────────────────────────────
ADMIN_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/keys" \
  -H "Content-Type: application/json" \
  -d '{"id": "k-admin", "name": "Admin Key", "raw_key": "sk-admin-secret-key-1234", "scopes": ["admin:write", "pipeline:run", "report:read", "usage:read"], "projects_allowed": ["*"]}')
check "Create admin key" "$([ "$ADMIN_CODE" = "201" ] && echo true || echo false)"

# ── 3. Create demo key (scoped to project "demo") ──────────────
DEMO_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/admin/keys" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-admin-secret-key-1234" \
  -d '{"id": "k-demo", "name": "Demo Key", "raw_key": "sk-demo-secret-key-1234", "scopes": ["pipeline:run", "report:read", "usage:read"], "projects_allowed": ["demo"]}')
check "Create demo key (scoped)" "$([ "$DEMO_CODE" = "201" ] && echo true || echo false)"

# ── 4. Bootstrap closed — unauth admin fails ───────────────────
echo ""
echo "-- Auth enforcement --"
UNAUTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/admin/projects")
check "Unauth admin rejected (bootstrap closed)" "$([ "$UNAUTH_CODE" = "401" ] && echo true || echo false)"

# ── 5. Demo key + wrong project → 403 ──────────────────────────
WRONG_PROJ=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/usage" \
  -H "Authorization: Bearer sk-demo-secret-key-1234" \
  -H "X-Project-ID: richemont")
check "Wrong project rejected (403)" "$([ "$WRONG_PROJ" = "403" ] && echo true || echo false)"

# ── 6. Demo key + correct project → 200 ────────────────────────
RIGHT_PROJ=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/usage" \
  -H "Authorization: Bearer sk-demo-secret-key-1234" \
  -H "X-Project-ID: demo")
check "Correct project allowed (200)" "$([ "$RIGHT_PROJ" = "200" ] && echo true || echo false)"

# ── 7. Scope enforcement: demo key lacks admin:write ────────────
SCOPE_FAIL=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/admin/projects" \
  -H "Authorization: Bearer sk-demo-secret-key-1234")
check "Scope enforcement (403)" "$([ "$SCOPE_FAIL" = "403" ] && echo true || echo false)"

# ── 8. Report with demo key ────────────────────────────────────
echo ""
echo "-- API operations with scoped key --"
REPORT_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/report" \
  -H "Authorization: Bearer sk-demo-secret-key-1234" \
  -H "Content-Type: application/json" \
  -H "X-Project-ID: demo" \
  -d "{\"dataset_path\": \"$TRAIN_DATA\", \"schema\": \"sft\"}")
check "Report with scoped key" "$([ "$REPORT_CODE" = "200" ] && echo true || echo false)"

# ── 9. Quota enforcement (limit = 5 req/day) ───────────────────
echo ""
echo "-- Quota enforcement (5 req/day limit) --"
# We've already used some requests. Fire more to hit the limit.
for i in $(seq 1 10); do
    QCODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/v1/usage" \
      -H "Authorization: Bearer sk-demo-secret-key-1234" \
      -H "X-Project-ID: demo")
    if [ "$QCODE" = "429" ]; then
        check "Quota exceeded (429 on request $i)" "true"
        break
    fi
done

if [ "$QCODE" != "429" ]; then
    check "Quota exceeded (expected 429)" "false"
fi

# Verify the 429 body
QUOTA_BODY=$(curl -s "$BASE_URL/v1/usage" \
  -H "Authorization: Bearer sk-demo-secret-key-1234" \
  -H "X-Project-ID: demo")
QUOTA_TYPE=$(echo "$QUOTA_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['error']['type'])" 2>/dev/null || echo "")
check "Quota error type=QUOTA_EXCEEDED" "$([ "$QUOTA_TYPE" = "QUOTA_EXCEEDED" ] && echo true || echo false)"

# ── 10. Key list shows redacted hashes ──────────────────────────
echo ""
echo "-- Admin key management --"
KEYS_BODY=$(curl -s "$BASE_URL/v1/admin/keys" \
  -H "Authorization: Bearer sk-admin-secret-key-1234")
NO_LEAK=$(echo "$KEYS_BODY" | python3 -c "
import sys, json
raw = json.dumps(json.load(sys.stdin))
# No raw key or real hash should appear in the response
print('true' if 'sk-admin-secret-key' not in raw and 'sk-demo-secret-key' not in raw else 'false')
" 2>/dev/null || echo "false")
check "Key list has no secret leaks" "$NO_LEAK"

# ── Summary ─────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "Enterprise Trust demo — ALL PASSED ($PASS/$PASS)"
    exit 0
else
    echo "Enterprise Trust demo — FAILED ($FAIL failures)"
    exit 1
fi
