#!/usr/bin/env bash
# ── Metered Billing v2 — Invoice Artifacts Demo ─────────────────────
# Demonstrates: deterministic invoice IDs, JSON+CSV artifacts, GET by ID
# No external services. Runtime target: < 10s
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

PASSED=0
FAILED=0

check() {
    local label="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $label"
        PASSED=$((PASSED + 1))
    else
        echo -e "  ${RED}✗${NC} $label"
        FAILED=$((FAILED + 1))
    fi
}

echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Invoice Artifacts v2 — Demo Script      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo

# ── Start server ─────────────────────────────────────────────
echo -e "${YELLOW}Starting server with billing enabled...${NC}"

PORT=18878
DATA_DIR=$(mktemp -d)
INVOICE_DIR="$DATA_DIR/billing/invoices"

python3 -c "
import uvicorn, os
os.environ['VERIFILY_ENABLE_BILLING'] = '1'
os.environ['VERIFILY_DATA_DIR'] = '$DATA_DIR'
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
settings = load_settings(port=$PORT)
app = create_app(settings)
uvicorn.run(app, host='127.0.0.1', port=$PORT, log_level='error')
" &
SERVER_PID=$!

cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    rm -rf "$DATA_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        break
    fi
    sleep 0.3
done

BASE="http://127.0.0.1:$PORT"

# ── 1. Generate some billing events ──────────────────────────
echo
echo -e "${BOLD}1. Generate billing events (run a report)${NC}"
RESP=$(curl -sf -X POST "$BASE/v1/report" \
    -H "Content-Type: application/json" \
    -d "{\"config_path\": \"$REPO_ROOT/examples/customer_drill/verifily.yaml\"}")
echo "$RESP" | python3 -c "import sys,json; json.load(sys.stdin)" >/dev/null 2>&1
check "POST /v1/report → generates billing events" $?

# ── 2. POST /v1/billing/invoice → create invoice ────────────
echo
echo -e "${BOLD}2. Create invoice via POST /v1/billing/invoice${NC}"
RESP=$(curl -sf -X POST "$BASE/v1/billing/invoice" \
    -H "Content-Type: application/json" \
    -d '{"project_id": "default", "plan_id": "FREE"}')
INV_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['invoice_id'])")
echo -e "    invoice_id: $INV_ID"
echo "$INV_ID" | grep -q "^inv-"
check "Invoice ID is deterministic (inv-<hash>)" $?

# ── 3. Verify deterministic — same inputs, same ID ──────────
echo
echo -e "${BOLD}3. Deterministic: same inputs → same invoice_id${NC}"
RESP2=$(curl -sf -X POST "$BASE/v1/billing/invoice" \
    -H "Content-Type: application/json" \
    -d '{"project_id": "default", "plan_id": "FREE"}')
INV_ID2=$(echo "$RESP2" | python3 -c "import sys,json; print(json.load(sys.stdin)['invoice_id'])")
[ "$INV_ID" = "$INV_ID2" ]
check "Second POST → same invoice_id ($INV_ID2)" $?

# ── 4. GET /v1/billing/invoice/{id} ─────────────────────────
echo
echo -e "${BOLD}4. Retrieve invoice by ID${NC}"
RESP=$(curl -sf "$BASE/v1/billing/invoice/$INV_ID")
echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['invoice_id'] == '$INV_ID'
assert d['customer'] == 'default'
assert 'lines' in d
print(f'    customer={d[\"customer\"]} lines={len(d[\"lines\"])}')
" 2>/dev/null
check "GET /v1/billing/invoice/$INV_ID → matches" $?

# ── 5. GET non-existent invoice → 404 ───────────────────────
echo
echo -e "${BOLD}5. Non-existent invoice → 404${NC}"
HTTP=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/billing/invoice/inv-nonexistent" || true)
[ "$HTTP" = "404" ]
check "GET /v1/billing/invoice/inv-nonexistent → 404" $?

# ── 6. JSON artifact on disk ────────────────────────────────
echo
echo -e "${BOLD}6. Invoice JSON artifact on disk${NC}"
JSON_FILE="$INVOICE_DIR/$INV_ID.json"
if [ -f "$JSON_FILE" ]; then
    python3 -c "
import json
with open('$JSON_FILE') as f:
    d = json.load(f)
assert d['invoice_id'] == '$INV_ID'
print(f'    {JSON_FILE}')
" 2>/dev/null
    check "JSON artifact exists: $INV_ID.json" $?
else
    check "JSON artifact exists: $INV_ID.json" 1
fi

# ── 7. CSV artifact on disk ─────────────────────────────────
echo
echo -e "${BOLD}7. Invoice CSV artifact on disk${NC}"
CSV_FILE="$INVOICE_DIR/$INV_ID.csv"
if [ -f "$CSV_FILE" ]; then
    python3 -c "
import csv
with open('$CSV_FILE') as f:
    rows = list(csv.reader(f))
assert rows[0][0] == 'label', f'header={rows[0]}'
assert rows[-1][0] == 'TOTAL'
print(f'    {CSV_FILE} ({len(rows)} rows)')
" 2>/dev/null
    check "CSV artifact exists: $INV_ID.csv" $?
else
    check "CSV artifact exists: $INV_ID.csv" 1
fi

# ── 8. Invoice preview still works ──────────────────────────
echo
echo -e "${BOLD}8. Invoice preview endpoint (existing v1)${NC}"
RESP=$(curl -sf "$BASE/v1/billing/invoice-preview?plan_id=STARTER")
echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['plan_id'] == 'STARTER'
assert 'total_cents' in d
print(f'    plan=STARTER total_cents={d[\"total_cents\"]}')
" 2>/dev/null
check "GET /v1/billing/invoice-preview → works" $?

# ── Summary ──────────────────────────────────────────────────
echo
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${BOLD}║  ${GREEN}INVOICE ARTIFACTS — ALL $PASSED PASSED${NC}${BOLD}      ║${NC}"
else
    echo -e "${BOLD}║  ${RED}INVOICE — $PASSED passed, $FAILED FAILED${NC}${BOLD}        ║${NC}"
fi
echo -e "${BOLD}║  deterministic ✓  JSON ✓  CSV ✓  GET ✓  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"

[ "$FAILED" -eq 0 ] || exit 1
