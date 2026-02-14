#!/usr/bin/env bash
# ── Billing v1 Demo ─────────────────────────────────────────────
# Demonstrates: plans, estimate, invoice, usage export
# Runtime target: < 10s
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
echo -e "${BOLD}║       Billing v1 — Demo Script           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo

# ── Ingest test data ─────────────────────────────────────────
echo -e "${YELLOW}Ingesting test data...${NC}"
TMPDIR=$(mktemp -d)
python3 -m verifily_cli_v1.commands.ingest \
    --input "$REPO_ROOT/examples/customer_drill/raw/support_tickets.csv" \
    --output "$TMPDIR" \
    --schema sft \
    --map question:subject answer:resolution context:body \
    --tag source:billing-demo \
    2>/dev/null
DS_PATH="$TMPDIR/dataset.jsonl"

# ── Start server ─────────────────────────────────────────────
echo -e "${YELLOW}Starting server with billing enabled...${NC}"

export VERIFILY_ENABLE_BILLING=1
export VERIFILY_BILLING_PERSIST=0
PORT=18799

python3 -c "
import uvicorn, os
os.environ['VERIFILY_ENABLE_BILLING'] = '1'
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
settings = load_settings(port=$PORT)
app = create_app(settings)
uvicorn.run(app, host='127.0.0.1', port=$PORT, log_level='error')
" &
SERVER_PID=$!

cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    rm -rf "$TMPDIR"
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

# ── 1. List plans ────────────────────────────────────────────
echo
echo -e "${BOLD}1. List plans${NC}"
RESP=$(curl -sf "$BASE/v1/billing/plans")
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['plans'])==4" 2>/dev/null
check "GET /v1/billing/plans → 4 plans" $?

# ── 2. Generate usage ───────────────────────────────────────
echo
echo -e "${BOLD}2. Generate usage${NC}"
curl -sf -X POST "$BASE/v1/report" \
    -H "Content-Type: application/json" \
    -d "{\"dataset_path\": \"$DS_PATH\", \"schema\": \"sft\"}" > /dev/null
check "POST /v1/report → usage recorded" $?

# ── 3. Estimate (FREE) ──────────────────────────────────────
echo
echo -e "${BOLD}3. Estimate (FREE plan)${NC}"
RESP=$(curl -sf "$BASE/v1/billing/estimate?plan=FREE")
TOTAL=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_cents'])")
[ "$TOTAL" = "0" ]
check "GET /v1/billing/estimate?plan=FREE → total=0" $?

# ── 4. Estimate (STARTER) ───────────────────────────────────
echo
echo -e "${BOLD}4. Estimate (STARTER plan)${NC}"
RESP=$(curl -sf "$BASE/v1/billing/estimate?plan=STARTER")
BASE_CENTS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['monthly_base_cents'])")
[ "$BASE_CENTS" = "9900" ]
check "GET /v1/billing/estimate?plan=STARTER → base=9900" $?

# ── 5. Generate invoice ─────────────────────────────────────
echo
echo -e "${BOLD}5. Generate invoice${NC}"
RESP=$(curl -sf -X POST "$BASE/v1/billing/invoice" \
    -H "Content-Type: application/json" \
    -d '{"plan":"STARTER","period_start":"2024-01-01T00:00:00+00:00","period_end":"2025-12-31T00:00:00+00:00"}')
INV_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['invoice_id'])")
echo "$INV_ID" | grep -q "^inv-"
check "POST /v1/billing/invoice → invoice_id starts with inv-" $?

LINES=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['lines']))")
[ "$LINES" = "4" ]
check "Invoice has 4 line items" $?

# ── 6. Export usage CSV ──────────────────────────────────────
echo
echo -e "${BOLD}6. Export usage (CSV)${NC}"
RESP=$(curl -sf "$BASE/v1/billing/usage_export?format=csv&period_days=30")
echo "$RESP" | head -1 | grep -q "date"
check "GET /v1/billing/usage_export?format=csv → has header" $?

CSV_ROWS=$(echo "$RESP" | wc -l | tr -d ' ')
[ "$CSV_ROWS" -ge 2 ]
check "CSV has data rows" $?

# ── 7. Export usage JSONL ────────────────────────────────────
echo
echo -e "${BOLD}7. Export usage (JSONL)${NC}"
RESP=$(curl -sf "$BASE/v1/billing/usage_export?format=jsonl&period_days=30")
echo "$RESP" | head -1 | python3 -c "import sys,json; json.loads(sys.stdin.readline())" 2>/dev/null
check "GET /v1/billing/usage_export?format=jsonl → valid JSON lines" $?

# ── 8. Billing events recorded ───────────────────────────────
echo
echo -e "${BOLD}8. Billing events${NC}"
RESP=$(curl -sf "$BASE/v1/billing/events")
EVT_COUNT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['total'])")
[ "$EVT_COUNT" -ge 1 ]
check "GET /v1/billing/events → events recorded" $?

# ── 9. No secrets in billing response ────────────────────────
echo
echo -e "${BOLD}9. Security checks${NC}"
RESP=$(curl -sf "$BASE/v1/billing/estimate?plan=STARTER")
echo "$RESP" | python3 -c "
import sys,json
d = json.load(sys.stdin)
s = json.dumps(d)
assert 'password' not in s.lower()
assert 'secret' not in s.lower()
" 2>/dev/null
check "No secrets in billing responses" $?

# ── 10. Invalid plan → 422 ──────────────────────────────────
echo
echo -e "${BOLD}10. Error handling${NC}"
HTTP=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/billing/estimate?plan=BOGUS" || true)
[ "$HTTP" = "422" ]
check "Unknown plan → 422" $?

# ── Summary ──────────────────────────────────────────────────
echo
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${BOLD}║  ${GREEN}BILLING v1 — ALL $PASSED PASSED${NC}${BOLD}              ║${NC}"
else
    echo -e "${BOLD}║  ${RED}BILLING v1 — $PASSED passed, $FAILED FAILED${NC}${BOLD}        ║${NC}"
fi
echo -e "${BOLD}║  estimate ✓  invoice ✓  export ✓        ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"

[ "$FAILED" -eq 0 ] || exit 1
