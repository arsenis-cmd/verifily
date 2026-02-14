#!/usr/bin/env bash
# ── Stripe Stub v1 Demo ─────────────────────────────────────────
# Demonstrates: checkout, webhooks, subscription state, soft gating
# No real Stripe calls -- uses mock mode.
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
echo -e "${BOLD}║     Stripe Stub v1 — Demo Script         ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo

# ── Start server ─────────────────────────────────────────────
echo -e "${YELLOW}Starting server with Stripe stub enabled...${NC}"

PORT=18899

export VERIFILY_STRIPE_ENABLED=1
export VERIFILY_ENABLE_BILLING=1
export VERIFILY_BILLING_ENFORCE=1
export STRIPE_SECRET_KEY=sk_test_mock_key_for_demo
export STRIPE_WEBHOOK_SECRET=whsec_mock_secret_for_demo
export STRIPE_PRICE_ID_PRO=price_mock_pro_for_demo

python3 -c "
import uvicorn, os
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.billing.stripe import MockStripeClient
settings = load_settings(port=$PORT)
app = create_app(settings)
app.state._stripe_client = MockStripeClient()
uvicorn.run(app, host='127.0.0.1', port=$PORT, log_level='error')
" &
SERVER_PID=$!

cleanup() {
    kill $SERVER_PID 2>/dev/null || true
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

# ── 1. Checkout creates session ──────────────────────────────
echo
echo -e "${BOLD}1. Start checkout${NC}"
RESP=$(curl -sf -X POST "$BASE/v1/billing/checkout" \
    -H "Content-Type: application/json" \
    -d '{"plan": "pro", "project_id": "proj-demo", "org_id": "org-demo"}')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'checkout_url' in d" 2>/dev/null
check "POST /v1/billing/checkout → checkout_url returned" $?

CHECKOUT_URL=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['checkout_url'])")
echo -e "    URL: $CHECKOUT_URL"

# ── 2. Subscription is INCOMPLETE before webhook ─────────────
echo
echo -e "${BOLD}2. Subscription before webhook${NC}"
RESP=$(curl -sf "$BASE/v1/billing/subscription?project_id=proj-demo")
STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
[ "$STATUS" = "incomplete" ]
check "GET /v1/billing/subscription → status=incomplete" $?

# ── 3. Gated endpoint returns 402 (no active sub) ───────────
echo
echo -e "${BOLD}3. Pro feature gating (no subscription)${NC}"
HTTP=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/billing/usage_export?format=csv&period_days=30" || true)
[ "$HTTP" = "402" ]
check "GET /v1/billing/usage_export → 402 PAYMENT_REQUIRED" $?

# ── 4. Simulate checkout.session.completed webhook ───────────
echo
echo -e "${BOLD}4. Webhook: checkout.session.completed${NC}"
RESP=$(curl -sf -X POST "$BASE/v1/billing/webhook" \
    -H "Content-Type: application/json" \
    -H "stripe-signature: valid" \
    -d '{
        "type": "checkout.session.completed",
        "data": {"object": {
            "customer": "cus_demo_123",
            "subscription": "sub_demo_456",
            "metadata": {"org_id": "org-demo", "project_id": "proj-demo"}
        }}
    }')
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('received') == True" 2>/dev/null
check "POST /v1/billing/webhook → received=true" $?

# ── 5. Subscription is now ACTIVE ────────────────────────────
echo
echo -e "${BOLD}5. Subscription after webhook${NC}"
RESP=$(curl -sf "$BASE/v1/billing/subscription?project_id=proj-demo")
STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
[ "$STATUS" = "active" ]
check "GET /v1/billing/subscription → status=active" $?

# ── 6. Gated endpoint now succeeds ──────────────────────────
echo
echo -e "${BOLD}6. Pro feature gating (active subscription)${NC}"
# Need to use the same org/project -- default org_id/project_id from request.state
# Since we don't have auth, request.state.org_id = "default" and project_id = "default"
# Let's activate for default/default too
curl -sf -X POST "$BASE/v1/billing/webhook" \
    -H "Content-Type: application/json" \
    -H "stripe-signature: valid" \
    -d '{
        "type": "checkout.session.completed",
        "data": {"object": {
            "customer": "cus_default",
            "subscription": "sub_default",
            "metadata": {"org_id": "default", "project_id": "default"}
        }}
    }' > /dev/null

HTTP=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/billing/usage_export?format=csv&period_days=30")
[ "$HTTP" = "200" ]
check "GET /v1/billing/usage_export → 200 (active sub)" $?

# ── 7. Simulate subscription deleted webhook ─────────────────
echo
echo -e "${BOLD}7. Webhook: customer.subscription.deleted${NC}"
curl -sf -X POST "$BASE/v1/billing/webhook" \
    -H "Content-Type: application/json" \
    -H "stripe-signature: valid" \
    -d '{
        "type": "customer.subscription.deleted",
        "data": {"object": {
            "customer": "cus_default",
            "subscription": "sub_default",
            "metadata": {"org_id": "default", "project_id": "default"}
        }}
    }' > /dev/null

RESP=$(curl -sf "$BASE/v1/billing/subscription?project_id=default")
STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
[ "$STATUS" = "canceled" ]
check "Subscription canceled after webhook" $?

# ── 8. Gated endpoint returns 402 again ──────────────────────
echo
echo -e "${BOLD}8. Pro feature gating (canceled)${NC}"
HTTP=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE/v1/billing/usage_export?format=csv&period_days=30" || true)
[ "$HTTP" = "402" ]
check "GET /v1/billing/usage_export → 402 after cancel" $?

# ── 9. Invalid webhook signature → 400 ──────────────────────
echo
echo -e "${BOLD}9. Invalid webhook signature${NC}"
HTTP=$(curl -sf -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/billing/webhook" \
    -H "Content-Type: application/json" \
    -H "stripe-signature: invalid" \
    -d '{"type":"test"}' || true)
[ "$HTTP" = "400" ]
check "Invalid signature → 400" $?

# ── 10. Invalid plan → 422 ──────────────────────────────────
echo
echo -e "${BOLD}10. Invalid checkout plan${NC}"
HTTP=$(curl -sf -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/billing/checkout" \
    -H "Content-Type: application/json" \
    -d '{"plan": "enterprise"}' || true)
[ "$HTTP" = "422" ]
check "Invalid plan → 422" $?

# ── Summary ──────────────────────────────────────────────────
echo
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${BOLD}║  ${GREEN}STRIPE STUB v1 — ALL $PASSED PASSED${NC}${BOLD}          ║${NC}"
else
    echo -e "${BOLD}║  ${RED}STRIPE STUB v1 — $PASSED passed, $FAILED FAILED${NC}${BOLD}  ║${NC}"
fi
echo -e "${BOLD}║  checkout ✓  webhook ✓  gating ✓        ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"

[ "$FAILED" -eq 0 ] || exit 1
