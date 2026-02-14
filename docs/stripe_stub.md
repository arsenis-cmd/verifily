# Stripe Stub v1 — Payment-Ready Integration

Verifily ships with a "Stripe-ready" billing layer. This means:

- Checkout sessions create real Stripe URLs (when configured)
- Webhooks activate/cancel subscriptions
- Pro features are gated behind active subscriptions
- Everything works locally with mocks for testing

No real Stripe calls happen unless you provide real API keys.

## Architecture

```
                     ┌──────────────────┐
  POST /checkout ──▶ │  StripeClient    │──▶ Stripe API (or mock)
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ SubscriptionsStore│  (in-memory + JSONL)
                     └──────────────────┘
                              │
  POST /webhook  ──▶ updates status (ACTIVE/CANCELED/INCOMPLETE)
                              │
  GET  /subscription ◀────── reads status
                              │
  GET  /usage_export ────────▶ checks require_active() if enforce=1
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VERIFILY_STRIPE_ENABLED` | Yes | Set to `1` to enable Stripe endpoints |
| `STRIPE_SECRET_KEY` | Yes* | Stripe secret API key (`sk_test_...` or `sk_live_...`) |
| `STRIPE_WEBHOOK_SECRET` | Yes* | Stripe webhook signing secret (`whsec_...`) |
| `STRIPE_PRICE_ID_PRO` | Yes* | Stripe Price ID for Pro plan |
| `STRIPE_SUCCESS_URL` | No | Redirect after successful checkout (default: localhost) |
| `STRIPE_CANCEL_URL` | No | Redirect after canceled checkout (default: localhost) |
| `VERIFILY_BILLING_ENFORCE` | No | Set to `1` to gate Pro features behind active subscription |
| `VERIFILY_SUBS_PERSIST` | No | Set to `1` to persist subscriptions to JSONL |
| `VERIFILY_SUBS_LOG_PATH` | No | Custom path for subscription JSONL log |

*Required when `VERIFILY_STRIPE_ENABLED=1`

## Quick Start

### 1. Enable Stripe (test mode)

```bash
export VERIFILY_STRIPE_ENABLED=1
export VERIFILY_ENABLE_BILLING=1
export STRIPE_SECRET_KEY=sk_test_your_key
export STRIPE_WEBHOOK_SECRET=whsec_your_secret
export STRIPE_PRICE_ID_PRO=price_your_pro_plan
verifily serve
```

### 2. Start checkout

```bash
curl -X POST http://localhost:8000/v1/billing/checkout \
  -H "Content-Type: application/json" \
  -d '{"plan": "pro", "project_id": "my-proj"}'
```

Response:
```json
{
  "checkout_url": "https://checkout.stripe.com/c/pay/...",
  "stripe_customer_id": "cus_abc123",
  "plan": "pro"
}
```

### 3. Set up webhook endpoint

Point Stripe webhooks to your server:

```
https://your-domain.com/v1/billing/webhook
```

Events handled:
- `checkout.session.completed` → subscription ACTIVE
- `customer.subscription.deleted` → subscription CANCELED
- `invoice.payment_failed` → subscription INCOMPLETE

### 4. Check subscription status

```bash
curl http://localhost:8000/v1/billing/subscription?project_id=my-proj
```

### 5. Enable Pro feature gating

```bash
export VERIFILY_BILLING_ENFORCE=1
```

When enabled, usage export (`/v1/billing/usage_export`) requires an active
subscription. Without one, the endpoint returns:

```json
{
  "detail": {
    "type": "PAYMENT_REQUIRED",
    "message": "Pro feature. Start checkout at /v1/billing/checkout"
  }
}
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/billing/checkout` | Start Stripe Checkout session |
| POST | `/v1/billing/webhook` | Receive Stripe webhook events |
| GET | `/v1/billing/subscription` | Get subscription status |

## CLI Commands

```bash
verifily checkout --plan pro --project my-proj
verifily subscription --project my-proj
```

## SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(base_url="http://localhost:8000")

# Start checkout
resp = client.checkout(plan="pro", project_id="my-proj")
print(f"Checkout: {resp.checkout_url}")

# Check subscription
sub = client.subscription(project_id="my-proj")
print(f"Status: {sub.status}")
```

## Testing Without Stripe

The `MockStripeClient` is used automatically when:
- The `stripe` Python package is not installed
- Or in test fixtures

Mock behavior:
- `create_customer()` → returns `cus_mock_NNNN`
- `create_checkout_session()` → returns `https://checkout.stripe.com/mock/NNNN`
- `construct_event()` → parses JSON payload directly (ignores signature unless `"invalid"`)

## Subscription States

| Status | Meaning |
|--------|---------|
| `active` | Payment confirmed, Pro features unlocked |
| `incomplete` | Checkout started but not completed, or payment failed |
| `canceled` | Subscription deleted |
| `none` | No subscription record exists |

## Persistence

Enable persistence to survive server restarts:

```bash
export VERIFILY_SUBS_PERSIST=1
export VERIFILY_SUBS_LOG_PATH=/var/log/verifily/subscriptions.jsonl
```

## How This Becomes Real Stripe Billing

1. Install `pip install stripe`
2. Set real keys (`sk_live_...`, `whsec_...`)
3. Create a Stripe Product + Price for your Pro plan
4. Point webhook URL to your server
5. The `StripeClient` class auto-detects the real `stripe` package

No code changes needed — the same endpoints work with real Stripe.

## Future: Usage-Based Billing

When ready for metered billing:

1. Create Stripe metered prices
2. Report usage via `stripe.UsageRecord.create()`
3. Use existing `BillingEvent` data as the source
4. Add a `billing/stripe_metering.py` that reads from `billing_store`

The invoice JSON from Billing v1 provides the line items and totals that
map directly to Stripe Invoice Items.
