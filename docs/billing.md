# Billing v1 — Usage-Based Accounting

Verifily tracks every API call as a billing event. This guide covers the
billing model, plans, endpoints, and how it maps to future Stripe integration.

## Billing Model

Every metered call records a **BillingEvent** with:

| Field | Description |
|-------|-------------|
| `ts` | Unix timestamp |
| `api_key_id` | Caller's API key ID |
| `project_id` | Target project |
| `endpoint` | e.g. `/v1/pipeline`, `/v1/report` |
| `units` | `{rows_in, rows_out, bytes_in, bytes_out, decisions}` |

No raw data or PII is stored in billing events.

## Billable Units

| Unit | How Counted |
|------|-------------|
| Requests | 1 per API call |
| Rows | `rows_in + rows_out` |
| Bytes | `bytes_in + bytes_out` |
| Decisions | pipeline ship/don't-ship/investigate counts |

## Plans

| Plan | Base $/mo | Incl. Requests | Incl. Rows | Incl. Bytes | Row Overage | Request Overage |
|------|-----------|---------------|------------|-------------|-------------|-----------------|
| **FREE** | $0 | 500 | 100K | 50MB | $0 | $0 |
| **STARTER** | $99 | 5K | 1M | 500MB | $0.05/1K | $0.01/req |
| **PRO** | $499 | 50K | 10M | 5GB | $0.03/1K | $0 |
| **ENTERPRISE** | $499 | 1M | 100M | 50GB | $0.01/1K | $0 |

## Quick Start

### 1. Enable billing

```bash
export VERIFILY_ENABLE_BILLING=1
verifily serve
```

### 2. List plans

```bash
curl http://localhost:8000/v1/billing/plans | jq
```

### 3. Generate some usage

```bash
curl -X POST http://localhost:8000/v1/report \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "path/to/dataset.jsonl", "schema": "sft"}'
```

### 4. Get an estimate

```bash
curl "http://localhost:8000/v1/billing/estimate?plan=STARTER&window_minutes=43200" | jq
```

### 5. Generate an invoice

```bash
curl -X POST http://localhost:8000/v1/billing/invoice \
  -H "Content-Type: application/json" \
  -d '{
    "plan": "STARTER",
    "period_start": "2024-01-01T00:00:00+00:00",
    "period_end": "2024-02-01T00:00:00+00:00"
  }' | jq
```

### 6. Export usage

```bash
# CSV
curl "http://localhost:8000/v1/billing/usage_export?format=csv&period_days=30"

# JSONL
curl "http://localhost:8000/v1/billing/usage_export?format=jsonl&period_days=30&group_by=day_project"
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/billing/plans` | List all plans |
| GET | `/v1/billing/events` | Query raw billing events |
| GET | `/v1/billing/invoice-preview` | Quick invoice preview |
| GET | `/v1/billing/estimate` | Estimate with current usage |
| POST | `/v1/billing/invoice` | Generate and persist invoice |
| GET | `/v1/billing/usage_export` | Export usage as CSV/JSONL |

## CLI Commands

```bash
verifily plans                          # list plans
verifily estimate --plan PRO            # estimate current usage
verifily invoice --plan STARTER --period-days 30
verifily usage-export --format csv --period-days 30 --out ./exports
verifily billing-events                 # raw events
verifily billing-preview --plan STARTER # invoice preview
```

## SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient(base_url="http://localhost:8000")

# List plans
plans = client.billing_plans()

# Estimate
est = client.billing_estimate(plan="STARTER", window_minutes=43200)
print(f"Estimated total: ${est.total_cents / 100:.2f}")

# Generate invoice
inv = client.billing_invoice(
    plan="STARTER",
    period_start="2024-01-01T00:00:00+00:00",
    period_end="2024-02-01T00:00:00+00:00",
)
print(f"Invoice {inv.invoice_id}: ${inv.total_cents / 100:.2f}")

# Export usage
csv_data = client.billing_usage_export(format="csv", period_days=30)
```

## Invoice JSON Structure

```json
{
  "invoice_id": "inv-a1b2c3d4e5f6",
  "plan_id": "STARTER",
  "project_id": "default",
  "period_start": "2024-01-01T00:00:00+00:00",
  "period_end": "2024-02-01T00:00:00+00:00",
  "lines": [
    {"label": "API Requests", "quantity": 150, "included": 5000, "overage": 0, "amount_cents": 0},
    {"label": "Rows Processed", "quantity": 50000, "included": 1000000, "overage": 0, "amount_cents": 0},
    {"label": "Data Processed", "quantity": 1000000, "included": 500000000, "overage": 0, "amount_cents": 0},
    {"label": "Pipeline Decisions", "quantity": 5, "included": 0, "overage": 5, "amount_cents": 50}
  ],
  "monthly_base_cents": 9900,
  "subtotal_cents": 9950,
  "tax_cents": 0,
  "total_cents": 9950
}
```

The invoice JSON is the **source of truth** for billing. It is deterministic:
same inputs always produce the same output.

## Mapping to Stripe (Future)

When Stripe integration is added:

1. **Invoice JSON → Stripe Invoice**: Each `InvoiceLine` maps to a Stripe
   `InvoiceItem` with `unit_amount = unit_price_cents`.

2. **Usage Export → Stripe Metered Billing**: The CSV/JSONL export can feed
   `stripe.UsageRecord.create()` calls.

3. **Plans → Stripe Products/Prices**: Each `PlanSpec` maps to a Stripe
   `Product` + `Price` with metered usage components.

4. **Events → Stripe Events**: BillingEvents provide the audit trail that
   backs up every charge.

No code changes needed in the billing core — only a new `billing/stripe.py`
integration layer that reads the existing invoice JSON.

## Persistence

Enable billing persistence to survive server restarts:

```bash
export VERIFILY_ENABLE_BILLING=1
export VERIFILY_BILLING_PERSIST=1
```

Events are stored as append-only JSONL in the data directory and replayed
on startup.

## Auth Behavior

- If auth is **disabled**: all billing endpoints are open (dev mode)
- If auth is **enabled** (simple/advanced/teams): billing endpoints require
  valid auth; in advanced mode, the `usage:read` scope is required
