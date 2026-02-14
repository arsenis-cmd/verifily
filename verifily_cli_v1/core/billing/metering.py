"""Invoice computation -- pure function, no side effects.

Takes a list of BillingEvents and a PlanSpec, returns an InvoicePreview.
Deterministic: same inputs always produce same output.
"""

from __future__ import annotations

from typing import List

from verifily_cli_v1.core.billing.models import BillingEvent, InvoiceLine, InvoicePreview
from verifily_cli_v1.core.billing.pricing import PlanSpec


def compute_invoice(
    events: List[BillingEvent],
    plan: PlanSpec,
    project_id: str,
    window_start: float,
    window_end: float,
) -> InvoicePreview:
    """Aggregate billing events into an invoice preview."""
    total_requests = len(events)
    total_rows = 0
    total_bytes = 0
    total_decisions = 0

    for evt in events:
        total_rows += evt.units.get("rows_in", 0) + evt.units.get("rows_out", 0)
        total_bytes += evt.units.get("bytes_in", 0) + evt.units.get("bytes_out", 0)
        total_decisions += evt.units.get("decisions", 0)

    lines: List[InvoiceLine] = []

    # Requests
    req_overage = max(0, total_requests - plan.included_requests)
    req_amount = req_overage * plan.price_per_request
    lines.append(InvoiceLine(
        label="API Requests",
        unit_type="requests",
        quantity=total_requests,
        included=plan.included_requests,
        overage=req_overage,
        unit_price_cents=plan.price_per_request,
        amount_cents=req_amount,
    ))

    # Rows (price per 1000 rows)
    row_overage = max(0, total_rows - plan.included_rows)
    row_blocks = (row_overage + 999) // 1000 if row_overage > 0 else 0
    row_amount = row_blocks * plan.price_per_1k_rows
    lines.append(InvoiceLine(
        label="Rows Processed",
        unit_type="rows",
        quantity=total_rows,
        included=plan.included_rows,
        overage=row_overage,
        unit_price_cents=plan.price_per_1k_rows,
        amount_cents=row_amount,
    ))

    # Bytes (price per MB)
    byte_overage = max(0, total_bytes - plan.included_bytes)
    byte_mb = (byte_overage + (1024 * 1024 - 1)) // (1024 * 1024) if byte_overage > 0 else 0
    byte_amount = byte_mb * plan.price_per_mb
    lines.append(InvoiceLine(
        label="Data Processed",
        unit_type="bytes",
        quantity=total_bytes,
        included=plan.included_bytes,
        overage=byte_overage,
        unit_price_cents=plan.price_per_mb,
        amount_cents=byte_amount,
    ))

    # Decisions (no included tier — all billable)
    decision_amount = total_decisions * plan.price_per_decision
    lines.append(InvoiceLine(
        label="Pipeline Decisions",
        unit_type="decisions",
        quantity=total_decisions,
        included=0,
        overage=total_decisions,
        unit_price_cents=plan.price_per_decision,
        amount_cents=decision_amount,
    ))

    overage_total = sum(line.amount_cents for line in lines)
    subtotal = plan.monthly_base_cents + overage_total
    return InvoicePreview(
        project_id=project_id,
        plan_id=plan.id,
        window_start=window_start,
        window_end=window_end,
        lines=lines,
        monthly_base_cents=plan.monthly_base_cents,
        subtotal_cents=subtotal,
        tax_cents=0,
        total_cents=subtotal,
    )
