"""Billing enforcement logic -- pure functions, no I/O.

Decides whether a request should PASS, trigger a WARN, or be BLOCKed
based on current usage vs. plan limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from verifily_cli_v1.core.billing.pricing import PlanLimits


class BillingStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class BillingDecision:
    """Result of a quota check."""

    status: BillingStatus
    reason: str
    remaining: int
    limit: int
    used: int


def check_quota(
    *,
    plan_limits: PlanLimits,
    current_processed_rows: int,
    additional_rows: int = 0,
) -> BillingDecision:
    """Check whether current + additional rows would exceed the plan cap.

    Returns BillingDecision with PASS, WARN, or BLOCK status.
    """
    cap = plan_limits.max_processed_rows_per_month
    after = current_processed_rows + additional_rows
    remaining = max(0, cap - current_processed_rows)

    if after > cap:
        return BillingDecision(
            status=BillingStatus.BLOCK,
            reason=(
                f"Monthly quota exceeded for plan={plan_limits.plan_id}. "
                f"{current_processed_rows:,}/{cap:,} rows used. Upgrade to Pro."
            ),
            remaining=remaining,
            limit=cap,
            used=current_processed_rows,
        )

    warn_at = int(cap * plan_limits.warn_threshold)
    if current_processed_rows >= warn_at:
        pct = current_processed_rows * 100 // cap if cap > 0 else 100
        return BillingDecision(
            status=BillingStatus.WARN,
            reason=(
                f"Approaching monthly limit: {current_processed_rows:,}/{cap:,} "
                f"({pct}% used)."
            ),
            remaining=remaining,
            limit=cap,
            used=current_processed_rows,
        )

    return BillingDecision(
        status=BillingStatus.PASS,
        reason="Within plan limits.",
        remaining=remaining,
        limit=cap,
        used=current_processed_rows,
    )
