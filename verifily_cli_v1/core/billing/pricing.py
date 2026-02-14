"""Plan catalog and unit pricing for Verifily billing.

All prices in cents (1/100 USD). Tax is always 0 for v1.
Plans are immutable frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PlanSpec:
    """Immutable plan definition."""

    id: str
    name: str
    monthly_base_cents: int
    included_requests: int
    included_rows: int
    included_bytes: int
    price_per_1k_rows: int
    price_per_request: int
    price_per_mb: int
    price_per_decision: int


# ── Plan Catalog ────────────────────────────────────────────────

FREE = PlanSpec(
    id="FREE",
    name="Free",
    monthly_base_cents=0,
    included_requests=500,
    included_rows=100_000,
    included_bytes=50_000_000,
    price_per_1k_rows=0,
    price_per_request=0,
    price_per_mb=0,
    price_per_decision=0,
)

STARTER = PlanSpec(
    id="STARTER",
    name="Starter",
    monthly_base_cents=9_900,
    included_requests=5_000,
    included_rows=1_000_000,
    included_bytes=500_000_000,
    price_per_1k_rows=5,
    price_per_request=1,
    price_per_mb=2,
    price_per_decision=10,
)

PRO = PlanSpec(
    id="PRO",
    name="Pro",
    monthly_base_cents=49_900,
    included_requests=50_000,
    included_rows=10_000_000,
    included_bytes=5_000_000_000,
    price_per_1k_rows=3,
    price_per_request=0,
    price_per_mb=1,
    price_per_decision=5,
)

ENTERPRISE = PlanSpec(
    id="ENTERPRISE",
    name="Enterprise",
    monthly_base_cents=49_900,
    included_requests=1_000_000,
    included_rows=100_000_000,
    included_bytes=50_000_000_000,
    price_per_1k_rows=1,
    price_per_request=0,
    price_per_mb=0,
    price_per_decision=2,
)

PLANS: Dict[str, PlanSpec] = {
    "FREE": FREE,
    "STARTER": STARTER,
    "PRO": PRO,
    "ENTERPRISE": ENTERPRISE,
}


def get_plan(plan_id: str) -> PlanSpec:
    """Lookup a plan by ID (case-insensitive). Raises KeyError if not found."""
    return PLANS[plan_id.upper()]


# ── Plan Enforcement Limits ────────────────────────────────────

import sys


@dataclass(frozen=True)
class PlanLimits:
    """Hard enforcement caps per billing period (month)."""

    plan_id: str
    max_processed_rows_per_month: int
    warn_threshold: float = 0.8  # warn at 80% of cap


PLAN_LIMITS: Dict[str, PlanLimits] = {
    "FREE": PlanLimits(plan_id="FREE", max_processed_rows_per_month=50_000),
    "STARTER": PlanLimits(plan_id="STARTER", max_processed_rows_per_month=1_000_000),
    "PRO": PlanLimits(plan_id="PRO", max_processed_rows_per_month=5_000_000),
    "ENTERPRISE": PlanLimits(plan_id="ENTERPRISE", max_processed_rows_per_month=sys.maxsize),
}


def get_plan_limits(plan_id: str) -> PlanLimits:
    """Lookup plan limits by ID (case-insensitive). Raises KeyError if not found."""
    return PLAN_LIMITS[plan_id.upper()]
