"""Billing data models -- events, invoice lines, invoice preview.

All models are plain dataclasses with to_dict() for serialization.
No Pydantic here (API Pydantic models live in core/api/models.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BillingEvent:
    """A single append-only billing event.

    Recorded for every metered API call. Never stores raw payloads or PII.
    """

    ts: float
    api_key_id: str = "anonymous"
    project_id: str = "default"
    endpoint: str = ""
    units: Dict[str, int] = field(default_factory=dict)
    request_id: Optional[str] = None
    job_id: Optional[str] = None
    status_code: int = 200

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ts": self.ts,
            "api_key_id": self.api_key_id,
            "project_id": self.project_id,
            "endpoint": self.endpoint,
            "units": dict(self.units),
            "status_code": self.status_code,
        }
        if self.request_id is not None:
            d["request_id"] = self.request_id
        if self.job_id is not None:
            d["job_id"] = self.job_id
        return d


@dataclass
class InvoiceLine:
    """A single line item on an invoice preview."""

    label: str
    unit_type: str
    quantity: int
    included: int
    overage: int
    unit_price_cents: int
    amount_cents: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "unit_type": self.unit_type,
            "quantity": self.quantity,
            "included": self.included,
            "overage": self.overage,
            "unit_price_cents": self.unit_price_cents,
            "amount_cents": self.amount_cents,
        }


@dataclass
class InvoicePreview:
    """Computed invoice preview for a project in a time window."""

    project_id: str
    plan_id: str
    window_start: float
    window_end: float
    lines: List[InvoiceLine]
    monthly_base_cents: int
    subtotal_cents: int
    tax_cents: int = 0
    total_cents: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "plan_id": self.plan_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "lines": [line.to_dict() for line in self.lines],
            "monthly_base_cents": self.monthly_base_cents,
            "subtotal_cents": self.subtotal_cents,
            "tax_cents": self.tax_cents,
            "total_cents": self.total_cents,
        }
