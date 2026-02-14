"""Invoice artifact generation -- JSON, CSV, optional PDF.

Deterministic invoice IDs: SHA256(project_id + period + plan_id).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from verifily_cli_v1.core.billing.models import InvoicePreview


@dataclass
class Invoice:
    """Full invoice extending InvoicePreview with metadata."""

    invoice_id: str
    customer: str  # project_id or org_id
    generated_at: float
    preview: InvoicePreview

    def to_dict(self) -> Dict[str, Any]:
        d = self.preview.to_dict()
        d["invoice_id"] = self.invoice_id
        d["customer"] = self.customer
        d["generated_at"] = self.generated_at
        return d


def deterministic_invoice_id(project_id: str, period: str, plan_id: str) -> str:
    """SHA256-based deterministic invoice ID."""
    raw = f"{project_id}:{period}:{plan_id}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"inv-{h}"


def create_invoice(
    preview: InvoicePreview,
    project_id: str,
    period: str,
    plan_id: str,
) -> Invoice:
    """Create an Invoice from an InvoicePreview."""
    return Invoice(
        invoice_id=deterministic_invoice_id(project_id, period, plan_id),
        customer=project_id,
        generated_at=time.time(),
        preview=preview,
    )


def write_invoice_json(invoice: Invoice, out_dir: str) -> str:
    """Write invoice as JSON. Returns file path."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"{invoice.invoice_id}.json"
    path.write_text(json.dumps(invoice.to_dict(), indent=2, default=str))
    return str(path)


def write_invoice_csv(invoice: Invoice, out_dir: str) -> str:
    """Write invoice line items as CSV. Returns file path."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"{invoice.invoice_id}.csv"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "label", "unit_type", "quantity", "included",
        "overage", "unit_price_cents", "amount_cents",
    ])
    for line in invoice.preview.lines:
        writer.writerow([
            line.label, line.unit_type, line.quantity, line.included,
            line.overage, line.unit_price_cents, line.amount_cents,
        ])
    writer.writerow(["TOTAL", "", "", "", "", "", invoice.preview.total_cents])
    path.write_text(buf.getvalue())
    return str(path)


def write_invoice_pdf(invoice: Invoice, out_dir: str) -> Optional[str]:
    """Write invoice as PDF if reportlab is available. Returns path or None."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as pdf_canvas
    except ImportError:
        return None

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    path = p / f"{invoice.invoice_id}.pdf"

    c = pdf_canvas.Canvas(str(path), pagesize=A4)
    _w, h = A4
    y = h - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Invoice {invoice.invoice_id}")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Customer: {invoice.customer}")
    y -= 15
    c.drawString(50, y, f"Plan: {invoice.preview.plan_id}")
    y -= 15
    c.drawString(50, y, f"Period: {invoice.preview.window_start} - {invoice.preview.window_end}")
    y -= 30

    c.setFont("Helvetica-Bold", 9)
    c.drawString(50, y, "Label")
    c.drawString(200, y, "Qty")
    c.drawString(270, y, "Included")
    c.drawString(340, y, "Overage")
    c.drawString(410, y, "Amount")
    y -= 15
    c.setFont("Helvetica", 9)
    for line in invoice.preview.lines:
        c.drawString(50, y, line.label)
        c.drawString(200, y, str(line.quantity))
        c.drawString(270, y, str(line.included))
        c.drawString(340, y, str(line.overage))
        c.drawString(410, y, f"${line.amount_cents / 100:.2f}")
        y -= 13

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(340, y, f"Total: ${invoice.preview.total_cents / 100:.2f}")
    c.save()
    return str(path)
