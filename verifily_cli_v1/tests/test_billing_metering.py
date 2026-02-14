"""Tests for Metered Billing v2 — periods, plan limits, enforcement, invoices."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.billing.enforcement import (
    BillingDecision,
    BillingStatus,
    check_quota,
)
from verifily_cli_v1.core.billing.invoice import (
    Invoice,
    create_invoice,
    deterministic_invoice_id,
    write_invoice_csv,
    write_invoice_json,
)
from verifily_cli_v1.core.billing.models import BillingEvent, InvoiceLine, InvoicePreview
from verifily_cli_v1.core.billing.periods import current_period, period_bounds, validate_period
from verifily_cli_v1.core.billing.pricing import (
    PLAN_LIMITS,
    PlanLimits,
    get_plan_limits,
)
from verifily_cli_v1.core.billing.store import BillingStore, billing_store


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _make_client(monkeypatch, **kwargs):
    """Create a TestClient with given settings overrides."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_ENABLE_BILLING", raising=False)
    monkeypatch.delenv("VERIFILY_BILLING_PERSIST", raising=False)
    monkeypatch.delenv("VERIFILY_BILLING_ENFORCE", raising=False)
    monkeypatch.delenv("VERIFILY_DEFAULT_PLAN", raising=False)
    settings = load_settings(**kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    return TestClient(app)


# ── Billing Periods ──────────────────────────────────────────────


class TestBillingPeriods:
    def test_current_period_format(self):
        p = current_period()
        assert len(p) == 7
        assert p[4] == "-"
        year, month = int(p[:4]), int(p[5:7])
        assert 2020 <= year <= 2099
        assert 1 <= month <= 12

    def test_period_bounds_january(self):
        start, end = period_bounds("2026-01")
        import datetime
        dt_start = datetime.datetime.fromtimestamp(start, tz=datetime.timezone.utc)
        dt_end = datetime.datetime.fromtimestamp(end, tz=datetime.timezone.utc)
        assert dt_start.month == 1 and dt_start.day == 1
        assert dt_end.month == 2 and dt_end.day == 1

    def test_period_bounds_december_wraps(self):
        start, end = period_bounds("2026-12")
        import datetime
        dt_end = datetime.datetime.fromtimestamp(end, tz=datetime.timezone.utc)
        assert dt_end.year == 2027 and dt_end.month == 1

    def test_validate_period_valid(self):
        assert validate_period("2026-02") is True
        assert validate_period("2025-12") is True

    def test_validate_period_invalid(self):
        assert validate_period("2026-13") is False
        assert validate_period("2026-00") is False
        assert validate_period("202602") is False
        assert validate_period("abcd-ef") is False
        assert validate_period("") is False


# ── Plan Limits ──────────────────────────────────────────────────


class TestPlanLimits:
    def test_free_cap_50k(self):
        lim = get_plan_limits("FREE")
        assert lim.max_processed_rows_per_month == 50_000

    def test_pro_cap_5m(self):
        lim = get_plan_limits("PRO")
        assert lim.max_processed_rows_per_month == 5_000_000

    def test_enterprise_unlimited(self):
        lim = get_plan_limits("ENTERPRISE")
        assert lim.max_processed_rows_per_month == sys.maxsize

    def test_unknown_plan_raises(self):
        with pytest.raises(KeyError):
            get_plan_limits("NONEXISTENT")


# ── Billing Decision ────────────────────────────────────────────


class TestBillingDecision:
    def test_pass_within_limit(self):
        lim = PlanLimits(plan_id="FREE", max_processed_rows_per_month=50_000)
        d = check_quota(plan_limits=lim, current_processed_rows=10_000)
        assert d.status == BillingStatus.PASS
        assert d.remaining == 40_000

    def test_warn_at_80_percent(self):
        lim = PlanLimits(plan_id="FREE", max_processed_rows_per_month=50_000)
        d = check_quota(plan_limits=lim, current_processed_rows=40_001)
        assert d.status == BillingStatus.WARN
        assert "Approaching" in d.reason

    def test_block_over_limit(self):
        lim = PlanLimits(plan_id="FREE", max_processed_rows_per_month=50_000)
        d = check_quota(plan_limits=lim, current_processed_rows=50_000, additional_rows=1)
        assert d.status == BillingStatus.BLOCK
        assert d.remaining == 0
        assert d.limit == 50_000
        assert d.used == 50_000

    def test_block_reason_includes_counts(self):
        lim = PlanLimits(plan_id="FREE", max_processed_rows_per_month=50_000)
        d = check_quota(plan_limits=lim, current_processed_rows=60_000)
        assert d.status == BillingStatus.BLOCK
        assert "50,000" in d.reason
        assert "60,000" in d.reason


# ── Usage For Period ─────────────────────────────────────────────


class TestUsageForPeriod:
    def test_aggregates_rows_in_plus_out(self):
        store = BillingStore()
        now = time.time()
        period = current_period()
        store._events.append(BillingEvent(
            ts=now, api_key_id="k1", project_id="p1",
            endpoint="/v1/pipeline", units={"rows_in": 100, "rows_out": 50},
        ))
        result = store.usage_for_period(period=period)
        assert result["processed_rows"] == 150
        assert result["requests"] == 1

    def test_filters_by_project(self):
        store = BillingStore()
        now = time.time()
        period = current_period()
        store._events.append(BillingEvent(
            ts=now, project_id="p1", units={"rows_in": 100},
        ))
        store._events.append(BillingEvent(
            ts=now, project_id="p2", units={"rows_in": 200},
        ))
        result = store.usage_for_period(project_id="p1", period=period)
        assert result["processed_rows"] == 100

    def test_filters_by_period(self):
        store = BillingStore()
        # Event in Jan 2025 should not appear in Feb 2026
        import datetime
        jan_ts = datetime.datetime(2025, 1, 15, tzinfo=datetime.timezone.utc).timestamp()
        store._events.append(BillingEvent(
            ts=jan_ts, units={"rows_in": 999},
        ))
        result = store.usage_for_period(period="2026-02")
        assert result["processed_rows"] == 0

    def test_empty_returns_zeros(self):
        store = BillingStore()
        result = store.usage_for_period(period="2026-02")
        assert result == {"processed_rows": 0, "bytes_processed": 0, "decisions": 0, "requests": 0}


# ── Billing Enforce Middleware ───────────────────────────────────


class TestBillingEnforceMiddleware:
    def test_exempt_health(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_exempt_get(self, monkeypatch):
        """GET requests to billable paths should not be blocked."""
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        resp = client.get("/v1/billing/plans")
        assert resp.status_code == 200

    def test_exempt_billing_prefix(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        resp = client.get("/v1/billing/events")
        assert resp.status_code == 200

    def test_pass_under_limit(self, monkeypatch):
        """POST to billable endpoint should pass when under quota."""
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        # POST to /v1/report with small data — billing_store has 0 rows
        resp = client.post("/v1/report", json={
            "config_path": str(DRILL_DIR / "verifily.yaml"),
        })
        # Should not be 402 — quota is 50k and we have 0 rows used
        assert resp.status_code != 402

    def test_block_over_limit_402(self, monkeypatch):
        """POST should return 402 when over quota."""
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        # Inject events to exceed free cap (50k rows)
        now = time.time()
        for _ in range(6):
            billing_store.record_event(
                api_key_id="test", project_id="default",
                endpoint="/v1/pipeline",
                units={"rows_in": 10_000},
            )
        # Now at 60k rows > 50k free cap
        resp = client.post("/v1/report", json={
            "config_path": str(DRILL_DIR / "verifily.yaml"),
        })
        assert resp.status_code == 402

    def test_402_body_structure(self, monkeypatch):
        """402 response should have the correct error envelope."""
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        now = time.time()
        for _ in range(6):
            billing_store.record_event(
                api_key_id="test", project_id="default",
                endpoint="/v1/pipeline",
                units={"rows_in": 10_000},
            )
        resp = client.post("/v1/report", json={
            "config_path": str(DRILL_DIR / "verifily.yaml"),
        })
        assert resp.status_code == 402
        body = resp.json()
        err = body["error"]
        assert err["type"] == "PAYMENT_REQUIRED"
        assert err["code"] == "billing_limit_exceeded"
        assert "period" in err
        assert "limit" in err
        assert "used" in err
        assert "plan" in err
        assert err["limit"] == 50_000
        assert err["used"] >= 60_000

    def test_warn_headers(self, monkeypatch):
        """When near limit, response should have warning headers."""
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=True)
        # Inject 41k rows — above 80% warn threshold (40k) but under 50k cap
        for _ in range(41):
            billing_store.record_event(
                api_key_id="test", project_id="default",
                endpoint="/v1/pipeline",
                units={"rows_in": 1_000},
            )
        resp = client.post("/v1/report", json={
            "config_path": str(DRILL_DIR / "verifily.yaml"),
        })
        # Should NOT be 402, but should have warn headers
        assert resp.status_code != 402
        assert "X-Billing-Warning" in resp.headers
        assert "X-Billing-Used" in resp.headers
        assert "X-Billing-Limit" in resp.headers

    def test_disabled_when_enforce_false(self, monkeypatch):
        """When billing_enforce=False, middleware should not block."""
        client = _make_client(monkeypatch, enable_billing=True, billing_enforce=False)
        # Inject tons of rows
        for _ in range(100):
            billing_store.record_event(
                api_key_id="test", project_id="default",
                endpoint="/v1/pipeline",
                units={"rows_in": 10_000},
            )
        resp = client.post("/v1/report", json={
            "config_path": str(DRILL_DIR / "verifily.yaml"),
        })
        # Should not be 402 because enforcement is off
        assert resp.status_code != 402


# ── Invoice Artifacts ────────────────────────────────────────────


class TestInvoiceArtifacts:
    def test_deterministic_id(self):
        id1 = deterministic_invoice_id("proj-1", "2026-02", "FREE")
        id2 = deterministic_invoice_id("proj-1", "2026-02", "FREE")
        id3 = deterministic_invoice_id("proj-2", "2026-02", "FREE")
        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("inv-")
        assert len(id1) == 20  # "inv-" + 16 hex chars

    def test_write_json_creates_file(self):
        preview = InvoicePreview(
            project_id="proj-1", plan_id="FREE",
            window_start=0.0, window_end=1.0,
            lines=[], monthly_base_cents=0,
            subtotal_cents=0, total_cents=0,
        )
        inv = create_invoice(preview, "proj-1", "2026-02", "FREE")
        with tempfile.TemporaryDirectory() as td:
            path = write_invoice_json(inv, td)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert data["invoice_id"] == inv.invoice_id
            assert data["customer"] == "proj-1"

    def test_write_csv_creates_file(self):
        lines = [
            InvoiceLine(
                label="API Requests", unit_type="requests",
                quantity=100, included=500, overage=0,
                unit_price_cents=0, amount_cents=0,
            ),
        ]
        preview = InvoicePreview(
            project_id="proj-1", plan_id="FREE",
            window_start=0.0, window_end=1.0,
            lines=lines, monthly_base_cents=0,
            subtotal_cents=0, total_cents=0,
        )
        inv = create_invoice(preview, "proj-1", "2026-02", "FREE")
        with tempfile.TemporaryDirectory() as td:
            path = write_invoice_csv(inv, td)
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "label" in content
            assert "TOTAL" in content
            assert "API Requests" in content

    def test_store_and_get_invoice(self):
        store = BillingStore()
        store.store_invoice("inv-abc", {"test": True})
        assert store.get_invoice("inv-abc") == {"test": True}

    def test_get_nonexistent_returns_none(self):
        store = BillingStore()
        assert store.get_invoice("inv-nope") is None


# ── Billing Usage Endpoint ───────────────────────────────────────


class TestBillingUsageEndpoint:
    def test_usage_returns_processed_rows(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        # Record some events
        billing_store.record_event(
            api_key_id="test", project_id="default",
            endpoint="/v1/pipeline",
            units={"rows_in": 100, "rows_out": 50},
        )
        period = current_period()
        resp = client.get("/v1/billing/usage", params={"period": period})
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == period
        assert data["usage"]["processed_rows"] == 150

    def test_invalid_period_422(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/usage", params={"period": "invalid"})
        assert resp.status_code == 422

    def test_disabled_404(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=False)
        resp = client.get("/v1/billing/usage")
        assert resp.status_code == 404


# ── Invoice Retrieval Endpoint ───────────────────────────────────


class TestInvoiceEndpoint:
    def test_get_invoice_after_create(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        # Create an invoice via POST
        now = time.time()
        import datetime
        start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).isoformat()
        end = datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc).isoformat()
        resp = client.post("/v1/billing/invoice", json={
            "plan": "FREE",
            "period_start": start,
            "period_end": end,
        })
        assert resp.status_code == 200
        invoice_id = resp.json()["invoice_id"]
        assert invoice_id.startswith("inv-")

        # Retrieve it
        resp2 = client.get(f"/v1/billing/invoice/{invoice_id}")
        assert resp2.status_code == 200
        assert resp2.json()["invoice_id"] == invoice_id

    def test_get_nonexistent_404(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/invoice/inv-doesnotexist")
        assert resp.status_code == 404

    def test_invoice_deterministic_id(self, monkeypatch):
        """Same params should produce the same invoice_id."""
        client = _make_client(monkeypatch, enable_billing=True)
        import datetime
        start = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).isoformat()
        end = datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc).isoformat()
        body = {"plan": "FREE", "period_start": start, "period_end": end}
        id1 = client.post("/v1/billing/invoice", json=body).json()["invoice_id"]
        id2 = client.post("/v1/billing/invoice", json=body).json()["invoice_id"]
        assert id1 == id2
