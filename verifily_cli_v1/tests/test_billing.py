"""Tests for Billing-Ready Accounting v1."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.billing.export import export_usage_csv, export_usage_jsonl, _bucket_events
from verifily_cli_v1.core.billing.metering import compute_invoice
from verifily_cli_v1.core.billing.models import BillingEvent, InvoiceLine, InvoicePreview
from verifily_cli_v1.core.billing.pricing import PLANS, PlanSpec, get_plan
from verifily_cli_v1.core.billing.store import BillingStore, billing_store


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _make_client(monkeypatch, **kwargs):
    """Create a TestClient with given settings overrides."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_ENABLE_BILLING", raising=False)
    monkeypatch.delenv("VERIFILY_BILLING_PERSIST", raising=False)
    settings = load_settings(**kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    return TestClient(app)


def _ingest_drill():
    """Ingest the demo support_tickets.csv, return dataset path."""
    from verifily_cli_v1.commands.ingest import run as ingest_run

    out_dir = tempfile.mkdtemp(prefix="billing_test_")
    ingest_run(
        input_path=str(DRILL_DIR / "raw" / "support_tickets.csv"),
        output_path=out_dir,
        schema="sft",
        map_args=["question:subject", "answer:resolution", "context:body"],
        tag_args=["source:test"],
        id_col=None,
        limit=None,
        strict=False,
        dry_run=False,
        verbose=False,
    )
    return str(Path(out_dir) / "dataset.jsonl")


# ── Pricing Tests ──────────────────────────────────────────────


class TestPricing:
    def test_all_plans_exist(self):
        assert len(PLANS) == 4
        for pid in ("FREE", "STARTER", "PRO", "ENTERPRISE"):
            assert pid in PLANS

    def test_get_plan_case_insensitive(self):
        p = get_plan("free")
        assert p.id == "FREE"
        p2 = get_plan("Starter")
        assert p2.id == "STARTER"

    def test_free_zero_base(self):
        p = get_plan("FREE")
        assert p.monthly_base_cents == 0
        assert p.price_per_request == 0
        assert p.price_per_1k_rows == 0

    def test_unknown_plan_raises(self):
        with pytest.raises(KeyError):
            get_plan("NONEXISTENT")

    def test_plan_spec_frozen(self):
        p = get_plan("STARTER")
        with pytest.raises(AttributeError):
            p.id = "HACKED"  # type: ignore


# ── Billing Models Tests ───────────────────────────────────────


class TestBillingModels:
    def test_event_to_dict_roundtrip(self):
        evt = BillingEvent(
            ts=1000.0,
            api_key_id="key-1",
            project_id="proj-1",
            endpoint="/v1/report",
            units={"rows_in": 100},
            request_id="req-1",
            status_code=200,
        )
        d = evt.to_dict()
        assert d["ts"] == 1000.0
        assert d["api_key_id"] == "key-1"
        assert d["units"]["rows_in"] == 100
        assert d["request_id"] == "req-1"

    def test_invoice_line_to_dict(self):
        line = InvoiceLine(
            label="Rows",
            unit_type="rows",
            quantity=2000,
            included=1000,
            overage=1000,
            unit_price_cents=5,
            amount_cents=5,
        )
        d = line.to_dict()
        assert d["label"] == "Rows"
        assert d["overage"] == 1000

    def test_invoice_preview_to_dict(self):
        preview = InvoicePreview(
            project_id="proj-1",
            plan_id="STARTER",
            window_start=0.0,
            window_end=100.0,
            lines=[],
            monthly_base_cents=9900,
            subtotal_cents=9900,
            total_cents=9900,
        )
        d = preview.to_dict()
        assert d["plan_id"] == "STARTER"
        assert d["total_cents"] == 9900


# ── Compute Invoice Tests ──────────────────────────────────────


class TestComputeInvoice:
    def test_within_free_tier_zero_total(self):
        plan = get_plan("FREE")
        events = [
            BillingEvent(ts=1.0, units={"rows_in": 10}),
            BillingEvent(ts=2.0, units={"rows_in": 5}),
        ]
        inv = compute_invoice(events, plan, "proj", 0.0, 100.0)
        assert inv.total_cents == 0
        assert inv.monthly_base_cents == 0

    def test_starter_overage_rows(self):
        plan = get_plan("STARTER")
        # Create enough events to exceed the included rows (1M)
        events = [
            BillingEvent(ts=1.0, units={"rows_in": 600_000, "rows_out": 600_000}),
        ]
        inv = compute_invoice(events, plan, "proj", 0.0, 100.0)
        # 1.2M total rows, 1M included, 200K overage
        # 200K / 1000 = 200 blocks * 5 cents = 1000 cents overage on rows
        assert inv.total_cents > plan.monthly_base_cents

    def test_deterministic(self):
        plan = get_plan("PRO")
        events = [
            BillingEvent(ts=1.0, units={"rows_in": 100, "decisions": 1}),
        ]
        inv1 = compute_invoice(events, plan, "proj", 0.0, 100.0)
        inv2 = compute_invoice(events, plan, "proj", 0.0, 100.0)
        assert inv1.total_cents == inv2.total_cents
        assert inv1.subtotal_cents == inv2.subtotal_cents

    def test_bytes_overage(self):
        plan = get_plan("STARTER")
        # 1GB = 1_000_000_000 bytes, included = 500MB = 500_000_000
        events = [
            BillingEvent(ts=1.0, units={"bytes_in": 800_000_000}),
        ]
        inv = compute_invoice(events, plan, "proj", 0.0, 100.0)
        # 800MB total, 500MB included, 300MB overage
        # ceil(300) * 2 cents/MB = 600 cents
        bytes_line = [l for l in inv.lines if l.unit_type == "bytes"][0]
        assert bytes_line.overage > 0
        assert bytes_line.amount_cents > 0

    def test_decisions_overage(self):
        plan = get_plan("STARTER")
        events = [
            BillingEvent(ts=float(i), units={"decisions": 1})
            for i in range(10)
        ]
        inv = compute_invoice(events, plan, "proj", 0.0, 100.0)
        decisions_line = [l for l in inv.lines if l.unit_type == "decisions"][0]
        assert decisions_line.quantity == 10


# ── BillingStore Tests ─────────────────────────────────────────


class TestBillingStore:
    def test_record_and_query(self):
        store = BillingStore()
        store.record_event(
            api_key_id="k1",
            project_id="p1",
            endpoint="/v1/report",
            units={"rows_in": 50},
        )
        events = store.query_events()
        assert len(events) == 1
        assert events[0]["endpoint"] == "/v1/report"

    def test_query_filter_project(self):
        store = BillingStore()
        store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={})
        store.record_event(api_key_id="k1", project_id="p2", endpoint="/v1/pipeline", units={})
        events = store.query_events(project_id="p1")
        assert len(events) == 1
        assert events[0]["project_id"] == "p1"

    def test_preview_invoice_free(self):
        store = BillingStore()
        store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={"rows_in": 10})
        preview = store.preview_invoice(project_id="p1", plan_id="FREE")
        assert preview["total_cents"] == 0

    def test_reset_clears(self):
        store = BillingStore()
        store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={})
        store.reset()
        events = store.query_events()
        assert len(events) == 0

    def test_persistence_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        store1 = BillingStore()
        store1.configure_persistence(path)
        store1.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={"rows_in": 42})
        store1.record_event(api_key_id="k2", project_id="p2", endpoint="/v1/pipeline", units={"decisions": 1})

        # New store replays from file
        store2 = BillingStore()
        store2.configure_persistence(path)
        events = store2.query_events()
        assert len(events) == 2

    def test_query_limit(self):
        store = BillingStore()
        for i in range(10):
            store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={})
        events = store.query_events(limit=3)
        assert len(events) == 3


# ── Billing API Tests ──────────────────────────────────────────


class TestBillingAPI:
    def test_events_empty_when_enabled(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["total"] == 0

    def test_invoice_preview_free(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/invoice-preview", params={"project_id": "default", "plan_id": "FREE"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_cents"] == 0
        assert data["plan_id"] == "FREE"

    def test_billing_disabled_404(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=False)
        resp = client.get("/v1/billing/events")
        assert resp.status_code == 404

        resp2 = client.get("/v1/billing/invoice-preview", params={"project_id": "default"})
        assert resp2.status_code == 404

    def test_events_after_report_call(self, monkeypatch):
        ds = _ingest_drill()
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.post("/v1/report", json={"dataset_path": ds, "schema": "sft"})
        assert resp.status_code == 200

        billing_resp = client.get("/v1/billing/events")
        assert billing_resp.status_code == 200
        events = billing_resp.json()["events"]
        assert len(events) >= 1
        assert events[0]["endpoint"] == "/v1/report"
        assert events[0]["units"]["rows_in"] > 0

    def test_events_after_pipeline_call(self, monkeypatch, tmp_path):
        ds = _ingest_drill()
        from verifily_cli_v1.core.io import write_yaml

        cfg = {
            "run_dir": str(DRILL_DIR / "runs" / "run_clean"),
            "train_data": ds,
            "eval_data": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
            "baseline_run": str(DRILL_DIR / "runs" / "run_clean"),
            "ship_if": {
                "min_f1": 0.65,
                "min_exact_match": 0.50,
                "max_f1_regression": 0.03,
                "max_pii_hits": 10,
            },
        }
        config_path = tmp_path / "pipeline.yaml"
        write_yaml(config_path, cfg)

        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.post("/v1/pipeline", json={"config_path": str(config_path), "ci": True, "plan": True})
        assert resp.status_code == 200

        billing_resp = client.get("/v1/billing/events")
        events = billing_resp.json()["events"]
        assert len(events) >= 1
        pipe_events = [e for e in events if e["endpoint"] == "/v1/pipeline"]
        assert len(pipe_events) >= 1

    def test_invalid_plan_422(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/invoice-preview", params={"project_id": "default", "plan_id": "BOGUS"})
        assert resp.status_code == 422

    def test_invoice_preview_starter(self, monkeypatch):
        ds = _ingest_drill()
        client = _make_client(monkeypatch, enable_billing=True)
        # Generate some usage
        client.post("/v1/report", json={"dataset_path": ds, "schema": "sft"})

        resp = client.get("/v1/billing/invoice-preview", params={"project_id": "default", "plan_id": "STARTER"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_id"] == "STARTER"
        assert data["monthly_base_cents"] == 9900
        assert len(data["lines"]) == 4


# ── Billing + Advanced Auth ────────────────────────────────────


class TestBillingAdvancedAuth:
    def test_billing_requires_usage_read_scope(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced",
            enable_admin=True,
            enable_billing=True,
            key_salt="test-salt",
        )
        from verifily_cli_v1.core.api.auth_registry import auth_registry

        # Create a key with NO usage:read scope
        auth_registry.create_key(
            id="k-no-usage",
            name="No Usage",
            raw_key="sk-test-no-usage-key",
            scopes=["pipeline:run"],
            projects_allowed=["*"],
        )

        resp = client.get(
            "/v1/billing/events",
            headers={"Authorization": "Bearer sk-test-no-usage-key"},
        )
        assert resp.status_code == 403

    def test_billing_allowed_with_usage_read(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            auth_mode="advanced",
            enable_admin=True,
            enable_billing=True,
            key_salt="test-salt",
        )
        from verifily_cli_v1.core.api.auth_registry import auth_registry

        auth_registry.create_key(
            id="k-with-usage",
            name="Usage Reader",
            raw_key="sk-test-usage-reader-key",
            scopes=["usage:read"],
            projects_allowed=["*"],
        )

        resp = client.get(
            "/v1/billing/events",
            headers={"Authorization": "Bearer sk-test-usage-reader-key"},
        )
        assert resp.status_code == 200


# ── Export Tests ──────────────────────────────────────────────────


class TestExport:
    def _sample_events(self):
        return [
            BillingEvent(ts=1704067200.0, api_key_id="k1", project_id="p1",
                         endpoint="/v1/report", units={"rows_in": 100, "bytes_in": 5000}),
            BillingEvent(ts=1704067200.0 + 3600, api_key_id="k2", project_id="p2",
                         endpoint="/v1/pipeline", units={"rows_in": 200, "rows_out": 50, "decisions": 3}),
            BillingEvent(ts=1704153600.0, api_key_id="k1", project_id="p1",
                         endpoint="/v1/report", units={"rows_in": 300}),
        ]

    def test_csv_header(self):
        csv_out = export_usage_csv(self._sample_events(), group_by="day")
        header = csv_out.splitlines()[0]
        assert "date" in header
        assert "api_key_id" in header
        assert "requests" in header
        assert "rows_in" in header

    def test_csv_day_grouping(self):
        csv_out = export_usage_csv(self._sample_events(), group_by="day")
        lines = csv_out.strip().splitlines()
        # header + 2 days
        assert len(lines) == 3

    def test_csv_day_project_grouping(self):
        csv_out = export_usage_csv(self._sample_events(), group_by="day_project")
        lines = csv_out.strip().splitlines()
        # header + 3 buckets (day1:p1, day1:p2, day2:p1)
        assert len(lines) == 4

    def test_csv_day_api_key_grouping(self):
        csv_out = export_usage_csv(self._sample_events(), group_by="day_api_key")
        lines = csv_out.strip().splitlines()
        # header + 3 buckets (day1:k1, day1:k2, day2:k1)
        assert len(lines) == 4

    def test_jsonl_output(self):
        jsonl_out = export_usage_jsonl(self._sample_events(), group_by="day")
        lines = [l for l in jsonl_out.strip().splitlines() if l]
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert "date" in row
        assert "requests" in row

    def test_empty_events_csv(self):
        csv_out = export_usage_csv([], group_by="day")
        lines = csv_out.strip().splitlines()
        assert len(lines) == 1  # header only

    def test_empty_events_jsonl(self):
        jsonl_out = export_usage_jsonl([], group_by="day")
        assert jsonl_out == ""

    def test_bucket_aggregation(self):
        events = self._sample_events()
        buckets = _bucket_events(events, group_by="day")
        # day 1: 2 events (100+200 rows_in)
        assert buckets[0]["requests"] == 2
        assert buckets[0]["rows_in"] == 300
        assert buckets[0]["decisions"] == 3


# ── Store Extended Tests ──────────────────────────────────────────


class TestBillingStoreExtended:
    def test_get_raw_events(self):
        store = BillingStore()
        store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={"rows_in": 10})
        store.record_event(api_key_id="k2", project_id="p2", endpoint="/v1/pipeline", units={"rows_in": 20})
        events = store.get_raw_events()
        assert len(events) == 2
        assert isinstance(events[0], BillingEvent)

    def test_get_raw_events_filter_project(self):
        store = BillingStore()
        store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report", units={})
        store.record_event(api_key_id="k1", project_id="p2", endpoint="/v1/pipeline", units={})
        events = store.get_raw_events(project_id="p1")
        assert len(events) == 1

    def test_generate_invoice(self):
        store = BillingStore()
        now = time.time()
        store.record_event(api_key_id="k1", project_id="p1", endpoint="/v1/report",
                           units={"rows_in": 50})
        inv = store.generate_invoice(
            project_id="p1", plan_id="FREE",
            period_start=now - 3600, period_end=now + 3600,
        )
        assert inv["plan_id"] == "FREE"
        assert inv["total_cents"] == 0


# ── Billing Extended API Tests ────────────────────────────────────


class TestBillingExtendedAPI:
    def test_plans_endpoint(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/plans")
        assert resp.status_code == 200
        data = resp.json()
        assert "plans" in data
        assert len(data["plans"]) == 4
        ids = {p["id"] for p in data["plans"]}
        assert "FREE" in ids
        assert "PRO" in ids

    def test_plans_disabled_404(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=False)
        resp = client.get("/v1/billing/plans")
        assert resp.status_code == 404

    def test_estimate_endpoint(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/estimate", params={"plan": "FREE"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_id"] == "FREE"
        assert "lines" in data
        assert data["total_cents"] == 0

    def test_estimate_with_usage(self, monkeypatch):
        ds = _ingest_drill()
        client = _make_client(monkeypatch, enable_billing=True)
        client.post("/v1/report", json={"dataset_path": ds, "schema": "sft"})
        resp = client.get("/v1/billing/estimate", params={"plan": "STARTER"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_id"] == "STARTER"
        assert data["monthly_base_cents"] == 9900

    def test_estimate_unknown_plan_422(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/estimate", params={"plan": "BOGUS"})
        assert resp.status_code == 422

    def test_invoice_endpoint(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.post("/v1/billing/invoice", json={
            "plan": "FREE",
            "period_start": "2024-01-01T00:00:00+00:00",
            "period_end": "2024-02-01T00:00:00+00:00",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "invoice_id" in data
        assert data["invoice_id"].startswith("inv-")
        assert data["plan_id"] == "FREE"

    def test_invoice_unknown_plan_422(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.post("/v1/billing/invoice", json={
            "plan": "BOGUS",
            "period_start": "2024-01-01T00:00:00+00:00",
            "period_end": "2024-02-01T00:00:00+00:00",
        })
        assert resp.status_code == 422

    def test_usage_export_csv(self, monkeypatch):
        ds = _ingest_drill()
        client = _make_client(monkeypatch, enable_billing=True)
        client.post("/v1/report", json={"dataset_path": ds, "schema": "sft"})
        resp = client.get("/v1/billing/usage_export", params={"format": "csv", "period_days": 30})
        assert resp.status_code == 200
        assert "date" in resp.text
        assert "rows_in" in resp.text

    def test_usage_export_jsonl(self, monkeypatch):
        ds = _ingest_drill()
        client = _make_client(monkeypatch, enable_billing=True)
        client.post("/v1/report", json={"dataset_path": ds, "schema": "sft"})
        resp = client.get("/v1/billing/usage_export", params={"format": "jsonl", "period_days": 30})
        assert resp.status_code == 200
        lines = [l for l in resp.text.strip().splitlines() if l]
        assert len(lines) >= 1
        row = json.loads(lines[0])
        assert "date" in row

    def test_usage_export_invalid_format(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=True)
        resp = client.get("/v1/billing/usage_export", params={"format": "xml"})
        assert resp.status_code == 422

    def test_usage_export_disabled_404(self, monkeypatch):
        client = _make_client(monkeypatch, enable_billing=False)
        resp = client.get("/v1/billing/usage_export", params={"format": "csv"})
        assert resp.status_code == 404
