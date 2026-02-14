"""Tests for enterprise audit export: store + API endpoint."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.audit import AuditEvent
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.security.audit_store import audit_store
from verifily_cli_v1.core.security.tokens import create_token


SECRET = "test-audit-secret"


def _make_client(monkeypatch, **kwargs):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_AUTH_MODE", raising=False)
    monkeypatch.delenv("VERIFILY_ENTERPRISE_SECURITY", raising=False)
    monkeypatch.delenv("VERIFILY_TOKEN_SECRET", raising=False)
    settings = load_settings(**kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    return TestClient(app)


def _token(role="owner", **overrides):
    payload = {
        "token_id": "tok_test",
        "role": role,
        "project_id": "proj1",
        "exp": time.time() + 3600,
    }
    payload.update(overrides)
    return create_token(SECRET, payload)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── AuditStore Unit Tests ────────────────────────────────────────


class TestAuditStore:
    def setup_method(self):
        audit_store.reset()

    def test_record_and_query(self):
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1"))
        events = audit_store.query()
        assert len(events) == 1
        assert events[0]["step"] == "PIPELINE"

    def test_project_filter(self):
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1", project="proj1"))
        audit_store.record(AuditEvent(step="REPORT", status="OK", run_id="r2", project="proj2"))
        events = audit_store.query(project_id="proj1")
        assert len(events) == 1
        assert events[0]["project"] == "proj1"

    def test_time_window(self):
        e1 = AuditEvent(step="PIPELINE", status="OK", run_id="r1")
        e2 = AuditEvent(step="REPORT", status="OK", run_id="r2")
        audit_store.record(e1)
        audit_store.record(e2)
        # Query with from_ts after both — should get 0
        future_ts = "2099-01-01T00:00:00.000000Z"
        events = audit_store.query(from_ts=future_ts)
        assert len(events) == 0

    def test_reset_clears(self):
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1"))
        audit_store.reset()
        assert audit_store.query() == []

    def test_limit(self):
        for i in range(10):
            audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id=f"r{i}"))
        events = audit_store.query(limit=3)
        assert len(events) == 3

    def test_most_recent_first(self):
        audit_store.record(AuditEvent(step="FIRST", status="OK", run_id="r1"))
        audit_store.record(AuditEvent(step="SECOND", status="OK", run_id="r2"))
        events = audit_store.query()
        assert events[0]["step"] == "SECOND"
        assert events[1]["step"] == "FIRST"


# ── Endpoint Tests ───────────────────────────────────────────────


class TestAuditExportEndpoint:
    def test_disabled_returns_404(self, monkeypatch):
        client = _make_client(monkeypatch)
        resp = client.get("/v1/audit/export")
        assert resp.status_code == 404

    def test_owner_can_export(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            enterprise_security=True,
            token_secret=SECRET,
        )
        # Seed an event
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1", project="proj1"))

        tok = _token("owner")
        resp = client.get("/v1/audit/export", headers=_auth(tok))
        assert resp.status_code == 200
        body = resp.json()
        assert "events" in body
        assert body["total"] >= 1

    def test_viewer_denied_403(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            enterprise_security=True,
            token_secret=SECRET,
        )
        tok = _token("viewer")
        resp = client.get("/v1/audit/export", headers=_auth(tok))
        assert resp.status_code == 403

    def test_jsonl_format(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            enterprise_security=True,
            token_secret=SECRET,
        )
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1"))

        tok = _token("owner")
        resp = client.get("/v1/audit/export?format=jsonl", headers=_auth(tok))
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert parsed["step"] == "PIPELINE"

    def test_project_isolation(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            enterprise_security=True,
            token_secret=SECRET,
        )
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1", project="proj1"))
        audit_store.record(AuditEvent(step="REPORT", status="OK", run_id="r2", project="proj2"))

        tok = _token("owner")
        resp = client.get("/v1/audit/export?project_id=proj1", headers=_auth(tok))
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["events"][0]["project"] == "proj1"

    def test_admin_can_export(self, monkeypatch):
        client = _make_client(
            monkeypatch,
            enterprise_security=True,
            token_secret=SECRET,
        )
        audit_store.record(AuditEvent(step="PIPELINE", status="OK", run_id="r1"))

        tok = _token("admin")
        resp = client.get("/v1/audit/export", headers=_auth(tok))
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1
