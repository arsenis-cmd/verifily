"""Tests for multi-tenant usage accounting.

Covers: compute_api_key_id, UsageStore record/query/reset,
project_id header precedence via TestClient, /v1/usage endpoint
(empty, after pipeline, group_by modes, window filter), auth behavior.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.usage_store import UsageStore, compute_api_key_id

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _ingest_drill(tmp_path: Path) -> Path:
    from verifily_cli_v1.commands.ingest import ingest

    ingest(
        DRILL_DIR / "raw" / "support_tickets.csv",
        tmp_path / "artifact",
        schema="sft",
        mapping={"question": "subject", "answer": "resolution", "context": "body"},
        extra_tags={"source": "customer_drill"},
    )
    return tmp_path / "artifact" / "dataset.jsonl"


def _write_pipeline_config(tmp_path: Path, train_path: Path, eval_file: str, run_name: str) -> Path:
    from verifily_cli_v1.core.io import write_yaml

    cfg = {
        "run_dir": str(DRILL_DIR / "runs" / run_name),
        "train_data": str(train_path),
        "eval_data": str(DRILL_DIR / "raw" / eval_file),
        "baseline_run": str(DRILL_DIR / "runs" / run_name),
        "ship_if": {
            "min_f1": 0.65, "min_exact_match": 0.50,
            "max_f1_regression": 0.03, "max_pii_hits": 10,
        },
    }
    config_path = tmp_path / "pipeline.yaml"
    write_yaml(config_path, cfg)
    return config_path


@pytest.fixture
def client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    from verifily_cli_v1.core.api.usage_store import usage_store
    from verifily_cli_v1.core.api.metrics import metrics
    usage_store.reset()
    metrics.reset()
    return TestClient(shared_app)


# ── compute_api_key_id ──────────────────────────────────────────

class TestComputeApiKeyId:
    def test_anonymous_for_none(self):
        assert compute_api_key_id(None) == "anonymous"

    def test_anonymous_for_empty(self):
        assert compute_api_key_id("") == "anonymous"

    def test_deterministic(self):
        kid1 = compute_api_key_id("my-secret-key")
        kid2 = compute_api_key_id("my-secret-key")
        assert kid1 == kid2

    def test_12_char_hex(self):
        kid = compute_api_key_id("test-key-123")
        assert len(kid) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", kid)

    def test_different_keys_different_ids(self):
        kid1 = compute_api_key_id("key-a")
        kid2 = compute_api_key_id("key-b")
        assert kid1 != kid2


# ── UsageStore unit tests ───────────────────────────────────────

class TestUsageStore:
    def test_record_and_query_alltime(self):
        store = UsageStore()
        store.record(api_key_id="abc", project_id="proj1", elapsed_ms=100, decision="SHIP", rows_in=10)
        result = store.query(group_by="key_project")
        assert "buckets" in result
        assert len(result["buckets"]) == 1
        b = result["buckets"][0]
        assert b["api_key_id"] == "abc"
        assert b["project_id"] == "proj1"
        assert b["requests"] == 1
        assert b["decisions_ship"] == 1
        assert b["rows_in"] == 10
        assert b["elapsed_ms_sum"] == 100

    def test_multiple_records_aggregate(self):
        store = UsageStore()
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=50, decision="SHIP")
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=30, decision="DONT_SHIP")
        result = store.query(group_by="key_project")
        b = result["buckets"][0]
        assert b["requests"] == 2
        assert b["decisions_ship"] == 1
        assert b["decisions_dont_ship"] == 1
        assert b["elapsed_ms_sum"] == 80

    def test_group_by_key(self):
        store = UsageStore()
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=10)
        store.record(api_key_id="k1", project_id="p2", elapsed_ms=20)
        result = store.query(group_by="key")
        assert len(result["buckets"]) == 1
        assert result["buckets"][0]["api_key_id"] == "k1"
        assert result["buckets"][0]["requests"] == 2

    def test_group_by_project(self):
        store = UsageStore()
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=10)
        store.record(api_key_id="k2", project_id="p1", elapsed_ms=20)
        result = store.query(group_by="project")
        assert len(result["buckets"]) == 1
        assert result["buckets"][0]["project_id"] == "p1"
        assert result["buckets"][0]["requests"] == 2

    def test_group_by_total(self):
        store = UsageStore()
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=10)
        store.record(api_key_id="k2", project_id="p2", elapsed_ms=20)
        result = store.query(group_by="total")
        assert "total" in result
        assert result["total"]["requests"] == 2
        assert result["total"]["elapsed_ms_sum"] == 30

    def test_window_filter(self):
        store = UsageStore()
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=10)
        # Query with a window that includes now
        result = store.query(window_minutes=5, group_by="total")
        assert result["total"]["requests"] == 1

    def test_reset(self):
        store = UsageStore()
        store.record(api_key_id="k1", project_id="p1", elapsed_ms=10)
        store.reset()
        result = store.query(group_by="total")
        assert result["total"]["requests"] == 0

    def test_empty_query(self):
        store = UsageStore()
        result = store.query(group_by="key_project")
        assert result["buckets"] == []


# ── /v1/usage endpoint ──────────────────────────────────────────

class TestUsageEndpoint:
    def test_usage_empty(self, client):
        resp = client.get("/v1/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["buckets"] == []

    def test_usage_after_report(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        client.post("/v1/report", json={"dataset_path": str(train)})
        resp = client.get("/v1/usage")
        data = resp.json()
        assert len(data["buckets"]) == 1
        assert data["buckets"][0]["requests"] == 1
        assert data["buckets"][0]["api_key_id"] == "anonymous"
        assert data["buckets"][0]["project_id"] == "default"

    def test_usage_after_pipeline(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        client.post("/v1/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        resp = client.get("/v1/usage")
        data = resp.json()
        assert len(data["buckets"]) == 1
        assert data["buckets"][0]["requests"] == 1
        assert data["buckets"][0]["decisions_ship"] == 1

    def test_usage_group_by_total(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        client.post("/v1/report", json={"dataset_path": str(train)})
        client.post("/v1/report", json={"dataset_path": str(train)})
        resp = client.get("/v1/usage", params={"group_by": "total"})
        data = resp.json()
        assert data["total"]["requests"] == 2

    def test_usage_window_filter(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        client.post("/v1/report", json={"dataset_path": str(train)})
        resp = client.get("/v1/usage", params={"window_minutes": 5, "group_by": "total"})
        data = resp.json()
        assert data["total"]["requests"] == 1

    def test_usage_invalid_group_by(self, client):
        resp = client.get("/v1/usage", params={"group_by": "invalid"})
        assert resp.status_code == 422


# ── project_id header precedence ─────────────────────────────────

class TestProjectIdPrecedence:
    def test_header_takes_precedence_over_body(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        client.post(
            "/v1/report",
            json={"dataset_path": str(train), "project_id": "body-project"},
            headers={"X-Project-ID": "header-project"},
        )
        resp = client.get("/v1/usage")
        data = resp.json()
        assert data["buckets"][0]["project_id"] == "header-project"

    def test_body_used_when_no_header(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        client.post(
            "/v1/report",
            json={"dataset_path": str(train), "project_id": "body-project"},
        )
        resp = client.get("/v1/usage")
        data = resp.json()
        assert data["buckets"][0]["project_id"] == "body-project"

    def test_default_when_none(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        client.post("/v1/report", json={"dataset_path": str(train)})
        resp = client.get("/v1/usage")
        data = resp.json()
        assert data["buckets"][0]["project_id"] == "default"


# ── auth behavior for /v1/usage ──────────────────────────────────

class TestUsageAuth:
    def test_usage_requires_auth_when_enabled(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_API_KEY", "secret-key")
        app = create_app()
        c = TestClient(app)
        resp = c.get("/v1/usage")
        assert resp.status_code == 401

    def test_usage_accessible_with_correct_key(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_API_KEY", "secret-key")
        app = create_app()
        c = TestClient(app)
        resp = c.get("/v1/usage", headers={"Authorization": "Bearer secret-key"})
        assert resp.status_code == 200

    def test_usage_accessible_without_auth_when_disabled(self, client):
        resp = client.get("/v1/usage")
        assert resp.status_code == 200
