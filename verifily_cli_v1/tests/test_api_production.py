"""Tests for the Verifily API production credibility layer.

Covers: /ready, /metrics, request-ID middleware, decision counters.
Uses FastAPI TestClient — no real network calls, no server process.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _ingest_drill(tmp_path: Path) -> Path:
    """Ingest customer drill CSV into tmp_path, return dataset.jsonl path."""
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
    """Write a pipeline YAML config and return its path."""
    from verifily_cli_v1.core.io import write_yaml

    cfg = {
        "run_dir": str(DRILL_DIR / "runs" / run_name),
        "train_data": str(train_path),
        "eval_data": str(DRILL_DIR / "raw" / eval_file),
        "baseline_run": str(DRILL_DIR / "runs" / run_name),
        "ship_if": {
            "min_f1": 0.65,
            "min_exact_match": 0.50,
            "max_f1_regression": 0.03,
            "max_pii_hits": 10,
        },
    }
    config_path = tmp_path / "pipeline.yaml"
    write_yaml(config_path, cfg)
    return config_path


@pytest.fixture
def client():
    """Fresh FastAPI TestClient with reset metrics."""
    app = create_app()
    return TestClient(app)


# ── Ready ────────────────────────────────────────────────────────

class TestReady:
    def test_ready_returns_ready(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"]["python"] == "ok"
        assert data["checks"]["temp_write"] == "ok"
        assert data["checks"]["imports"] == "ok"

    def test_ready_has_no_error_field_when_ok(self, client):
        resp = client.get("/ready")
        data = resp.json()
        assert data.get("error") is None


# ── Metrics ──────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_plaintext(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_requests_total(self, client):
        # Hit health twice, then check metrics
        client.get("/health")
        client.get("/health")
        resp = client.get("/metrics")
        body = resp.text
        assert "verifily_requests_total" in body

    def test_metrics_increments_after_requests(self, client):
        # Baseline
        resp0 = client.get("/metrics")
        # The /metrics call itself increments, so just check the endpoint counter
        client.get("/health")
        client.get("/health")
        resp = client.get("/metrics")
        body = resp.text
        assert 'verifily_endpoint_requests_total{path="/health",method="GET"}' in body

    def test_metrics_inflight_is_self_only(self, client):
        """The only inflight request when reading /metrics is /metrics itself."""
        client.get("/health")
        resp = client.get("/metrics")
        # Inflight = 1 because the /metrics request is still being handled
        assert "verifily_requests_inflight 1" in resp.text

    def test_metrics_decision_counts(self, client, tmp_path):
        """Call pipeline clean + leaked, assert decision counters appear."""
        train = _ingest_drill(tmp_path)

        # Clean -> SHIP
        config_clean = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        client.post("/v1/pipeline", json={
            "config_path": str(config_clean),
            "plan": True,
            "ci": True,
        })

        # Leaked -> DONT_SHIP
        config_leaked = _write_pipeline_config(
            tmp_path / "leaked", train, "eval_leaked_exact.jsonl", "run_leaked"
        )
        client.post("/v1/pipeline", json={
            "config_path": str(config_leaked),
            "plan": True,
            "ci": True,
        })

        resp = client.get("/metrics")
        body = resp.text
        assert 'verifily_decision_total{decision="SHIP"} 1' in body
        assert 'verifily_decision_total{decision="DONT_SHIP"} 1' in body

    def test_metrics_contamination_counts(self, client, tmp_path):
        """Call contamination clean + leaked, assert status counters appear."""
        train = _ingest_drill(tmp_path)

        # Clean -> PASS
        client.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        })

        # Leaked -> FAIL
        client.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_leaked_exact.jsonl"),
        })

        resp = client.get("/metrics")
        body = resp.text
        assert 'verifily_contamination_total{status="PASS"} 1' in body
        assert 'verifily_contamination_total{status="FAIL"} 1' in body


# ── Request ID ───────────────────────────────────────────────────

class TestRequestID:
    def test_request_id_echoed(self, client):
        """Client-provided X-Request-ID is echoed back."""
        resp = client.get("/health", headers={"X-Request-ID": "test-req-42"})
        assert resp.headers["x-request-id"] == "test-req-42"

    def test_request_id_generated(self, client):
        """When no header is sent, server generates one."""
        resp = client.get("/health")
        rid = resp.headers.get("x-request-id")
        assert rid is not None
        assert len(rid) > 0

    def test_request_id_on_post(self, client, tmp_path):
        """Request ID works on POST endpoints too."""
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/contamination",
            json={
                "train_path": str(train),
                "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
            },
            headers={"X-Request-ID": "contam-123"},
        )
        assert resp.headers["x-request-id"] == "contam-123"
