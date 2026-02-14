"""Tests for API key authentication middleware.

Covers:
  - Auth disabled (no env var) -> all endpoints work as before
  - Auth enabled -> missing key rejected, correct key accepted, wrong key rejected
  - Public endpoints always accessible
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"

TEST_API_KEY = "test-verifily-key-42"


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


def _write_pipeline_config(tmp_path: Path, train_path: Path) -> Path:
    from verifily_cli_v1.core.io import write_yaml

    cfg = {
        "run_dir": str(DRILL_DIR / "runs" / "run_clean"),
        "train_data": str(train_path),
        "eval_data": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        "baseline_run": str(DRILL_DIR / "runs" / "run_clean"),
        "ship_if": {
            "min_f1": 0.65, "min_exact_match": 0.50,
            "max_f1_regression": 0.03, "max_pii_hits": 10,
        },
    }
    config_path = tmp_path / "pipeline.yaml"
    write_yaml(config_path, cfg)
    return config_path


@pytest.fixture
def client_no_auth(shared_app, monkeypatch):
    """Client with auth DISABLED (no VERIFILY_API_KEY)."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    return TestClient(shared_app)


@pytest.fixture
def client_with_auth(shared_app, monkeypatch):
    """Client with auth ENABLED."""
    monkeypatch.setenv("VERIFILY_API_KEY", TEST_API_KEY)
    return TestClient(shared_app)


# ── Auth disabled ────────────────────────────────────────────────

class TestAuthDisabled:
    def test_health_works(self, client_no_auth):
        assert client_no_auth.get("/health").status_code == 200

    def test_ready_works(self, client_no_auth):
        assert client_no_auth.get("/ready").status_code == 200

    def test_metrics_works(self, client_no_auth):
        assert client_no_auth.get("/metrics").status_code == 200

    def test_pipeline_works_without_key(self, client_no_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train)
        resp = client_no_auth.post("/v1/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        assert resp.status_code == 200

    def test_contamination_works_without_key(self, client_no_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client_no_auth.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        })
        assert resp.status_code == 200


# ── Auth enabled ─────────────────────────────────────────────────

class TestAuthEnabled:
    def test_public_endpoints_no_key_needed(self, client_with_auth):
        assert client_with_auth.get("/health").status_code == 200
        assert client_with_auth.get("/ready").status_code == 200
        assert client_with_auth.get("/metrics").status_code == 200

    def test_pipeline_rejected_without_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train)
        resp = client_with_auth.post("/v1/pipeline", json={
            "config_path": str(config), "plan": True,
        })
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "AUTH_ERROR"

    def test_pipeline_rejected_with_wrong_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train)
        resp = client_with_auth.post(
            "/v1/pipeline",
            json={"config_path": str(config), "plan": True},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_pipeline_accepted_with_correct_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train)
        resp = client_with_auth.post(
            "/v1/pipeline",
            json={"config_path": str(config), "plan": True, "ci": True},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["exit_code"] == 0

    def test_contamination_rejected_without_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client_with_auth.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        })
        assert resp.status_code == 401

    def test_contamination_accepted_with_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client_with_auth.post(
            "/v1/contamination",
            json={
                "train_path": str(train),
                "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200

    def test_report_rejected_without_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client_with_auth.post("/v1/report", json={
            "dataset_path": str(train),
        })
        assert resp.status_code == 401

    def test_401_body_format(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client_with_auth.post("/v1/report", json={
            "dataset_path": str(train),
        })
        body = resp.json()
        assert "error" in body
        assert isinstance(body["error"], dict)
        assert body["error"]["type"] == "AUTH_ERROR"
        assert "message" in body["error"]
        assert "request_id" in body["error"]
