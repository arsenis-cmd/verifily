"""Tests for the Verifily local API server.

Uses FastAPI TestClient — no real network calls, no server process.
All tests are deterministic and fast (<2s each).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app, validate_host

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"

# Reuse the ingest helper to produce training data for pipeline tests
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
def client(shared_app):
    """FastAPI TestClient (reuses module-scoped app)."""
    return TestClient(shared_app)


# ── Health ───────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["mode"] == "local"
        assert "version" in data
        assert "time" in data

    def test_health_version_matches(self, client):
        from verifily_cli_v1 import __version__
        resp = client.get("/health")
        assert resp.json()["version"] == __version__


# ── Pipeline ─────────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_plan_returns_decision(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": True,
            "ci": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "exit_code" in data
        assert "decision" in data
        assert data["decision"]["recommendation"] in ("SHIP", "DONT_SHIP", "INVESTIGATE")
        assert data["output_dir"] is None  # plan mode writes nothing

    def test_pipeline_plan_false_writes_outputs(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": False,
            "ci": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["output_dir"] is not None
        assert Path(data["output_dir"]).exists()
        # Clean up
        shutil.rmtree(data["output_dir"], ignore_errors=True)

    def test_pipeline_clean_ships(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": True,
            "ci": True,
        })
        data = resp.json()
        assert data["exit_code"] == 0
        assert data["decision"]["recommendation"] == "SHIP"

    def test_pipeline_leaked_blocks(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": True,
            "ci": True,
        })
        data = resp.json()
        assert data["exit_code"] == 1
        assert data["decision"]["recommendation"] == "DONT_SHIP"
        assert "contamination_fail" in data["decision"]["risk_flags"]

    def test_pipeline_includes_contamination_summary(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": True,
            "ci": True,
        })
        data = resp.json()
        assert data["contamination"] is not None
        assert data["contamination"]["status"] == "FAIL"
        assert data["contamination"]["exact_overlaps"] == 4

    def test_pipeline_includes_contract_summary(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": True,
            "ci": True,
        })
        data = resp.json()
        assert data["contract"] is not None
        assert data["contract"]["valid"] is True

    def test_pipeline_has_elapsed_ms(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/pipeline", json={
            "config_path": str(config),
            "plan": True,
            "ci": True,
        })
        data = resp.json()
        assert "elapsed_ms" in data
        assert isinstance(data["elapsed_ms"], int)
        assert data["elapsed_ms"] >= 0

    def test_pipeline_missing_config_returns_404(self, client):
        resp = client.post("/v1/pipeline", json={
            "config_path": "/nonexistent/verifily.yaml",
            "plan": True,
        })
        assert resp.status_code == 404

    def test_pipeline_no_path_returns_422(self, client):
        resp = client.post("/v1/pipeline", json={"plan": True})
        assert resp.status_code == 422


# ── Contamination ────────────────────────────────────────────────

class TestContamination:
    def test_contamination_clean(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "PASS"
        assert data["exit_code"] == 0

    def test_contamination_exact_leak(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_leaked_exact.jsonl"),
        })
        data = resp.json()
        assert data["status"] == "FAIL"
        assert data["exit_code"] == 1
        assert data["exact_overlaps"] == 4

    def test_contamination_near_leak(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_leaked_near.jsonl"),
        })
        data = resp.json()
        assert data["status"] == "WARN"
        assert data["exit_code"] == 2
        assert data["near_duplicates"] == 3


# ── Report ───────────────────────────────────────────────────────

class TestReport:
    def test_report_returns_stats(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/report", json={
            "dataset_path": str(train),
            "schema": "sft",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["row_count"] == 28
        assert data["schema"] == "sft"
        assert "field_stats" in data
        assert "pii_summary" in data
        assert data["exit_code"] == 0

    def test_report_pii_summary_counts_only(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/report", json={
            "dataset_path": str(train),
        })
        data = resp.json()
        # PII summary should be counts, never raw PII values
        for pii_type, count in data["pii_summary"].items():
            assert isinstance(count, int)

    def test_report_with_sample(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/report", json={
            "dataset_path": str(train),
            "sample": 3,
        })
        data = resp.json()
        assert data["sample_rows"] is not None
        assert len(data["sample_rows"]) == 3


# ── Host safety ──────────────────────────────────────────────────

class TestHostSafety:
    def test_localhost_allowed(self):
        # Should not raise
        validate_host("127.0.0.1", allow_nonlocal=False)
        validate_host("localhost", allow_nonlocal=False)

    def test_nonlocal_rejected_by_default(self):
        with pytest.raises(ValueError, match="Refusing to bind"):
            validate_host("0.0.0.0", allow_nonlocal=False)

    def test_nonlocal_allowed_with_flag(self):
        # Should not raise
        validate_host("0.0.0.0", allow_nonlocal=True)

    def test_custom_ip_rejected(self):
        with pytest.raises(ValueError, match="Refusing to bind"):
            validate_host("192.168.1.100", allow_nonlocal=False)

    def test_custom_ip_allowed_with_flag(self):
        validate_host("192.168.1.100", allow_nonlocal=True)
