"""Tests for RETRAIN job type.

Covers:
  - Submit retrain job → QUEUED → drain → SUCCEEDED
  - Mock artifacts exist on disk and pass contract validation
  - Deterministic outputs (same dataset+seed → same metrics)
  - Decision is SHIP/DONT_SHIP based on mock metrics
  - Real mode blocked without env var
  - Sync endpoint works
  - Monitor retrain trigger
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.monitor_store import MonitorConfig, monitor_store
from verifily_cli_v1.core.io import read_json

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
    return tmp_path / "artifact"


@pytest.fixture
def client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_ENABLE_REAL_TRAIN", raising=False)
    jobs_store.reset()
    monitor_store.reset()
    return TestClient(shared_app)


# ── Async job submit + drain ─────────────────────────────────────

class TestRetrainSubmit:
    def test_submit_returns_queued(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "QUEUED"
        assert "job_id" in data

    def test_drain_transitions_to_succeeded(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        })
        job_id = resp.json()["job_id"]
        jobs_store.drain()

        meta = client.get(f"/v1/jobs/{job_id}").json()
        assert meta["status"] == "SUCCEEDED"

    def test_result_includes_run_dir_and_decision(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        })
        job_id = resp.json()["job_id"]
        jobs_store.drain()

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert "run_dir" in result
        assert "decision" in result
        assert result["decision"]["recommendation"] in ("SHIP", "DONT_SHIP", "INVESTIGATE")
        assert result["exit_code"] in (0, 1, 2, 3, 4)


# ── Mock artifacts ───────────────────────────────────────────────

class TestRetrainMockArtifacts:
    def test_all_contract_files_exist(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        })
        assert resp.status_code == 200
        result = resp.json()
        run_dir = Path(result["run_dir"])

        # Contract files
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "hashes.json").exists()
        assert (run_dir / "environment.json").exists()
        assert (run_dir / "eval" / "eval_results.json").exists()
        assert (run_dir / "run_meta.json").exists()

    def test_decision_artifacts_exist(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        result = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        }).json()
        run_dir = Path(result["run_dir"])

        assert (run_dir / "decision" / "decision.json").exists()
        assert (run_dir / "decision" / "decision.txt").exists()
        assert (run_dir / "usage.json").exists()
        assert (run_dir / "audit_log.jsonl").exists()

    def test_contract_validates(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        result = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        }).json()

        from verifily_cli_v1.commands.contract_check import validate_run_contract
        contract = validate_run_contract(result["run_dir"])
        assert contract["valid"] is True
        assert contract["has_eval"] is True

    def test_run_meta_completed(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        result = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        }).json()

        meta = read_json(Path(result["run_dir"]) / "run_meta.json")
        assert meta["status"] == "completed"


# ── Deterministic outputs ────────────────────────────────────────

class TestRetrainDeterministic:
    def test_same_seed_same_metrics(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        r1 = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "out1"),
            "seed": 42,
        }).json()
        r2 = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "out2"),
            "seed": 42,
        }).json()

        assert r1["eval_summary"] == r2["eval_summary"]
        assert r1["decision"]["recommendation"] == r2["decision"]["recommendation"]

    def test_different_seed_different_metrics(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        r1 = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "out1"),
            "seed": 42,
        }).json()
        r2 = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "out2"),
            "seed": 99,
        }).json()

        # Different seeds should produce different metrics
        assert r1["eval_summary"]["f1"] != r2["eval_summary"]["f1"]


# ── Real mode gate ───────────────────────────────────────────────

class TestRetrainRealMode:
    def test_real_mode_blocked_without_env(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "mode": "real",
        })
        assert resp.status_code == 400
        assert "VERIFILY_ENABLE_REAL_TRAIN" in resp.json()["error"]["message"]


# ── Sync endpoint ────────────────────────────────────────────────

class TestRetrainSync:
    def test_sync_returns_result(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = client.post("/v1/retrain", json={
            "dataset_dir": str(ds),
            "output_dir": str(tmp_path / "retrain_out"),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_type"] == "RETRAIN"
        assert "run_dir" in data
        assert "decision" in data
        assert "artifacts" in data

    def test_missing_dataset_returns_404(self, client):
        resp = client.post("/v1/retrain", json={
            "dataset_dir": "/nonexistent/path",
        })
        assert resp.status_code == 404


# ── Monitor retrain trigger ──────────────────────────────────────

class TestMonitorRetrain:
    def _write_pipeline_config(self, tmp_path, train_path):
        from verifily_cli_v1.core.io import write_yaml

        cfg = {
            "run_dir": str(DRILL_DIR / "runs" / "run_clean"),
            "train_data": str(train_path),
            "eval_data": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
            "baseline_run": str(DRILL_DIR / "runs" / "run_clean"),
            "ship_if": {
                "min_f1": 0.50,
                "min_exact_match": 0.40,
                "max_f1_regression": 0.05,
                "max_pii_hits": 10,
            },
        }
        config_path = tmp_path / "pipeline.yaml"
        write_yaml(config_path, cfg)
        return config_path

    def test_monitor_triggers_retrain_when_allowed(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        train_path = ds / "dataset.jsonl"
        config = self._write_pipeline_config(tmp_path, train_path)

        mid = uuid.uuid4().hex[:12]
        mc = MonitorConfig(
            monitor_id=mid,
            project_id="test",
            config_path=str(config),
            max_ticks=1,
            allow_retrain=True,
            retrain_dataset_dir=str(ds),
        )
        monitor_store.start(mc, paused=True)

        tick = monitor_store.tick_once(mid)
        # If decision is SHIP, retrain should be triggered
        if tick.decision == "SHIP":
            assert tick.retrain_submitted is True
            assert tick.retrain_run_dir is not None
            assert Path(tick.retrain_run_dir).exists()

    def test_monitor_no_retrain_when_not_allowed(self, client, tmp_path):
        ds = _ingest_drill(tmp_path)
        train_path = ds / "dataset.jsonl"
        config = self._write_pipeline_config(tmp_path, train_path)

        mid = uuid.uuid4().hex[:12]
        mc = MonitorConfig(
            monitor_id=mid,
            project_id="test",
            config_path=str(config),
            max_ticks=1,
            allow_retrain=False,  # disabled
        )
        monitor_store.start(mc, paused=True)

        tick = monitor_store.tick_once(mid)
        assert tick.retrain_submitted is False
        assert tick.retrain_run_dir is None
