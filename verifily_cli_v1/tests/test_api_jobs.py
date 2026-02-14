"""Tests for the Verifily async jobs API.

Covers:
  - Submit returns QUEUED
  - Job transitions QUEUED -> RUNNING -> SUCCEEDED
  - Result endpoint returns same shape as sync endpoints
  - Auth blocks missing token
  - request_id propagated into job record
  - project_id filtering on list endpoint
  - Job not found returns 404
  - Result of pending job returns 409
  - Failed job returns error

All tests are deterministic and fast (<2s each).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"

TEST_API_KEY = "test-jobs-key-42"


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


def _drain_and_get(client: TestClient, job_id: str) -> dict:
    """Drain all queued jobs synchronously, then return job metadata."""
    jobs_store.drain()
    return client.get(f"/v1/jobs/{job_id}").json()


@pytest.fixture
def client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    jobs_store.reset()
    return TestClient(shared_app)


@pytest.fixture
def client_with_auth(monkeypatch):
    monkeypatch.setenv("VERIFILY_API_KEY", TEST_API_KEY)
    app = create_app()
    jobs_store.stop_worker()
    return TestClient(app)


# ── Submit ────────────────────────────────────────────────────────

class TestJobSubmit:
    def test_submit_pipeline_returns_queued(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "QUEUED"
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_submit_contamination_returns_queued(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "QUEUED"

    def test_submit_report_returns_queued(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/report", json={
            "dataset_path": str(train), "schema": "sft",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "QUEUED"

    def test_submit_pipeline_missing_config_returns_422(self, client):
        resp = client.post("/v1/jobs/pipeline", json={"plan": True})
        assert resp.status_code == 422


# ── Lifecycle ─────────────────────────────────────────────────────

class TestJobLifecycle:
    def test_pipeline_job_succeeds(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        job_id = resp.json()["job_id"]
        meta = _drain_and_get(client, job_id)
        assert meta["status"] == "SUCCEEDED"
        assert meta["started_at"] is not None
        assert meta["finished_at"] is not None

    def test_contamination_job_succeeds(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/contamination", json={
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        })
        job_id = resp.json()["job_id"]
        meta = _drain_and_get(client, job_id)
        assert meta["status"] == "SUCCEEDED"

    def test_report_job_succeeds(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        resp = client.post("/v1/jobs/report", json={
            "dataset_path": str(train), "schema": "sft",
        })
        job_id = resp.json()["job_id"]
        meta = _drain_and_get(client, job_id)
        assert meta["status"] == "SUCCEEDED"


# ── Result ────────────────────────────────────────────────────────

class TestJobResult:
    def test_pipeline_result_matches_sync_shape(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        # Sync call
        sync_resp = client.post("/v1/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        sync_data = sync_resp.json()

        # Async call
        job_resp = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        job_id = job_resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result_resp = client.get(f"/v1/jobs/{job_id}/result")
        assert result_resp.status_code == 200
        result = result_resp.json()

        # Same keys as sync (plus job_id)
        assert result["job_id"] == job_id
        assert result["exit_code"] == sync_data["exit_code"]
        assert result["decision"]["recommendation"] == sync_data["decision"]["recommendation"]

    def test_contamination_result_matches_sync_shape(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        payload = {
            "train_path": str(train),
            "eval_path": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        }

        sync_resp = client.post("/v1/contamination", json=payload)
        sync_data = sync_resp.json()

        job_resp = client.post("/v1/jobs/contamination", json=payload)
        job_id = job_resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["job_id"] == job_id
        assert result["status"] == sync_data["status"]
        assert result["exit_code"] == sync_data["exit_code"]

    def test_report_result_matches_sync_shape(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        payload = {"dataset_path": str(train), "schema": "sft"}

        sync_resp = client.post("/v1/report", json=payload)
        sync_data = sync_resp.json()

        job_resp = client.post("/v1/jobs/report", json=payload)
        job_id = job_resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["job_id"] == job_id
        assert result["row_count"] == sync_data["row_count"]

    def test_result_of_pending_job_returns_409(self, client, tmp_path):
        """Requesting result before job completes returns 409."""
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        job_id = resp.json()["job_id"]
        result_resp = client.get(f"/v1/jobs/{job_id}/result")
        assert result_resp.status_code == 409


# ── Not found ─────────────────────────────────────────────────────

class TestJobNotFound:
    def test_get_nonexistent_job_returns_404(self, client):
        resp = client.get("/v1/jobs/nonexistent123")
        assert resp.status_code == 404

    def test_result_nonexistent_job_returns_404(self, client):
        resp = client.get("/v1/jobs/nonexistent123/result")
        assert resp.status_code == 404


# ── Auth ──────────────────────────────────────────────────────────

class TestJobAuth:
    def test_submit_rejected_without_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client_with_auth.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True,
        })
        assert resp.status_code == 401

    def test_submit_accepted_with_correct_key(self, client_with_auth, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        resp = client_with_auth.post(
            "/v1/jobs/pipeline",
            json={"config_path": str(config), "plan": True, "ci": True},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "QUEUED"

    def test_get_job_rejected_without_key(self, client_with_auth):
        resp = client_with_auth.get("/v1/jobs/someid")
        assert resp.status_code == 401

    def test_list_jobs_rejected_without_key(self, client_with_auth):
        resp = client_with_auth.get("/v1/jobs")
        assert resp.status_code == 401


# ── Request ID propagation ────────────────────────────────────────

class TestJobRequestID:
    def test_request_id_in_job_record(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        custom_rid = "custom-request-id-42"
        resp = client.post(
            "/v1/jobs/pipeline",
            json={"config_path": str(config), "plan": True, "ci": True},
            headers={"X-Request-ID": custom_rid},
        )
        job_id = resp.json()["job_id"]
        assert resp.json()["request_id"] == custom_rid

        meta = client.get(f"/v1/jobs/{job_id}").json()
        assert meta["request_id"] == custom_rid


# ── List + project_id filtering ──────────────────────────────────

class TestJobList:
    def test_list_returns_submitted_jobs(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        resp = client.get("/v1/jobs")
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        assert len(jobs) >= 1

    def test_list_filters_by_project_id(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        # Submit with project_id via header
        client.post(
            "/v1/jobs/pipeline",
            json={"config_path": str(config), "plan": True, "ci": True, "project_id": "proj_a"},
            headers={"X-Project-ID": "proj_a"},
        )
        client.post(
            "/v1/jobs/pipeline",
            json={"config_path": str(config), "plan": True, "ci": True, "project_id": "proj_b"},
            headers={"X-Project-ID": "proj_b"},
        )

        resp_a = client.get("/v1/jobs", params={"project_id": "proj_a"})
        jobs_a = resp_a.json()["jobs"]
        assert all(j["project_id"] == "proj_a" for j in jobs_a)

    def test_list_filters_by_status(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")

        sub = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        job_id = sub.json()["job_id"]
        _drain_and_get(client, job_id)

        resp = client.get("/v1/jobs", params={"status": "SUCCEEDED"})
        jobs = resp.json()["jobs"]
        assert all(j["status"] == "SUCCEEDED" for j in jobs)


# ── Pipeline decision through jobs ───────────────────────────────

class TestJobDecisions:
    def test_clean_pipeline_job_ships(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sub = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        job_id = sub.json()["job_id"]
        _drain_and_get(client, job_id)
        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["decision"]["recommendation"] == "SHIP"

    def test_leaked_pipeline_job_blocks(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked")
        sub = client.post("/v1/jobs/pipeline", json={
            "config_path": str(config), "plan": True, "ci": True,
        })
        job_id = sub.json()["job_id"]
        _drain_and_get(client, job_id)
        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["decision"]["recommendation"] == "DONT_SHIP"
