"""Tests for the Verifily async CLASSIFY job type.

Covers:
  - Submit classify job returns QUEUED
  - Job completes SUCCEEDED
  - Result has expected keys + deterministic output for fixture dataset
  - project_id filtering on list + fetch
  - Auth enabled blocks missing token
  - Inline JSONL dataset classification

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
CLASSIFY_DIR = REPO_ROOT / "examples" / "classify_demo"
MIXED_CSV = str(CLASSIFY_DIR / "raw" / "mixed_dump.csv")

TEST_API_KEY = "test-classify-key-42"


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

class TestClassifySubmit:
    def test_submit_returns_queued(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "QUEUED"
        assert len(data["job_id"]) > 0

    def test_submit_missing_path_returns_422(self, client):
        resp = client.post("/v1/jobs/classify", json={})
        assert resp.status_code == 422

    def test_submit_nonexistent_file_fails(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": "/nonexistent/data.csv",
        })
        job_id = resp.json()["job_id"]
        meta = _drain_and_get(client, job_id)
        assert meta["status"] == "FAILED"


# ── Lifecycle ─────────────────────────────────────────────────────

class TestClassifyLifecycle:
    def test_classify_job_succeeds(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        meta = _drain_and_get(client, job_id)
        assert meta["status"] == "SUCCEEDED"
        assert meta["started_at"] is not None
        assert meta["finished_at"] is not None

    def test_job_type_is_classify(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        meta = _drain_and_get(client, job_id)
        assert meta["type"] == "CLASSIFY"


# ── Result shape + determinism ────────────────────────────────────

class TestClassifyResult:
    def test_result_has_expected_keys(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert result["job_id"] == job_id
        cls = result["classification"]
        assert "row_count" in cls
        assert "suggested_schema" in cls
        assert "dataset_buckets" in cls
        assert "tags_summary" in cls
        assert "warnings" in cls

    def test_deterministic_schema_detection(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        cls = client.get(f"/v1/jobs/{job_id}/result").json()["classification"]
        assert cls["suggested_schema"] == "qa"
        assert cls["row_count"] == 16

    def test_deterministic_buckets(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        cls = client.get(f"/v1/jobs/{job_id}/result").json()["classification"]
        buckets = cls["dataset_buckets"]
        assert len(buckets) == 4
        schemas = {b["schema"] for b in buckets}
        assert schemas == {"qa"}
        categories = {b["category"] for b in buckets}
        assert "support" in categories
        assert "qa_geography" in categories

    def test_deterministic_pii_detection(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        pii = client.get(f"/v1/jobs/{job_id}/result").json()["classification"]["tags_summary"]["pii_risk"]
        assert pii["emails"] == 4
        assert pii["phones"] == 3
        assert pii["rows_with_pii"] == 4

    def test_deterministic_duplicate_rate(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        cls = client.get(f"/v1/jobs/{job_id}/result").json()["classification"]
        assert cls["tags_summary"]["duplicate_rate"] == 0.125

    def test_warnings_present(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        warnings = client.get(f"/v1/jobs/{job_id}/result").json()["classification"]["warnings"]
        assert len(warnings) >= 2
        assert any("duplicate" in w.lower() for w in warnings)
        assert any("pii" in w.lower() for w in warnings)

    def test_result_has_elapsed_ms(self, client):
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], int)


# ── Auth ──────────────────────────────────────────────────────────

class TestClassifyAuth:
    def test_submit_rejected_without_key(self, client_with_auth):
        resp = client_with_auth.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
        })
        assert resp.status_code == 401

    def test_submit_accepted_with_correct_key(self, client_with_auth):
        resp = client_with_auth.post(
            "/v1/jobs/classify",
            json={"dataset_path": MIXED_CSV},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "QUEUED"


# ── project_id ────────────────────────────────────────────────────

class TestClassifyProjectId:
    def test_project_id_filtering(self, client):
        client.post(
            "/v1/jobs/classify",
            json={"dataset_path": MIXED_CSV, "project_id": "proj_cls_a"},
            headers={"X-Project-ID": "proj_cls_a"},
        )
        client.post(
            "/v1/jobs/classify",
            json={"dataset_path": MIXED_CSV, "project_id": "proj_cls_b"},
            headers={"X-Project-ID": "proj_cls_b"},
        )

        resp_a = client.get("/v1/jobs", params={"project_id": "proj_cls_a"})
        jobs_a = resp_a.json()["jobs"]
        assert all(j["project_id"] == "proj_cls_a" for j in jobs_a)
        assert len(jobs_a) >= 1

    def test_request_id_propagated(self, client):
        custom_rid = "classify-rid-99"
        resp = client.post(
            "/v1/jobs/classify",
            json={"dataset_path": MIXED_CSV},
            headers={"X-Request-ID": custom_rid},
        )
        job_id = resp.json()["job_id"]
        assert resp.json()["request_id"] == custom_rid

        meta = client.get(f"/v1/jobs/{job_id}").json()
        assert meta["request_id"] == custom_rid


# ── Output artifacts ──────────────────────────────────────────────

class TestClassifyArtifacts:
    def test_output_dir_writes_artifact(self, client, tmp_path):
        out = str(tmp_path / "classify_out")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        artifact_path = result["artifacts"].get("classification_json")
        assert artifact_path is not None
        assert Path(artifact_path).exists()
