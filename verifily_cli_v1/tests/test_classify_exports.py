"""Tests for CLASSIFY v1.1 bucket export + routing artifacts.

Covers:
  - export_buckets=false writes only classification.json (no buckets/)
  - export_buckets=true writes per-bucket JSONL files
  - JSONL format is valid (one JSON object per line)
  - suggested_next_steps.json is written and stable
  - min_bucket_rows filters small buckets
  - export_summary appears in result when export_buckets=true
  - SDK passes export_buckets through correctly

All tests are deterministic and fast (<2s each).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLASSIFY_DIR = REPO_ROOT / "examples" / "classify_demo"
MIXED_CSV = str(CLASSIFY_DIR / "raw" / "mixed_dump.csv")


def _drain_and_get(client: TestClient, job_id: str) -> dict:
    """Drain all queued jobs synchronously, then return job metadata."""
    jobs_store.drain()
    return client.get(f"/v1/jobs/{job_id}").json()


@pytest.fixture
def client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    jobs_store.reset()
    return TestClient(shared_app)


# ── export_buckets=false (default) ────────────────────────────────

class TestExportBucketsFalse:
    def test_no_buckets_dir_by_default(self, client, tmp_path):
        out = str(tmp_path / "out_no_export")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        # classification.json should exist
        assert "classification_json" in result["artifacts"]
        assert Path(result["artifacts"]["classification_json"]).exists()
        # buckets/ directory should NOT exist
        buckets_dir = Path(out) / "classification" / "buckets"
        assert not buckets_dir.exists()
        # no export_summary key
        assert "export_summary" not in result

    def test_no_suggested_next_steps_by_default(self, client, tmp_path):
        out = str(tmp_path / "out_no_steps")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        next_steps = Path(out) / "classification" / "suggested_next_steps.json"
        assert not next_steps.exists()


# ── export_buckets=true ───────────────────────────────────────────

class TestExportBucketsTrue:
    def test_bucket_files_written(self, client, tmp_path):
        out = str(tmp_path / "out_export")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        assert "export_summary" in result
        summary = result["export_summary"]
        assert summary["total_rows_written"] == 16
        assert len(summary["bucket_paths"]) == 4  # 4 categories in mixed_dump

        # All bucket files exist
        for name, path in summary["bucket_paths"].items():
            assert Path(path).exists(), f"Bucket file missing: {path}"

    def test_bucket_jsonl_valid_format(self, client, tmp_path):
        out = str(tmp_path / "out_jsonl")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        for name, path in result["export_summary"]["bucket_paths"].items():
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) > 0, f"Bucket {name} is empty"
            for i, line in enumerate(lines):
                obj = json.loads(line.strip())
                assert isinstance(obj, dict), f"Line {i} in {name} is not a dict"

    def test_bucket_rows_sum_to_total(self, client, tmp_path):
        out = str(tmp_path / "out_sum")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        summary = client.get(f"/v1/jobs/{job_id}/result").json()["export_summary"]
        total = sum(summary["rows_per_bucket"].values())
        assert total == 16
        assert total == summary["total_rows_written"]

    def test_bucket_schemas_correct(self, client, tmp_path):
        out = str(tmp_path / "out_schemas")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        summary = client.get(f"/v1/jobs/{job_id}/result").json()["export_summary"]
        # mixed_dump.csv is all QA schema
        for name, schema in summary["schemas_per_bucket"].items():
            assert schema == "qa"

    def test_suggested_next_steps_written(self, client, tmp_path):
        out = str(tmp_path / "out_steps")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        result = client.get(f"/v1/jobs/{job_id}/result").json()
        steps_path = result["export_summary"]["suggested_next_steps_path"]
        assert steps_path is not None
        assert Path(steps_path).exists()

        with open(steps_path, "r", encoding="utf-8") as f:
            steps = json.load(f)
        assert steps["suggested_schema"] == "qa"
        assert steps["total_buckets"] == 4
        assert steps["total_rows_exported"] == 16
        assert len(steps["steps"]) > 0

    def test_suggested_next_steps_stable(self, client, tmp_path):
        """Two runs produce identical suggested_next_steps.json content."""
        results = []
        for i in range(2):
            out = str(tmp_path / f"out_stable_{i}")
            resp = client.post("/v1/jobs/classify", json={
                "dataset_path": MIXED_CSV,
                "output_dir": out,
                "export_buckets": True,
            })
            job_id = resp.json()["job_id"]
            _drain_and_get(client, job_id)
            result = client.get(f"/v1/jobs/{job_id}/result").json()
            with open(result["export_summary"]["suggested_next_steps_path"]) as f:
                results.append(json.load(f))

        # Compare steps (paths will differ but structure should match)
        assert results[0]["suggested_schema"] == results[1]["suggested_schema"]
        assert results[0]["total_buckets"] == results[1]["total_buckets"]
        assert results[0]["total_rows_exported"] == results[1]["total_rows_exported"]
        assert len(results[0]["steps"]) == len(results[1]["steps"])

    def test_artifacts_include_bucket_keys(self, client, tmp_path):
        out = str(tmp_path / "out_artkeys")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        artifacts = client.get(f"/v1/jobs/{job_id}/result").json()["artifacts"]
        assert "classification_json" in artifacts
        assert "suggested_next_steps" in artifacts
        bucket_keys = [k for k in artifacts if k.startswith("bucket_")]
        assert len(bucket_keys) == 4


# ── min_bucket_rows filtering ────────────────────────────────────

class TestMinBucketRows:
    def test_min_bucket_rows_filters_small_buckets(self, client, tmp_path):
        out = str(tmp_path / "out_min")
        # mixed_dump has: support=6, qa_geography=4, qa_tech=4, qa_science=2
        # min_bucket_rows=3 should skip qa_science (2 rows)
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
            "min_bucket_rows": 3,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        summary = client.get(f"/v1/jobs/{job_id}/result").json()["export_summary"]
        assert len(summary["bucket_paths"]) == 3  # qa_science excluded
        for name, count in summary["rows_per_bucket"].items():
            assert count >= 3

    def test_min_bucket_rows_high_exports_nothing(self, client, tmp_path):
        out = str(tmp_path / "out_high_min")
        resp = client.post("/v1/jobs/classify", json={
            "dataset_path": MIXED_CSV,
            "output_dir": out,
            "export_buckets": True,
            "min_bucket_rows": 100,
        })
        job_id = resp.json()["job_id"]
        _drain_and_get(client, job_id)

        summary = client.get(f"/v1/jobs/{job_id}/result").json()["export_summary"]
        assert len(summary["bucket_paths"]) == 0
        assert summary["total_rows_written"] == 0


# ── SDK integration ──────────────────────────────────────────────

class TestSDKExportBuckets:
    def test_sdk_export_buckets(self, tmp_path, monkeypatch):
        """SDK submit_classify_job passes export_buckets through."""
        import sys
        sys.path.insert(0, str(REPO_ROOT / "verifily_sdk"))
        from starlette.testclient import TestClient as StarletteTestClient
        from verifily_sdk import VerifilyClient

        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        app = create_app()
        jobs_store.stop_worker()
        inner = StarletteTestClient(app, base_url="http://testserver")
        sdk = VerifilyClient.__new__(VerifilyClient)
        sdk._base_url = "http://testserver"
        sdk._api_key = None
        sdk._timeout = 60.0
        sdk._retries = 0
        sdk._client = inner

        out = str(tmp_path / "sdk_export")
        sub = sdk.submit_classify_job(
            dataset_path=MIXED_CSV,
            output_dir=out,
            export_buckets=True,
        )
        assert sub.status == "QUEUED"

        jobs_store.drain()
        meta = sdk.get_job(sub.job_id)
        assert meta.status == "SUCCEEDED"

        result = sdk.get_job_result(sub.job_id)
        assert "export_summary" in result
        assert result["export_summary"]["total_rows_written"] == 16
        assert len(result["export_summary"]["bucket_paths"]) == 4
