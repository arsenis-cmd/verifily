"""Tests for the Verifily SDK async jobs methods.

Uses starlette TestClient as the underlying httpx.Client transport.
No real server process, no network calls.  All tests are fast (<1s).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient as StarletteTestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "verifily_sdk"))

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_sdk import VerifilyClient

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


def _make_sdk_client(app, api_key=None):
    inner = StarletteTestClient(app, base_url="http://testserver")
    client = VerifilyClient.__new__(VerifilyClient)
    client._base_url = "http://testserver"
    client._api_key = api_key
    client._timeout = 60.0
    client._retries = 0
    client._client = inner
    return client


@pytest.fixture
def sdk_client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    jobs_store.reset()
    return _make_sdk_client(shared_app)


# ── Submit + wait ─────────────────────────────────────────────────

class TestSDKJobSubmit:
    def test_submit_pipeline_job(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sub = sdk_client.submit_pipeline_job(config_path=str(config), plan=True, ci=True)
        assert sub.status == "QUEUED"
        assert len(sub.job_id) > 0

    def test_submit_and_wait_pipeline(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sub = sdk_client.submit_pipeline_job(config_path=str(config), plan=True, ci=True)
        jobs_store.drain()
        meta = sdk_client.get_job(sub.job_id)
        assert meta.status == "SUCCEEDED"

    def test_submit_and_get_result(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sub = sdk_client.submit_pipeline_job(config_path=str(config), plan=True, ci=True)
        jobs_store.drain()
        result = sdk_client.get_job_result(sub.job_id)
        assert result["job_id"] == sub.job_id
        assert result["decision"]["recommendation"] == "SHIP"

    def test_submit_contamination_job(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sub = sdk_client.submit_contamination_job(
            train_path=str(train),
            eval_path=str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        )
        jobs_store.drain()
        meta = sdk_client.get_job(sub.job_id)
        assert meta.status == "SUCCEEDED"
        result = sdk_client.get_job_result(sub.job_id)
        assert result["status"] == "PASS"

    def test_submit_report_job(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sub = sdk_client.submit_report_job(dataset_path=str(train), schema="sft")
        jobs_store.drain()
        meta = sdk_client.get_job(sub.job_id)
        assert meta.status == "SUCCEEDED"
        result = sdk_client.get_job_result(sub.job_id)
        assert result["row_count"] == 28


# ── List ──────────────────────────────────────────────────────────

class TestSDKJobList:
    def test_list_jobs(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sdk_client.submit_pipeline_job(config_path=str(config), plan=True, ci=True)
        listing = sdk_client.list_jobs()
        assert len(listing.jobs) >= 1
