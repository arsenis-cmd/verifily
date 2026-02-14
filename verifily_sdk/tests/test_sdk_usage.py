"""Tests for SDK usage accounting integration.

Covers: client.usage() returns UsageResponse, usage after pipeline,
window/group_by params, project_id header.
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
from verifily_sdk import VerifilyClient
from verifily_sdk.models import UsageResponse

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
    from verifily_cli_v1.core.api.usage_store import usage_store
    from verifily_cli_v1.core.api.metrics import metrics
    usage_store.reset()
    metrics.reset()
    return _make_sdk_client(shared_app)


# ── Basic usage endpoint ────────────────────────────────────────

class TestSDKUsage:
    def test_usage_returns_usage_response(self, sdk_client):
        resp = sdk_client.usage()
        assert isinstance(resp, UsageResponse)

    def test_usage_empty(self, sdk_client):
        resp = sdk_client.usage()
        assert resp.buckets == []
        assert resp.total is None

    def test_usage_after_report(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.report(dataset_path=str(train), schema="sft")
        resp = sdk_client.usage()
        assert resp.buckets is not None
        assert len(resp.buckets) == 1
        assert resp.buckets[0].requests == 1

    def test_usage_after_pipeline(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sdk_client.pipeline(config_path=str(config), plan=True, ci=True)
        resp = sdk_client.usage()
        assert resp.buckets is not None
        assert len(resp.buckets) == 1
        assert resp.buckets[0].decisions_ship == 1


# ── Window and group_by params ──────────────────────────────────

class TestSDKUsageParams:
    def test_group_by_total(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.report(dataset_path=str(train), schema="sft")
        sdk_client.report(dataset_path=str(train), schema="sft")
        resp = sdk_client.usage(group_by="total")
        assert resp.total is not None
        assert resp.total["requests"] == 2
        assert resp.buckets is None

    def test_window_filter(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.report(dataset_path=str(train), schema="sft")
        resp = sdk_client.usage(window_minutes=5, group_by="total")
        assert resp.total is not None
        assert resp.total["requests"] == 1

    def test_group_by_key(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.report(dataset_path=str(train), schema="sft")
        resp = sdk_client.usage(group_by="key")
        assert resp.buckets is not None
        assert len(resp.buckets) == 1
        assert resp.buckets[0].api_key_id == "anonymous"

    def test_group_by_project(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.report(dataset_path=str(train), schema="sft")
        resp = sdk_client.usage(group_by="project")
        assert resp.buckets is not None
        assert len(resp.buckets) == 1
        assert resp.buckets[0].project_id == "default"


# ── project_id header ───────────────────────────────────────────

class TestSDKProjectId:
    def test_project_id_sent_via_sdk(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.report(dataset_path=str(train), schema="sft", project_id="my-project")
        resp = sdk_client.usage()
        assert resp.buckets is not None
        assert resp.buckets[0].project_id == "my-project"

    def test_pipeline_with_project_id(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        sdk_client.pipeline(config_path=str(config), plan=True, ci=True, project_id="ci-project")
        resp = sdk_client.usage()
        assert resp.buckets is not None
        assert resp.buckets[0].project_id == "ci-project"

    def test_contamination_with_project_id(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        sdk_client.contamination(
            train_path=str(train),
            eval_path=str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
            project_id="contam-project",
        )
        resp = sdk_client.usage()
        assert resp.buckets is not None
        assert resp.buckets[0].project_id == "contam-project"
