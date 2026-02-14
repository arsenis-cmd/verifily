"""Tests for SDK retrain methods.

Covers: submit_retrain_job + drain, sync retrain, project_id passthrough.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient as StarletteTestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "verifily_sdk"))

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.monitor_store import monitor_store
from verifily_sdk import VerifilyClient
from verifily_sdk.models import RetrainResponse, JobSubmitResponse

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


def _make_sdk_client(app):
    inner = StarletteTestClient(app, base_url="http://testserver")
    client = VerifilyClient.__new__(VerifilyClient)
    client._base_url = "http://testserver"
    client._api_key = None
    client._timeout = 60.0
    client._retries = 0
    client._client = inner
    return client


@pytest.fixture
def sdk_client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_ENABLE_REAL_TRAIN", raising=False)
    jobs_store.reset()
    monitor_store.reset()
    return _make_sdk_client(shared_app)


class TestSDKRetrain:
    def test_submit_and_drain(self, sdk_client, tmp_path):
        ds = _ingest_drill(tmp_path)
        sub = sdk_client.submit_retrain_job(
            dataset_dir=str(ds),
            output_dir=str(tmp_path / "retrain_out"),
        )
        assert isinstance(sub, JobSubmitResponse)
        assert sub.status == "QUEUED"

        jobs_store.drain()
        meta = sdk_client.get_job(sub.job_id)
        assert meta.status == "SUCCEEDED"

        result = sdk_client.get_job_result(sub.job_id)
        assert "run_dir" in result
        assert "decision" in result

    def test_sync_retrain(self, sdk_client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = sdk_client.retrain(
            dataset_dir=str(ds),
            output_dir=str(tmp_path / "retrain_out"),
        )
        assert isinstance(resp, RetrainResponse)
        assert resp.job_type == "RETRAIN"
        assert resp.decision["recommendation"] in ("SHIP", "DONT_SHIP", "INVESTIGATE")

    def test_project_id_passthrough(self, sdk_client, tmp_path):
        ds = _ingest_drill(tmp_path)
        resp = sdk_client.retrain(
            dataset_dir=str(ds),
            output_dir=str(tmp_path / "retrain_out"),
            project_id="my-retrain-project",
        )
        assert isinstance(resp, RetrainResponse)
        assert resp.run_dir is not None
