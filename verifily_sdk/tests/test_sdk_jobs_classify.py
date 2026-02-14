"""Tests for the Verifily SDK CLASSIFY job methods.

Uses starlette TestClient — no real server, no network.  Fast (<1s).
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

CLASSIFY_DIR = REPO_ROOT / "examples" / "classify_demo"
MIXED_CSV = str(CLASSIFY_DIR / "raw" / "mixed_dump.csv")


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

class TestSDKClassifyJob:
    def test_submit_classify_job(self, sdk_client):
        sub = sdk_client.submit_classify_job(dataset_path=MIXED_CSV)
        assert sub.status == "QUEUED"
        assert len(sub.job_id) > 0

    def test_submit_and_wait(self, sdk_client):
        sub = sdk_client.submit_classify_job(dataset_path=MIXED_CSV)
        jobs_store.drain()
        meta = sdk_client.get_job(sub.job_id)
        assert meta.status == "SUCCEEDED"

    def test_submit_and_get_result(self, sdk_client):
        sub = sdk_client.submit_classify_job(dataset_path=MIXED_CSV)
        jobs_store.drain()
        result = sdk_client.get_job_result(sub.job_id)
        assert result["job_id"] == sub.job_id
        cls = result["classification"]
        assert cls["suggested_schema"] == "qa"
        assert cls["row_count"] == 16
        assert cls["tags_summary"]["pii_risk"]["emails"] == 4
        assert cls["tags_summary"]["duplicate_rate"] == 0.125

    def test_classify_appears_in_job_list(self, sdk_client):
        sdk_client.submit_classify_job(dataset_path=MIXED_CSV)
        listing = sdk_client.list_jobs()
        assert any(j.type == "CLASSIFY" for j in listing.jobs)
