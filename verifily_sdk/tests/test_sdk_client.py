"""Tests for the Verifily SDK client.

Uses starlette TestClient as the underlying httpx.Client transport.
No real server process, no network calls.  All tests are fast (<1s).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient as StarletteTestClient

# Ensure repo modules are importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "verifily_sdk"))

from verifily_cli_v1.core.api.server import create_app
from verifily_sdk import VerifilyClient
from verifily_sdk.errors import AuthError, NotFoundError, ValidationError

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
    """Create an SDK client backed by a starlette TestClient (no real network)."""
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
    """SDK client wired directly to the FastAPI app — no auth."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    return _make_sdk_client(shared_app)


@pytest.fixture
def sdk_client_with_auth(shared_app, monkeypatch):
    """SDK client with auth enabled on server and correct key."""
    monkeypatch.setenv("VERIFILY_API_KEY", "sdk-test-key")
    return _make_sdk_client(shared_app, api_key="sdk-test-key")


# ── Basic endpoints ──────────────────────────────────────────────

class TestSDKBasics:
    def test_health(self, sdk_client):
        h = sdk_client.health()
        assert h.status == "ok"
        assert h.mode == "local"

    def test_ready(self, sdk_client):
        r = sdk_client.ready()
        assert r.status == "ready"

    def test_metrics_returns_text(self, sdk_client):
        m = sdk_client.metrics()
        assert "verifily_requests_total" in m


# ── Headers ──────────────────────────────────────────────────────

class TestSDKHeaders:
    def test_sets_request_id(self, sdk_client):
        resp = sdk_client._get("/health")
        assert "x-request-id" in resp.headers

    def test_auth_header_sent(self, sdk_client_with_auth):
        h = sdk_client_with_auth.health()
        assert h.status == "ok"


# ── Pipeline ─────────────────────────────────────────────────────

class TestSDKPipeline:
    def test_pipeline_clean_ships(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_clean.jsonl", "run_clean")
        r = sdk_client.pipeline(config_path=str(config), plan=True, ci=True)
        assert r.exit_code == 0
        assert r.decision["recommendation"] == "SHIP"

    def test_pipeline_leaked_blocks(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_pipeline_config(tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked")
        r = sdk_client.pipeline(config_path=str(config), plan=True, ci=True)
        assert r.exit_code == 1
        assert r.decision["recommendation"] == "DONT_SHIP"


# ── Error handling ───────────────────────────────────────────────

class TestSDKErrors:
    def test_404_raises_not_found(self, sdk_client):
        with pytest.raises(NotFoundError):
            sdk_client.pipeline(config_path="/nonexistent/verifily.yaml", plan=True)

    def test_422_raises_validation_error(self, sdk_client):
        with pytest.raises(ValidationError):
            sdk_client.pipeline(plan=True)

    def test_401_raises_auth_error(self, monkeypatch):
        """SDK raises AuthError when server requires key but SDK has wrong one."""
        monkeypatch.setenv("VERIFILY_API_KEY", "real-key")
        client = _make_sdk_client(create_app(), api_key="wrong-key")
        with pytest.raises(AuthError):
            client.pipeline(config_path="/some/path", plan=True)


# ── Contamination & Report ───────────────────────────────────────

class TestSDKContamination:
    def test_contamination_clean(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        r = sdk_client.contamination(
            train_path=str(train),
            eval_path=str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        )
        assert r.status == "PASS"
        assert r.exit_code == 0


class TestSDKReport:
    def test_report_returns_stats(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        r = sdk_client.report(dataset_path=str(train), schema="sft")
        assert r.row_count == 28
        assert r.pii_clean is not None
