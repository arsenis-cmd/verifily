"""Tests for SDK monitor methods.

Covers: start_monitor, stop_monitor, monitor_status, monitor_history.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from starlette.testclient import TestClient as StarletteTestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "verifily_sdk"))

from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.monitor_store import MonitorConfig, monitor_store
from verifily_cli_v1.core.io import write_yaml
from verifily_sdk import VerifilyClient
from verifily_sdk.models import MonitorStartResponse, MonitorStatusResponse, MonitorHistoryResponse

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


def _write_config(tmp_path: Path, train_path: Path) -> Path:
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


def _make_sdk_client(app):
    inner = StarletteTestClient(app, base_url="http://testserver")
    client = VerifilyClient.__new__(VerifilyClient)
    client._base_url = "http://testserver"
    client._api_key = None
    client._timeout = 60.0
    client._retries = 0
    client._client = inner
    return client


def _paused_monitor(config_path: str, **kwargs) -> str:
    """Register a monitor without starting its background thread."""
    mid = uuid.uuid4().hex[:12]
    config = MonitorConfig(
        monitor_id=mid,
        project_id=kwargs.get("project_id", "default"),
        config_path=config_path,
        interval_seconds=kwargs.get("interval_seconds", 60),
        max_ticks=kwargs.get("max_ticks", 0),
        rolling_window=kwargs.get("rolling_window", 20),
    )
    monitor_store.start(config, paused=True)
    return mid


@pytest.fixture
def sdk_client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    jobs_store.reset()
    monitor_store.reset()
    return _make_sdk_client(shared_app)


class TestSDKMonitor:
    def test_start_monitor(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        resp = sdk_client.start_monitor(
            config_path=str(config),
            max_ticks=5,
        )
        assert isinstance(resp, MonitorStartResponse)
        assert resp.status == "running"

    def test_stop_monitor(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        start = sdk_client.start_monitor(config_path=str(config), max_ticks=5)
        resp = sdk_client.stop_monitor(start.monitor_id)
        assert resp["status"] == "stopped"

    def test_monitor_status(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=5)
        resp = sdk_client.monitor_status(mid)
        assert isinstance(resp, MonitorStatusResponse)
        assert resp.monitor_id == mid

    def test_monitor_history_after_ticks(self, sdk_client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=10)

        monitor_store.tick_once(mid)
        monitor_store.tick_once(mid)

        resp = sdk_client.monitor_history(mid)
        assert isinstance(resp, MonitorHistoryResponse)
        assert resp.total_ticks == 2
        assert len(resp.ticks) == 2
