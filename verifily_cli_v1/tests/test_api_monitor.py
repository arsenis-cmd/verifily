"""Tests for Continuous Gating v1 — MONITOR endpoints.

Covers:
  - POST /v1/monitor/start + stop
  - GET /v1/monitor/status
  - GET /v1/monitor/history
  - tick_once() test hook
  - regression detection
  - max_ticks auto-stop
  - rolling_window capping

Tests that need deterministic tick execution use monitor_store.start(paused=True)
+ tick_once().  API tests use the HTTP endpoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.monitor_store import MonitorConfig, monitor_store
from verifily_cli_v1.core.io import write_yaml

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
    return tmp_path / "artifact" / "dataset.jsonl"


def _write_config(tmp_path: Path, train_path: Path, run_name: str = "run_clean") -> Path:
    cfg = {
        "run_dir": str(DRILL_DIR / "runs" / run_name),
        "train_data": str(train_path),
        "eval_data": str(DRILL_DIR / "raw" / "eval_clean.jsonl"),
        "baseline_run": str(DRILL_DIR / "runs" / run_name),
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


def _paused_monitor(config_path: str, **kwargs) -> str:
    """Register a monitor without starting its background thread."""
    import uuid
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
def client(shared_app, monkeypatch):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    jobs_store.reset()
    monitor_store.reset()
    return TestClient(shared_app)


# ── Start / Stop ─────────────────────────────────────────────────

class TestMonitorStartStop:
    def test_start_returns_monitor_id(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        resp = client.post("/v1/monitor/start", json={
            "config_path": str(config),
            "interval_seconds": 60,
            "max_ticks": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "monitor_id" in data
        assert data["status"] == "running"

    def test_stop_works(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        start = client.post("/v1/monitor/start", json={
            "config_path": str(config),
            "max_ticks": 5,
        })
        mid = start.json()["monitor_id"]
        resp = client.post(f"/v1/monitor/stop?monitor_id={mid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_unknown_returns_404(self, client):
        resp = client.post("/v1/monitor/stop?monitor_id=nonexistent")
        assert resp.status_code == 404

    def test_status_after_start(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config))
        resp = client.get(f"/v1/monitor/status?monitor_id={mid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["monitor_id"] == mid
        assert data["config"]["config_path"] == str(config)


# ── tick_once ────────────────────────────────────────────────────

class TestMonitorTick:
    def test_tick_once_produces_result(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=10)

        tick = monitor_store.tick_once(mid)
        assert tick.tick_number == 1
        assert tick.decision in ("SHIP", "DONT_SHIP", "INVESTIGATE")
        assert tick.contract_pass is True

    def test_tick_has_metric_value(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=10)

        tick = monitor_store.tick_once(mid)
        assert tick.metric_value is not None
        assert tick.metric_value > 0.0


# ── History ──────────────────────────────────────────────────────

class TestMonitorHistory:
    def test_history_accumulates(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=10)

        monitor_store.tick_once(mid)
        monitor_store.tick_once(mid)
        monitor_store.tick_once(mid)

        resp = client.get(f"/v1/monitor/history?monitor_id={mid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_ticks"] == 3
        assert len(data["ticks"]) == 3

    def test_last_n_filters(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=10)

        for _ in range(5):
            monitor_store.tick_once(mid)

        resp = client.get(f"/v1/monitor/history?monitor_id={mid}&last_n=2")
        data = resp.json()
        assert data["total_ticks"] == 2
        assert data["ticks"][0]["tick_number"] == 4

    def test_rolling_window_caps_history(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=100, rolling_window=3)

        for _ in range(6):
            monitor_store.tick_once(mid)

        history = monitor_store.get_history(mid)
        assert len(history) == 3
        assert history[0].tick_number == 4  # oldest kept


# ── Regression detection ─────────────────────────────────────────

class TestMonitorRegression:
    def test_no_regression_on_stable_metrics(self, client, tmp_path):
        train = _ingest_drill(tmp_path)
        config = _write_config(tmp_path, train)
        mid = _paused_monitor(str(config), max_ticks=10)

        t1 = monitor_store.tick_once(mid)
        t2 = monitor_store.tick_once(mid)
        # Same config, same data => same metrics => no regression
        assert t2.regression_detected is False
        assert t2.delta is not None
