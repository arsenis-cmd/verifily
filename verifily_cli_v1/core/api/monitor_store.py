"""Continuous gating monitor store for Verifily API.

Runs the gate (CONTRACT → REPORT → CONTAMINATION → DECISION) periodically,
producing rolling history and regression alerts.

Thread-safe singleton.  Each monitor gets its own background thread.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("verifily.api")


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class MonitorConfig:
    monitor_id: str
    project_id: str
    config_path: str
    interval_seconds: int = 60
    max_ticks: int = 0  # 0 = unlimited
    rolling_window: int = 20
    allow_retrain: bool = False
    retrain_dataset_dir: Optional[str] = None


@dataclass
class MonitorTickResult:
    tick_number: int
    timestamp: float
    decision: str  # SHIP / DONT_SHIP / INVESTIGATE
    metric_value: Optional[float] = None  # primary metric (f1)
    delta: Optional[float] = None  # vs previous tick
    regression_detected: bool = False
    contamination_pass: bool = True
    contract_pass: bool = True
    retrain_submitted: bool = False
    retrain_run_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Internal monitor state ────────────────────────────────────────

class _MonitorState:
    """Per-monitor runtime state."""

    __slots__ = (
        "config", "status", "thread", "stop_event",
        "history", "tick_count", "error",
    )

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.status: str = "running"
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.history: List[MonitorTickResult] = []
        self.tick_count: int = 0
        self.error: Optional[str] = None


# ── Monitor store (singleton) ────────────────────────────────────

class MonitorStore:
    """Thread-safe monitor store.  One background thread per active monitor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._monitors: Dict[str, _MonitorState] = {}

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self, config: MonitorConfig, *, paused: bool = False) -> str:
        """Start a new monitor.  Returns monitor_id.

        Args:
            paused: If True, register monitor but don't launch thread.
                    Use tick_once() for deterministic execution.  TEST-ONLY.
        """
        mid = config.monitor_id
        state = _MonitorState(config)

        with self._lock:
            if mid in self._monitors:
                raise ValueError(f"Monitor {mid} already exists.")
            self._monitors[mid] = state

        if not paused:
            # Launch background thread
            t = threading.Thread(target=self._run_loop, args=(mid,), daemon=True)
            state.thread = t
            t.start()
        return mid

    def stop(self, monitor_id: str) -> None:
        """Signal monitor to stop."""
        with self._lock:
            state = self._monitors.get(monitor_id)
            if state is None:
                raise KeyError(f"Monitor {monitor_id} not found.")
            state.stop_event.set()
            state.status = "stopped"

    def status(self, monitor_id: str) -> Dict[str, Any]:
        """Return monitor status summary."""
        with self._lock:
            state = self._monitors.get(monitor_id)
            if state is None:
                raise KeyError(f"Monitor {monitor_id} not found.")
            last_tick = None
            if state.history:
                last_tick = state.history[-1].to_dict()
            return {
                "monitor_id": monitor_id,
                "status": state.status,
                "tick_count": state.tick_count,
                "last_tick": last_tick,
                "config": asdict(state.config),
                "error": state.error,
            }

    def get_history(self, monitor_id: str, last_n: int = 0) -> List[MonitorTickResult]:
        """Return rolling tick history."""
        with self._lock:
            state = self._monitors.get(monitor_id)
            if state is None:
                raise KeyError(f"Monitor {monitor_id} not found.")
            history = list(state.history)
        if last_n > 0:
            history = history[-last_n:]
        return history

    def list_monitors(self) -> List[str]:
        with self._lock:
            return list(self._monitors.keys())

    def reset(self) -> None:
        """Stop all monitors and clear state (for test isolation)."""
        with self._lock:
            for state in self._monitors.values():
                state.stop_event.set()
            monitors = list(self._monitors.values())
        # Join threads outside lock
        for state in monitors:
            if state.thread and state.thread.is_alive():
                state.thread.join(timeout=2.0)
        with self._lock:
            self._monitors.clear()

    # ── TEST-ONLY: synchronous single tick ────────────────────

    def tick_once(self, monitor_id: str) -> MonitorTickResult:
        """Execute exactly one tick synchronously.  TEST-ONLY.

        Bypasses the background thread — no sleep, no interval.
        """
        with self._lock:
            state = self._monitors.get(monitor_id)
            if state is None:
                raise KeyError(f"Monitor {monitor_id} not found.")
        return self._execute_tick(state)

    # ── Background loop ──────────────────────────────────────

    def _run_loop(self, monitor_id: str) -> None:
        with self._lock:
            state = self._monitors.get(monitor_id)
        if state is None:
            return

        while not state.stop_event.is_set():
            self._execute_tick(state)

            # Check max_ticks
            if state.config.max_ticks > 0 and state.tick_count >= state.config.max_ticks:
                with self._lock:
                    state.status = "completed"
                return

            # Sleep with early exit on stop
            state.stop_event.wait(timeout=state.config.interval_seconds)

        with self._lock:
            state.status = "stopped"

    def _execute_tick(self, state: _MonitorState) -> MonitorTickResult:
        """Run one pipeline tick and record the result."""
        from verifily_cli_v1.core.api.runners import run_pipeline_api

        tick_num = state.tick_count + 1
        ts = time.time()

        try:
            result = run_pipeline_api(
                config_path=state.config.config_path,
                plan=True,
                ci=True,
                project_id=state.config.project_id,
            )

            decision_data = result.get("decision", {})
            decision = decision_data.get("recommendation", "UNKNOWN")

            # Extract primary metric (f1)
            metrics = decision_data.get("metrics", {})
            metric_value = metrics.get("f1")

            # Contamination / contract pass
            contam = result.get("contamination")
            contam_pass = contam.get("status", "PASS") == "PASS" if contam else True
            contract = result.get("contract")
            contract_pass = contract.get("valid", True) if contract else True

            # Regression detection: compare to previous tick
            delta = None
            regression = False
            with self._lock:
                if state.history and metric_value is not None:
                    prev = state.history[-1].metric_value
                    if prev is not None:
                        delta = metric_value - prev
                        if delta < -0.02:  # > 2% drop = regression
                            regression = True

            tick = MonitorTickResult(
                tick_number=tick_num,
                timestamp=ts,
                decision=decision,
                metric_value=metric_value,
                delta=delta,
                regression_detected=regression,
                contamination_pass=contam_pass,
                contract_pass=contract_pass,
            )

            # Retrain trigger: if allowed and gates pass
            if (
                state.config.allow_retrain
                and state.config.retrain_dataset_dir
                and decision == "SHIP"
                and contam_pass
                and contract_pass
            ):
                try:
                    from verifily_cli_v1.core.api.retrain import run_retrain_api

                    retrain_result = run_retrain_api({
                        "project_id": state.config.project_id,
                        "dataset_dir": state.config.retrain_dataset_dir,
                        "mode": "mock",
                        "seed": 42,
                    })
                    tick.retrain_submitted = True
                    tick.retrain_run_dir = retrain_result.get("run_dir")
                except Exception as retrain_exc:
                    logger.warning("monitor %s retrain failed: %s", state.config.monitor_id, retrain_exc)

        except Exception as exc:
            logger.warning("monitor %s tick %d failed: %s", state.config.monitor_id, tick_num, exc)
            tick = MonitorTickResult(
                tick_number=tick_num,
                timestamp=ts,
                decision="ERROR",
                regression_detected=False,
                contamination_pass=False,
                contract_pass=False,
            )
            with self._lock:
                state.error = str(exc)

        # Record
        with self._lock:
            state.tick_count = tick_num
            state.history.append(tick)
            # Cap at rolling_window
            if len(state.history) > state.config.rolling_window:
                state.history = state.history[-state.config.rolling_window:]

        # Persist to JSONL
        self._persist_tick(state.config, tick)

        # Best-effort billing recording
        try:
            from verifily_cli_v1.core.billing.store import billing_store
            if tick.decision != "ERROR":
                billing_store.record_event(
                    api_key_id="monitor",
                    project_id=state.config.project_id,
                    endpoint="/v1/monitor/tick",
                    units={"decisions": 1},
                )
        except Exception:
            pass

        return tick

    def _persist_tick(self, config: MonitorConfig, tick: MonitorTickResult) -> None:
        """Append tick to monitor_history.jsonl alongside the pipeline config."""
        try:
            config_dir = Path(config.config_path).parent
            history_path = config_dir / "monitor_history.jsonl"
            entry = tick.to_dict()
            entry["monitor_id"] = config.monitor_id
            with open(history_path, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
                f.flush()
        except Exception:
            logger.warning("monitor: failed to persist tick", exc_info=True)


# Singleton
monitor_store = MonitorStore()
