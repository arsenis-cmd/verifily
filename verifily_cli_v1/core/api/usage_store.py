"""Multi-tenant usage accounting store for Verifily API.

Thread-safe in-memory store: api_key_id = "who pays", project_id = "what consumes".
Dual storage: O(1) all-time buckets + event list for time-windowed queries.

Opt-in file persistence (VERIFILY_USAGE_PERSIST=1):
  - append-only JSONL events file
  - replays on startup to rebuild buckets
  - never writes raw payloads, PII, or full API keys
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("verifily.api")


def compute_api_key_id(api_key: Optional[str]) -> str:
    """Derive a short, non-reversible identifier from an API key.

    Returns ``sha256("verifily:" + key)[:12]`` or ``"anonymous"`` when None/empty.
    """
    if not api_key:
        return "anonymous"
    digest = hashlib.sha256(f"verifily:{api_key}".encode()).hexdigest()
    return digest[:12]


class _BucketCounters:
    """Mutable counters for a single (api_key_id, project_id) bucket."""

    __slots__ = (
        "requests",
        "decisions_ship",
        "decisions_dont_ship",
        "decisions_investigate",
        "rows_in",
        "rows_out",
        "bytes_in",
        "bytes_out",
        "elapsed_ms_sum",
    )

    def __init__(self) -> None:
        self.requests: int = 0
        self.decisions_ship: int = 0
        self.decisions_dont_ship: int = 0
        self.decisions_investigate: int = 0
        self.rows_in: int = 0
        self.rows_out: int = 0
        self.bytes_in: int = 0
        self.bytes_out: int = 0
        self.elapsed_ms_sum: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "requests": self.requests,
            "decisions_ship": self.decisions_ship,
            "decisions_dont_ship": self.decisions_dont_ship,
            "decisions_investigate": self.decisions_investigate,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "elapsed_ms_sum": self.elapsed_ms_sum,
        }


class UsageStore:
    """Thread-safe in-memory usage accounting store.

    Follows the MetricsStore pattern: ``threading.Lock``, ``reset()`` for test isolation.
    Supports opt-in file persistence via ``configure_persistence()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, str], _BucketCounters] = {}
        self._events: List[Tuple[float, str, str, Dict[str, int]]] = []
        self._persist_path: Optional[str] = None

    def configure_persistence(self, path: Optional[str]) -> None:
        """Enable file-backed persistence and replay existing events.

        Args:
            path: Path to JSONL events file, or None to disable.
        """
        with self._lock:
            self._persist_path = path
        if path:
            self._replay(path)

    def _replay(self, path: str) -> None:
        """Replay events from a JSONL file to rebuild in-memory state."""
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    ts = evt.get("ts", 0.0)
                    kid = evt.get("api_key_id", "anonymous")
                    pid = evt.get("project_id", "default")
                    cd = evt.get("counters", {})
                    with self._lock:
                        key = (kid, pid)
                        bucket = self._buckets.get(key)
                        if bucket is None:
                            bucket = _BucketCounters()
                            self._buckets[key] = bucket
                        _merge_counters_dict(bucket, cd)
                        self._events.append((ts, kid, pid, cd))
        except Exception:
            logger.warning("usage_store: failed to replay %s", path, exc_info=True)

    def _persist_event(
        self, ts: float, api_key_id: str, project_id: str, counters_dict: Dict[str, int]
    ) -> None:
        """Append a single event to the JSONL file (caller does NOT hold lock)."""
        path = self._persist_path
        if not path:
            return
        line = json.dumps({
            "ts": ts,
            "api_key_id": api_key_id,
            "project_id": project_id,
            "counters": counters_dict,
        }, separators=(",", ":"))
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            logger.warning("usage_store: failed to persist event to %s", path, exc_info=True)

    def record(
        self,
        *,
        api_key_id: str,
        project_id: str,
        elapsed_ms: int = 0,
        decision: Optional[str] = None,
        rows_in: int = 0,
        rows_out: int = 0,
        bytes_in: int = 0,
        bytes_out: int = 0,
    ) -> None:
        """Record a usage event, incrementing both bucket aggregates and the event list."""
        counters_dict: Dict[str, int] = {
            "requests": 1,
            "decisions_ship": 1 if decision == "SHIP" else 0,
            "decisions_dont_ship": 1 if decision == "DONT_SHIP" else 0,
            "decisions_investigate": 1 if decision == "INVESTIGATE" else 0,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "elapsed_ms_sum": elapsed_ms,
        }

        ts = time.time()
        key = (api_key_id, project_id)

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _BucketCounters()
                self._buckets[key] = bucket

            bucket.requests += 1
            bucket.decisions_ship += counters_dict["decisions_ship"]
            bucket.decisions_dont_ship += counters_dict["decisions_dont_ship"]
            bucket.decisions_investigate += counters_dict["decisions_investigate"]
            bucket.rows_in += rows_in
            bucket.rows_out += rows_out
            bucket.bytes_in += bytes_in
            bucket.bytes_out += bytes_out
            bucket.elapsed_ms_sum += elapsed_ms

            self._events.append((ts, api_key_id, project_id, counters_dict))

        # Persist outside lock to avoid holding lock during I/O
        self._persist_event(ts, api_key_id, project_id, counters_dict)

    def query(
        self,
        *,
        window_minutes: int = 0,
        group_by: str = "key_project",
    ) -> Dict[str, Any]:
        """Query usage data.

        Args:
            window_minutes: If >0, only include events from the last N minutes.
                            If 0, use all-time bucket aggregates.
            group_by: One of ``"key_project"``, ``"key"``, ``"project"``, ``"total"``.

        Returns:
            ``{"buckets": [...]}`` for grouped queries or ``{"total": {...}}`` for total.
        """
        with self._lock:
            if window_minutes > 0:
                return self._query_windowed(window_minutes, group_by)
            return self._query_alltime(group_by)

    def _query_alltime(self, group_by: str) -> Dict[str, Any]:
        """Query all-time aggregates (caller holds lock)."""
        if group_by == "total":
            total = _BucketCounters()
            for bucket in self._buckets.values():
                _merge_counters(total, bucket)
            return {"total": total.to_dict()}

        grouped: Dict[str, _BucketCounters] = {}
        for (kid, pid), bucket in self._buckets.items():
            gk = _group_key(kid, pid, group_by)
            if gk not in grouped:
                grouped[gk] = _BucketCounters()
            _merge_counters(grouped[gk], bucket)

        buckets_list = []
        for gk, counters in grouped.items():
            entry = counters.to_dict()
            if group_by == "key_project":
                kid, pid = gk.split("|", 1)
                entry["api_key_id"] = kid
                entry["project_id"] = pid
            elif group_by == "key":
                entry["api_key_id"] = gk
            elif group_by == "project":
                entry["project_id"] = gk
            buckets_list.append(entry)

        return {"buckets": buckets_list}

    def _query_windowed(self, window_minutes: int, group_by: str) -> Dict[str, Any]:
        """Query events within a time window (caller holds lock)."""
        cutoff = time.time() - (window_minutes * 60)

        if group_by == "total":
            total = _BucketCounters()
            for ts, kid, pid, cd in self._events:
                if ts >= cutoff:
                    _merge_counters_dict(total, cd)
            return {"total": total.to_dict()}

        grouped: Dict[str, _BucketCounters] = {}
        for ts, kid, pid, cd in self._events:
            if ts >= cutoff:
                gk = _group_key(kid, pid, group_by)
                if gk not in grouped:
                    grouped[gk] = _BucketCounters()
                _merge_counters_dict(grouped[gk], cd)

        buckets_list = []
        for gk, counters in grouped.items():
            entry = counters.to_dict()
            if group_by == "key_project":
                kid, pid = gk.split("|", 1)
                entry["api_key_id"] = kid
                entry["project_id"] = pid
            elif group_by == "key":
                entry["api_key_id"] = gk
            elif group_by == "project":
                entry["project_id"] = gk
            buckets_list.append(entry)

        return {"buckets": buckets_list}

    def reset(self) -> None:
        """Clear all data and disable persistence (for test isolation)."""
        with self._lock:
            self._buckets.clear()
            self._events.clear()
            self._persist_path = None


def _group_key(api_key_id: str, project_id: str, group_by: str) -> str:
    if group_by == "key":
        return api_key_id
    if group_by == "project":
        return project_id
    return f"{api_key_id}|{project_id}"


def _merge_counters(target: _BucketCounters, source: _BucketCounters) -> None:
    target.requests += source.requests
    target.decisions_ship += source.decisions_ship
    target.decisions_dont_ship += source.decisions_dont_ship
    target.decisions_investigate += source.decisions_investigate
    target.rows_in += source.rows_in
    target.rows_out += source.rows_out
    target.bytes_in += source.bytes_in
    target.bytes_out += source.bytes_out
    target.elapsed_ms_sum += source.elapsed_ms_sum


def _merge_counters_dict(target: _BucketCounters, cd: Dict[str, int]) -> None:
    target.requests += cd.get("requests", 0)
    target.decisions_ship += cd.get("decisions_ship", 0)
    target.decisions_dont_ship += cd.get("decisions_dont_ship", 0)
    target.decisions_investigate += cd.get("decisions_investigate", 0)
    target.rows_in += cd.get("rows_in", 0)
    target.rows_out += cd.get("rows_out", 0)
    target.bytes_in += cd.get("bytes_in", 0)
    target.bytes_out += cd.get("bytes_out", 0)
    target.elapsed_ms_sum += cd.get("elapsed_ms_sum", 0)


# Singleton instance — one per process.
usage_store = UsageStore()
