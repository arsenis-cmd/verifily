"""Per-project daily quota enforcement for advanced auth mode.

Tracks (api_key_id, project_id, day) counters for requests, rows, and bytes.
Thread-safe. Counters reset daily (UTC).
"""

from __future__ import annotations

import datetime
import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger("verifily.api")


@dataclass(frozen=True)
class QuotaLimits:
    requests_per_day: int = 2000
    rows_per_day: int = 2_000_000
    bytes_per_day: int = 200_000_000


# (api_key_id, project_id, day_str) → {"requests": N, "rows": N, "bytes": N}
_CounterKey = Tuple[str, str, str]


class QuotaStore:
    """In-memory daily quota counters. Only active in advanced auth mode."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[_CounterKey, Dict[str, int]] = {}
        self._limits = QuotaLimits()

    def configure_limits(
        self,
        requests_per_day: int = 2000,
        rows_per_day: int = 2_000_000,
        bytes_per_day: int = 200_000_000,
    ) -> None:
        self._limits = QuotaLimits(
            requests_per_day=requests_per_day,
            rows_per_day=rows_per_day,
            bytes_per_day=bytes_per_day,
        )

    def check_and_increment(
        self,
        api_key_id: str,
        project_id: str,
        rows_in: int = 0,
        bytes_in: int = 0,
    ) -> Optional[str]:
        """Check quotas and increment counters.

        Returns None if OK, or a human-readable error message if exceeded.
        """
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        key = (api_key_id, project_id, day)
        limits = self._limits

        with self._lock:
            c = self._counters.get(key)
            if c is None:
                c = {"requests": 0, "rows": 0, "bytes": 0}
                self._counters[key] = c

            # Check before increment
            if limits.requests_per_day > 0 and c["requests"] >= limits.requests_per_day:
                return f"Daily request quota exceeded ({limits.requests_per_day}/day)."

            if limits.rows_per_day > 0 and rows_in > 0 and c["rows"] + rows_in > limits.rows_per_day:
                return f"Daily row quota exceeded ({limits.rows_per_day}/day)."

            if limits.bytes_per_day > 0 and bytes_in > 0 and c["bytes"] + bytes_in > limits.bytes_per_day:
                return f"Daily byte quota exceeded ({limits.bytes_per_day}/day)."

            # Increment
            c["requests"] += 1
            c["rows"] += rows_in
            c["bytes"] += bytes_in

        return None

    def get_usage(self, api_key_id: str, project_id: str) -> Dict[str, int]:
        """Return current day's usage for a key/project pair."""
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        key = (api_key_id, project_id, day)
        with self._lock:
            c = self._counters.get(key)
            if c is None:
                return {"requests": 0, "rows": 0, "bytes": 0}
            return dict(c)

    def seconds_until_reset(self) -> int:
        """Seconds until midnight UTC (for Retry-After header)."""
        now = datetime.datetime.now(datetime.timezone.utc)
        midnight = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(1, int((midnight - now).total_seconds()))

    def reset(self) -> None:
        """Clear all counters. For test isolation."""
        with self._lock:
            self._counters.clear()
            self._limits = QuotaLimits()


# Singleton
quota_store = QuotaStore()
