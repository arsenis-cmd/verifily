"""Central in-memory audit event store for enterprise audit export API.

Thread-safe singleton following the same pattern as billing_store, usage_store.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from verifily_cli_v1.core.audit import AuditEvent


class AuditStore:
    """Thread-safe in-memory audit event store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        """Record an audit event."""
        with self._lock:
            self._events.append(event)

    def query(
        self,
        *,
        project_id: Optional[str] = None,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query events with optional filters.

        Returns safe dicts (redacted), most-recent-first.
        ISO timestamp strings are compared lexicographically (valid because
        the format is always ``YYYY-MM-DDTHH:MM:SS.ffffffZ``).
        """
        from verifily_cli_v1.core.secrets import redact_dict

        with self._lock:
            result: List[Dict[str, Any]] = []
            for event in reversed(self._events):
                if project_id and event.project != project_id:
                    continue
                if from_ts and event.ts < from_ts:
                    continue
                if to_ts and event.ts > to_ts:
                    continue
                result.append(redact_dict(event.to_dict()))
                if len(result) >= limit:
                    break
        return result

    def reset(self) -> None:
        """Clear all events (for test isolation)."""
        with self._lock:
            self._events.clear()


# Singleton
audit_store = AuditStore()
