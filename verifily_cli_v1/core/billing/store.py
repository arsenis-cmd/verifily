"""BillingStore -- thread-safe append-only billing event store.

Same persistence pattern as usage_store, auth_registry, and jobs_store:
  - configure_persistence(path) -> _replay(path) on boot
  - Append-only JSONL events
  - Thread-safe with threading.Lock()
  - reset() for test isolation
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from verifily_cli_v1.core.billing.metering import compute_invoice
from verifily_cli_v1.core.billing.models import BillingEvent
from verifily_cli_v1.core.billing.pricing import get_plan

logger = logging.getLogger("verifily.api")


class BillingStore:
    """Thread-safe in-memory billing event store with optional JSONL persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[BillingEvent] = []
        self._invoices: Dict[str, Dict[str, Any]] = {}
        self._persist_path: Optional[str] = None

    def configure_persistence(self, path: Optional[str]) -> None:
        with self._lock:
            self._persist_path = path
        if path:
            self._replay(path)

    def _replay(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    evt = BillingEvent(
                        ts=d.get("ts", 0.0),
                        request_id=d.get("request_id"),
                        api_key_id=d.get("api_key_id", "anonymous"),
                        project_id=d.get("project_id", "default"),
                        endpoint=d.get("endpoint", ""),
                        units=d.get("units", {}),
                        job_id=d.get("job_id"),
                        status_code=d.get("status_code", 200),
                    )
                    with self._lock:
                        self._events.append(evt)
        except Exception:
            logger.warning("billing_store: replay failed for %s", path, exc_info=True)

    def _persist_event(self, evt: BillingEvent) -> None:
        path = self._persist_path
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(evt.to_dict(), separators=(",", ":")) + "\n")
                f.flush()
        except Exception:
            logger.warning("billing_store: persist failed", exc_info=True)

    def record_event(
        self,
        *,
        api_key_id: str,
        project_id: str,
        endpoint: str,
        units: Dict[str, int],
        request_id: Optional[str] = None,
        job_id: Optional[str] = None,
        status_code: int = 200,
    ) -> None:
        evt = BillingEvent(
            ts=time.time(),
            request_id=request_id,
            api_key_id=api_key_id,
            project_id=project_id,
            endpoint=endpoint,
            units=units,
            job_id=job_id,
            status_code=status_code,
        )
        with self._lock:
            self._events.append(evt)
        self._persist_event(evt)

    def query_events(
        self,
        *,
        project_id: Optional[str] = None,
        window_minutes: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query events with optional filters. Returns most-recent-first."""
        cutoff = time.time() - (window_minutes * 60) if window_minutes > 0 else 0.0
        with self._lock:
            filtered = []
            for evt in reversed(self._events):
                if cutoff > 0 and evt.ts < cutoff:
                    continue
                if project_id and evt.project_id != project_id:
                    continue
                filtered.append(evt.to_dict())
                if len(filtered) >= limit:
                    break
        return filtered

    def preview_invoice(
        self,
        *,
        project_id: str,
        plan_id: str = "FREE",
        window_minutes: int = 43200,
    ) -> Dict[str, Any]:
        plan = get_plan(plan_id)
        now = time.time()
        cutoff = now - (window_minutes * 60)
        with self._lock:
            events = [
                e for e in self._events
                if e.project_id == project_id and e.ts >= cutoff
            ]
        preview = compute_invoice(events, plan, project_id, cutoff, now)
        return preview.to_dict()

    def get_raw_events(
        self,
        *,
        window_minutes: int = 0,
        project_id: Optional[str] = None,
    ) -> List[BillingEvent]:
        """Return raw BillingEvent objects (for export/invoice generation)."""
        cutoff = time.time() - (window_minutes * 60) if window_minutes > 0 else 0.0
        with self._lock:
            out = []
            for evt in self._events:
                if cutoff > 0 and evt.ts < cutoff:
                    continue
                if project_id and evt.project_id != project_id:
                    continue
                out.append(evt)
        return out

    def generate_invoice(
        self,
        *,
        project_id: str,
        plan_id: str = "FREE",
        period_start: float,
        period_end: float,
    ) -> Dict[str, Any]:
        """Generate an invoice for a specific period (start/end as unix timestamps)."""
        plan = get_plan(plan_id)
        with self._lock:
            events = [
                e for e in self._events
                if e.project_id == project_id
                and e.ts >= period_start
                and e.ts <= period_end
            ]
        preview = compute_invoice(events, plan, project_id, period_start, period_end)
        return preview.to_dict()

    def usage_for_period(
        self,
        *,
        project_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        period: str,
    ) -> Dict[str, int]:
        """Aggregate usage for a YYYY-MM billing period.

        Returns dict with keys: processed_rows, bytes_processed,
        decisions, requests.
        """
        from verifily_cli_v1.core.billing.periods import period_bounds

        start_ts, end_ts = period_bounds(period)
        totals = {"processed_rows": 0, "bytes_processed": 0, "decisions": 0, "requests": 0}
        with self._lock:
            for evt in self._events:
                if evt.ts < start_ts or evt.ts >= end_ts:
                    continue
                if project_id and evt.project_id != project_id:
                    continue
                if api_key_id and evt.api_key_id != api_key_id:
                    continue
                totals["processed_rows"] += evt.units.get("rows_in", 0) + evt.units.get("rows_out", 0)
                totals["bytes_processed"] += evt.units.get("bytes_in", 0) + evt.units.get("bytes_out", 0)
                totals["decisions"] += evt.units.get("decisions", 0)
                totals["requests"] += 1
        return totals

    def store_invoice(self, invoice_id: str, invoice_data: Dict[str, Any]) -> None:
        """Store a generated invoice in memory for later retrieval."""
        with self._lock:
            self._invoices[invoice_id] = invoice_data

    def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored invoice by ID."""
        with self._lock:
            return self._invoices.get(invoice_id)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._invoices.clear()
            self._persist_path = None


billing_store = BillingStore()
