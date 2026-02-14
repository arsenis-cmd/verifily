"""Usage export helpers -- CSV and JSONL output.

Produces deterministic, no-PII exports suitable for finance or Stripe import.
Groups events by day / project / api_key depending on caller's needs.
"""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from verifily_cli_v1.core.billing.models import BillingEvent

# CSV header order -- stable across versions.
CSV_COLUMNS = [
    "date",
    "api_key_id",
    "project_id",
    "requests",
    "rows_in",
    "rows_out",
    "bytes_in",
    "bytes_out",
    "decisions",
]


def _day_key(ts: float) -> str:
    """ISO date string for a unix timestamp."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _bucket_events(
    events: List[BillingEvent],
    group_by: str = "day",
) -> List[Dict[str, Any]]:
    """Aggregate events into buckets.

    group_by:
      - "day"            → one row per calendar day
      - "day_project"    → one row per day+project
      - "day_api_key"    → one row per day+api_key
    """
    buckets: Dict[str, Dict[str, Any]] = {}

    for evt in events:
        day = _day_key(evt.ts)
        if group_by == "day_project":
            key = f"{day}|{evt.project_id}"
        elif group_by == "day_api_key":
            key = f"{day}|{evt.api_key_id}"
        else:
            key = day

        if key not in buckets:
            buckets[key] = {
                "date": day,
                "api_key_id": evt.api_key_id if group_by == "day_api_key" else "*",
                "project_id": evt.project_id if group_by == "day_project" else "*",
                "requests": 0,
                "rows_in": 0,
                "rows_out": 0,
                "bytes_in": 0,
                "bytes_out": 0,
                "decisions": 0,
            }

        b = buckets[key]
        b["requests"] += 1
        b["rows_in"] += evt.units.get("rows_in", 0)
        b["rows_out"] += evt.units.get("rows_out", 0)
        b["bytes_in"] += evt.units.get("bytes_in", 0)
        b["bytes_out"] += evt.units.get("bytes_out", 0)
        b["decisions"] += evt.units.get("decisions", 0)

    return sorted(buckets.values(), key=lambda r: r["date"])


def export_usage_csv(
    events: List[BillingEvent],
    group_by: str = "day",
) -> str:
    """Return usage as a CSV string (header + data rows)."""
    rows = _bucket_events(events, group_by)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def export_usage_jsonl(
    events: List[BillingEvent],
    group_by: str = "day",
) -> str:
    """Return usage as newline-delimited JSON."""
    rows = _bucket_events(events, group_by)
    lines = [json.dumps(row, separators=(",", ":")) for row in rows]
    return "\n".join(lines) + "\n" if lines else ""
