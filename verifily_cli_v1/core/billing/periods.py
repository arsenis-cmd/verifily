"""Billing period helpers -- YYYY-MM string, start/end timestamps.

Pure functions, no I/O. A billing period is always a UTC calendar month.
"""

from __future__ import annotations

import datetime
from typing import Tuple


def current_period() -> str:
    """Return current billing period as 'YYYY-MM' in UTC."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m")


def period_bounds(period: str) -> Tuple[float, float]:
    """Return (start_ts, end_ts) for a 'YYYY-MM' period string.

    start_ts = midnight UTC on the 1st of the month.
    end_ts   = midnight UTC on the 1st of the next month.
    """
    year, month = int(period[:4]), int(period[5:7])
    start = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
    if month == 12:
        end = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    else:
        end = datetime.datetime(year, month + 1, 1, tzinfo=datetime.timezone.utc)
    return start.timestamp(), end.timestamp()


def validate_period(period: str) -> bool:
    """Return True if period matches YYYY-MM format with valid ranges."""
    if len(period) != 7 or period[4] != "-":
        return False
    try:
        y, m = int(period[:4]), int(period[5:7])
        return 2020 <= y <= 2099 and 1 <= m <= 12
    except ValueError:
        return False
