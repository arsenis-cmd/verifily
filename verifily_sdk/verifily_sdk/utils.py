"""Utilities: retry/backoff, request-ID helpers, safe logging."""

from __future__ import annotations

import time
import uuid
from typing import Optional


def generate_request_id() -> str:
    """Generate a short UUID4 hex string for X-Request-ID."""
    return uuid.uuid4().hex[:12]


def retry_with_backoff(
    fn,
    *,
    retries: int = 2,
    backoff_base: float = 0.5,
    retryable_statuses: frozenset = frozenset({502, 503, 504}),
):
    """Call fn() with exponential backoff on retryable HTTP status codes.

    fn must return an httpx.Response.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = fn()
            if resp.status_code not in retryable_statuses:
                return resp
            if attempt < retries:
                time.sleep(backoff_base * (2 ** attempt))
                continue
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_base * (2 ** attempt))
                continue
            raise
    raise last_exc  # type: ignore[misc]
