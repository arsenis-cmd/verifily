"""Per-api_key_id rate limiting middleware for Verifily API.

Opt-in via env var: VERIFILY_RATE_LIMIT_RPM=<int>  (requests per minute).
If unset or 0: limiter OFF, all requests pass through.

Uses fixed 60-second windows per api_key_id.
Public endpoints (/health, /ready, /metrics, /docs, /openapi.json) are never limited.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths exempt from rate limiting (same set as auth).
_EXEMPT_PATHS = frozenset({"/health", "/ready", "/metrics", "/docs", "/openapi.json"})


class _WindowEntry:
    __slots__ = ("window_start", "count")

    def __init__(self, window_start: int, count: int) -> None:
        self.window_start = window_start
        self.count = count


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-api_key_id rate limiter.

    Reads ``VERIFILY_RATE_LIMIT_RPM`` on each request so tests can
    toggle via ``monkeypatch.setenv``.
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        self._lock = threading.Lock()
        self._windows: Dict[str, _WindowEntry] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        limit_str = os.environ.get("VERIFILY_RATE_LIMIT_RPM", "")
        if not limit_str:
            return await call_next(request)

        try:
            limit = int(limit_str)
        except ValueError:
            return await call_next(request)

        if limit <= 0:
            return await call_next(request)

        # Exempt endpoints
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        api_key_id = getattr(request.state, "api_key_id", "anonymous")
        now = time.time()
        window_start = int(now / 60) * 60  # fixed 60s windows

        with self._lock:
            entry = self._windows.get(api_key_id)
            if entry is None or entry.window_start != window_start:
                entry = _WindowEntry(window_start, 0)
                self._windows[api_key_id] = entry
            entry.count += 1
            count = entry.count

        if count > limit:
            retry_after = window_start + 60 - int(now)
            if retry_after < 1:
                retry_after = 1
            request_id = getattr(request.state, "request_id", None)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "RATE_LIMITED",
                        "message": f"Rate limit exceeded ({limit} requests/minute). Try again later.",
                        "request_id": request_id,
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    def reset(self) -> None:
        """Clear all window state (for test isolation)."""
        with self._lock:
            self._windows.clear()
