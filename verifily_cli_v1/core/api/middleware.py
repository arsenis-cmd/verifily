"""Request-ID middleware and structured request logging for Verifily API.

Adds X-Request-ID to every response (reads from header or generates UUID4).
Logs one compact line per request: request_id, method, path, status, elapsed_ms.
Never logs request bodies, dataset content, or PII.

Supports VERIFILY_LOG_FORMAT=json for machine-readable JSON log lines.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from verifily_cli_v1.core.api.metrics import metrics
from verifily_cli_v1.core.api.usage_store import compute_api_key_id
from verifily_cli_v1.core.secrets import safe_log_json

logger = logging.getLogger("verifily.api")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach request ID, log metadata, update metrics counters."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Request ID: use client-provided or generate
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id

        # Usage accounting: extract api_key_id + project_id
        raw_key = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            raw_key = auth_header[7:]
        request.state.api_key_id = compute_api_key_id(raw_key)
        request.state.project_id = request.headers.get("x-project-id")

        # Metrics: inflight + total
        metrics.inc_requests_total()
        metrics.inc_inflight()

        t0 = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            metrics.dec_inflight()

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Per-endpoint metrics (counter + latency + errors)
        is_error = response.status_code >= 400
        metrics.record_endpoint_request(
            request.url.path, 
            request.method, 
            elapsed_ms,
            is_error=is_error
        )

        # Add header to response
        response.headers["x-request-id"] = request_id

        # Structured log — metadata only, never bodies or PII
        log_format = os.environ.get("VERIFILY_LOG_FORMAT", "text")
        if log_format == "json":
            log_event = safe_log_json({
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "level": "INFO",
                "request_id": request_id,
                "api_key_id": request.state.api_key_id,
                "project_id": request.state.project_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            })
            logger.info(json.dumps(log_event, separators=(",", ":")))
        else:
            logger.info(
                "request_id=%s method=%s path=%s status=%d elapsed_ms=%d api_key_id=%s project_id=%s",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                request.state.api_key_id,
                request.state.project_id,
            )

        return response
