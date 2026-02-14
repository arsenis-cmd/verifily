"""Normalized error envelope for Verifily API.

Every API error follows:
    {"error": {"type": "<CODE>", "message": "<human readable>", "request_id": "<id>"}}

Stable error types:
    VALIDATION_ERROR, NOT_FOUND, AUTH_ERROR, RATE_LIMITED,
    CONTRACT_FAIL, TOOL_ERROR, INTERNAL_ERROR
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Request
from verifily_cli_v1.core.secrets import redact_text
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

_STATUS_TO_TYPE: Dict[int, str] = {
    400: "VALIDATION_ERROR",
    401: "AUTH_ERROR",
    403: "AUTH_ERROR",
    404: "NOT_FOUND",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    503: "NOT_READY",
}


def make_error_envelope(
    error_type: str, message: str, request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Build the standard error envelope dict."""
    return {
        "error": {
            "type": error_type,
            "message": message,
            "request_id": request_id,
        }
    }


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException to the normalized error envelope."""
    request_id = getattr(request.state, "request_id", None)
    error_type = _STATUS_TO_TYPE.get(exc.status_code, "INTERNAL_ERROR")

    if isinstance(exc.detail, dict):
        raw = exc.detail.get("error") or exc.detail.get("message") or exc.detail
        message = raw if isinstance(raw, str) else str(raw)
    elif isinstance(exc.detail, str):
        message = exc.detail
    else:
        message = str(exc.detail)

    message = redact_text(message)

    return JSONResponse(
        status_code=exc.status_code,
        content=make_error_envelope(error_type, message, request_id),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert Pydantic validation errors to the normalized envelope."""
    request_id = getattr(request.state, "request_id", None)
    errors = exc.errors()
    if errors:
        parts = []
        for err in errors:
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "")
            parts.append(f"{loc}: {msg}" if loc else msg)
        message = "; ".join(parts)
    else:
        message = str(exc)

    message = redact_text(message)

    return JSONResponse(
        status_code=422,
        content=make_error_envelope("VALIDATION_ERROR", message, request_id),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — return 500 with envelope."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content=make_error_envelope("INTERNAL_ERROR", "Internal server error.", request_id),
    )
