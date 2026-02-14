"""Structured exceptions for the Verifily SDK."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ApiError(Exception):
    """Base exception for all Verifily API errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        detail: Any = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.detail = detail
        self.request_id = request_id
        super().__init__(f"[{status_code}] {message}")


class AuthError(ApiError):
    """401 Unauthorized — missing or invalid API key."""
    pass


class ForbiddenError(ApiError):
    """403 Forbidden — insufficient permissions."""
    pass


class ValidationError(ApiError):
    """422 Unprocessable Entity — invalid request parameters."""
    pass


class NotFoundError(ApiError):
    """404 Not Found — config or data file missing."""
    pass


class QuotaExceededError(ApiError):
    """429 Too Many Requests — daily quota exceeded."""
    pass


class BudgetExceededError(ApiError):
    """402 Payment Required — budget limit exceeded."""
    pass


class ServerError(ApiError):
    """500+ — server-side error."""
    pass
