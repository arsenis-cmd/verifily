"""Verifily Python SDK — typed client for the Verifily API."""

from verifily_sdk.client import VerifilyClient
from verifily_sdk.async_client import AsyncVerifilyClient
from verifily_sdk.errors import (
    ApiError, 
    AuthError, 
    NotFoundError, 
    ServerError, 
    ValidationError,
    QuotaExceededError,
    BudgetExceededError,
    ForbiddenError,
)

__version__ = "1.0.0"

__all__ = [
    "VerifilyClient",
    "AsyncVerifilyClient",
    "ApiError",
    "AuthError",
    "NotFoundError",
    "ServerError",
    "ValidationError",
    "QuotaExceededError",
    "BudgetExceededError",
    "ForbiddenError",
]
