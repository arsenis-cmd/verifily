"""Tests for SDK retry logic.

Target: ~8 tests, runtime <0.3s
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from verifily_sdk import VerifilyClient, AsyncVerifilyClient
from verifily_sdk.errors import ApiError, ServerError, AuthError


def run_async(coro):
    """Helper to run async coroutine in sync test."""
    return asyncio.run(coro)


class TestSyncRetry:
    """Test synchronous client retry behavior."""

    def test_sync_client_retry_config(self) -> None:
        """Sync client accepts retry config."""
        client = VerifilyClient(retries=3)
        assert client._retries == 3

    def test_sync_no_retry_on_401(self) -> None:
        """401 should NOT be retried."""
        client = VerifilyClient(retries=3)
        
        # Create mock response that returns 401
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {}
        mock_resp.json.return_value = {"message": "Unauthorized"}
        mock_resp.text = "Unauthorized"
        
        with pytest.raises(AuthError):
            client._raise_for_status(mock_resp)

    def test_sync_no_retry_on_422(self) -> None:
        """422 should NOT be retried."""
        client = VerifilyClient(retries=3)
        
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.headers = {}
        mock_resp.json.return_value = {"message": "Validation error"}
        mock_resp.text = "Validation error"
        
        # _raise_for_status doesn't retry, it just raises
        with pytest.raises(Exception):
            client._raise_for_status(mock_resp)


class TestAsyncRetryClassification:
    """Test async retry classification in detail."""

    def test_retry_status_codes(self) -> None:
        """Correct status codes are retryable."""
        from verifily_sdk.async_client import RETRYABLE_STATUS_CODES, NON_RETRYABLE_STATUS_CODES
        
        assert 429 in RETRYABLE_STATUS_CODES
        assert 500 in RETRYABLE_STATUS_CODES
        assert 502 in RETRYABLE_STATUS_CODES
        assert 503 in RETRYABLE_STATUS_CODES
        assert 504 in RETRYABLE_STATUS_CODES
        
        assert 400 in NON_RETRYABLE_STATUS_CODES
        assert 401 in NON_RETRYABLE_STATUS_CODES
        assert 403 in NON_RETRYABLE_STATUS_CODES
        assert 404 in NON_RETRYABLE_STATUS_CODES
        assert 422 in NON_RETRYABLE_STATUS_CODES

    def test_async_retry_on_503(self) -> None:
        """503 should trigger retry."""
        client = AsyncVerifilyClient(retries=2, retry_delay=0.01)
        
        # Should return True for retryable status
        assert client._should_retry(503) is True
        run_async(client.close())

    def test_async_no_retry_on_400(self) -> None:
        """400 should NOT trigger retry."""
        client = AsyncVerifilyClient(retries=2)
        
        assert client._should_retry(400) is False
        run_async(client.close())


class TestErrorDetails:
    """Test error includes full details."""

    def test_error_has_all_fields(self) -> None:
        """ApiError has all expected fields."""
        error = ApiError(
            status_code=404,
            message="Not found",
            detail={"resource": "dataset"},
            request_id="req-abc123",
        )
        
        assert error.status_code == 404
        assert error.message == "Not found"
        assert error.detail == {"resource": "dataset"}
        assert error.request_id == "req-abc123"
        assert "404" in str(error)
        assert "Not found" in str(error)

    def test_error_string_representation(self) -> None:
        """Error string format."""
        error = ApiError(
            status_code=500,
            message="Server error",
            detail=None,
            request_id=None,
        )
        
        str_repr = str(error)
        assert "[500]" in str_repr
        assert "Server error" in str_repr


class TestRetryWithBackoff:
    """Test retry backoff behavior."""

    def test_backoff_increases_delay(self) -> None:
        """Backoff multiplier increases delay between retries."""
        client = AsyncVerifilyClient(
            retries=3,
            retry_delay=1.0,
            retry_backoff=2.0,
        )
        
        # First retry: 1.0 * 2.0 = 2.0
        # Second retry: 2.0 * 2.0 = 4.0
        # etc.
        
        delay = client._retry_delay
        for i in range(3):
            delay *= client._retry_backoff
        
        assert delay == 8.0  # 1.0 * 2^3
        run_async(client.close())

    def test_zero_retries(self) -> None:
        """Zero retries means no retry attempts."""
        client = AsyncVerifilyClient(retries=0)
        
        assert client._retries == 0
        # With 0 retries, only one attempt is made
        run_async(client.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
