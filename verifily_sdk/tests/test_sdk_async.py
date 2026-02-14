"""Tests for AsyncVerifilyClient.

Target: ~8 tests, runtime <0.5s
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from verifily_sdk import AsyncVerifilyClient, ApiError, ServerError


def run_async(coro):
    """Helper to run async coroutine in sync test."""
    return asyncio.run(coro)


class TestAsyncClientBasics:
    """Test async client basic functionality."""

    def test_async_client_init(self) -> None:
        """Async client can be initialized."""
        client = AsyncVerifilyClient(
            base_url="http://localhost:8080",
            api_key="test-key",
            timeout=30.0,
        )
        assert client._base_url == "http://localhost:8080"
        assert client._api_key == "test-key"
        assert client._timeout == 30.0
        run_async(client.close())

    def test_async_client_context_manager(self) -> None:
        """Async client works as context manager."""
        async def test_ctx():
            async with AsyncVerifilyClient(base_url="http://localhost:8080") as client:
                assert isinstance(client, AsyncVerifilyClient)
        run_async(test_ctx())

    def test_headers_generation(self) -> None:
        """Headers include auth and request ID."""
        client = AsyncVerifilyClient(api_key="test-key")
        headers = client._headers()
        
        assert "Authorization" in headers
        assert "X-Request-ID" in headers
        assert headers["Authorization"] == "Bearer test-key"
        run_async(client.close())

    def test_headers_with_extra(self) -> None:
        """Extra headers are merged."""
        client = AsyncVerifilyClient(api_key="test-key")
        headers = client._headers({"X-Custom": "value"})
        
        assert headers["X-Custom"] == "value"
        assert "Authorization" in headers
        run_async(client.close())


class TestAsyncRetryLogic:
    """Test async retry classification."""

    def test_should_retry_429(self) -> None:
        """429 should be retried."""
        client = AsyncVerifilyClient()
        assert client._should_retry(429) is True
        run_async(client.close())

    def test_should_retry_5xx(self) -> None:
        """5xx errors should be retried."""
        client = AsyncVerifilyClient()
        assert client._should_retry(500) is True
        assert client._should_retry(502) is True
        assert client._should_retry(503) is True
        assert client._should_retry(504) is True
        run_async(client.close())

    def test_should_not_retry_4xx(self) -> None:
        """4xx errors should NOT be retried (except 429)."""
        client = AsyncVerifilyClient()
        assert client._should_retry(400) is False
        assert client._should_retry(401) is False
        assert client._should_retry(403) is False
        assert client._should_retry(404) is False
        assert client._should_retry(422) is False
        run_async(client.close())


class TestAsyncErrorHandling:
    """Test async error handling with endpoint info."""

    def test_error_includes_endpoint(self) -> None:
        """Error message includes endpoint information."""
        client = AsyncVerifilyClient()
        
        # Create mock response
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"x-request-id": "req-123"}
        mock_resp.json.return_value = {"message": "Not found"}
        
        with pytest.raises(ApiError) as exc_info:
            client._raise_for_status(mock_resp, "GET /v1/test")
        
        assert "GET /v1/test" in str(exc_info.value)
        assert exc_info.value.status_code == 404
        assert exc_info.value.request_id == "req-123"
        run_async(client.close())

    def test_error_includes_request_id(self) -> None:
        """Error includes request ID from response header."""
        client = AsyncVerifilyClient()
        
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.headers = {"x-request-id": "test-req-id"}
        mock_resp.json.return_value = {"message": "Server error"}
        
        with pytest.raises(ServerError) as exc_info:
            client._raise_for_status(mock_resp, "POST /v1/test")
        
        assert exc_info.value.request_id == "test-req-id"
        run_async(client.close())


class TestAsyncClientConfig:
    """Test async client configuration options."""

    def test_retry_config(self) -> None:
        """Retry configuration is respected."""
        client = AsyncVerifilyClient(
            retries=5,
            retry_delay=2.0,
            retry_backoff=3.0,
        )
        
        assert client._retries == 5
        assert client._retry_delay == 2.0
        assert client._retry_backoff == 3.0
        run_async(client.close())

    def test_default_retries(self) -> None:
        """Default retry config."""
        client = AsyncVerifilyClient()
        
        assert client._retries == 3  # Default from constructor
        assert client._retry_delay == 1.0
        assert client._retry_backoff == 2.0
        run_async(client.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
