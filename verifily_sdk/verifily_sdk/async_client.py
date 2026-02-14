"""AsyncVerifilyClient — asynchronous Python SDK for the Verifily API.

Requires: pip install httpx[async]
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypeVar, Union

import httpx

from verifily_sdk.auth import build_auth_headers
from verifily_sdk.errors import ApiError, AuthError, BudgetExceededError, ForbiddenError, NotFoundError, QuotaExceededError, ServerError, ValidationError
from verifily_sdk.models import (
    HealthResponse,
    ReadyResponse,
    PipelineResponse,
    ContaminationResponse,
    ReportResponse,
    UsageResponse,
    JobSubmitResponse,
    JobMetaResponse,
    JobListResponse,
    MonitorStartResponse,
    MonitorStatusResponse,
    MonitorHistoryResponse,
)
from verifily_sdk.utils import generate_request_id

T = TypeVar('T')


# Status codes that warrant a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# Status codes that should NOT be retried
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


class AsyncVerifilyClient:
    """Asynchronous client for the Verifily API.

    Usage::

        import asyncio
        from verifily_sdk import AsyncVerifilyClient

        async def main():
            c = AsyncVerifilyClient(base_url="http://localhost:8080", api_key="...")
            r = await c.pipeline(config_path="/path/to/verifily.yaml", plan=True)
            print(r.decision)

        asyncio.run(main())
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
    ) -> None:
        """Initialize async client.

        Args:
            base_url: Verifily API base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            retries: Number of retries for retryable errors (429, 5xx)
            retry_delay: Initial delay between retries (seconds)
            retry_backoff: Backoff multiplier for retries
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._retries = retries
        self._retry_delay = retry_delay
        self._retry_backoff = retry_backoff
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)

    # ── Internal helpers ─────────────────────────────────────────

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        headers.update(build_auth_headers(self._api_key))
        if extra:
            headers.update(extra)
        if "x-request-id" not in {k.lower() for k in headers}:
            headers["X-Request-ID"] = generate_request_id()
        return headers

    def _should_retry(self, status_code: int) -> bool:
        """Determine if a request should be retried based on status code."""
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        return status_code in RETRYABLE_STATUS_CODES

    def _raise_for_status(self, resp: httpx.Response, endpoint: str) -> None:
        """Raise appropriate exception based on status code."""
        if resp.status_code < 400:
            return

        request_id = resp.headers.get("x-request-id")
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                message = err.get("message", str(body))
            else:
                message = body.get("message") or body.get("detail") or str(body)
        else:
            message = str(body)

        # Include endpoint info in error
        full_message = f"{message} (endpoint: {endpoint})"

        if resp.status_code == 401:
            raise AuthError(resp.status_code, full_message, body, request_id)
        if resp.status_code == 402:
            raise BudgetExceededError(resp.status_code, full_message, body, request_id)
        if resp.status_code == 403:
            raise ForbiddenError(resp.status_code, full_message, body, request_id)
        if resp.status_code == 404:
            raise NotFoundError(resp.status_code, full_message, body, request_id)
        if resp.status_code == 422:
            raise ValidationError(resp.status_code, full_message, body, request_id)
        if resp.status_code == 429:
            raise QuotaExceededError(resp.status_code, full_message, body, request_id)
        if resp.status_code >= 500:
            raise ServerError(resp.status_code, full_message, body, request_id)
        raise ApiError(resp.status_code, full_message, body, request_id)

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic.

        Retries on 429 and 5xx errors with exponential backoff.
        Does NOT retry on 4xx client errors.
        """
        endpoint = f"{method} {path}"
        headers = self._headers(kwargs.pop("headers", None))
        
        last_error: Optional[Exception] = None
        delay = self._retry_delay

        for attempt in range(self._retries + 1):
            try:
                if method == "GET":
                    resp = await self._client.get(path, headers=headers, **kwargs)
                elif method == "POST":
                    resp = await self._client.post(path, headers=headers, **kwargs)
                elif method == "PUT":
                    resp = await self._client.put(path, headers=headers, **kwargs)
                elif method == "DELETE":
                    resp = await self._client.delete(path, headers=headers, **kwargs)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Check if we should retry
                if resp.status_code >= 400 and attempt < self._retries:
                    if self._should_retry(resp.status_code):
                        last_error = self._create_error_from_response(resp, endpoint)
                        await asyncio.sleep(delay)
                        delay *= self._retry_backoff
                        continue

                self._raise_for_status(resp, endpoint)
                return resp

            except (httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt < self._retries:
                    last_error = e
                    await asyncio.sleep(delay)
                    delay *= self._retry_backoff
                    continue
                raise ServerError(0, f"Network error after {self._retries} retries: {e}", None, None)

        # If we exhausted retries
        if last_error:
            raise last_error
        raise ServerError(0, f"Request failed after {self._retries} retries", None, None)

    def _create_error_from_response(self, resp: httpx.Response, endpoint: str) -> ApiError:
        """Create appropriate error from response without raising."""
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        if isinstance(body, dict):
            message = body.get("message") or body.get("detail") or str(body)
        else:
            message = str(body)

        full_message = f"{message} (endpoint: {endpoint})"
        return ApiError(resp.status_code, full_message, body, resp.headers.get("x-request-id"))

    async def _get(self, path: str, **kwargs) -> httpx.Response:
        return await self._request_with_retry("GET", path, **kwargs)

    async def _post(self, path: str, **kwargs) -> httpx.Response:
        return await self._request_with_retry("POST", path, **kwargs)

    # ── Public API ───────────────────────────────────────────────

    async def health(self) -> HealthResponse:
        """GET /health"""
        resp = await self._get("/health")
        return HealthResponse(**resp.json())

    async def ready(self) -> ReadyResponse:
        """GET /ready"""
        resp = await self._get("/ready")
        return ReadyResponse(**resp.json())

    async def metrics(self) -> str:
        """GET /metrics — returns raw plaintext."""
        resp = await self._get("/metrics")
        return resp.text

    async def pipeline(
        self,
        *,
        config_path: Optional[str] = None,
        project_path: Optional[str] = None,
        plan: bool = False,
        ci: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
    ) -> PipelineResponse:
        """POST /v1/pipeline"""
        body: Dict[str, Any] = {"plan": plan, "ci": ci}
        if config_path:
            body["config_path"] = config_path
        if project_path:
            body["project_path"] = project_path
        if overrides:
            body["overrides"] = overrides
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = await self._post("/v1/pipeline", json=body, headers=extra)
        return PipelineResponse(**resp.json())

    async def contamination(
        self,
        *,
        train_path: str,
        eval_path: str,
        jaccard_cutoff: float = 0.70,
        no_write: bool = True,
        project_id: Optional[str] = None,
    ) -> ContaminationResponse:
        """POST /v1/contamination"""
        body: Dict[str, Any] = {
            "train_path": train_path,
            "eval_path": eval_path,
            "jaccard_cutoff": jaccard_cutoff,
            "no_write": no_write,
        }
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = await self._post("/v1/contamination", json=body, headers=extra)
        return ContaminationResponse(**resp.json())

    async def report(
        self,
        *,
        dataset_path: str,
        schema: str = "sft",
        sample: int = 0,
        project_id: Optional[str] = None,
    ) -> ReportResponse:
        """POST /v1/report"""
        body: Dict[str, Any] = {
            "dataset_path": dataset_path,
            "schema": schema,
            "sample": sample,
        }
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = await self._post("/v1/report", json=body, headers=extra)
        return ReportResponse(**resp.json())

    async def usage(
        self,
        *,
        window_minutes: int = 0,
        group_by: str = "key_project",
    ) -> UsageResponse:
        """GET /v1/usage"""
        params: Dict[str, Any] = {
            "window_minutes": window_minutes,
            "group_by": group_by,
        }
        resp = await self._get("/v1/usage", params=params)
        return UsageResponse(**resp.json())

    # ── Async Jobs ─────────────────────────────────────────────

    async def submit_pipeline_job(
        self,
        *,
        config_path: Optional[str] = None,
        project_path: Optional[str] = None,
        plan: bool = False,
        ci: bool = True,
        overrides: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
    ) -> JobSubmitResponse:
        """POST /v1/jobs/pipeline — submit async pipeline job."""
        body: Dict[str, Any] = {"plan": plan, "ci": ci}
        if config_path:
            body["config_path"] = config_path
        if project_path:
            body["project_path"] = project_path
        if overrides:
            body["overrides"] = overrides
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = await self._post("/v1/jobs/pipeline", json=body, headers=extra)
        return JobSubmitResponse(**resp.json())

    async def get_job(self, job_id: str) -> JobMetaResponse:
        """GET /v1/jobs/{job_id}"""
        resp = await self._get(f"/v1/jobs/{job_id}")
        return JobMetaResponse(**resp.json())

    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """GET /v1/jobs/{job_id}/result"""
        resp = await self._get(f"/v1/jobs/{job_id}/result")
        return resp.json()

    async def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 50,
    ) -> JobListResponse:
        """GET /v1/jobs"""
        params: Dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if project_id:
            params["project_id"] = project_id
        resp = await self._get("/v1/jobs", params=params)
        return JobListResponse(**resp.json())

    async def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_s: float = 30.0,
        poll_s: float = 0.5,
    ) -> JobMetaResponse:
        """Poll GET /v1/jobs/{job_id} until terminal status or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            meta = await self.get_job(job_id)
            if meta.status in ("SUCCEEDED", "FAILED"):
                return meta
            await asyncio.sleep(poll_s)
        return await self.get_job(job_id)

    # ── Monitor ────────────────────────────────────────────────

    async def start_monitor(
        self,
        *,
        config_path: str,
        interval_seconds: int = 60,
        max_ticks: int = 0,
        rolling_window: int = 20,
        project_id: Optional[str] = None,
    ) -> MonitorStartResponse:
        """POST /v1/monitor/start"""
        body: Dict[str, Any] = {
            "config_path": config_path,
            "interval_seconds": interval_seconds,
            "max_ticks": max_ticks,
            "rolling_window": rolling_window,
        }
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = await self._post("/v1/monitor/start", json=body, headers=extra)
        return MonitorStartResponse(**resp.json())

    async def stop_monitor(
        self,
        monitor_id: str,
        *,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/monitor/stop"""
        extra: Optional[Dict[str, str]] = None
        if project_id:
            extra = {"X-Project-ID": project_id}
        resp = await self._post(
            "/v1/monitor/stop",
            json={},
            params={"monitor_id": monitor_id},
            headers=extra,
        )
        return resp.json()

    async def monitor_status(
        self,
        monitor_id: str,
        *,
        project_id: Optional[str] = None,
    ) -> MonitorStatusResponse:
        """GET /v1/monitor/status"""
        params: Dict[str, Any] = {"monitor_id": monitor_id}
        extra: Optional[Dict[str, str]] = None
        if project_id:
            extra = {"X-Project-ID": project_id}
        resp = await self._get("/v1/monitor/status", params=params, headers=extra)
        return MonitorStatusResponse(**resp.json())

    # ── Context Manager ─────────────────────────────────────────

    async def __aenter__(self) -> "AsyncVerifilyClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
