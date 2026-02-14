"""VerifilyClient — typed Python SDK for the Verifily API."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from verifily_sdk.auth import build_auth_headers
from verifily_sdk.errors import ApiError, AuthError, BudgetExceededError, ForbiddenError, NotFoundError, QuotaExceededError, ServerError, ValidationError
from verifily_sdk.models import (
    AdminKeyListResponse,
    AdminKeyResponse,
    AdminProjectListResponse,
    AdminProjectResponse,
    BillingEstimateResponse,
    BillingEventsResponse,
    BillingInvoiceResponse,
    BillingPlansResponse,
    BudgetStatusResponse,
    CheckoutResponse,
    SubscriptionResponse,
    ContaminationResponse,
    EffectiveConfigResponse,
    HealthIndexResponse,
    HealthResponse,
    InvoicePreviewResponse,
    JobListResponse,
    JobMetaResponse,
    JobSubmitResponse,
    LineageResponse,
    ListModelsResponse,
    ModelHistoryResponse,
    MonitorHistoryResponse,
    MonitorStartResponse,
    MonitorStatusResponse,
    OrgListResponse,
    OrgResponse,
    PipelineResponse,
    ProjectListResponse,
    ProjectResponse,
    PromoteModelResponse,
    ReadyResponse,
    RegisterModelResponse,
    ReportResponse,
    RetrainResponse,
    RiskScoreResponse,
    ScoreResponse,
    TeamsAddMembershipResponse,
    TeamsCreateOrgResponse,
    TeamsCreateProjectResponse,
    TeamsCreateUserResponse,
    TeamsIssueApiKeyResponse,
    TeamsProjectListResponse,
    TeamsWhoamiResponse,
    UsageResponse,
)
from verifily_sdk.utils import generate_request_id, retry_with_backoff


class VerifilyClient:
    """Synchronous client for the Verifily API.

    Usage::

        from verifily_sdk import VerifilyClient

        c = VerifilyClient(base_url="http://localhost:8080", api_key="...")
        r = c.pipeline(config_path="/path/to/verifily.yaml", plan=True)
        print(r.decision)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        retries: int = 0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._retries = retries
        self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout)

    # ── Internal helpers ─────────────────────────────────────────

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        headers.update(build_auth_headers(self._api_key))
        if extra:
            headers.update(extra)
        if "x-request-id" not in {k.lower() for k in headers}:
            headers["X-Request-ID"] = generate_request_id()
        return headers

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        request_id = resp.headers.get("x-request-id")
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        if isinstance(body, dict):
            # Support normalized error envelope: {"error": {"type": ..., "message": ...}}
            err = body.get("error")
            if isinstance(err, dict):
                message = err.get("message", str(body))
            else:
                message = body.get("message") or body.get("detail") or str(body)
        else:
            message = str(body)

        if resp.status_code == 401:
            raise AuthError(resp.status_code, message, body, request_id)
        if resp.status_code == 402:
            raise BudgetExceededError(resp.status_code, message, body, request_id)
        if resp.status_code == 403:
            raise ForbiddenError(resp.status_code, message, body, request_id)
        if resp.status_code == 404:
            raise NotFoundError(resp.status_code, message, body, request_id)
        if resp.status_code == 422:
            raise ValidationError(resp.status_code, message, body, request_id)
        if resp.status_code == 429:
            raise QuotaExceededError(resp.status_code, message, body, request_id)
        if resp.status_code >= 500:
            raise ServerError(resp.status_code, message, body, request_id)
        raise ApiError(resp.status_code, message, body, request_id)

    def _get(self, path: str, **kwargs) -> httpx.Response:
        headers = self._headers(kwargs.pop("headers", None))

        def do():
            return self._client.get(path, headers=headers, **kwargs)

        if self._retries > 0:
            resp = retry_with_backoff(do, retries=self._retries)
        else:
            resp = do()
        self._raise_for_status(resp)
        return resp

    def _post(self, path: str, json: Dict[str, Any], **kwargs) -> httpx.Response:
        headers = self._headers(kwargs.pop("headers", None))

        def do():
            return self._client.post(path, json=json, headers=headers, **kwargs)

        if self._retries > 0:
            resp = retry_with_backoff(do, retries=self._retries)
        else:
            resp = do()
        self._raise_for_status(resp)
        return resp

    # ── Public API ───────────────────────────────────────────────

    def health(self) -> HealthResponse:
        """GET /health"""
        resp = self._get("/health")
        return HealthResponse(**resp.json())

    def ready(self) -> ReadyResponse:
        """GET /ready"""
        resp = self._get("/ready")
        return ReadyResponse(**resp.json())

    def metrics(self) -> str:
        """GET /metrics — returns raw plaintext."""
        resp = self._get("/metrics")
        return resp.text

    def pipeline(
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
        resp = self._post("/v1/pipeline", json=body, headers=extra)
        return PipelineResponse(**resp.json())

    def contamination(
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
        resp = self._post("/v1/contamination", json=body, headers=extra)
        return ContaminationResponse(**resp.json())

    def report(
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
        resp = self._post("/v1/report", json=body, headers=extra)
        return ReportResponse(**resp.json())

    def usage(
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
        resp = self._get("/v1/usage", params=params)
        return UsageResponse(**resp.json())

    # ── Async Jobs ─────────────────────────────────────────────

    def submit_pipeline_job(
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
        resp = self._post("/v1/jobs/pipeline", json=body, headers=extra)
        return JobSubmitResponse(**resp.json())

    def submit_contamination_job(
        self,
        *,
        train_path: str,
        eval_path: str,
        jaccard_cutoff: float = 0.70,
        no_write: bool = True,
        project_id: Optional[str] = None,
    ) -> JobSubmitResponse:
        """POST /v1/jobs/contamination — submit async contamination job."""
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
        resp = self._post("/v1/jobs/contamination", json=body, headers=extra)
        return JobSubmitResponse(**resp.json())

    def submit_report_job(
        self,
        *,
        dataset_path: str,
        schema: str = "sft",
        sample: int = 0,
        project_id: Optional[str] = None,
    ) -> JobSubmitResponse:
        """POST /v1/jobs/report — submit async report job."""
        body: Dict[str, Any] = {
            "dataset_path": dataset_path,
            "schema": schema,
            "sample": sample,
        }
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = self._post("/v1/jobs/report", json=body, headers=extra)
        return JobSubmitResponse(**resp.json())

    def submit_classify_job(
        self,
        *,
        dataset_path: str,
        output_dir: Optional[str] = None,
        max_rows_scan: int = 500,
        export_buckets: bool = False,
        min_bucket_rows: int = 1,
        project_id: Optional[str] = None,
    ) -> JobSubmitResponse:
        """POST /v1/jobs/classify — submit async classify job."""
        body: Dict[str, Any] = {
            "dataset_path": dataset_path,
            "max_rows_scan": max_rows_scan,
            "export_buckets": export_buckets,
            "min_bucket_rows": min_bucket_rows,
        }
        if output_dir:
            body["output_dir"] = output_dir
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = self._post("/v1/jobs/classify", json=body, headers=extra)
        return JobSubmitResponse(**resp.json())

    def get_job(self, job_id: str) -> JobMetaResponse:
        """GET /v1/jobs/{job_id}"""
        resp = self._get(f"/v1/jobs/{job_id}")
        return JobMetaResponse(**resp.json())

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """GET /v1/jobs/{job_id}/result"""
        resp = self._get(f"/v1/jobs/{job_id}/result")
        return resp.json()

    def list_jobs(
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
        resp = self._get("/v1/jobs", params=params)
        return JobListResponse(**resp.json())

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_s: float = 30.0,
        poll_s: float = 0.5,
    ) -> JobMetaResponse:
        """Poll GET /v1/jobs/{job_id} until terminal status or timeout."""
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            meta = self.get_job(job_id)
            if meta.status in ("SUCCEEDED", "FAILED"):
                return meta
            time.sleep(poll_s)
        return self.get_job(job_id)

    # ── Retrain ────────────────────────────────────────────────

    def retrain(
        self,
        *,
        dataset_dir: str,
        base_run_dir: Optional[str] = None,
        contaminated_run_dir: Optional[str] = None,
        metric: str = "f1",
        mode: str = "mock",
        output_dir: Optional[str] = None,
        seed: int = 42,
        notes: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> RetrainResponse:
        """POST /v1/retrain — synchronous retrain."""
        body: Dict[str, Any] = {
            "dataset_dir": dataset_dir,
            "metric": metric,
            "mode": mode,
            "seed": seed,
        }
        if base_run_dir:
            body["base_run_dir"] = base_run_dir
        if contaminated_run_dir:
            body["contaminated_run_dir"] = contaminated_run_dir
        if output_dir:
            body["output_dir"] = output_dir
        if notes:
            body["notes"] = notes
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = self._post("/v1/retrain", json=body, headers=extra)
        return RetrainResponse(**resp.json())

    def submit_retrain_job(
        self,
        *,
        dataset_dir: str,
        base_run_dir: Optional[str] = None,
        contaminated_run_dir: Optional[str] = None,
        metric: str = "f1",
        mode: str = "mock",
        output_dir: Optional[str] = None,
        seed: int = 42,
        notes: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> JobSubmitResponse:
        """POST /v1/jobs/retrain — async retrain job."""
        body: Dict[str, Any] = {
            "dataset_dir": dataset_dir,
            "metric": metric,
            "mode": mode,
            "seed": seed,
        }
        if base_run_dir:
            body["base_run_dir"] = base_run_dir
        if contaminated_run_dir:
            body["contaminated_run_dir"] = contaminated_run_dir
        if output_dir:
            body["output_dir"] = output_dir
        if notes:
            body["notes"] = notes
        extra: Optional[Dict[str, str]] = None
        if project_id:
            body["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = self._post("/v1/jobs/retrain", json=body, headers=extra)
        return JobSubmitResponse(**resp.json())

    # ── Monitor ────────────────────────────────────────────────

    def start_monitor(
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
        resp = self._post("/v1/monitor/start", json=body, headers=extra)
        return MonitorStartResponse(**resp.json())

    def stop_monitor(
        self,
        monitor_id: str,
        *,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/monitor/stop"""
        extra: Optional[Dict[str, str]] = None
        if project_id:
            extra = {"X-Project-ID": project_id}
        resp = self._post(
            "/v1/monitor/stop",
            json={},
            params={"monitor_id": monitor_id},
            headers=extra,
        )
        return resp.json()

    def monitor_status(
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
        resp = self._get("/v1/monitor/status", params=params, headers=extra)
        return MonitorStatusResponse(**resp.json())

    def monitor_history(
        self,
        monitor_id: str,
        *,
        last_n: int = 0,
        project_id: Optional[str] = None,
    ) -> MonitorHistoryResponse:
        """GET /v1/monitor/history"""
        params: Dict[str, Any] = {"monitor_id": monitor_id, "last_n": last_n}
        extra: Optional[Dict[str, str]] = None
        if project_id:
            extra = {"X-Project-ID": project_id}
        resp = self._get("/v1/monitor/history", params=params, headers=extra)
        return MonitorHistoryResponse(**resp.json())

    # ── Orgs / Projects ─────────────────────────────────────────

    def create_org(self, *, name: str) -> OrgResponse:
        """POST /v1/orgs — create an organization."""
        resp = self._post("/v1/orgs", json={"name": name})
        return OrgResponse(**resp.json())

    def list_orgs(self) -> OrgListResponse:
        """GET /v1/orgs — list organizations you belong to."""
        resp = self._get("/v1/orgs")
        return OrgListResponse(**resp.json())

    def create_project(self, *, org_id: str, name: str) -> ProjectResponse:
        """POST /v1/projects — create a project in an organization."""
        resp = self._post("/v1/projects", json={"org_id": org_id, "name": name})
        return ProjectResponse(**resp.json())

    def list_projects(self, *, org_id: Optional[str] = None) -> ProjectListResponse:
        """GET /v1/projects — list projects."""
        params: Dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        resp = self._get("/v1/projects", params=params)
        return ProjectListResponse(**resp.json())

    # ── Admin (Enterprise Trust) ─────────────────────────────────

    def admin_create_project(self, *, id: str, name: str) -> AdminProjectResponse:
        """POST /v1/admin/projects — create a project in the auth registry."""
        resp = self._post("/v1/admin/projects", json={"id": id, "name": name})
        return AdminProjectResponse(**resp.json())

    def admin_list_projects(self) -> AdminProjectListResponse:
        """GET /v1/admin/projects — list registry projects."""
        resp = self._get("/v1/admin/projects")
        return AdminProjectListResponse(**resp.json())

    def admin_create_key(
        self,
        *,
        id: str,
        name: str,
        raw_key: str,
        scopes: list,
        projects_allowed: list,
    ) -> AdminKeyResponse:
        """POST /v1/admin/keys — create a scoped API key."""
        resp = self._post("/v1/admin/keys", json={
            "id": id,
            "name": name,
            "raw_key": raw_key,
            "scopes": scopes,
            "projects_allowed": projects_allowed,
        })
        return AdminKeyResponse(**resp.json())

    def admin_list_keys(self) -> AdminKeyListResponse:
        """GET /v1/admin/keys — list all keys (hashes redacted)."""
        resp = self._get("/v1/admin/keys")
        return AdminKeyListResponse(**resp.json())

    def admin_disable_key(self, key_id: str) -> AdminKeyResponse:
        """POST /v1/admin/keys/{key_id}/disable"""
        resp = self._post(f"/v1/admin/keys/{key_id}/disable", json={})
        return AdminKeyResponse(**resp.json())

    def admin_rotate_key(self, key_id: str, *, raw_key: str) -> AdminKeyResponse:
        """POST /v1/admin/keys/{key_id}/rotate"""
        resp = self._post(f"/v1/admin/keys/{key_id}/rotate", json={"raw_key": raw_key})
        return AdminKeyResponse(**resp.json())

    # ── Billing ────────────────────────────────────────────────

    def billing_events(
        self,
        *,
        project_id: Optional[str] = None,
        window_minutes: int = 0,
        limit: int = 100,
    ) -> BillingEventsResponse:
        """GET /v1/billing/events"""
        params: Dict[str, Any] = {"window_minutes": window_minutes, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        resp = self._get("/v1/billing/events", params=params)
        return BillingEventsResponse(**resp.json())

    def invoice_preview(
        self,
        *,
        project_id: str = "default",
        plan_id: str = "FREE",
        window_minutes: int = 43200,
    ) -> InvoicePreviewResponse:
        """GET /v1/billing/invoice-preview"""
        params: Dict[str, Any] = {
            "project_id": project_id,
            "plan_id": plan_id,
            "window_minutes": window_minutes,
        }
        resp = self._get("/v1/billing/invoice-preview", params=params)
        return InvoicePreviewResponse(**resp.json())

    def billing_plans(self) -> BillingPlansResponse:
        """GET /v1/billing/plans"""
        resp = self._get("/v1/billing/plans")
        return BillingPlansResponse(**resp.json())

    def billing_estimate(
        self,
        *,
        plan: str = "FREE",
        window_minutes: int = 43200,
        project_id: Optional[str] = None,
    ) -> BillingEstimateResponse:
        """GET /v1/billing/estimate"""
        params: Dict[str, Any] = {"plan": plan, "window_minutes": window_minutes}
        if project_id:
            params["project_id"] = project_id
        resp = self._get("/v1/billing/estimate", params=params)
        return BillingEstimateResponse(**resp.json())

    def billing_invoice(
        self,
        *,
        plan: str,
        period_start: str,
        period_end: str,
        project_id: Optional[str] = None,
    ) -> BillingInvoiceResponse:
        """POST /v1/billing/invoice"""
        body: Dict[str, Any] = {
            "plan": plan,
            "period_start": period_start,
            "period_end": period_end,
        }
        if project_id:
            body["project_id"] = project_id
        resp = self._post("/v1/billing/invoice", json=body)
        return BillingInvoiceResponse(**resp.json())

    def billing_usage_export(
        self,
        *,
        format: str = "csv",
        period_days: int = 30,
        group_by: str = "day",
    ) -> str:
        """GET /v1/billing/usage_export — returns raw content string."""
        params: Dict[str, Any] = {
            "format": format,
            "period_days": period_days,
            "group_by": group_by,
        }
        resp = self._get("/v1/billing/usage_export", params=params)
        return resp.text

    def billing_usage(
        self,
        *,
        period: Optional[str] = None,
        group_by: str = "total",
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET /v1/billing/usage — query processed rows for a billing period."""
        params: Dict[str, Any] = {"group_by": group_by}
        if period:
            params["period"] = period
        if project_id:
            params["project_id"] = project_id
        resp = self._get("/v1/billing/usage", params=params)
        return resp.json()

    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """GET /v1/billing/invoice/{invoice_id} — retrieve a stored invoice."""
        resp = self._get(f"/v1/billing/invoice/{invoice_id}")
        return resp.json()

    # ── Teams (RBAC) ─────────────────────────────────────────────

    def whoami(self) -> TeamsWhoamiResponse:
        """GET /v1/admin/whoami — return identity of current API key."""
        resp = self._get("/v1/admin/whoami")
        return TeamsWhoamiResponse(**resp.json())

    def admin_create_org(self, *, id: str, name: str) -> TeamsCreateOrgResponse:
        """POST /v1/admin/orgs — create an organization (teams)."""
        resp = self._post("/v1/admin/orgs", json={"id": id, "name": name})
        return TeamsCreateOrgResponse(**resp.json())

    def admin_create_user(self, *, id: str, email: str, name: str) -> TeamsCreateUserResponse:
        """POST /v1/admin/users — create a user (teams)."""
        resp = self._post("/v1/admin/users", json={"id": id, "email": email, "name": name})
        return TeamsCreateUserResponse(**resp.json())

    def admin_add_membership(
        self,
        *,
        user_id: str,
        org_id: str,
        role: str = "member",
    ) -> TeamsAddMembershipResponse:
        """POST /v1/admin/memberships — add a membership (teams)."""
        resp = self._post("/v1/admin/memberships", json={
            "user_id": user_id, "org_id": org_id, "role": role,
        })
        return TeamsAddMembershipResponse(**resp.json())

    def admin_create_team_project(
        self,
        *,
        id: str,
        org_id: str,
        name: str,
    ) -> TeamsCreateProjectResponse:
        """POST /v1/admin/team-projects — create a project (teams)."""
        resp = self._post("/v1/admin/team-projects", json={
            "id": id, "org_id": org_id, "name": name,
        })
        return TeamsCreateProjectResponse(**resp.json())

    def admin_list_team_projects(self) -> TeamsProjectListResponse:
        """GET /v1/admin/team-projects — list team projects."""
        resp = self._get("/v1/admin/team-projects")
        return TeamsProjectListResponse(**resp.json())

    def admin_issue_api_key(
        self,
        *,
        id: str,
        org_id: str,
        name: str,
        raw_key: str,
        scopes: list,
        project_ids: list,
        created_by: str,
    ) -> TeamsIssueApiKeyResponse:
        """POST /v1/admin/api-keys — issue a scoped API key (teams)."""
        resp = self._post("/v1/admin/api-keys", json={
            "id": id, "org_id": org_id, "name": name,
            "raw_key": raw_key, "scopes": scopes,
            "project_ids": project_ids, "created_by": created_by,
        })
        return TeamsIssueApiKeyResponse(**resp.json())

    # ── Workspaces ─────────────────────────────────────────────

    def ws_create_org(
        self, *, name: str, bootstrap_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """POST /v1/orgs — create an organization (workspaces mode)."""
        extra: Dict[str, str] = {}
        if bootstrap_token:
            extra["X-Bootstrap-Token"] = bootstrap_token
        resp = self._post("/v1/orgs", json={"name": name}, headers=extra)
        return resp.json()

    def ws_create_project(
        self, *, org_id: str, name: str, billing_plan: str = "free"
    ) -> Dict[str, Any]:
        """POST /v1/projects — create a project (workspaces mode)."""
        resp = self._post(
            "/v1/projects",
            json={"org_id": org_id, "name": name, "billing_plan": billing_plan},
        )
        return resp.json()

    def ws_create_key(
        self, *, project_id: str, role: str
    ) -> Dict[str, Any]:
        """POST /v1/keys — create an API key (workspaces mode)."""
        resp = self._post(
            "/v1/keys", json={"project_id": project_id, "role": role}
        )
        return resp.json()

    def ws_revoke_key(
        self, *, project_id: str, api_key_id: str
    ) -> Dict[str, Any]:
        """POST /v1/keys/revoke — revoke an API key (workspaces mode)."""
        resp = self._post(
            "/v1/keys/revoke",
            json={"project_id": project_id, "api_key_id": api_key_id},
        )
        return resp.json()

    def ws_me(self) -> Dict[str, Any]:
        """GET /v1/me — return identity of calling key (workspaces mode)."""
        resp = self._get("/v1/me")
        return resp.json()

    # ── Effective Config ───────────────────────────────────────

    def effective_config(self) -> EffectiveConfigResponse:
        """GET /v1/config/effective — return current effective configuration."""
        resp = self._get("/v1/config/effective")
        return EffectiveConfigResponse(**resp.json())

    # ── Registry ────────────────────────────────────────────────

    def register_model(
        self,
        *,
        run_dir: str,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
    ) -> RegisterModelResponse:
        """POST /v1/registry/register — register a model from a run.

        Args:
            run_dir: Path to run directory
            model_id: Optional model ID (auto-detected if not provided)
            version: Optional version (auto-generated if not provided)

        Returns:
            RegisterModelResponse with success status and record/error
        """
        body: Dict[str, Any] = {"run_dir": run_dir}
        if model_id:
            body["model_id"] = model_id
        if version:
            body["version"] = version
        resp = self._post("/v1/registry/register", json=body)
        return RegisterModelResponse(**resp.json())

    def promote_model(
        self,
        *,
        model_id: str,
        version: str,
        target_stage: str,
        reason: str = "",
    ) -> PromoteModelResponse:
        """POST /v1/registry/promote — promote a model to a new stage.

        Args:
            model_id: Model identifier
            version: Model version
            target_stage: Target stage (none/staging/production/archived)
            reason: Optional promotion reason

        Returns:
            PromoteModelResponse with success status and record/error
        """
        body = {
            "model_id": model_id,
            "version": version,
            "target_stage": target_stage,
            "reason": reason,
        }
        resp = self._post("/v1/registry/promote", json=body)
        return PromoteModelResponse(**resp.json())

    def list_models(
        self,
        *,
        stage: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> ListModelsResponse:
        """GET /v1/registry/models — list registered models.

        Args:
            stage: Optional stage filter
            model_id: Optional model ID filter

        Returns:
            ListModelsResponse with list of models
        """
        params: Dict[str, Any] = {}
        if stage:
            params["stage"] = stage
        if model_id:
            params["model_id"] = model_id
        resp = self._get("/v1/registry/models", params=params)
        return ListModelsResponse(**resp.json())

    def model_history(self, model_id: str) -> ModelHistoryResponse:
        """GET /v1/registry/history/{model_id} — get model history.

        Args:
            model_id: Model identifier

        Returns:
            ModelHistoryResponse with all versions
        """
        resp = self._get(f"/v1/registry/history/{model_id}")
        return ModelHistoryResponse(**resp.json())

    # ── Score ───────────────────────────────────────────────────

    def score(
        self,
        *,
        run_dir: str,
        project_id: Optional[str] = None,
    ) -> ScoreResponse:
        """GET /v1/score — return Risk Score and Health Index for a run.

        Args:
            run_dir: Path to run directory
            project_id: Optional project identifier

        Returns:
            ScoreResponse with risk_score, health_index, verdict, recommendations
        """
        params: Dict[str, Any] = {"run_dir": run_dir}
        extra: Optional[Dict[str, str]] = None
        if project_id:
            params["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = self._get("/v1/score", params=params, headers=extra)
        return ScoreResponse(**resp.json())

    # ── Lineage ────────────────────────────────────────────────

    def lineage(
        self,
        *,
        run_dir: str,
        project_id: Optional[str] = None,
    ) -> LineageResponse:
        """GET /v1/lineage — return lineage graph for a run.

        Args:
            run_dir: Path to run directory
            project_id: Optional project identifier

        Returns:
            Lineage graph with nodes and edges
        """
        params: Dict[str, Any] = {"run_dir": run_dir}
        extra: Optional[Dict[str, str]] = None
        if project_id:
            params["project_id"] = project_id
            extra = {"X-Project-ID": project_id}
        resp = self._get("/v1/lineage", params=params, headers=extra)
        return LineageResponse(**resp.json())

    # ── Budget ─────────────────────────────────────────────────

    def budget_status(
        self,
        *,
        project_id: str = "default",
    ) -> BudgetStatusResponse:
        """GET /v1/budget/status — return budget status for a project.

        Args:
            project_id: Project identifier (default: "default")

        Returns:
            BudgetStatusResponse with policy, usage, and reset time
        """
        params: Dict[str, Any] = {"project_id": project_id}
        resp = self._get("/v1/budget/status", params=params)
        return BudgetStatusResponse(**resp.json())

    # ── Org Mode (Multi-tenant Admin) ────────────────────────────

    def om_create_org(self, *, name: str) -> Dict[str, Any]:
        """POST /v1/orgs — create a new organization (ADMIN only, org_mode).

        Args:
            name: Organization name

        Returns:
            Dict with org_id, name, created_at
        """
        resp = self._post("/v1/orgs", json={"name": name})
        return resp.json()

    def om_create_project(self, *, org_id: str, name: str) -> Dict[str, Any]:
        """POST /v1/projects — create a new project (ADMIN only, org_mode).

        Args:
            org_id: Parent organization ID
            name: Project name

        Returns:
            Dict with project_id, org_id, name, created_at
        """
        resp = self._post("/v1/projects", json={"org_id": org_id, "name": name})
        return resp.json()

    def om_list_projects(self, *, org_id: Optional[str] = None) -> Dict[str, Any]:
        """GET /v1/projects — list projects (org_mode).

        Args:
            org_id: Optional filter by org

        Returns:
            Dict with projects list
        """
        params: Dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        resp = self._get("/v1/projects", params=params)
        return resp.json()

    def om_create_key(
        self,
        *,
        project_id: str,
        role: str,  # "admin", "dev", "viewer"
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/keys — create a new API key (ADMIN only).

        IMPORTANT: The secret is returned ONCE and never stored.
        Save it immediately - you cannot retrieve it later.

        Args:
            project_id: Project to scope the key to
            role: Permission level (admin, dev, viewer)
            label: Optional human-readable label

        Returns:
            Dict with secret, key_id, org_id, project_id, role, etc.
        """
        body: Dict[str, Any] = {
            "project_id": project_id,
            "role": role,
        }
        if label:
            body["label"] = label
        resp = self._post("/v1/keys", json=body)
        return resp.json()

    def om_revoke_key(self, *, key_id: str) -> Dict[str, Any]:
        """POST /v1/keys/{key_id}/revoke — revoke an API key (ADMIN only, org_mode).

        Args:
            key_id: Key to revoke

        Returns:
            Dict with key_id, revoked_at, was_active
        """
        resp = self._post(f"/v1/keys/{key_id}/revoke", json={})
        return resp.json()

    def om_list_keys(self, *, project_id: Optional[str] = None) -> Dict[str, Any]:
        """GET /v1/keys — list API keys (ADMIN only, org_mode).

        Args:
            project_id: Optional filter by project

        Returns:
            Dict with keys list
        """
        params: Dict[str, Any] = {}
        if project_id:
            params["project_id"] = project_id
        resp = self._get("/v1/keys", params=params)
        return resp.json()

    # ── Stripe / Subscriptions ────────────────────────────────────

    def checkout(
        self,
        *,
        plan: str = "pro",
        project_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> CheckoutResponse:
        """POST /v1/billing/checkout — start a Stripe Checkout session."""
        body: Dict[str, Any] = {"plan": plan}
        if project_id:
            body["project_id"] = project_id
        if org_id:
            body["org_id"] = org_id
        resp = self._post("/v1/billing/checkout", json=body)
        return CheckoutResponse(**resp.json())

    def subscription(
        self,
        *,
        project_id: str = "default",
    ) -> SubscriptionResponse:
        """GET /v1/billing/subscription — get subscription status."""
        resp = self._get("/v1/billing/subscription", params={"project_id": project_id})
        return SubscriptionResponse(**resp.json())

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
