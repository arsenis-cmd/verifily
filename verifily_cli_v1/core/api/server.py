"""Verifily local-only HTTP API server (FastAPI + uvicorn)."""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from verifily_cli_v1 import __version__
from verifily_cli_v1.core.api.auth import ApiKeyMiddleware
from verifily_cli_v1.core.api.budget_middleware import BudgetMiddleware
from verifily_cli_v1.core.api.errors import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from verifily_cli_v1.core.api.metrics import metrics
from verifily_cli_v1.core.api.middleware import RequestIDMiddleware
from verifily_cli_v1.core.security.audit_store import audit_store
from verifily_cli_v1.core.api.models import (
    BudgetStatusResponse,
    AddMembershipRequest,
    AdminCreateKeyRequest,
    AdminCreateProjectRequest,
    AdminKeyListResponse,
    AdminKeyResponse,
    AdminProjectListResponse,
    AdminProjectResponse,
    AdminRotateKeyRequest,
    BillingEventsResponse,
    BillingInvoiceRequest,
    CheckoutRequest,
    CheckoutResponse,
    SubscriptionResponse,
    ClassifyRequest,
    CreateKeyRequest,
    CreateKeyResponse,
    CreateOrgRequest,
    CreateOrgResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    ListKeysResponse,
    ListProjectsResponse,
    RevokeKeyResponse,
    TeamsAddMembershipRequest,
    TeamsAddMembershipResponse,
    TeamsCreateOrgRequest,
    TeamsCreateOrgResponse,
    TeamsCreateProjectRequest,
    TeamsCreateProjectResponse,
    TeamsCreateUserRequest,
    TeamsCreateUserResponse,
    TeamsIssueApiKeyRequest,
    TeamsIssueApiKeyResponse,
    TeamsProjectListResponse,
    TeamsWhoamiResponse,
    ContaminationRequest,
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
    MembershipListResponse,
    MembershipResponse,
    ModelHistoryResponse,
    ModelRecordResponse,
    MonitorHistoryResponse,
    MonitorStartRequest,
    MonitorStartResponse,
    MonitorStatusResponse,
    OrgListResponse,
    OrgResponse,
    PipelineRequest,
    PipelineResponse,
    ProjectListResponse,
    ProjectResponse,
    PromoteModelRequest,
    PromoteModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
    ReadyResponse,
    ReportRequest,
    ReportResponse,
    RetrainRequest,
    RetrainResponse,
    RiskScoreResponse,
    UsageResponse,
)
from verifily_cli_v1.core.api.monitor_store import MonitorConfig, monitor_store
from verifily_cli_v1.core.api.identity import Role
from verifily_cli_v1.core.api.org_store import org_store as org_mode_store
from verifily_cli_v1.core.api.orgs import org_store, Role as OrgsRole
from verifily_cli_v1.core.api.rate_limit import RateLimitMiddleware
from verifily_cli_v1.core.api.settings import load_production_settings, ProductionSettings
from verifily_cli_v1.core.api.usage_store import usage_store
from verifily_cli_v1.core.api.jobs import (
    JobStatus,
    JobType,
    jobs_store,
    register_executor,
)
from verifily_cli_v1.core.api.workspace import validate_workspace_exists
from verifily_cli_v1.core.runtime_paths import get_runtime_paths
from verifily_cli_v1.core.deploy_config import (
    load_deploy_config,
    validate_deploy_config,
    ConfigError,
)
from verifily_cli_v1.core.api.retrain import run_retrain_api
from verifily_cli_v1.core.api.runners import (
    run_classify_api,
    run_contamination_api,
    run_pipeline_api,
    run_report_api,
)

from verifily_cli_v1.core.api.auth_registry import SCOPES, auth_registry
from verifily_cli_v1.core.api.quotas import quota_store
from verifily_cli_v1.core.api.settings import Settings, load_settings
from verifily_cli_v1.core.api.startup_checks import (
    StartupCheckResult,
    format_check_result,
    run_startup_checks,
)
from verifily_cli_v1.core.billing.store import billing_store
from verifily_cli_v1.core.billing.subscriptions import SubscriptionStatus, subscriptions_store
from verifily_cli_v1.core.teams.store import teams_store
from verifily_cli_v1.core.workspaces.models import (
    WsCreateOrgRequest,
    WsCreateProjectRequest,
    WsCreateKeyRequest,
    WsRevokeKeyRequest,
)
from verifily_cli_v1.core.workspaces.store import (
    workspaces_store,
    OrgNotFoundError,
    ProjectNotFoundError,
    KeyNotFoundError,
    InvalidRoleError,
    ROLE_REVERSE,
)

logger = logging.getLogger("verifily.api")

# Global state for readiness tracking
_startup_check_result: Optional[StartupCheckResult] = None
_server_shutting_down: bool = False
_deploy_config_valid: bool = True
_deploy_config_errors: list[str] = []

_DEFAULT_USAGE_LOG = str(Path(tempfile.gettempdir()) / "verifily_usage_events.jsonl")


def _shutdown_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global _server_shutting_down
    _server_shutting_down = True
    logger.info("Received shutdown signal %s, initiating graceful shutdown...", signum)


@asynccontextmanager
async def _lifespan_context(app: FastAPI) -> AsyncGenerator[Dict[str, Any], None]:
    """Lifespan context manager for startup checks and graceful shutdown.
    
    Runs startup validation before the server accepts connections.
    Handles graceful shutdown when the server is stopping.
    """
    global _startup_check_result, _server_shutting_down, _deploy_config_valid, _deploy_config_errors
    
    settings: Settings = app.state.settings
    console = logging.getLogger("verifily.api")
    
    # ── Enterprise Deployment Config ──────────────────────────────
    console.info("Loading deployment configuration...")
    try:
        deploy_config = load_deploy_config()
        config_valid, config_errors = validate_deploy_config(deploy_config)
        _deploy_config_valid = config_valid
        _deploy_config_errors = config_errors
        
        if not config_valid:
            console.error("Deployment configuration errors:")
            for err in config_errors:
                console.error(f"  - {err}")
        else:
            console.info("Deployment configuration loaded successfully")
            
            # Validate runtime paths
            runtime_paths = get_runtime_paths()
            path_errors = runtime_paths.validate_writable()
            if path_errors:
                _deploy_config_valid = False
                _deploy_config_errors.extend(path_errors)
                console.error("Runtime path errors:")
                for err in path_errors:
                    console.error(f"  - {err}")
            else:
                runtime_paths.ensure_directories()
                console.info(f"Runtime directories ready: {runtime_paths.home}")
                
    except ConfigError as e:
        _deploy_config_valid = False
        _deploy_config_errors.append(str(e))
        console.error(f"Configuration error: {e}")
    
    # ── Startup Checks ────────────────────────────────────────────
    console.info("Running startup checks...")
    
    result = run_startup_checks(settings)
    _startup_check_result = result
    
    # Log formatted results
    from rich.console import Console as RichConsole
    rich_console = RichConsole(stderr=True)
    rich_console.print()
    for line in format_check_result(result).split("\n"):
        rich_console.print(f"  {line}")
    rich_console.print()
    
    if not result.ok:
        # Fatal errors - refuse to start
        error_msg = f"Startup checks failed: {'; '.join(result.errors)}"
        console.error(error_msg)
        raise RuntimeError(error_msg)
    
    if result.warnings:
        for warning in result.warnings:
            console.warning("Startup warning: %s", warning)
    
    console.info("Startup checks passed. Server initializing...")
    
    # Register signal handlers for graceful shutdown (skip in tests)
    if os.environ.get("VERIFILY_SKIP_SIGNALS") != "1":
        try:
            signal.signal(signal.SIGTERM, _shutdown_handler)
            signal.signal(signal.SIGINT, _shutdown_handler)
        except ValueError:
            # Signal only works in main thread (e.g., in tests)
            pass
    
    yield {"startup_checks": result}
    
    # ── Shutdown ───────────────────────────────────────────────────
    console.info("Verifily shutting down gracefully...")
    _server_shutting_down = True
    
    shutdown_errors = []
    
    # Stop JobsStore worker thread
    try:
        jobs_store.stop_worker(timeout=5.0)
        console.info("  ✓ JobsStore stopped")
    except Exception as e:
        shutdown_errors.append(f"JobsStore: {e}")
        console.error("  ✗ JobsStore stop failed: %s", e)
    
    # Stop MonitorStore threads
    try:
        monitor_store.reset()  # This stops all monitors and joins threads
        console.info("  ✓ MonitorStore stopped")
    except Exception as e:
        shutdown_errors.append(f"MonitorStore: {e}")
        console.error("  ✗ MonitorStore stop failed: %s", e)
    
    # UsageStore persistence is flushed per-event, but we verify it's healthy
    try:
        # Force a test write to ensure all buffers are flushed
        usage_store.record(
            api_key_id="system",
            project_id="shutdown",
            elapsed_ms=0,
        )
        console.info("  ✓ UsageStore healthy")
    except Exception as e:
        shutdown_errors.append(f"UsageStore: {e}")
        console.error("  ✗ UsageStore check failed: %s", e)
    
    console.info("Graceful shutdown complete.")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and return the FastAPI application."""
    if settings is None:
        settings = load_settings()

    docs_url = "/docs" if settings.enable_docs else None
    openapi_url = "/openapi.json" if settings.enable_docs else None

    app = FastAPI(
        title="Verifily API",
        description="Local-only API for Verifily pipeline, contamination, and report.",
        version=__version__,
        docs_url=docs_url,
        openapi_url=openapi_url,
        redoc_url=None,
        lifespan=_lifespan_context,
    )

    app.state.settings = settings

    # ── Normalized error envelope (always-on) ────────────────────
    from verifily_cli_v1.core.api.jobs import BudgetExceededError
    
    def budget_exception_handler(request: Request, exc: BudgetExceededError) -> Any:
        return JSONResponse(
            status_code=402,
            content={
                "error": {
                    "type": "BUDGET_EXCEEDED",
                    "message": str(exc),
                    "code": "budget_limit_reached",
                    "remaining_daily": exc.budget_result.remaining_daily_units,
                    "remaining_monthly": exc.budget_result.remaining_monthly_units,
                    "reset_time": exc.budget_result.reset_time_utc,
                }
            },
            headers={
                "Retry-After": str(3600),  # 1 hour default
                "X-Budget-Remaining-Daily": str(exc.budget_result.remaining_daily_units),
                "X-Budget-Remaining-Monthly": str(exc.budget_result.remaining_monthly_units),
            },
        )
    
    app.add_exception_handler(BudgetExceededError, budget_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ── Middleware ────────────────────────────────────────────────
    # Starlette processes in reverse add order (last added = outermost).
    # Desired order: RequestID → Auth → RateLimit → BillingEnforce → Budget → handler
    app.add_middleware(BudgetMiddleware)
    if settings.billing_enforce and settings.enable_billing:
        from verifily_cli_v1.core.api.billing_enforce import BillingEnforceMiddleware
        app.add_middleware(BillingEnforceMiddleware, default_plan=settings.default_plan)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Reset globals for test isolation ─────────────────────────
    global _startup_check_result, _server_shutting_down, _deploy_config_valid, _deploy_config_errors
    _startup_check_result = None
    _server_shutting_down = False
    _deploy_config_valid = True
    _deploy_config_errors = []

    # ── Reset singletons for test isolation ──────────────────────
    metrics.reset()
    usage_store.reset()
    jobs_store.reset()
    monitor_store.reset()
    org_store.reset()
    auth_registry.reset()
    quota_store.reset()
    billing_store.reset()
    subscriptions_store.reset()
    teams_store.reset()
    audit_store.reset()
    workspaces_store.reset()

    # ── Configure auth registry (advanced mode) ──────────────────
    if settings.auth_mode == "advanced":
        auth_registry.configure_salt(settings.key_salt)
        if settings.auth_persist:
            auth_persist_path = str(Path(settings.data_dir) / "verifily_auth_events.jsonl")
            auth_registry.configure_persistence(auth_persist_path)
        quota_store.configure_limits(
            requests_per_day=settings.quota_req_per_day,
            rows_per_day=settings.quota_rows_per_day,
            bytes_per_day=settings.quota_bytes_per_day,
        )

    # ── Opt-in billing persistence ────────────────────────────────
    if settings.enable_billing and settings.billing_persist:
        billing_persist_path = str(Path(settings.data_dir) / "verifily_billing_events.jsonl")
        billing_store.configure_persistence(billing_persist_path)

    # ── Opt-in subscriptions persistence ──────────────────────────
    if settings.subs_persist:
        subs_persist_path = os.environ.get(
            "VERIFILY_SUBS_LOG_PATH",
            str(Path(settings.data_dir) / "verifily_subscriptions.jsonl"),
        )
        subscriptions_store.configure_persistence(subs_persist_path)

    # ── Opt-in teams persistence ──────────────────────────────────
    if settings.teams_enabled:
        teams_store.configure_salt(settings.key_salt)
        if settings.teams_persist:
            teams_persist_path = str(Path(settings.data_dir) / "verifily_teams_events.jsonl")
            teams_store.configure_persistence(teams_persist_path)

    # ── Opt-in workspaces store ──────────────────────────────────
    if settings.workspaces_enabled:
        ws_path = settings.workspaces_store_path or str(
            Path(tempfile.gettempdir()) / "verifily_workspaces_store.json"
        )
        workspaces_store.configure(path=ws_path, salt=settings.key_salt)

    # ── Opt-in usage persistence ─────────────────────────────────
    if settings.usage_persist:
        default_path = os.environ.get(
            "VERIFILY_USAGE_LOG_PATH",
            str(Path(settings.data_dir) / "verifily_usage_events.jsonl"),
        )
        usage_store.configure_persistence(default_path)

    # ── Routes ───────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "time": datetime.datetime.utcnow().isoformat() + "Z",
            "mode": "local",
        }

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> Any:
        """Readiness probe endpoint.
        
        Returns ready=True only when:
        - Startup checks passed
        - Deployment config is valid
        - Runtime paths are writable
        - Worker threads are alive (JobsStore)
        - Persistence initialized correctly
        - Server is not shutting down
        """
        global _startup_check_result, _server_shutting_down, _deploy_config_valid, _deploy_config_errors
        
        checks: Dict[str, Any] = {}
        errors: List[str] = []

        # 1. Check server is not shutting down
        if _server_shutting_down:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "checks": {"shutdown": True},
                    "error": "Server is shutting down",
                },
            )

        # 2. Check deployment configuration
        if not _deploy_config_valid:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "checks": {"deploy_config": "invalid"},
                    "error": f"Deployment configuration invalid: {'; '.join(_deploy_config_errors)}",
                },
            )
        checks["deploy_config"] = "ok"

        # 3. Check startup checks passed (skip if in test mode with no startup result)
        # In production, startup checks run via lifespan context before accepting requests.
        # In tests, we may not have a startup result, so we run a quick check here.
        if _startup_check_result is None:
            # Test mode: run quick startup check
            from verifily_cli_v1.core.api.startup_checks import run_startup_checks
            _startup_check_result = run_startup_checks(settings)
        
        if not _startup_check_result.ok:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "checks": {"startup": "failed"},
                    "error": f"Startup checks failed: {'; '.join(_startup_check_result.errors)}",
                },
            )
        
        checks["startup"] = "ok"

        # 3. Check Python version >= 3.9
        if sys.version_info >= (3, 9):
            checks["python"] = "ok"
        else:
            checks["python"] = "fail"
            errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.9")

        # 4. Check temp dir write access
        try:
            tmp = Path(tempfile.gettempdir()) / "verifily_ready_probe"
            tmp.write_text("ok")
            tmp.unlink()
            checks["temp_write"] = "ok"
        except Exception as e:
            checks["temp_write"] = "fail"
            errors.append(f"Cannot write to temp dir: {e}")

        # 5. Check critical imports
        try:
            from verifily_cli_v1.commands import pipeline as _p  # noqa: F401
            from verifily_cli_v1.commands import contamination as _c  # noqa: F401
            from verifily_cli_v1.commands import report as _r  # noqa: F401
            checks["imports"] = "ok"
        except ImportError as e:
            checks["imports"] = "fail"
            errors.append(f"Import error: {e}")

        # 6. Check subsystem stores are initialized and healthy
        checks["jobs_store"] = "ok" if jobs_store is not None else "unavailable"
        checks["usage_store"] = "ok" if usage_store is not None else "unavailable"
        checks["monitor_store"] = "ok" if monitor_store is not None else "unavailable"

        # 7. Check JobsStore worker thread is alive
        if jobs_store._worker is not None:
            if jobs_store._worker.is_alive():
                checks["jobs_worker"] = "alive"
            else:
                checks["jobs_worker"] = "dead"
                errors.append("JobsStore worker thread is not running")
        else:
            checks["jobs_worker"] = "not_started"
            # Worker not started yet is ok during startup

        # 8. Check persistence initialization if enabled
        if settings.usage_persist:
            checks["usage_persistence"] = (
                "configured" if usage_store._persist_path else "not_configured"
            )
        else:
            checks["usage_persistence"] = "disabled"

        if settings.jobs_persist:
            checks["jobs_persistence"] = (
                "configured" if jobs_store._persist_path else "not_configured"
            )
        else:
            checks["jobs_persistence"] = "disabled"

        if errors:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "checks": checks,
                    "error": "; ".join(errors),
                },
            )

        return {"status": "ready", "checks": checks}

    @app.get("/metrics", response_class=PlainTextResponse)
    def get_metrics() -> str:
        return metrics.format_metrics()

    # ── Enterprise permission helper ─────────────────────────────
    def _check_enterprise_permission(request: Request, perm_name: str) -> None:
        """If enterprise token is present, check permission. Otherwise no-op."""
        user_ctx = getattr(request.state, "user_ctx", None)
        if user_ctx is None:
            return
        from verifily_cli_v1.core.security.rbac import Permission, require_permission
        require_permission(user_ctx, Permission(perm_name))

    def _int_env_or_none(key: str) -> Optional[int]:
        val = os.environ.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    def _float_env_or_none(key: str) -> Optional[float]:
        val = os.environ.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    @app.post("/v1/pipeline", response_model=PipelineResponse)
    def pipeline(req: PipelineRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "run_pipeline")
        if not req.config_path and not req.project_path:
            raise HTTPException(
                status_code=422,
                detail="Either config_path or project_path must be provided.",
            )
        request_id = getattr(request.state, "request_id", None)
        api_key_id = getattr(request.state, "api_key_id", "anonymous")
        # Resolve project_id: header > body > "default"
        project_id = (
            getattr(request.state, "project_id", None)
            or req.project_id
            or "default"
        )
        try:
            result = run_pipeline_api(
                config_path=req.config_path,
                project_path=req.project_path,
                plan=req.plan,
                ci=req.ci,
                overrides=req.overrides,
                request_id=request_id,
                project_id=project_id,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Enterprise policy enforcement
        if settings.enterprise_security:
            from verifily_cli_v1.core.security.policies import PolicyConfig, evaluate_policies
            policy_cfg = PolicyConfig(
                require_contamination_pass=_bool_env("VERIFILY_POLICY_REQUIRE_CONTAM_PASS", False),
                require_reproducibility=_bool_env("VERIFILY_POLICY_REQUIRE_REPRO", False),
                block_if_pii_hits=_int_env_or_none("VERIFILY_POLICY_MAX_PII_HITS"),
                min_f1_threshold=_float_env_or_none("VERIFILY_POLICY_MIN_F1"),
            )
            policy_result = evaluate_policies(policy_cfg, result)
            if not policy_result.allowed:
                if "decision" not in result:
                    result["decision"] = {}
                result["decision"]["recommendation"] = "DONT_SHIP"
                result["decision"]["exit_code"] = 1
                result["policy_violations"] = policy_result.violations

        # Track decision counter
        decision_label = result.get("decision", {}).get("recommendation")
        if decision_label:
            metrics.inc_decision(decision_label)

        # Record usage
        usage_data = result.get("usage", {})
        billable = usage_data.get("billable_units", {})
        timing = usage_data.get("timing_ms", {})
        usage_store.record(
            api_key_id=api_key_id,
            project_id=project_id,
            elapsed_ms=timing.get("total", result.get("elapsed_ms", 0)),
            decision=decision_label,
            rows_in=billable.get("rows_in", 0),
            rows_out=billable.get("rows_out", 0),
            bytes_in=billable.get("bytes_in", 0),
            bytes_out=billable.get("bytes_out", 0),
        )

        if settings.enable_billing:
            billing_store.record_event(
                api_key_id=api_key_id,
                project_id=project_id,
                endpoint="/v1/pipeline",
                units={
                    "rows_in": billable.get("rows_in", 0),
                    "rows_out": billable.get("rows_out", 0),
                    "bytes_in": billable.get("bytes_in", 0),
                    "bytes_out": billable.get("bytes_out", 0),
                    "decisions": 1 if decision_label else 0,
                },
                request_id=request_id,
            )

        # Enterprise audit hook
        if settings.enterprise_security:
            from verifily_cli_v1.core.audit import AuditEvent
            audit_store.record(AuditEvent(
                step="PIPELINE", status="OK" if result.get("exit_code", 4) == 0 else "WARN",
                run_id=request_id or "unknown", request_id=request_id,
                project=project_id, summary={"decision": decision_label},
            ))

        return result

    @app.post("/v1/contamination", response_model=ContaminationResponse)
    def contamination_check(req: ContaminationRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "run_contamination")
        api_key_id = getattr(request.state, "api_key_id", "anonymous")
        project_id = (
            getattr(request.state, "project_id", None)
            or req.project_id
            or "default"
        )
        out_path = None if req.no_write else req.out_path
        try:
            result = run_contamination_api(
                train_path=req.train_path,
                eval_path=req.eval_path,
                jaccard_cutoff=req.jaccard_cutoff,
                out_path=out_path,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Track contamination counter
        contam_status = result.get("status")
        if contam_status:
            metrics.inc_contamination(contam_status)

        # Record usage
        usage_store.record(
            api_key_id=api_key_id,
            project_id=project_id,
            elapsed_ms=result.get("elapsed_ms", 0),
            rows_in=result.get("train_rows", 0) + result.get("eval_rows", 0),
        )

        if settings.enable_billing:
            billing_store.record_event(
                api_key_id=api_key_id,
                project_id=project_id,
                endpoint="/v1/contamination",
                units={
                    "rows_in": result.get("train_rows", 0) + result.get("eval_rows", 0),
                },
                request_id=getattr(request.state, "request_id", None),
            )

        if settings.enterprise_security:
            from verifily_cli_v1.core.audit import AuditEvent
            audit_store.record(AuditEvent(
                step="CONTAMINATION", status="OK",
                run_id=getattr(request.state, "request_id", None) or "unknown",
                request_id=getattr(request.state, "request_id", None),
                project=project_id, summary={"status": result.get("status")},
            ))

        return result

    @app.post("/v1/report", response_model=ReportResponse)
    def dataset_report(req: ReportRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "run_report")
        api_key_id = getattr(request.state, "api_key_id", "anonymous")
        project_id = (
            getattr(request.state, "project_id", None)
            or req.project_id
            or "default"
        )
        try:
            result = run_report_api(
                dataset_path=req.dataset_path,
                schema=req.schema_type,
                sample=req.sample,
                out_dir=req.out_dir,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Record usage
        usage_store.record(
            api_key_id=api_key_id,
            project_id=project_id,
            elapsed_ms=result.get("elapsed_ms", 0),
            rows_in=result.get("row_count", 0),
        )

        if settings.enable_billing:
            billing_store.record_event(
                api_key_id=api_key_id,
                project_id=project_id,
                endpoint="/v1/report",
                units={
                    "rows_in": result.get("row_count", 0),
                },
                request_id=getattr(request.state, "request_id", None),
            )

        if settings.enterprise_security:
            from verifily_cli_v1.core.audit import AuditEvent
            audit_store.record(AuditEvent(
                step="REPORT", status="OK",
                run_id=getattr(request.state, "request_id", None) or "unknown",
                request_id=getattr(request.state, "request_id", None),
                project=project_id, summary={"row_count": result.get("row_count")},
            ))

        return result

    @app.get("/v1/usage", response_model=UsageResponse)
    def get_usage(
        request: Request,
        window_minutes: int = Query(0, ge=0, description="Time window in minutes (0 = all time)."),
        group_by: str = Query("key_project", description="Grouping: key_project, key, project, or total."),
    ) -> Dict[str, Any]:
        _check_enterprise_permission(request, "view_usage")
        if group_by not in ("key_project", "key", "project", "total"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid group_by: {group_by}. Must be key_project, key, project, or total.",
            )
        return usage_store.query(window_minutes=window_minutes, group_by=group_by)

    # ── Async Jobs ────────────────────────────────────────────────

    # Opt-in jobs persistence
    if settings.jobs_persist:
        jobs_persist_path = os.environ.get(
            "VERIFILY_JOBS_LOG_PATH",
            str(Path(settings.data_dir) / "verifily_jobs_events.jsonl"),
        )
        jobs_store.configure_persistence(jobs_persist_path)

    # Register executors that the worker thread calls
    def _exec_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
        return run_pipeline_api(**payload)

    def _exec_contamination(payload: Dict[str, Any]) -> Dict[str, Any]:
        return run_contamination_api(**payload)

    def _exec_report(payload: Dict[str, Any]) -> Dict[str, Any]:
        return run_report_api(**payload)

    def _exec_classify(payload: Dict[str, Any]) -> Dict[str, Any]:
        result = run_classify_api(**payload)
        if settings.enable_billing:
            try:
                row_count = result.get("classification", {}).get("row_count", 0)
                billing_store.record_event(
                    api_key_id=payload.get("api_key_id", "anonymous"),
                    project_id=payload.get("project_id", "default"),
                    endpoint="/v1/jobs/classify",
                    units={"rows_in": row_count},
                )
            except Exception:
                pass  # billing recording is best-effort
        return result

    def _exec_retrain(payload: Dict[str, Any]) -> Dict[str, Any]:
        return run_retrain_api(payload)

    register_executor(JobType.PIPELINE, _exec_pipeline)
    register_executor(JobType.CONTAMINATION, _exec_contamination)
    register_executor(JobType.REPORT, _exec_report)
    register_executor(JobType.CLASSIFY, _exec_classify)
    register_executor(JobType.RETRAIN, _exec_retrain)

    # Start the worker thread
    jobs_store.start_worker()

    def _extract_job_ctx(request: Request, body_project_id: Optional[str] = None) -> Dict[str, Any]:
        """Extract request_id, api_key_id, project_id from the request."""
        return {
            "request_id": getattr(request.state, "request_id", None),
            "api_key_id": getattr(request.state, "api_key_id", "anonymous"),
            "project_id": (
                getattr(request.state, "project_id", None)
                or body_project_id
                or "default"
            ),
        }

    @app.post("/v1/jobs/pipeline", response_model=JobSubmitResponse)
    def submit_pipeline_job(req: PipelineRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "submit_jobs")
        if not req.config_path and not req.project_path:
            raise HTTPException(
                status_code=422,
                detail="Either config_path or project_path must be provided.",
            )
        ctx = _extract_job_ctx(request, req.project_id)
        payload: Dict[str, Any] = {
            "config_path": req.config_path,
            "project_path": req.project_path,
            "plan": req.plan,
            "ci": req.ci,
            "overrides": req.overrides,
            "request_id": ctx["request_id"],
            "project_id": ctx["project_id"],
        }
        job_id = jobs_store.submit(JobType.PIPELINE, payload, **ctx)
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "request_id": ctx["request_id"],
            "project_id": ctx["project_id"],
        }

    @app.post("/v1/jobs/contamination", response_model=JobSubmitResponse)
    def submit_contamination_job(req: ContaminationRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "submit_jobs")
        ctx = _extract_job_ctx(request, req.project_id)
        out_path = None if req.no_write else req.out_path
        payload: Dict[str, Any] = {
            "train_path": req.train_path,
            "eval_path": req.eval_path,
            "jaccard_cutoff": req.jaccard_cutoff,
            "out_path": out_path,
        }
        job_id = jobs_store.submit(JobType.CONTAMINATION, payload, **ctx)
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "request_id": ctx["request_id"],
            "project_id": ctx["project_id"],
        }

    @app.post("/v1/jobs/report", response_model=JobSubmitResponse)
    def submit_report_job(req: ReportRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "submit_jobs")
        ctx = _extract_job_ctx(request, req.project_id)
        payload: Dict[str, Any] = {
            "dataset_path": req.dataset_path,
            "schema": req.schema_type,
            "sample": req.sample,
            "out_dir": req.out_dir,
        }
        job_id = jobs_store.submit(JobType.REPORT, payload, **ctx)
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "request_id": ctx["request_id"],
            "project_id": ctx["project_id"],
        }

    @app.post("/v1/jobs/classify", response_model=JobSubmitResponse)
    def submit_classify_job(req: ClassifyRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "submit_jobs")
        if not req.dataset_path:
            raise HTTPException(
                status_code=422,
                detail="dataset_path must be provided.",
            )
        ctx = _extract_job_ctx(request, req.project_id)
        payload: Dict[str, Any] = {
            "dataset_path": req.dataset_path,
            "output_dir": req.output_dir,
            "max_rows_scan": req.max_rows_scan,
            "export_buckets": req.export_buckets,
            "min_bucket_rows": req.min_bucket_rows,
        }
        job_id = jobs_store.submit(JobType.CLASSIFY, payload, **ctx)
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "request_id": ctx["request_id"],
            "project_id": ctx["project_id"],
        }

    @app.post("/v1/retrain", response_model=RetrainResponse)
    def retrain_sync(req: RetrainRequest, request: Request) -> Dict[str, Any]:
        ctx = _extract_job_ctx(request, req.project_id)
        payload = {
            "project_id": ctx["project_id"],
            "dataset_dir": req.dataset_dir,
            "base_run_dir": req.base_run_dir,
            "contaminated_run_dir": req.contaminated_run_dir,
            "metric": req.metric,
            "mode": req.mode,
            "output_dir": req.output_dir,
            "seed": req.seed,
            "notes": req.notes,
            "request_id": ctx["request_id"],
        }
        try:
            result = run_retrain_api(payload)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Record usage
        usage_store.record(
            api_key_id=ctx["api_key_id"],
            project_id=ctx["project_id"],
            elapsed_ms=result.get("elapsed_ms", 0),
            decision=result.get("decision", {}).get("recommendation"),
        )

        if settings.enable_billing:
            billing_store.record_event(
                api_key_id=ctx["api_key_id"],
                project_id=ctx["project_id"],
                endpoint="/v1/retrain",
                units={
                    "decisions": 1 if result.get("decision", {}).get("recommendation") else 0,
                },
                request_id=ctx["request_id"],
            )

        return result

    @app.post("/v1/jobs/retrain", response_model=JobSubmitResponse)
    def submit_retrain_job(req: RetrainRequest, request: Request) -> Dict[str, Any]:
        _check_enterprise_permission(request, "submit_jobs")
        ctx = _extract_job_ctx(request, req.project_id)
        payload = {
            "project_id": ctx["project_id"],
            "dataset_dir": req.dataset_dir,
            "base_run_dir": req.base_run_dir,
            "contaminated_run_dir": req.contaminated_run_dir,
            "metric": req.metric,
            "mode": req.mode,
            "output_dir": req.output_dir,
            "seed": req.seed,
            "notes": req.notes,
            "request_id": ctx["request_id"],
        }
        job_id = jobs_store.submit(JobType.RETRAIN, payload, **ctx)
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "request_id": ctx["request_id"],
            "project_id": ctx["project_id"],
        }

    @app.get("/v1/jobs/{job_id}", response_model=JobMetaResponse)
    def get_job(job_id: str) -> Dict[str, Any]:
        job = jobs_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        return job.to_meta()

    @app.get("/v1/jobs/{job_id}/result")
    def get_job_result(job_id: str) -> Dict[str, Any]:
        job = jobs_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        if job.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=job.error or "Job failed.")
        if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} is still {job.status.value}. Try again later.",
            )
        result = jobs_store.result(job_id)
        if result is None:
            raise HTTPException(status_code=500, detail="No result available.")
        result["job_id"] = job_id
        return result

    @app.get("/v1/jobs", response_model=JobListResponse)
    def list_jobs(
        status: Optional[str] = Query(None, description="Filter by status: QUEUED, RUNNING, SUCCEEDED, FAILED."),
        project_id: Optional[str] = Query(None, description="Filter by project_id."),
        limit: int = Query(50, ge=1, le=500, description="Max results."),
    ) -> Dict[str, Any]:
        if status and status not in ("QUEUED", "RUNNING", "SUCCEEDED", "FAILED"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status filter: {status}.",
            )
        items = jobs_store.list_jobs(status=status, project_id=project_id, limit=limit)
        return {"jobs": items}

    # ── Monitor Endpoints ────────────────────────────────────────

    @app.post("/v1/monitor/start", response_model=MonitorStartResponse)
    def start_monitor(req: MonitorStartRequest, request: Request) -> Dict[str, Any]:
        ctx = _extract_job_ctx(request, req.project_id)
        import uuid as _uuid
        mid = _uuid.uuid4().hex[:12]
        config = MonitorConfig(
            monitor_id=mid,
            project_id=ctx["project_id"],
            config_path=req.config_path,
            interval_seconds=req.interval_seconds,
            max_ticks=req.max_ticks,
            rolling_window=req.rolling_window,
        )
        try:
            monitor_store.start(config)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"monitor_id": mid, "status": "running"}

    @app.post("/v1/monitor/stop")
    def stop_monitor(request: Request, monitor_id: str = Query(..., description="Monitor ID to stop.")) -> Dict[str, Any]:
        try:
            monitor_store.stop(monitor_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"monitor_id": monitor_id, "status": "stopped"}

    @app.get("/v1/monitor/status", response_model=MonitorStatusResponse)
    def get_monitor_status(monitor_id: str = Query(..., description="Monitor ID.")) -> Dict[str, Any]:
        try:
            return monitor_store.status(monitor_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/v1/monitor/history", response_model=MonitorHistoryResponse)
    def get_monitor_history(
        monitor_id: str = Query(..., description="Monitor ID."),
        last_n: int = Query(0, ge=0, description="Return last N ticks (0=all)."),
    ) -> Dict[str, Any]:
        try:
            ticks = monitor_store.get_history(monitor_id, last_n=last_n)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {
            "monitor_id": monitor_id,
            "ticks": [t.to_dict() for t in ticks],
            "total_ticks": len(ticks),
        }

    # ── Orgs / Projects ────────────────────────────────────────

    _org_mode = os.environ.get("VERIFILY_ORG_MODE", "0") == "1"
    if not settings.workspaces_enabled and not _org_mode:
        @app.post("/v1/orgs", response_model=OrgResponse, status_code=201)
        def create_org(req: CreateOrgRequest, request: Request) -> Dict[str, Any]:
            api_key_id = getattr(request.state, "api_key_id", "anonymous")
            org = org_store.create_org(name=req.name, api_key_id=api_key_id)
            return org.to_dict()

        @app.get("/v1/orgs", response_model=OrgListResponse)
        def list_orgs(request: Request) -> Dict[str, Any]:
            api_key_id = getattr(request.state, "api_key_id", "anonymous")
            orgs = org_store.list_orgs(api_key_id=api_key_id)
            return {"orgs": [o.to_dict() for o in orgs]}

        @app.post("/v1/projects", response_model=ProjectResponse, status_code=201)
        def create_project(req: CreateProjectRequest, request: Request) -> Dict[str, Any]:
            api_key_id = getattr(request.state, "api_key_id", "anonymous")
            role = org_store.check_access(req.org_id, api_key_id)
            if role is None:
                raise HTTPException(status_code=403, detail="Not a member of this organization.")
            try:
                project = org_store.create_project(org_id=req.org_id, name=req.name, api_key_id=api_key_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return project.to_dict()

        @app.get("/v1/projects", response_model=ProjectListResponse)
        def list_projects(
            org_id: Optional[str] = Query(None, description="Filter by organization ID."),
        ) -> Dict[str, Any]:
            projects = org_store.list_projects(org_id=org_id)
            return {"projects": [p.to_dict() for p in projects]}

        @app.post("/v1/orgs/{org_id}/memberships", response_model=MembershipResponse, status_code=201)
        def add_membership(org_id: str, req: AddMembershipRequest, request: Request) -> Dict[str, Any]:
            caller_key_id = getattr(request.state, "api_key_id", "anonymous")
            caller_role = org_store.check_access(org_id, caller_key_id)
            if caller_role not in (OrgsRole.OWNER, OrgsRole.ADMIN):
                raise HTTPException(status_code=403, detail="Only OWNER or ADMIN can add members.")
            try:
                role = OrgsRole(req.role)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid role: {req.role}. Must be OWNER, ADMIN, or MEMBER.")
            try:
                membership = org_store.add_membership(org_id=org_id, api_key_id=req.api_key_id, role=role)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return membership.to_dict()

        @app.get("/v1/orgs/{org_id}/memberships", response_model=MembershipListResponse)
        def list_memberships(org_id: str) -> Dict[str, Any]:
            memberships = org_store.list_memberships(org_id)
            return {"memberships": [m.to_dict() for m in memberships]}

    # ── Workspaces Endpoints ──────────────────────────────────

    if settings.workspaces_enabled:

        @app.post("/v1/orgs", status_code=201)
        def ws_create_org(req: WsCreateOrgRequest) -> Dict[str, Any]:
            """Create a new organization (bootstrap or admin)."""
            return workspaces_store.create_org(name=req.name)

        @app.post("/v1/projects", status_code=201)
        def ws_create_project(req: WsCreateProjectRequest) -> Dict[str, Any]:
            """Create a project within an org (admin only)."""
            try:
                return workspaces_store.create_project(
                    org_id=req.org_id,
                    name=req.name,
                    billing_plan=req.billing_plan or "free",
                )
            except OrgNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.post("/v1/keys", status_code=201)
        def ws_create_key(req: WsCreateKeyRequest) -> Dict[str, Any]:
            """Create an API key for a project (admin only)."""
            try:
                return workspaces_store.create_api_key(
                    project_id=req.project_id,
                    role=req.role,
                )
            except ProjectNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except InvalidRoleError as e:
                raise HTTPException(status_code=422, detail=str(e))

        @app.post("/v1/keys/revoke")
        def ws_revoke_key(req: WsRevokeKeyRequest) -> Dict[str, Any]:
            """Revoke an API key (admin only)."""
            try:
                return workspaces_store.revoke_api_key(
                    project_id=req.project_id,
                    api_key_id=req.api_key_id,
                )
            except (ProjectNotFoundError, KeyNotFoundError) as e:
                raise HTTPException(status_code=404, detail=str(e))

        @app.get("/v1/me")
        def ws_me(request: Request) -> Dict[str, Any]:
            """Return identity of the calling key."""
            ws_role = ROLE_REVERSE.get(
                getattr(request.state, "role", None), "unknown"
            )
            return {
                "org_id": getattr(request.state, "org_id", ""),
                "project_id": getattr(request.state, "project_id", ""),
                "role": ws_role,
                "api_key_id": getattr(request.state, "api_key_id", ""),
            }

    # ── Effective Config ──────────────────────────────────────

    @app.get("/v1/config/effective", response_model=EffectiveConfigResponse)
    def effective_config() -> Dict[str, Any]:
        """Return current effective configuration (redacted, no secrets)."""
        from verifily_cli_v1.core.secrets import redact_dict

        base_config = {
            "privacy_mode": os.environ.get("VERIFILY_PRIVACY_MODE", "local"),
            "log_format": os.environ.get("VERIFILY_LOG_FORMAT", "text"),
            "rate_limit_rpm": os.environ.get("VERIFILY_RATE_LIMIT_RPM"),
            "usage_persist": os.environ.get("VERIFILY_USAGE_PERSIST", "0"),
            "jobs_persist": os.environ.get("VERIFILY_JOBS_PERSIST", "0"),
            "auth_enabled": bool(os.environ.get("VERIFILY_API_KEY")),
        }
        return {"config": redact_dict(base_config)}

    # ── Production Config Endpoint ─────────────────────────────

    @app.get("/v1/config")
    def get_config(request: Request) -> Dict[str, Any]:
        """Return production configuration (safe, no secrets).
        
        This endpoint is useful for debugging deployments and verifying
        that configuration is loaded correctly.
        """
        try:
            settings = load_production_settings()
            config = settings.as_safe_dict()
        except ValueError as e:
            # Settings not valid - return partial info
            config = {"error": str(e), "partial": True}
        
        return {
            "config": config,
            "request_id": getattr(request.state, "request_id", None),
        }

    # ── Lineage Endpoint ───────────────────────────────────────

    @app.get("/v1/lineage", response_model=LineageResponse)
    def get_lineage(
        run_dir: str = Query(..., description="Path to run directory."),
        request: Request = None,
    ) -> Dict[str, Any]:
        """Get lineage graph for a run.

        Returns the complete data lineage: raw data → transforms → dataset →
        contamination check → train → evaluation → decision.
        """
        from verifily_cli_v1.core.lineage_graph import build_lineage_graph

        run_path = Path(run_dir)
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run directory not found: {run_dir}")

        try:
            graph = build_lineage_graph(run_path)
            return graph.to_dict()
        except Exception as e:
            logger.warning("Lineage build failed for %s: %s", run_dir, e)
            raise HTTPException(status_code=500, detail=f"Failed to build lineage graph: {e}")

    # ── Score Endpoint ─────────────────────────────────────────

    @app.get("/v1/score", response_model=Dict[str, Any])
    def get_score(
        run_dir: str = Query(..., description="Path to run directory."),
        request: Request = None,
    ) -> Dict[str, Any]:
        """Get Risk Score and Health Index for a run.

        Computes quantitative assessments:
        - Risk Score (0-100, higher=riskier): Dataset safety
        - Health Index (0-100, higher=healthier): Model readiness
        """
        from verifily_cli_v1.core.scoring import (
            compute_dataset_risk,
            compute_model_health,
            compute_verdict,
        )
        from verifily_cli_v1.core.io import read_json

        run_path = Path(run_dir)
        if not run_path.exists():
            raise HTTPException(status_code=404, detail=f"Run directory not found: {run_dir}")

        try:
            # Load artifacts
            report = _safe_read_json(run_path / "report.json")
            contamination = _safe_read_json(run_path / "contamination_results.json")
            contract = _safe_read_json(run_path / "contract.json")
            decision = _safe_read_json(run_path / "decision.json")
            eval_results = _safe_read_json(run_path / "eval" / "eval_results.json")

            # Compute scores
            risk_score = compute_dataset_risk(
                report_result=report,
                contamination_result=contamination,
                contract_result=contract,
            )
            health_index = compute_model_health(
                decision_result=decision,
                eval_results=eval_results,
                reproducibility_ok=contract.get("valid", False) if contract else False,
            )
            verdict, recommendations = compute_verdict(risk_score, health_index)

            return {
                "risk_score": risk_score.to_dict(),
                "health_index": health_index.to_dict(),
                "verdict": verdict,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.warning("Score computation failed for %s: %s", run_dir, e)
            raise HTTPException(status_code=500, detail=f"Failed to compute scores: {e}")

    def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
        """Safely read JSON file, return None if missing/invalid."""
        if not path.exists():
            return None
        try:
            return read_json(path)
        except Exception:
            return None

    # ── Registry Endpoints ────────────────────────────────────

    # Configure registry persistence
    if settings.jobs_persist or settings.usage_persist:
        from verifily_cli_v1.core.registry import configure_registry_persistence
        registry_path = Path(settings.data_dir) / "registry.jsonl"
        configure_registry_persistence(registry_path)

    @app.post("/v1/registry/register", response_model=RegisterModelResponse)
    def register_model(
        req: RegisterModelRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Register a model from a run directory."""
        from verifily_cli_v1.core.registry import (
            RegistrationError,
            RegistrationRequest,
            registry_store,
        )

        ctx = _extract_job_ctx(request)

        try:
            reg_request = RegistrationRequest(
                run_dir=req.run_dir,
                model_id=req.model_id,
                version=req.version,
                registered_by=ctx["api_key_id"],
            )
            record = registry_store.register(reg_request)

            return {
                "success": True,
                "record": record.to_dict(),
            }
        except RegistrationError as e:
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.warning("Registry registration failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

    @app.post("/v1/registry/promote", response_model=PromoteModelResponse)
    def promote_model(
        req: PromoteModelRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Promote a model to a new stage."""
        from verifily_cli_v1.core.registry import (
            PromotionError,
            PromotionRequest,
            registry_store,
        )

        ctx = _extract_job_ctx(request)

        try:
            promo_request = PromotionRequest(
                model_id=req.model_id,
                version=req.version,
                target_stage=req.target_stage,
                promoted_by=ctx["api_key_id"],
                reason=req.reason,
            )
            record = registry_store.promote(promo_request)

            return {
                "success": True,
                "record": record.to_dict(),
            }
        except PromotionError as e:
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.warning("Registry promotion failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")

    @app.get("/v1/registry/models", response_model=ListModelsResponse)
    def list_models(
        stage: Optional[str] = Query(None, description="Filter by stage."),
        model_id: Optional[str] = Query(None, description="Filter by model ID."),
        request: Request = None,
    ) -> Dict[str, Any]:
        """List registered models."""
        from verifily_cli_v1.core.registry import registry_store

        try:
            records = registry_store.list(stage=stage, model_id=model_id)
            return {
                "models": [r.to_dict() for r in records],
                "total": len(records),
            }
        except Exception as e:
            logger.warning("Registry list failed: %s", e)
            raise HTTPException(status_code=500, detail=f"List failed: {e}")

    @app.get("/v1/registry/history/{model_id}", response_model=ModelHistoryResponse)
    def get_model_history(
        model_id: str,
        request: Request = None,
    ) -> Dict[str, Any]:
        """Get history of a model (all versions)."""
        from verifily_cli_v1.core.registry import registry_store

        try:
            records = registry_store.history(model_id)
            return {
                "model_id": model_id,
                "versions": [r.to_dict() for r in records],
            }
        except Exception as e:
            logger.warning("Registry history failed for %s: %s", model_id, e)
            raise HTTPException(status_code=500, detail=f"History failed: {e}")

    # ── Admin Endpoints (Enterprise Trust) ────────────────────

    if settings.enable_admin:

        @app.post("/v1/admin/projects", response_model=AdminProjectResponse, status_code=201)
        def admin_create_project(req: AdminCreateProjectRequest) -> Dict[str, Any]:
            rec = auth_registry.create_project(id=req.id, name=req.name)
            return rec.to_dict()

        @app.get("/v1/admin/projects", response_model=AdminProjectListResponse)
        def admin_list_projects() -> Dict[str, Any]:
            projects = auth_registry.list_projects()
            return {"projects": [p.to_dict() for p in projects]}

        @app.post("/v1/admin/keys", response_model=AdminKeyResponse, status_code=201)
        def admin_create_key(req: AdminCreateKeyRequest) -> Dict[str, Any]:
            invalid = set(req.scopes) - SCOPES
            if invalid:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid scopes: {sorted(invalid)}. Valid: {sorted(SCOPES)}",
                )
            rec = auth_registry.create_key(
                id=req.id,
                name=req.name,
                raw_key=req.raw_key,
                scopes=req.scopes,
                projects_allowed=req.projects_allowed,
            )
            return {
                "id": rec.id,
                "name": rec.name,
                "created_at": rec.created_at,
                "scopes": rec.scopes,
                "projects_allowed": rec.projects_allowed,
                "disabled": rec.disabled,
            }

        @app.get("/v1/admin/keys", response_model=AdminKeyListResponse)
        def admin_list_keys() -> Dict[str, Any]:
            keys = auth_registry.list_keys()
            return {"keys": keys}

        @app.post("/v1/admin/keys/{key_id}/disable", response_model=AdminKeyResponse)
        def admin_disable_key(key_id: str) -> Dict[str, Any]:
            try:
                rec = auth_registry.disable_key(key_id)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Key {key_id} not found.")
            return {
                "id": rec.id,
                "name": rec.name,
                "created_at": rec.created_at,
                "scopes": rec.scopes,
                "projects_allowed": rec.projects_allowed,
                "disabled": rec.disabled,
            }

        @app.post("/v1/admin/keys/{key_id}/rotate", response_model=AdminKeyResponse)
        def admin_rotate_key(key_id: str, req: AdminRotateKeyRequest) -> Dict[str, Any]:
            try:
                rec = auth_registry.rotate_key(key_id, req.raw_key)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Key {key_id} not found.")
            return {
                "id": rec.id,
                "name": rec.name,
                "created_at": rec.created_at,
                "scopes": rec.scopes,
                "projects_allowed": rec.projects_allowed,
                "disabled": rec.disabled,
            }

    # ── Teams Admin Endpoints ────────────────────────────────

    if settings.teams_enabled:
        from verifily_cli_v1.core.teams.scopes import TEAMS_SCOPES

        @app.post("/v1/admin/orgs", response_model=TeamsCreateOrgResponse, status_code=201)
        def teams_create_org(req: TeamsCreateOrgRequest) -> Dict[str, Any]:
            org = teams_store.create_org(id=req.id, name=req.name)
            return org.to_dict()

        @app.post("/v1/admin/users", response_model=TeamsCreateUserResponse, status_code=201)
        def teams_create_user(req: TeamsCreateUserRequest) -> Dict[str, Any]:
            user = teams_store.create_user(id=req.id, email=req.email, name=req.name)
            return user.to_dict()

        @app.post("/v1/admin/memberships", response_model=TeamsAddMembershipResponse, status_code=201)
        def teams_add_membership(req: TeamsAddMembershipRequest) -> Dict[str, Any]:
            try:
                mem = teams_store.add_membership(
                    user_id=req.user_id, org_id=req.org_id, role=req.role,
                )
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            return mem.to_dict()

        @app.post("/v1/admin/team-projects", response_model=TeamsCreateProjectResponse, status_code=201)
        def teams_create_project(req: TeamsCreateProjectRequest) -> Dict[str, Any]:
            proj = teams_store.create_project(id=req.id, org_id=req.org_id, name=req.name)
            return proj.to_dict()

        @app.get("/v1/admin/team-projects", response_model=TeamsProjectListResponse)
        def teams_list_projects() -> Dict[str, Any]:
            projects = teams_store.list_projects()
            return {"projects": [p.to_dict() for p in projects]}

        @app.post("/v1/admin/api-keys", response_model=TeamsIssueApiKeyResponse, status_code=201)
        def teams_issue_api_key(req: TeamsIssueApiKeyRequest) -> Dict[str, Any]:
            invalid = set(req.scopes) - TEAMS_SCOPES
            if invalid:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid scopes: {sorted(invalid)}. Valid: {sorted(TEAMS_SCOPES)}",
                )
            rec = teams_store.create_api_key(
                id=req.id, org_id=req.org_id, name=req.name,
                raw_key=req.raw_key, scopes=req.scopes,
                project_ids=req.project_ids, created_by=req.created_by,
            )
            return rec.to_dict()

        @app.get("/v1/admin/whoami", response_model=TeamsWhoamiResponse)
        def teams_whoami(request: Request) -> Dict[str, Any]:
            api_key_id = getattr(request.state, "api_key_id", None)
            org_id = getattr(request.state, "org_id", None)
            # If we got here through super admin, return admin info
            if api_key_id is None:
                return {
                    "api_key_id": "super-admin",
                    "org_id": "*",
                    "scopes": sorted(TEAMS_SCOPES),
                    "project_ids": ["*"],
                }
            # Look up the key record for full info
            with teams_store._lock:
                rec = teams_store._keys.get(api_key_id)
            if rec:
                return {
                    "api_key_id": rec.id,
                    "org_id": rec.org_id,
                    "scopes": rec.scopes,
                    "project_ids": rec.project_ids,
                }
            return {
                "api_key_id": api_key_id or "unknown",
                "org_id": org_id or "unknown",
                "scopes": [],
                "project_ids": [],
            }

    # ── Budget Endpoints ───────────────────────────────────────

    @app.get("/v1/budget/status", response_model=BudgetStatusResponse)
    def budget_status(
        project_id: str = Query("default", description="Project to check budget for."),
    ) -> Dict[str, Any]:
        """Get current budget status for a project."""
        from verifily_cli_v1.core.budget import budget_store
        
        status = budget_store.get_status(project_id, usage_store)
        return status

    # ── Org Mode Endpoints (Multi-tenant) ─────────────────────

    @app.post("/v1/orgs", response_model=CreateOrgResponse)
    def create_org(
        req: CreateOrgRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new organization (ADMIN only, or bootstrap if no orgs exist)."""
        import os
        
        # Allow bootstrap if no orgs exist or VERIFILY_ORG_BOOTSTRAP=1
        bootstrap_mode = os.environ.get("VERIFILY_ORG_BOOTSTRAP", "0") == "1"
        no_orgs = len(org_mode_store.list_orgs()) == 0
        
        if not (bootstrap_mode or no_orgs):
            # Require admin role
            request_role: Optional[Role] = getattr(request.state, "role", None)
            if request_role != Role.ADMIN:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": {
                            "type": "FORBIDDEN",
                            "message": "Admin role required to create organizations",
                            "code": "admin_required",
                        }
                    }
                )
        
        org = org_mode_store.create_org(name=req.name)
        return {
            "org_id": org.org_id,
            "name": org.name,
            "created_at": org.created_at,
        }

    @app.post("/v1/projects", response_model=CreateProjectResponse)
    def create_project(
        req: CreateProjectRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new project within an organization (ADMIN only)."""
        # Verify requesting key's org matches or is admin
        request_org_id: Optional[str] = getattr(request.state, "org_id", None)
        request_role: Optional[Role] = getattr(request.state, "role", None)
        
        # Check admin permission
        if request_role != Role.ADMIN:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "FORBIDDEN",
                        "message": "Admin role required",
                        "code": "admin_required",
                    }
                }
            )
        
        # Only ADMIN keys for the same org can create projects
        if request_org_id is not None and request_org_id != req.org_id:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "FORBIDDEN",
                        "message": "Can only create projects in your organization",
                        "code": "org_mismatch",
                    }
                }
            )
        
        project = org_mode_store.create_project(org_id=req.org_id, name=req.name)
        return {
            "project_id": project.project_id,
            "org_id": project.org_id,
            "name": project.name,
            "created_at": project.created_at,
        }

    @app.get("/v1/projects", response_model=ListProjectsResponse)
    def list_projects(
        request: Request,
        org_id: Optional[str] = Query(None, description="Filter by org_id."),
    ) -> Dict[str, Any]:
        """List projects (ADMIN, DEV, VIEWER)."""
        request_org_id: Optional[str] = getattr(request.state, "org_id", None)
        
        # Can only see projects in your org
        effective_org_id = org_id or request_org_id
        if request_org_id and effective_org_id != request_org_id:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "FORBIDDEN",
                        "message": "Can only view projects in your organization",
                        "code": "org_mismatch",
                    }
                }
            )
        
        projects = org_mode_store.list_projects(org_id=effective_org_id)
        return {
            "projects": [
                {
                    "project_id": p.project_id,
                    "org_id": p.org_id,
                    "name": p.name,
                    "created_at": p.created_at,
                }
                for p in projects
            ]
        }

    @app.post("/v1/keys", response_model=CreateKeyResponse)
    def create_key(
        req: CreateKeyRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new API key (ADMIN only)."""
        from verifily_cli_v1.core.api.rbac import require_role
        require_role(Role.ADMIN)(lambda: None)
        
        # Verify requesting key's org matches target project
        request_org_id: Optional[str] = getattr(request.state, "org_id", None)
        
        if request_org_id:
            try:
                project = org_mode_store.get_project(req.project_id)
                if project.org_id != request_org_id:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": {
                                "type": "FORBIDDEN",
                                "message": "Can only create keys for projects in your organization",
                                "code": "org_mismatch",
                            }
                        }
                    )
            except Exception:
                pass
        
        try:
            role = Role(req.role)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "type": "VALIDATION_ERROR",
                        "message": f"Invalid role: {req.role}. Must be admin, dev, or viewer.",
                        "code": "invalid_role",
                    }
                }
            )
        
        project = org_mode_store.get_project(req.project_id)
        secret, api_key = org_mode_store.create_key(
            org_id=project.org_id,
            project_id=req.project_id,
            role=role,
            label=req.label,
        )
        
        return {
            "secret": secret,
            "key_id": api_key.key_id,
            "org_id": api_key.org_id,
            "project_id": api_key.project_id,
            "role": api_key.role.value,
            "created_at": api_key.created_at,
            "label": api_key.label,
        }

    @app.post("/v1/keys/{key_id}/revoke", response_model=RevokeKeyResponse)
    def revoke_key(
        key_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Revoke an API key (ADMIN only)."""
        from verifily_cli_v1.core.api.rbac import require_role
        require_role(Role.ADMIN)(lambda: None)
        
        # Verify requesting key's org matches target key
        request_org_id: Optional[str] = getattr(request.state, "org_id", None)
        
        try:
            target_key = org_mode_store.get_key(key_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "type": "NOT_FOUND",
                        "message": f"Key {key_id} not found",
                        "code": "key_not_found",
                    }
                }
            )
        
        if request_org_id and target_key.org_id != request_org_id:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "FORBIDDEN",
                        "message": "Can only revoke keys in your organization",
                        "code": "org_mismatch",
                    }
                }
            )
        
        was_active = target_key.is_active
        revoked_key = org_mode_store.revoke_key(key_id)
        
        return {
            "key_id": revoked_key.key_id,
            "revoked_at": revoked_key.revoked_at or "",
            "was_active": was_active,
        }

    @app.get("/v1/keys", response_model=ListKeysResponse)
    def list_keys(
        request: Request,
        project_id: Optional[str] = Query(None, description="Filter by project_id."),
    ) -> Dict[str, Any]:
        """List API keys (ADMIN only)."""
        from verifily_cli_v1.core.api.rbac import require_role
        require_role(Role.ADMIN)(lambda: None)
        
        request_org_id: Optional[str] = getattr(request.state, "org_id", None)
        
        keys = org_mode_store.list_keys(org_id=request_org_id, project_id=project_id)
        return {
            "keys": [k.to_dict() for k in keys]
        }

    # ── Billing Endpoints ─────────────────────────────────────

    @app.get("/v1/billing/events", response_model=BillingEventsResponse)
    def billing_events(
        project_id: Optional[str] = Query(None, description="Filter by project_id."),
        window_minutes: int = Query(0, ge=0, description="Time window in minutes (0=all)."),
        limit: int = Query(100, ge=1, le=1000, description="Max events to return."),
    ) -> Dict[str, Any]:
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        events = billing_store.query_events(
            project_id=project_id, window_minutes=window_minutes, limit=limit,
        )
        return {"events": events, "total": len(events)}

    @app.get("/v1/billing/invoice-preview", response_model=InvoicePreviewResponse)
    def billing_invoice_preview(
        project_id: str = Query("default", description="Project to preview."),
        plan_id: str = Query("FREE", description="Plan ID: FREE, STARTER, PRO, ENTERPRISE."),
        window_minutes: int = Query(43200, ge=1, description="Window in minutes (default 30 days)."),
    ) -> Dict[str, Any]:
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        try:
            return billing_store.preview_invoice(
                project_id=project_id, plan_id=plan_id, window_minutes=window_minutes,
            )
        except KeyError:
            raise HTTPException(status_code=422, detail=f"Unknown plan: {plan_id}")

    # ── Billing v1 Extended Endpoints ─────────────────────────

    @app.get("/v1/billing/plans")
    def billing_plans() -> Dict[str, Any]:
        """List all available billing plans."""
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        from verifily_cli_v1.core.billing.pricing import PLANS
        from dataclasses import asdict
        return {
            "plans": [asdict(p) for p in PLANS.values()]
        }

    @app.get("/v1/billing/estimate")
    def billing_estimate(
        plan: str = Query("FREE", description="Plan ID."),
        window_minutes: int = Query(43200, ge=1, description="Window in minutes (default 30d)."),
        project_id: Optional[str] = Query(None, description="Filter by project."),
    ) -> Dict[str, Any]:
        """Estimate invoice for current usage."""
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        from verifily_cli_v1.core.billing.pricing import get_plan as _get_plan
        from verifily_cli_v1.core.billing.metering import compute_invoice as _compute
        try:
            plan_spec = _get_plan(plan)
        except KeyError:
            raise HTTPException(status_code=422, detail=f"Unknown plan: {plan}")
        now = time.time()
        cutoff = now - (window_minutes * 60)
        events = billing_store.get_raw_events(
            window_minutes=window_minutes,
            project_id=project_id,
        )
        preview = _compute(events, plan_spec, project_id or "all", cutoff, now)
        d = preview.to_dict()
        d["plan_id"] = d.pop("plan_id", plan_spec.id)
        return d

    @app.post("/v1/billing/invoice")
    def billing_generate_invoice(
        request: Request,
        body: "BillingInvoiceRequest",
    ) -> Dict[str, Any]:
        """Generate and persist an invoice with deterministic ID + artifacts."""
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        from verifily_cli_v1.core.billing.pricing import get_plan as _get_plan
        from verifily_cli_v1.core.billing.invoice import (
            deterministic_invoice_id,
            write_invoice_json,
            write_invoice_csv,
        )
        try:
            _get_plan(body.plan)
        except KeyError:
            raise HTTPException(status_code=422, detail=f"Unknown plan: {body.plan}")
        try:
            period_start = datetime.datetime.fromisoformat(body.period_start).timestamp()
            period_end = datetime.datetime.fromisoformat(body.period_end).timestamp()
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid ISO datetime in period_start or period_end.")

        project_id = body.project_id or "default"
        invoice_data = billing_store.generate_invoice(
            project_id=project_id,
            plan_id=body.plan,
            period_start=period_start,
            period_end=period_end,
        )

        # Deterministic invoice ID
        period_str = body.period_start[:7]  # YYYY-MM
        invoice_id = deterministic_invoice_id(project_id, period_str, body.plan)
        invoice_data["invoice_id"] = invoice_id
        invoice_data["period_start"] = body.period_start
        invoice_data["period_end"] = body.period_end

        # Store for later retrieval via GET /v1/billing/invoice/{id}
        billing_store.store_invoice(invoice_id, invoice_data)

        # Write artifacts to disk
        invoice_path = None
        try:
            invoice_dir = str(Path(settings.data_dir) / "billing" / "invoices")
            # Build a lightweight Invoice-like object for writers
            from verifily_cli_v1.core.billing.models import InvoiceLine, InvoicePreview
            from verifily_cli_v1.core.billing.invoice import Invoice
            lines = [
                InvoiceLine(**ln) for ln in invoice_data.get("lines", [])
            ]
            preview = InvoicePreview(
                project_id=project_id,
                plan_id=body.plan,
                window_start=period_start,
                window_end=period_end,
                lines=lines,
                monthly_base_cents=invoice_data.get("monthly_base_cents", 0),
                subtotal_cents=invoice_data.get("subtotal_cents", 0),
                tax_cents=invoice_data.get("tax_cents", 0),
                total_cents=invoice_data.get("total_cents", 0),
            )
            inv_obj = Invoice(
                invoice_id=invoice_id,
                customer=project_id,
                generated_at=time.time(),
                preview=preview,
            )
            json_path = write_invoice_json(inv_obj, invoice_dir)
            csv_path = write_invoice_csv(inv_obj, invoice_dir)
            invoice_data["invoice_path"] = json_path
            invoice_data["csv_path"] = csv_path
        except Exception:
            logger.warning("Failed to persist invoice %s", invoice_id, exc_info=True)

        return invoice_data

    @app.get("/v1/billing/invoice/{invoice_id}")
    def billing_get_invoice(invoice_id: str) -> Dict[str, Any]:
        """Retrieve a previously generated invoice by ID."""
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        inv = billing_store.get_invoice(invoice_id)
        if inv is None:
            raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found.")
        return inv

    @app.get("/v1/billing/usage")
    def billing_usage_query(
        period: Optional[str] = Query(None, description="YYYY-MM billing period (default: current)."),
        group_by: str = Query("total", description="Grouping: total, api_key, project."),
        project_id: Optional[str] = Query(None, description="Filter by project."),
    ) -> Dict[str, Any]:
        """Query processed rows usage for a billing period."""
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")
        from verifily_cli_v1.core.billing.periods import current_period, validate_period
        effective_period = period or current_period()
        if not validate_period(effective_period):
            raise HTTPException(status_code=422, detail=f"Invalid period format: {effective_period}. Use YYYY-MM.")
        if group_by not in ("total", "api_key", "project"):
            raise HTTPException(status_code=422, detail="group_by must be total, api_key, or project.")
        usage = billing_store.usage_for_period(
            project_id=project_id,
            period=effective_period,
        )
        return {
            "period": effective_period,
            "group_by": group_by,
            "project_id": project_id,
            "usage": usage,
        }

    @app.get("/v1/billing/usage_export")
    def billing_usage_export(
        request: Request,
        format: str = Query("csv", description="Export format: csv or jsonl."),
        period_days: int = Query(30, ge=1, le=365, description="Period in days."),
        group_by: str = Query("day", description="Grouping: day, day_project, day_api_key."),
    ) -> Any:
        """Export usage data as CSV or JSONL. Pro feature when billing_enforce=1."""
        if not settings.enable_billing:
            raise HTTPException(status_code=404, detail="Billing is not enabled.")

        # Soft gate: usage_export is a Pro feature
        if settings.billing_enforce:
            org_id = getattr(request.state, "org_id", None) or "default"
            project_id = getattr(request.state, "project_id", None) or "default"
            if not subscriptions_store.require_active(org_id, project_id):
                raise HTTPException(
                    status_code=402,
                    detail={
                        "type": "PAYMENT_REQUIRED",
                        "message": "Pro feature. Start checkout at /v1/billing/checkout",
                    },
                )

        if format not in ("csv", "jsonl"):
            raise HTTPException(status_code=422, detail="format must be 'csv' or 'jsonl'.")
        if group_by not in ("day", "day_project", "day_api_key"):
            raise HTTPException(status_code=422, detail="group_by must be 'day', 'day_project', or 'day_api_key'.")

        from verifily_cli_v1.core.billing.export import export_usage_csv, export_usage_jsonl
        window_minutes = period_days * 24 * 60
        events = billing_store.get_raw_events(window_minutes=window_minutes)

        if format == "csv":
            content = export_usage_csv(events, group_by=group_by)
            media_type = "text/csv"
        else:
            content = export_usage_jsonl(events, group_by=group_by)
            media_type = "application/x-ndjson"

        row_count = len(content.strip().splitlines()) - (1 if format == "csv" else 0)
        return PlainTextResponse(content=content, media_type=media_type, headers={
            "X-Export-Rows": str(max(0, row_count)),
            "X-Export-Format": format,
        })

    # ── Stripe / Subscription Endpoints ───────────────────────

    @app.post("/v1/billing/checkout", response_model=CheckoutResponse)
    def billing_checkout(body: CheckoutRequest, request: Request) -> Dict[str, Any]:
        """Create a Stripe Checkout session for Pro subscription."""
        if not settings.stripe_enabled:
            raise HTTPException(
                status_code=501,
                detail="Stripe is not enabled. Set VERIFILY_STRIPE_ENABLED=1.",
            )

        from verifily_cli_v1.core.billing.stripe import StripeConfig

        stripe_config = StripeConfig.from_env()
        config_err = stripe_config.validate()
        if config_err:
            raise HTTPException(status_code=500, detail=config_err)

        # Get or create stripe client (use app.state to cache)
        if not hasattr(app.state, "_stripe_client"):
            from verifily_cli_v1.core.billing.stripe import MockStripeClient
            # Use mock if no real stripe package
            try:
                from verifily_cli_v1.core.billing.stripe import StripeClient
                app.state._stripe_client = StripeClient(stripe_config)
            except Exception:
                app.state._stripe_client = MockStripeClient(stripe_config)

        stripe_client = app.state._stripe_client

        if body.plan.lower() != "pro":
            raise HTTPException(status_code=422, detail="Only 'pro' plan supports checkout.")

        org_id = body.org_id or getattr(request.state, "org_id", "default")
        project_id = body.project_id or getattr(request.state, "project_id", "default")

        # Check if we already have a customer mapping
        existing = subscriptions_store.get(org_id, project_id)
        customer_id = existing.stripe_customer_id if existing else None

        if not customer_id:
            customer_id = stripe_client.create_customer(
                name=f"{org_id}/{project_id}",
                metadata={"org_id": org_id, "project_id": project_id},
            )

        session = stripe_client.create_checkout_session(
            customer_id=customer_id,
            price_id=stripe_config.price_id_pro,
            success_url=stripe_config.success_url,
            cancel_url=stripe_config.cancel_url,
            metadata={"org_id": org_id, "project_id": project_id},
        )

        # Record as INCOMPLETE until webhook confirms
        subscriptions_store.set_status(
            org_id=org_id,
            project_id=project_id,
            status=SubscriptionStatus.INCOMPLETE,
            stripe_customer_id=customer_id,
            plan="pro",
        )

        return {
            "checkout_url": session["url"],
            "stripe_customer_id": customer_id,
            "plan": "pro",
        }

    @app.post("/v1/billing/webhook")
    async def billing_webhook(request: Request) -> Any:
        """Receive Stripe webhook events."""
        if not settings.stripe_enabled:
            raise HTTPException(
                status_code=501,
                detail="Stripe is not enabled.",
            )

        sig_header = request.headers.get("stripe-signature", "")
        payload = await request.body()

        # Use the stripe client from app.state if available
        if hasattr(app.state, "_stripe_client"):
            stripe_client = app.state._stripe_client
        else:
            from verifily_cli_v1.core.billing.stripe import StripeConfig, MockStripeClient
            stripe_config = StripeConfig.from_env()
            stripe_client = MockStripeClient(stripe_config)
            app.state._stripe_client = stripe_client

        try:
            event = stripe_client.construct_event(payload, sig_header)
        except (ValueError, Exception) as e:
            logger.warning("webhook_invalid_signature: %s", str(e))
            return JSONResponse(status_code=400, content={"error": "Invalid signature"})

        event_type = event.get("type", "")
        event_data = event.get("data", {})

        org_id = event_data.get("metadata", {}).get("org_id", "default")
        project_id = event_data.get("metadata", {}).get("project_id", "default")
        subscription_id = event_data.get("subscription", "") or event_data.get("id", "")
        customer_id = event_data.get("customer", "")

        if event_type == "checkout.session.completed":
            subscriptions_store.set_status(
                org_id=org_id,
                project_id=project_id,
                status=SubscriptionStatus.ACTIVE,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                plan="pro",
            )
            logger.info("subscription_activated org=%s project=%s", org_id, project_id)

        elif event_type == "customer.subscription.deleted":
            subscriptions_store.set_status(
                org_id=org_id,
                project_id=project_id,
                status=SubscriptionStatus.CANCELED,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
            )
            logger.info("subscription_canceled org=%s project=%s", org_id, project_id)

        elif event_type == "invoice.payment_failed":
            subscriptions_store.set_status(
                org_id=org_id,
                project_id=project_id,
                status=SubscriptionStatus.INCOMPLETE,
                stripe_customer_id=customer_id,
            )
            logger.warning("subscription_payment_failed org=%s project=%s", org_id, project_id)

        else:
            logger.info("webhook_unhandled_event type=%s", event_type)

        return JSONResponse(status_code=200, content={"received": True})

    @app.get("/v1/billing/subscription")
    def billing_subscription(
        request: Request,
        project_id: str = Query("default", description="Project ID."),
        org_id: Optional[str] = Query(None, description="Organization ID."),
    ) -> Dict[str, Any]:
        """Get subscription status for a project."""
        if not settings.stripe_enabled:
            raise HTTPException(
                status_code=501,
                detail="Stripe is not enabled. Set VERIFILY_STRIPE_ENABLED=1.",
            )

        effective_org = org_id or getattr(request.state, "org_id", "default")
        rec = subscriptions_store.get(effective_org, project_id)
        if rec is None:
            return {
                "org_id": effective_org,
                "project_id": project_id,
                "status": "none",
                "plan": "free",
            }
        return rec.to_dict()

    # ── Audit Export ──────────────────────────────────────────────
    @app.get("/v1/audit/export")
    def audit_export(
        request: Request,
        project_id: Optional[str] = Query(None, description="Filter by project ID."),
        from_ts: Optional[str] = Query(None, description="ISO start timestamp."),
        to_ts: Optional[str] = Query(None, description="ISO end timestamp."),
        format: str = Query("json", description="Output format: json or jsonl."),
    ) -> Any:
        """Export audit events (enterprise security only)."""
        if not settings.enterprise_security:
            return JSONResponse(
                status_code=404,
                content={"error": {"type": "NOT_FOUND", "message": "Not found."}},
            )
        _check_enterprise_permission(request, "export_audit")
        events = audit_store.query(
            project_id=project_id,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        if format == "jsonl":
            import json as _json
            lines = [_json.dumps(e) for e in events]
            return PlainTextResponse(
                "\n".join(lines) + ("\n" if lines else ""),
                media_type="application/x-ndjson",
            )
        return {"events": events, "total": len(events)}

    return app


def validate_host(host: str, allow_nonlocal: bool) -> None:
    """Refuse to bind to non-localhost unless explicitly allowed."""
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in local_hosts and not allow_nonlocal:
        raise ValueError(
            f"Refusing to bind to non-local host '{host}'. "
            f"Verifily API is designed for local use only. "
            f"Pass --allow-nonlocal to override this safety check."
        )


def start_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_nonlocal: bool = False,
    reload: bool = False,
    settings: Optional[Settings] = None,
) -> None:
    """Validate host, create app, and start uvicorn.
    
    Startup checks are now handled by the lifespan context manager,
    but we run a pre-flight check here to fail fast with a clear error.
    """
    import uvicorn

    from verifily_cli_v1.core.api.settings import print_startup_warnings

    validate_host(host, allow_nonlocal)

    if settings is None:
        settings = load_settings(
            bind=host, port=port, allow_nonlocal=allow_nonlocal,
        )

    # Run pre-flight startup checks for fast failure
    from rich.console import Console
    console = Console(stderr=True)
    
    console.print("\n[bold]Verifily Pre-flight Checks[/bold]")
    result = run_startup_checks(settings)
    console.print(format_check_result(result))
    
    if not result.ok:
        console.print("\n[red bold]Startup failed:[/red bold] Environment validation errors detected.")
        console.print("Fix the issues above and try again.")
        raise RuntimeError(f"Startup checks failed: {'; '.join(result.errors)}")
    
    if result.warnings:
        console.print(f"\n[yellow]Warnings ({len(result.warnings)}):[/yellow]")
        for w in result.warnings:
            console.print(f"  • {w}")
    
    console.print("\n[green]✓ Startup checks passed[/green]")
    
    print_startup_warnings(settings)

    # Configure structured logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Create app factory closure for uvicorn
    def app_factory() -> FastAPI:
        return create_app(settings)

    uvicorn.run(
        app_factory,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        factory=True,
    )
