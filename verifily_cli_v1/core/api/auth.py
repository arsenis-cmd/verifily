"""API key authentication middleware for Verifily API.

Supports three modes:
  - simple (default): Single VERIFILY_API_KEY env var, hmac comparison.
  - advanced: Scoped key registry with per-key scopes and project binding.
  - org_mode: Multi-tenant org/project/key scoping with RBAC.

Auth is disabled when no key is configured (simple) or no keys exist (advanced/org).
Public endpoints (/health, /ready, /metrics, /docs) are always open.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from verifily_cli_v1.core.api.auth_registry import auth_registry, resolve_scope
from verifily_cli_v1.core.api.errors import make_error_envelope
from verifily_cli_v1.core.api.identity import Role
from verifily_cli_v1.core.api.org_store import org_store
from verifily_cli_v1.core.api.quotas import quota_store
from verifily_cli_v1.core.teams.scopes import resolve_teams_scope
from verifily_cli_v1.core.teams.store import teams_store

logger = logging.getLogger("verifily.api")

# Paths that never require auth.
PUBLIC_PATHS = frozenset({"/health", "/ready", "/metrics", "/docs", "/openapi.json"})


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Enforce Bearer token auth on /v1/* endpoints.

    In simple mode: uses VERIFILY_API_KEY env var.
    In advanced mode: uses auth_registry with scopes and project binding.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Public endpoints skip auth.
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Enterprise token detection — runs before any auth mode.
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer vft_"):
            return await self._enterprise_token_auth(request, call_next)

        auth_mode = getattr(getattr(request, "app", None), "state", None)
        if auth_mode and hasattr(auth_mode, "settings"):
            auth_mode = auth_mode.settings.auth_mode
        else:
            auth_mode = os.environ.get("VERIFILY_AUTH_MODE", "simple")
        
        # Workspaces mode (takes priority over org_mode)
        _app_state = getattr(getattr(request, "app", None), "state", None)
        ws_enabled = False
        if _app_state and hasattr(_app_state, "settings"):
            ws_enabled = _app_state.settings.workspaces_enabled
        if ws_enabled:
            return await self._workspaces_auth(request, call_next)

        # Check for org mode (new multi-tenant mode)
        org_mode = os.environ.get("VERIFILY_ORG_MODE", "0") == "1"

        if org_mode:
            return await self._org_mode_auth(request, call_next)
        elif auth_mode == "teams":
            return await self._teams_auth(request, call_next)
        elif auth_mode == "advanced":
            return await self._advanced_auth(request, call_next)
        else:
            return await self._simple_auth(request, call_next)

    async def _simple_auth(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Original single-key auth behavior."""
        expected_key = os.environ.get("VERIFILY_API_KEY", "")

        # If no key is configured, auth is disabled — pass through.
        if not expected_key:
            return await call_next(request)

        request_id = getattr(request.state, "request_id", None)

        # Extract Bearer token.
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=missing_token",
                request_id or "?",
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Missing or invalid Authorization header.",
                    request_id,
                ),
            )

        provided_key = auth_header[7:]  # strip "Bearer "

        # Constant-time comparison to prevent timing attacks.
        if not hmac.compare_digest(provided_key, expected_key):
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=invalid_key",
                request_id or "?",
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Invalid API key.",
                    request_id,
                ),
            )

        return await call_next(request)

    async def _teams_auth(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Teams RBAC auth: super admin key for /v1/admin/*, scoped API keys for everything else."""
        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method

        settings = getattr(getattr(request, "app", None), "state", None)
        super_admin_key = ""
        if settings and hasattr(settings, "settings"):
            super_admin_key = settings.settings.super_admin_key

        # Admin endpoints: require super admin key
        if path.startswith("/v1/admin/"):
            if not super_admin_key:
                # No super admin key configured -- admin endpoints are open (bootstrap)
                return await call_next(request)

            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content=make_error_envelope(
                        "AUTH_ERROR",
                        "Missing or invalid Authorization header.",
                        request_id,
                    ),
                )

            provided = auth_header[7:]
            if not hmac.compare_digest(provided, super_admin_key):
                return JSONResponse(
                    status_code=401,
                    content=make_error_envelope(
                        "AUTH_ERROR",
                        "Invalid super admin key.",
                        request_id,
                    ),
                )

            request.state.api_key_id = "super-admin"
            request.state.org_id = "*"
            return await call_next(request)

        # Non-admin: require scoped API key from teams_store
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=missing_token mode=teams",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Missing or invalid Authorization header.",
                    request_id,
                ),
            )

        raw_key = auth_header[7:]
        record = teams_store.resolve_key(raw_key)
        if record is None:
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=invalid_key mode=teams",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Invalid API key.",
                    request_id,
                ),
            )

        # Scope enforcement
        required_scope = resolve_teams_scope(method, path)
        if required_scope and required_scope not in record.scopes:
            logger.warning(
                "auth_forbidden request_id=%s key=%s path=%s required_scope=%s mode=teams",
                request_id or "?",
                record.id,
                path,
                required_scope,
            )
            return JSONResponse(
                status_code=403,
                content=make_error_envelope(
                    "AUTH_FORBIDDEN",
                    f"Key lacks required scope: {required_scope}.",
                    request_id,
                ),
            )

        # Project enforcement: check query param or header
        project_id = request.query_params.get("project_id") or request.headers.get("x-project-id", "default")
        if "*" not in record.project_ids and project_id not in record.project_ids:
            logger.warning(
                "auth_forbidden request_id=%s key=%s project=%s allowed=%s mode=teams",
                request_id or "?",
                record.id,
                project_id,
                record.project_ids,
            )
            return JSONResponse(
                status_code=403,
                content=make_error_envelope(
                    "AUTH_FORBIDDEN",
                    f"Key not authorized for project: {project_id}.",
                    request_id,
                ),
            )

        # Set state for downstream handlers
        request.state.api_key_id = record.id
        request.state.org_id = record.org_id
        request.state.project_id = project_id

        return await call_next(request)

    async def _advanced_auth(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Scoped key auth via auth_registry."""
        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method

        # Bootstrap mode: if admin endpoints and no admin keys exist,
        # allow unauthenticated access to /v1/admin/* so the first key can be created.
        if path.startswith("/v1/admin/") and not auth_registry.has_admin_keys():
            return await call_next(request)

        # Extract Bearer token.
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=missing_token mode=advanced",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Missing or invalid Authorization header.",
                    request_id,
                ),
            )

        raw_key = auth_header[7:]

        # Authenticate against registry.
        record = auth_registry.authenticate(raw_key)
        if record is None:
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=invalid_key mode=advanced",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Invalid API key.",
                    request_id,
                ),
            )

        # Set state for downstream handlers.
        request.state.api_key_id = record.id
        request.state.scopes = record.scopes
        request.state.projects_allowed = record.projects_allowed

        # Scope enforcement.
        required_scope = resolve_scope(path, method)
        if required_scope and required_scope not in record.scopes:
            logger.warning(
                "auth_forbidden request_id=%s key=%s path=%s required_scope=%s",
                request_id or "?",
                record.id,
                path,
                required_scope,
            )
            return JSONResponse(
                status_code=403,
                content=make_error_envelope(
                    "AUTH_FORBIDDEN",
                    f"Key lacks required scope: {required_scope}.",
                    request_id,
                ),
            )

        # Project scoping.
        project_id = request.headers.get("x-project-id", "default")
        if "*" not in record.projects_allowed and project_id not in record.projects_allowed:
            logger.warning(
                "auth_forbidden request_id=%s key=%s project=%s allowed=%s",
                request_id or "?",
                record.id,
                project_id,
                record.projects_allowed,
            )
            return JSONResponse(
                status_code=403,
                content=make_error_envelope(
                    "AUTH_FORBIDDEN",
                    f"Key not authorized for project: {project_id}.",
                    request_id,
                ),
            )

        # Quota enforcement.
        quota_msg = quota_store.check_and_increment(record.id, project_id)
        if quota_msg:
            logger.warning(
                "quota_exceeded request_id=%s key=%s project=%s",
                request_id or "?",
                record.id,
                project_id,
            )
            return JSONResponse(
                status_code=429,
                content=make_error_envelope(
                    "QUOTA_EXCEEDED",
                    quota_msg,
                    request_id,
                ),
                headers={"Retry-After": str(quota_store.seconds_until_reset())},
            )

        return await call_next(request)


    async def _org_mode_auth(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Org mode auth: Multi-tenant org/project/key scoping with RBAC.
        
        Keys are scoped to a specific project within an org.
        ADMIN keys can override project via X-Project-ID header (same org only).
        """
        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method

        # Bootstrap mode: Allow first org creation without auth
        # Also allow if VERIFILY_ORG_BOOTSTRAP=1 is set
        bootstrap_mode = os.environ.get("VERIFILY_ORG_BOOTSTRAP", "0") == "1"
        is_org_endpoint = path == "/v1/orgs" and method == "POST"
        no_orgs_exist = len(org_store.list_orgs()) == 0
        
        if is_org_endpoint and (bootstrap_mode or no_orgs_exist):
            return await call_next(request)

        # Extract Bearer token.
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=missing_token mode=org",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Missing or invalid Authorization header.",
                    request_id,
                ),
            )

        raw_key = auth_header[7:]

        # Authenticate against org_store.
        api_key = org_store.resolve_key(raw_key)
        if api_key is None:
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=invalid_or_revoked_key mode=org",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Invalid or revoked API key.",
                    request_id,
                ),
            )

        # Determine effective project_id
        effective_project_id = api_key.project_id
        
        # ADMIN can override project_id via header (same org only)
        if api_key.role == Role.ADMIN:
            override_project_id = request.headers.get("x-project-id")
            if override_project_id:
                try:
                    target_project = org_store.get_project(override_project_id)
                    # Verify same org
                    if target_project.org_id == api_key.org_id:
                        effective_project_id = override_project_id
                    else:
                        logger.warning(
                            "auth_forbidden request_id=%s key=%s target_project=%s wrong_org mode=org",
                            request_id or "?",
                            api_key.key_id,
                            override_project_id,
                        )
                        return JSONResponse(
                            status_code=403,
                            content=make_error_envelope(
                                "AUTH_FORBIDDEN",
                                "Cannot access project in different organization.",
                                request_id,
                            ),
                        )
                except Exception:
                    # Invalid project override, use key's project
                    pass

        # Set state for downstream handlers
        request.state.api_key_id = api_key.key_id
        request.state.org_id = api_key.org_id
        request.state.project_id = effective_project_id
        request.state.role = api_key.role

        # RBAC enforcement
        from verifily_cli_v1.core.api.rbac import check_permission
        try:
            check_permission(request)
        except HTTPException as e:
            logger.warning(
                "rbac_denied request_id=%s key=%s role=%s path=%s method=%s",
                request_id or "?",
                api_key.key_id,
                api_key.role.value,
                path,
                method,
            )
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail,
            )

        # Quota enforcement (bucketed by key_id)
        quota_msg = quota_store.check_and_increment(api_key.key_id, effective_project_id)
        if quota_msg:
            logger.warning(
                "quota_exceeded request_id=%s key=%s project=%s mode=org",
                request_id or "?",
                api_key.key_id,
                effective_project_id,
            )
            return JSONResponse(
                status_code=429,
                content=make_error_envelope(
                    "QUOTA_EXCEEDED",
                    quota_msg,
                    request_id,
                ),
                headers={"Retry-After": str(quota_store.seconds_until_reset())},
            )

        return await call_next(request)

    async def _workspaces_auth(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Workspaces auth: JSON-file-backed org/project/key with RBAC."""
        from verifily_cli_v1.core.workspaces.store import workspaces_store, ROLE_MAP

        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method

        settings = getattr(getattr(request, "app", None), "state", None)
        s = getattr(settings, "settings", None) if settings else None

        # Bootstrap: allow first org creation without auth
        if path == "/v1/orgs" and method == "POST" and workspaces_store.is_empty():
            # If bootstrap_token configured, require it
            if s and s.bootstrap_token:
                bt = request.headers.get("x-bootstrap-token", "")
                if not hmac.compare_digest(bt, s.bootstrap_token):
                    return JSONResponse(
                        status_code=401,
                        content=make_error_envelope(
                            "AUTH_ERROR",
                            "Invalid bootstrap token.",
                            request_id,
                        ),
                    )
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=missing_token mode=workspaces",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Missing or invalid Authorization header.",
                    request_id,
                ),
            )

        raw_key = auth_header[7:]
        resolved = workspaces_store.resolve_request(raw_key)
        if resolved is None:
            logger.warning(
                "auth_rejected request_id=%s path=%s reason=invalid_key mode=workspaces",
                request_id or "?",
                path,
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Invalid or revoked API key.",
                    request_id,
                ),
            )

        # Map workspace role to identity.Role and set state
        mapped_role = ROLE_MAP.get(resolved["role"])
        request.state.api_key_id = resolved["api_key_id"]
        request.state.org_id = resolved["org_id"]
        request.state.project_id = resolved["project_id"]
        request.state.role = mapped_role

        # RBAC enforcement via existing check_permission
        from verifily_cli_v1.core.api.rbac import check_permission
        try:
            check_permission(request)
        except HTTPException as e:
            logger.warning(
                "rbac_denied request_id=%s key=%s role=%s path=%s method=%s mode=workspaces",
                request_id or "?",
                resolved["api_key_id"],
                resolved["role"],
                path,
                method,
            )
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail,
            )

        return await call_next(request)

    async def _enterprise_token_auth(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Authenticate an enterprise scoped token (vft_ prefix)."""
        request_id = getattr(request.state, "request_id", None)

        settings = getattr(getattr(request, "app", None), "state", None)
        s = getattr(settings, "settings", None) if settings else None

        if not s or not s.enterprise_security or not s.token_secret:
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    "Enterprise security not enabled.",
                    request_id,
                ),
            )

        raw_token = request.headers.get("authorization", "")[7:]  # strip "Bearer "

        from verifily_cli_v1.core.security.tokens import verify_token
        from verifily_cli_v1.core.security.rbac import UserContext

        try:
            payload = verify_token(s.token_secret, raw_token)
        except ValueError as e:
            logger.warning(
                "enterprise_auth_rejected request_id=%s path=%s reason=%s",
                request_id or "?",
                request.url.path,
                str(e),
            )
            return JSONResponse(
                status_code=401,
                content=make_error_envelope(
                    "AUTH_ERROR",
                    f"Invalid enterprise token: {e}",
                    request_id,
                ),
            )

        user_ctx = UserContext(
            user_id=payload.token_id,
            role=payload.role,
            api_key_id=payload.token_id,
            project_id=payload.project_id,
            permissions_override=payload.permissions_override,
        )
        request.state.user_ctx = user_ctx
        request.state.api_key_id = payload.token_id
        request.state.project_id = payload.project_id

        return await call_next(request)
