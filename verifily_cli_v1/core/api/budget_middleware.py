"""Budget enforcement middleware for FastAPI.

Intercepts requests to billable endpoints and enforces budget limits.
Returns 402 Payment Required when budget is exceeded (or 429 with Retry-After).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from verifily_cli_v1.core.budget import BudgetMode, BudgetStore, budget_store
from verifily_cli_v1.core.api.usage_store import UsageStore, usage_store

logger = logging.getLogger("verifily.budget_middleware")

# Billable endpoint patterns
BILLABLE_ENDPOINTS: List[tuple] = [
    (r"^/v1/pipeline$", ["POST"]),
    (r"^/v1/report$", ["POST"]),
    (r"^/v1/contamination$", ["POST"]),
    (r"^/v1/jobs$", ["POST", "PUT"]),
    (r"^/v1/jobs/.*", ["POST", "PUT", "DELETE"]),
    (r"^/v1/monitor/start$", ["POST"]),
]

# Non-billable health endpoints (always allowed)
NON_BILLABLE_PATHS = [
    "/health",
    "/ready",
    "/metrics",
    "/v1/budget",
]


class BudgetMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing budget limits on billable endpoints.
    
    Adds X-Budget-* headers to responses and blocks over-budget requests.
    """
    
    def __init__(
        self,
        app,
        budget_store_instance: Optional[BudgetStore] = None,
        usage_store_instance: Optional[UsageStore] = None,
    ):
        super().__init__(app)
        self._budget_store = budget_store_instance or budget_store
        self._usage_store = usage_store_instance or usage_store
    
    def _is_billable(self, path: str, method: str) -> bool:
        """Check if endpoint is billable."""
        # Always allow health/non-billable endpoints
        for non_billable in NON_BILLABLE_PATHS:
            if path.startswith(non_billable):
                return False
        
        # Check against billable patterns
        for pattern, methods in BILLABLE_ENDPOINTS:
            if re.match(pattern, path) and method in methods:
                return True
        
        return False
    
    def _extract_project_id(self, request: Request) -> Optional[str]:
        """Extract project ID from request."""
        # Priority: query param > header > default
        project_id = request.query_params.get("project_id")
        if project_id:
            return project_id
        
        project_id = request.headers.get("X-Project-ID")
        if project_id:
            return project_id
        
        # Try to extract from body for POST requests (async)
        # Default to "default" project
        return "default"
    
    async def _get_request_body(self, request: Request) -> Optional[Dict[str, Any]]:
        """Safely peek at request body for project_id."""
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    # Reset body for downstream processing
                    async def receive():
                        return {"type": "http.request", "body": body}
                    request._receive = receive
                    
                    import json
                    data = json.loads(body)
                    return data
        except Exception:
            pass
        return None
    
    def _add_budget_headers(self, response: Response, check_result: Any) -> None:
        """Add budget headers to response."""
        response.headers["X-Budget-Mode"] = check_result.mode
        response.headers["X-Budget-Remaining-Daily"] = str(check_result.remaining_daily_units)
        response.headers["X-Budget-Remaining-Monthly"] = str(check_result.remaining_monthly_units)
        
        if check_result.reset_time_utc:
            response.headers["X-Budget-Reset-Time"] = check_result.reset_time_utc
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with budget check."""
        path = request.url.path
        method = request.method
        
        # Skip budget check for non-billable endpoints
        if not self._is_billable(path, method):
            return await call_next(request)
        
        # Extract project ID
        project_id = self._extract_project_id(request)
        if not project_id:
            # No project specified, allow but warn
            logger.debug("No project_id found, skipping budget check")
            return await call_next(request)
        
        # Check budget
        try:
            check_result = self._budget_store.check_budget(
                project_id, self._usage_store
            )
        except Exception as e:
            logger.error("Budget check failed: %s", e)
            # Fail open (allow request) if budget check errors
            response = await call_next(request)
            response.headers["X-Budget-Error"] = "check_failed"
            return response
        
        # Handle BLOCK mode
        if check_result.mode == BudgetMode.BLOCK:
            logger.warning(
                "Budget exceeded for project %s: %s", project_id, check_result.reason
            )
            
            # Calculate retry after (seconds until reset)
            import datetime
            from datetime import datetime as dt
            
            retry_after = 3600  # Default 1 hour
            if check_result.reset_time_utc:
                try:
                    reset_dt = dt.fromisoformat(check_result.reset_time_utc.replace("Z", "+00:00"))
                    now = dt.utcnow().replace(tzinfo=datetime.timezone.utc)
                    retry_after = max(1, int((reset_dt - now).total_seconds()))
                except Exception:
                    pass
            
            return JSONResponse(
                status_code=402,  # Payment Required
                content={
                    "error": {
                        "type": "BUDGET_EXCEEDED",
                        "message": check_result.reason,
                        "code": "budget_limit_reached",
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-Budget-Remaining-Daily": str(check_result.remaining_daily_units),
                    "X-Budget-Remaining-Monthly": str(check_result.remaining_monthly_units),
                    "X-Budget-Reset-Time": check_result.reset_time_utc,
                },
            )
        
        # Handle WARN mode (allow but flag)
        if check_result.mode == BudgetMode.WARN:
            logger.info("Budget warning for project %s: %s", project_id, check_result.reason)
            request.state.budget_warning = True
        
        # Process request
        response = await call_next(request)
        
        # Add budget headers to successful response
        self._add_budget_headers(response, check_result)
        
        # Add warning header if applicable
        if check_result.mode == BudgetMode.WARN:
            response.headers["X-Budget-Warning"] = check_result.reason
        
        return response


def create_budget_middleware(
    budget_store_instance: Optional[BudgetStore] = None,
    usage_store_instance: Optional[UsageStore] = None,
) -> Callable:
    """Factory to create BudgetMiddleware with custom stores."""
    def _middleware(app):
        return BudgetMiddleware(
            app,
            budget_store_instance=budget_store_instance,
            usage_store_instance=usage_store_instance,
        )
    return _middleware
