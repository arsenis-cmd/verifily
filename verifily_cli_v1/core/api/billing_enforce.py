"""Billing enforcement middleware -- plan-tier monthly caps.

Intercepts requests to billable endpoints before they execute.
Returns 402 PAYMENT_REQUIRED when processed_rows exceeds the plan cap.

Enabled by VERIFILY_BILLING_ENFORCE=1 + VERIFILY_ENABLE_BILLING=1.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from verifily_cli_v1.core.billing.enforcement import (
    BillingStatus,
    check_quota,
)
from verifily_cli_v1.core.billing.periods import current_period
from verifily_cli_v1.core.billing.pricing import get_plan_limits
from verifily_cli_v1.core.billing.store import billing_store
from verifily_cli_v1.core.billing.subscriptions import subscriptions_store

logger = logging.getLogger("verifily.billing_enforce")

# Billable endpoint patterns (expensive POST endpoints)
_BILLABLE: List[Tuple[str, List[str]]] = [
    (r"^/v1/pipeline$", ["POST"]),
    (r"^/v1/report$", ["POST"]),
    (r"^/v1/contamination$", ["POST"]),
    (r"^/v1/retrain$", ["POST"]),
    (r"^/v1/jobs/.*", ["POST"]),
    (r"^/v1/monitor/start$", ["POST"]),
]

_EXEMPT_PREFIXES = (
    "/health", "/ready", "/metrics", "/docs", "/openapi.json",
    "/v1/billing/",
)


class BillingEnforceMiddleware(BaseHTTPMiddleware):
    """Enforce monthly plan caps on billable endpoints."""

    def __init__(self, app, *, default_plan: str = "FREE"):
        super().__init__(app)
        self._default_plan = default_plan

    def _is_billable(self, path: str, method: str) -> bool:
        for prefix in _EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return False
        if method == "GET":
            return False
        for pattern, methods in _BILLABLE:
            if re.match(pattern, path) and method in methods:
                return True
        return False

    def _resolve_plan_id(self, project_id: str) -> str:
        """Determine the plan for a project via subscription store or default."""
        for rec in subscriptions_store.list_all():
            if rec.project_id == project_id and rec.status.value == "active":
                return rec.plan.upper() if rec.plan else self._default_plan
        return self._default_plan

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        path = request.url.path
        method = request.method

        if not self._is_billable(path, method):
            return await call_next(request)

        project_id = (
            getattr(request.state, "project_id", None)
            or request.query_params.get("project_id")
            or request.headers.get("x-project-id")
            or "default"
        )

        plan_id = self._resolve_plan_id(project_id)
        try:
            plan_limits = get_plan_limits(plan_id)
        except KeyError:
            plan_limits = get_plan_limits("FREE")

        period = current_period()
        usage = billing_store.usage_for_period(project_id=project_id, period=period)
        current_rows = usage["processed_rows"]

        decision = check_quota(
            plan_limits=plan_limits,
            current_processed_rows=current_rows,
        )

        if decision.status == BillingStatus.BLOCK:
            logger.warning(
                "billing_enforce_block project=%s plan=%s period=%s used=%d limit=%d",
                project_id, plan_id, period, decision.used, decision.limit,
            )
            return JSONResponse(
                status_code=402,
                content={
                    "error": {
                        "type": "PAYMENT_REQUIRED",
                        "message": decision.reason,
                        "code": "billing_limit_exceeded",
                        "period": period,
                        "limit": decision.limit,
                        "used": decision.used,
                        "attempted": 0,
                        "project_id": project_id,
                        "plan": plan_id,
                    }
                },
            )

        response = await call_next(request)

        if decision.status == BillingStatus.WARN:
            response.headers["X-Billing-Warning"] = decision.reason
            response.headers["X-Billing-Used"] = str(decision.used)
            response.headers["X-Billing-Limit"] = str(decision.limit)
            response.headers["X-Billing-Remaining"] = str(decision.remaining)

        return response
