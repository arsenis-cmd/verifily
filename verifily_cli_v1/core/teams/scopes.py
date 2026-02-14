"""Teams scope resolution -- 4 simplified scopes for RBAC.

Maps (method, path) to a required scope string, or None if the
endpoint is unscoped (public or admin-gated separately).
"""

from __future__ import annotations

from typing import Optional

TEAMS_SCOPES = frozenset({"run:write", "run:read", "usage:read", "admin:write"})

# Exact (method, path) -> scope
_TEAMS_ENDPOINT_SCOPES = {
    ("POST", "/v1/pipeline"): "run:write",
    ("POST", "/v1/contamination"): "run:write",
    ("POST", "/v1/report"): "run:write",
    ("POST", "/v1/retrain"): "run:write",
    ("POST", "/v1/jobs/pipeline"): "run:write",
    ("POST", "/v1/jobs/report"): "run:write",
    ("POST", "/v1/jobs/contamination"): "run:write",
    ("POST", "/v1/jobs/classify"): "run:write",
    ("POST", "/v1/jobs/retrain"): "run:write",
    ("GET", "/v1/usage"): "usage:read",
    ("GET", "/v1/usage/aggregated"): "usage:read",
    ("GET", "/v1/billing/events"): "usage:read",
    ("GET", "/v1/billing/invoice-preview"): "usage:read",
}

# Prefix matches: (prefix, method, scope)
_TEAMS_PREFIX_SCOPES = [
    ("/v1/jobs/", "GET", "run:read"),
    ("/v1/jobs", "GET", "run:read"),
    ("/v1/monitor/", "POST", "run:write"),
    ("/v1/monitor/", "GET", "run:read"),
]


def resolve_teams_scope(method: str, path: str) -> Optional[str]:
    """Return required teams scope for an endpoint, or None if unscoped."""
    method = method.upper()

    # Exact match first
    key = (method, path)
    if key in _TEAMS_ENDPOINT_SCOPES:
        return _TEAMS_ENDPOINT_SCOPES[key]

    # Prefix match
    for prefix, method_filter, scope in _TEAMS_PREFIX_SCOPES:
        if path.startswith(prefix):
            if method_filter is None or method == method_filter:
                return scope

    return None
