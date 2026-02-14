"""Enterprise RBAC: fine-grained permission model for scoped tokens.

Activated only when enterprise_security=True and a scoped token (vft_ prefix)
is presented.  Legacy API key / org_mode auth continues to use core/api/rbac.py.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import FrozenSet, Optional

from fastapi import HTTPException


class EnterpriseRole(str, enum.Enum):
    """Enterprise roles — distinct from identity.Role to avoid collision."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(str, enum.Enum):
    """Fine-grained enterprise permissions."""

    RUN_PIPELINE = "run_pipeline"
    RUN_REPORT = "run_report"
    RUN_CONTAMINATION = "run_contamination"
    SUBMIT_JOBS = "submit_jobs"
    VIEW_USAGE = "view_usage"
    EXPORT_AUDIT = "export_audit"


# ── Permission matrix ────────────────────────────────────────────────

_ROLE_PERMISSIONS: dict[EnterpriseRole, FrozenSet[Permission]] = {
    EnterpriseRole.OWNER: frozenset(Permission),
    EnterpriseRole.ADMIN: frozenset(Permission),
    EnterpriseRole.MEMBER: frozenset({
        Permission.RUN_PIPELINE,
        Permission.RUN_REPORT,
        Permission.RUN_CONTAMINATION,
        Permission.SUBMIT_JOBS,
    }),
    EnterpriseRole.VIEWER: frozenset({
        Permission.VIEW_USAGE,
    }),
}


@dataclass(frozen=True)
class UserContext:
    """Attached to request.state.user_ctx when an enterprise token is verified."""

    user_id: str
    role: EnterpriseRole
    api_key_id: str
    project_id: str
    permissions_override: Optional[FrozenSet[Permission]] = None


def has_permission(ctx: UserContext, permission: Permission) -> bool:
    """Check if user context grants the given permission."""
    if ctx.permissions_override is not None:
        return permission in ctx.permissions_override
    return permission in _ROLE_PERMISSIONS.get(ctx.role, frozenset())


def require_permission(ctx: UserContext, permission: Permission) -> None:
    """Raise 403 if user context lacks permission."""
    if not has_permission(ctx, permission):
        raise HTTPException(
            status_code=403,
            detail=f"Permission required: {permission.value}",
        )
