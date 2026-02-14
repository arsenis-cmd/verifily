"""Tests for enterprise RBAC: roles, permissions, user context."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from verifily_cli_v1.core.security.rbac import (
    EnterpriseRole,
    Permission,
    UserContext,
    has_permission,
    require_permission,
)


def _ctx(role: str, override=None) -> UserContext:
    po = frozenset(Permission(p) for p in override) if override is not None else None
    return UserContext(
        user_id="u1",
        role=EnterpriseRole(role),
        api_key_id="k1",
        project_id="proj1",
        permissions_override=po,
    )


class TestRolePermissions:
    def test_owner_has_all_6(self):
        ctx = _ctx("owner")
        for perm in Permission:
            assert has_permission(ctx, perm), f"owner should have {perm}"

    def test_admin_has_all_6(self):
        ctx = _ctx("admin")
        for perm in Permission:
            assert has_permission(ctx, perm), f"admin should have {perm}"

    def test_member_has_4_run_perms(self):
        ctx = _ctx("member")
        assert has_permission(ctx, Permission.RUN_PIPELINE)
        assert has_permission(ctx, Permission.RUN_REPORT)
        assert has_permission(ctx, Permission.RUN_CONTAMINATION)
        assert has_permission(ctx, Permission.SUBMIT_JOBS)

    def test_member_lacks_export_audit(self):
        ctx = _ctx("member")
        assert not has_permission(ctx, Permission.EXPORT_AUDIT)

    def test_member_lacks_view_usage(self):
        ctx = _ctx("member")
        assert not has_permission(ctx, Permission.VIEW_USAGE)

    def test_viewer_has_view_usage_only(self):
        ctx = _ctx("viewer")
        assert has_permission(ctx, Permission.VIEW_USAGE)
        for perm in Permission:
            if perm != Permission.VIEW_USAGE:
                assert not has_permission(ctx, perm), f"viewer should lack {perm}"


class TestRequirePermission:
    def test_passes_when_allowed(self):
        ctx = _ctx("owner")
        require_permission(ctx, Permission.RUN_PIPELINE)  # no exception

    def test_raises_403_when_denied(self):
        ctx = _ctx("viewer")
        with pytest.raises(HTTPException) as exc_info:
            require_permission(ctx, Permission.RUN_PIPELINE)
        assert exc_info.value.status_code == 403
        assert "Permission required" in str(exc_info.value.detail)


class TestPermissionsOverride:
    def test_override_narrows(self):
        ctx = _ctx("owner", override=["view_usage"])
        assert has_permission(ctx, Permission.VIEW_USAGE)
        assert not has_permission(ctx, Permission.RUN_PIPELINE)

    def test_override_empty_denies_all(self):
        ctx = _ctx("admin", override=[])
        for perm in Permission:
            assert not has_permission(ctx, perm)


class TestUserContext:
    def test_frozen(self):
        ctx = _ctx("viewer")
        with pytest.raises(AttributeError):
            ctx.role = EnterpriseRole.OWNER  # type: ignore[misc]
