"""RBAC (Role-Based Access Control) for Org & Access v1.

Defines permissions for VIEWER, DEV, and ADMIN roles.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

from fastapi import HTTPException, Request

from verifily_cli_v1.core.api.identity import Role, can_admin, can_read, can_write


# Endpoint method classifications
READ_ENDPOINTS = {
    "GET": True,
    "HEAD": True,
}

WRITE_ENDPOINTS = {
    "POST": True,
    "PUT": True,
    "PATCH": True,
    "DELETE": True,
}

# Admin-only endpoints (key management, org/project creation)
ADMIN_ENDPOINTS = {
    "/v1/orgs",
    "/v1/projects",
    "/v1/keys",
}

# Read-only endpoints (allowed for VIEWER)
READ_ONLY_PATHS = {
    "/health",
    "/ready",
    "/metrics",
    "/v1/usage",
    "/v1/jobs",  # GET only
    "/v1/monitor/status",
    "/v1/monitor/history",
    "/v1/registry/models",
    "/v1/registry/history",
    "/v1/lineage",
    "/v1/score",
    "/v1/budget/status",
    "/v1/me",
}

# Write endpoints (allowed for DEV and ADMIN)
WRITE_PATHS = {
    "/v1/pipeline",
    "/v1/report",
    "/v1/contamination",
    "/v1/retrain",
    "/v1/jobs/pipeline",
    "/v1/jobs/report",
    "/v1/jobs/contamination",
    "/v1/jobs/classify",
    "/v1/jobs/retrain",
    "/v1/monitor/start",
    "/v1/monitor/stop",
    "/v1/registry/register",
    "/v1/registry/promote",
}


def is_admin_endpoint(path: str) -> bool:
    """Check if endpoint requires ADMIN role."""
    for admin_path in ADMIN_ENDPOINTS:
        if path.startswith(admin_path):
            return True
    return False


def is_write_endpoint(method: str, path: str) -> bool:
    """Check if endpoint is a write operation."""
    if method not in WRITE_ENDPOINTS:
        return False
    
    # Check if it's a known write path
    for write_path in WRITE_PATHS:
        if path.startswith(write_path):
            return True
    
    # POST/PUT/DELETE on jobs is write
    if path.startswith("/v1/jobs") and method != "GET":
        return True
    
    return False


def check_permission(request: Request) -> None:
    """Check if request has permission based on role.
    
    Args:
        request: FastAPI request with state.api_key_id, state.role, etc.
        
    Raises:
        HTTPException: 403 if permission denied
    """
    role: Optional[Role] = getattr(request.state, "role", None)
    
    # No role set (legacy mode) - allow
    if role is None:
        return
    
    method = request.method
    path = request.url.path
    
    # ADMIN can do everything
    if role == Role.ADMIN:
        return
    
    # Check admin endpoints
    if is_admin_endpoint(path):
        # GET/HEAD on admin endpoints is allowed for any authenticated role (read access)
        if method in READ_ENDPOINTS:
            pass  # allow — individual endpoints handle their own checks
        else:
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
    
    # Check write endpoints
    if is_write_endpoint(method, path):
        if not can_write(role):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "FORBIDDEN",
                        "message": f"Write permission required (current role: {role.value})",
                        "code": "write_permission_required",
                    }
                }
            )
    
    # VIEWER can only read
    if not can_read(role):
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "type": "FORBIDDEN",
                    "message": "Read permission required",
                    "code": "read_permission_required",
                }
            }
        )


def require_role(min_role: Role) -> Callable:
    """Decorator to require minimum role for an endpoint.
    
    Usage:
        @app.post("/v1/orgs")
        @require_role(Role.ADMIN)
        def create_org(...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Get request from args (first arg for endpoint handlers)
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                # Try kwargs
                request = kwargs.get("request")
            
            if request is not None:
                role: Optional[Role] = getattr(request.state, "role", None)
                
                if role is None:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": {
                                "type": "FORBIDDEN",
                                "message": "Authentication required",
                                "code": "auth_required",
                            }
                        }
                    )
                
                if not has_role_permission(role, min_role):
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": {
                                "type": "FORBIDDEN",
                                "message": f"{min_role.value} role required (current: {role.value})",
                                "code": "insufficient_role",
                            }
                        }
                    )
            
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def has_role_permission(user_role: Role, required_role: Role) -> bool:
    """Check if user role meets minimum required role.
    
    Args:
        user_role: Role of the user
        required_role: Minimum required role
        
    Returns:
        True if user has sufficient permissions
    """
    hierarchy = {
        Role.VIEWER: 0,
        Role.DEV: 1,
        Role.ADMIN: 2,
    }
    return hierarchy.get(user_role, -1) >= hierarchy.get(required_role, 0)


def get_effective_project_id(request: Request) -> Optional[str]:
    """Get the effective project ID for a request.
    
    In org mode:
    - Uses project_id from API key
    - ADMIN can override via X-Project-ID header (same org only)
    
    Args:
        request: FastAPI request
        
    Returns:
        Project ID or None
    """
    role: Optional[Role] = getattr(request.state, "role", None)
    key_project_id: Optional[str] = getattr(request.state, "project_id", None)
    
    # Not in org mode
    if role is None:
        # Fall back to header for legacy compatibility
        return request.headers.get("X-Project-ID")
    
    # Check for ADMIN override
    if role == Role.ADMIN:
        override_project_id = request.headers.get("X-Project-ID")
        if override_project_id:
            # Verify override project is in same org
            key_org_id: Optional[str] = getattr(request.state, "org_id", None)
            
            if key_org_id and key_project_id:
                # Import here to avoid circular dependency
                from verifily_cli_v1.core.api.org_store import org_store
                try:
                    project = org_store.get_project(override_project_id)
                    if project.org_id == key_org_id:
                        return override_project_id
                except Exception:
                    pass
            # If override invalid, fall through to key's project
    
    return key_project_id
