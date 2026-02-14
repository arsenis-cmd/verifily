"""Workspace security model for safe file IO.

Prevents path traversal and enforces that all file operations stay within
the workspace root directory.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

from verifily_cli_v1.core.api.settings import load_production_settings, ProductionSettings

logger = logging.getLogger("verifily.workspace")


class WorkspaceSecurityError(Exception):
    """Security violation in workspace path resolution."""
    pass


class PathTraversalError(WorkspaceSecurityError):
    """Path traversal attempt detected."""
    pass


class AbsolutePathError(WorkspaceSecurityError):
    """Absolute paths not allowed in production mode."""
    pass


class OutsideWorkspaceError(WorkspaceSecurityError):
    """Resolved path is outside workspace root."""
    pass


class WorkspacePath:
    """A validated path within the workspace.
    
    Use resolve_workspace_path() to create instances.
    """
    
    def __init__(self, resolved_path: Path, workspace_root: Path, original: str) -> None:
        self.path = resolved_path
        self.workspace_root = workspace_root
        self.original = original
    
    def __str__(self) -> str:
        return str(self.path)
    
    def __fspath__(self) -> str:
        return str(self.path)
    
    def relative_to_workspace(self) -> Path:
        """Return path relative to workspace root."""
        return self.path.relative_to(self.workspace_root)
    
    def exists(self) -> bool:
        """Check if path exists."""
        return self.path.exists()
    
    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """Create directory if it doesn't exist."""
        self.path.mkdir(parents=parents, exist_ok=exist_ok)
    
    def ensure_parent_exists(self) -> None:
        """Ensure parent directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)


def _contains_traversal(user_path: str) -> bool:
    """Check if path contains traversal attempts.
    
    Detects patterns like:
    - ../etc/passwd
    - ..\\windows\\system32
    - /etc/passwd (absolute)
    - ../../../
    """
    # Normalize path separators
    normalized = user_path.replace("\\", "/")
    
    # Check for .. components
    parts = normalized.split("/")
    for part in parts:
        if part == "..":
            return True
    
    return False


def _is_absolute_path(user_path: str) -> bool:
    """Check if path is absolute."""
    # Unix absolute
    if user_path.startswith("/"):
        return True
    # Windows absolute (C:\ or \\server\share)
    if len(user_path) >= 2 and user_path[1] == ":":
        return True
    if user_path.startswith("\\\\"):
        return True
    return False


def resolve_workspace_path(
    user_path: str,
    workspace_root: Optional[Path] = None,
    allow_absolute: Optional[bool] = None,
    prod_mode: Optional[bool] = None,
) -> WorkspacePath:
    """Resolve a user-provided path to a safe workspace path.
    
    Security rules (enforced in order):
    1. No path traversal (.. components)
    2. No absolute paths (unless explicitly allowed)
    3. Must resolve within workspace root
    
    Args:
        user_path: Path from user input (API request)
        workspace_root: Workspace root directory (default from settings)
        allow_absolute: Override allow_abspath setting
        prod_mode: Override prod_mode setting
        
    Returns:
        WorkspacePath: Validated path within workspace
        
    Raises:
        PathTraversalError: If path contains .. components
        AbsolutePathError: If absolute paths not allowed
        OutsideWorkspaceError: If resolved path escapes workspace
    """
    try:
        settings = load_production_settings()
    except ValueError:
        # Fallback to defaults if settings aren't valid yet
        settings = ProductionSettings()
    
    if workspace_root is None:
        workspace_root = settings.workspace_root
    
    if allow_absolute is None:
        allow_absolute = settings.allow_abspath
    
    if prod_mode is None:
        prod_mode = settings.prod_mode
    
    # Rule 1: Check for path traversal
    if _contains_traversal(user_path):
        logger.warning(
            "Path traversal blocked: %r (workspace: %s)",
            user_path,
            workspace_root,
        )
        raise PathTraversalError(
            f"Path traversal not allowed: {user_path}\n"
            "Use relative paths within the workspace."
        )
    
    # Rule 2: Check for absolute paths
    if _is_absolute_path(user_path):
        if prod_mode or not allow_absolute:
            logger.warning(
                "Absolute path blocked in %s mode: %r",
                "prod" if prod_mode else "dev",
                user_path,
            )
            mode_str = "production" if prod_mode else "this configuration"
            raise AbsolutePathError(
                f"Absolute paths not allowed in {mode_str}: {user_path}\n"
                "Use relative paths from workspace root, or set ALLOW_ABSPATH=1 in dev."
            )
        # In dev mode with allow_absolute, we'll still try to make it relative
        logger.warning(
            "Absolute path used in dev mode: %r (consider using relative paths)",
            user_path,
        )
    
    # Resolve the path
    # Strip leading ./ and /
    clean_path = user_path.lstrip("./\\")
    
    # Build absolute path within workspace
    resolved = (workspace_root / clean_path).resolve()
    
    # Rule 3: Ensure path is within workspace
    # Use Path.parts comparison to avoid symlink issues
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError:
        logger.error(
            "Path escaped workspace: %r resolved to %s (workspace: %s)",
            user_path,
            resolved,
            workspace_root,
        )
        raise OutsideWorkspaceError(
            f"Path outside workspace: {user_path}\n"
            f"Resolved to: {resolved}\n"
            f"Workspace: {workspace_root}"
        )
    
    logger.debug(
        "Workspace path resolved: %r -> %s",
        user_path,
        resolved,
    )
    
    return WorkspacePath(resolved, workspace_root, user_path)


def validate_workspace_exists() -> None:
    """Validate that workspace root exists.
    
    Called during server startup in production mode.
    """
    try:
        settings = load_production_settings()
    except ValueError:
        settings = ProductionSettings()
    
    root = settings.workspace_root
    
    if not root.exists():
        if settings.prod_mode:
            raise WorkspaceSecurityError(
                f"Workspace root does not exist: {root}\n"
                f"Create it: mkdir -p {root}"
            )
        else:
            # In dev mode, auto-create
            logger.info("Creating workspace directory: %s", root)
            root.mkdir(parents=True, exist_ok=True)


def _get_settings_safe() -> ProductionSettings:
    """Get settings or return defaults."""
    try:
        return load_production_settings()
    except ValueError:
        return ProductionSettings()


def get_runs_directory() -> Path:
    """Get the runs output directory (workspace/runs)."""
    settings = _get_settings_safe()
    runs_dir = settings.workspace_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_datasets_directory() -> Path:
    """Get the datasets directory (workspace/datasets)."""
    settings = _get_settings_safe()
    datasets_dir = settings.workspace_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_temp_directory() -> Path:
    """Get a temporary workspace directory."""
    settings = _get_settings_safe()
    temp_dir = settings.workspace_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


# Backwards compatibility helpers

def safe_path_for_api(user_path: str, operation: str = "access") -> str:
    """Resolve path for API use, returning string or raising.
    
    This is a convenience wrapper for existing endpoints.
    """
    try:
        wp = resolve_workspace_path(user_path)
        return str(wp.path)
    except WorkspaceSecurityError as e:
        # Re-raise with operation context
        raise WorkspaceSecurityError(f"Cannot {operation} path: {e}") from e


def dev_mode_path_warning(user_path: str) -> None:
    """Warn about path usage in dev mode."""
    settings = _get_settings_safe()
    if not settings.prod_mode and _is_absolute_path(user_path):
        logger.warning(
            "DEV MODE: Using absolute path %r. "
            "In production, use relative paths from workspace root.",
            user_path,
        )
