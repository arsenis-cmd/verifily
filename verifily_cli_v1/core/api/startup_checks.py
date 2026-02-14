"""Startup environment validation for Verifily API.

Ensures the server environment is safe to start before binding to any port.
All checks are deterministic and produce actionable error messages.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from verifily_cli_v1.core.api.settings import Settings, load_settings


REQUIRED_PYTHON_VERSION = (3, 9)


class StartupCheckResult(BaseModel):
    """Result of startup environment validation.
    
    Attributes:
        ok: True if server should start (no errors)
        warnings: Non-fatal issues that were logged but allow startup
        errors: Fatal issues that must prevent startup
    """
    ok: bool
    warnings: List[str]
    errors: List[str]


def _check_python_version() -> tuple[bool, Optional[str]]:
    """Check Python version meets minimum requirement.
    
    Returns:
        (passed, error_message_or_none)
    """
    if sys.version_info >= REQUIRED_PYTHON_VERSION:
        return True, None
    return False, (
        f"Python {sys.version_info.major}.{sys.version_info.minor} < "
        f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}"
    )


def _check_write_access(path: str, purpose: str) -> tuple[bool, Optional[str]]:
    """Check write access to a directory.
    
    Args:
        path: Directory path to check
        purpose: Human-readable purpose for error messages
        
    Returns:
        (passed, error_message_or_none)
    """
    try:
        p = Path(path)
        # Create directory if it doesn't exist
        p.mkdir(parents=True, exist_ok=True)
        
        # Try to write a test file
        test_file = p / ".verifily_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return True, None
    except PermissionError:
        # Try fallback to temp directory
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "verifily" / Path(path).name
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            test_file = fallback / ".verifily_write_test"
            test_file.write_text("ok")
            test_file.unlink()
            # Return warning but pass - we'll use temp fallback
            return True, f"Using temp fallback for {purpose}: {fallback}"
        except Exception:
            return False, f"No write permission for {purpose}: {path} (temp fallback also failed)"
    except OSError as e:
        # Try fallback to temp directory
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "verifily" / Path(path).name
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            test_file = fallback / ".verifily_write_test"
            test_file.write_text("ok")
            test_file.unlink()
            return True, f"Using temp fallback for {purpose}: {fallback}"
        except Exception:
            return False, f"Cannot access {purpose}: {path} ({e}) (temp fallback also failed)"


def _check_api_key(settings: Settings) -> tuple[bool, Optional[str]]:
    """Check VERIFILY_API_KEY if set is non-empty.
    
    Returns:
        (passed, error_message_or_none)
    """
    api_key_env = os.environ.get("VERIFILY_API_KEY")
    if api_key_env is None:
        # Not set is fine - auth will be disabled
        return True, None
    if api_key_env.strip():
        return True, None
    return False, "VERIFILY_API_KEY is set but empty"


def _check_rate_limit_config(settings: Settings) -> tuple[bool, Optional[str], Optional[str]]:
    """Check rate limit configuration is sane.
    
    Returns:
        (passed, warning_or_none, error_or_none)
    """
    rpm = settings.rate_limit_rpm
    
    if rpm == 0:
        # 0 means disabled - warn but allow
        return True, "Rate limit not configured (unlimited)", None
    
    if rpm < 0:
        return False, None, f"Rate limit RPM must be >= 0, got {rpm}"
    
    # RPM > 0 is valid
    return True, None, None


def _get_persistence_paths(settings: Settings) -> List[tuple[str, str]]:
    """Get list of (path, purpose) tuples for persistence checks.
    
    Returns paths that need write access based on current settings.
    """
    paths: List[tuple[str, str]] = []
    
    # Data directory always needed for various subsystems
    paths.append((settings.data_dir, "data directory"))
    
    # Usage persistence
    if settings.usage_persist:
        usage_path = os.environ.get(
            "VERIFILY_USAGE_LOG_PATH",
            str(Path(settings.data_dir) / "verifily_usage_events.jsonl")
        )
        paths.append((str(Path(usage_path).parent), "usage log directory"))
    
    # Jobs persistence
    if settings.jobs_persist:
        jobs_path = os.environ.get(
            "VERIFILY_JOBS_LOG_PATH",
            str(Path(settings.data_dir) / "verifily_jobs_events.jsonl")
        )
        paths.append((str(Path(jobs_path).parent), "jobs log directory"))
    
    # Auth persistence
    if settings.auth_persist:
        paths.append((settings.data_dir, "auth persistence directory"))
    
    # Billing persistence
    if settings.enable_billing and settings.billing_persist:
        paths.append((settings.data_dir, "billing persistence directory"))
    
    # Teams persistence
    if settings.teams_enabled and settings.teams_persist:
        paths.append((settings.data_dir, "teams persistence directory"))
    
    return paths


def run_startup_checks(settings: Optional[Settings] = None) -> StartupCheckResult:
    """Run all startup environment validation checks.
    
    Validates:
    - Python version >= 3.9
    - Write access to persistence paths (usage/jobs/monitor data)
    - VERIFILY_API_KEY is non-empty if set
    - Rate limit config is sane (RPM >= 0)
    
    Args:
        settings: Settings to validate against, or None to load from environment
        
    Returns:
        StartupCheckResult with ok/warnings/errors
    """
    if settings is None:
        settings = load_settings()
    
    warnings: List[str] = []
    errors: List[str] = []
    
    # 1. Python version check
    passed, error = _check_python_version()
    if not passed:
        errors.append(error)
    
    # 2. Write access checks for persistence paths
    for path, purpose in _get_persistence_paths(settings):
        passed, error = _check_write_access(path, purpose)
        if not passed:
            errors.append(error)
    
    # 3. Temp directory check (always needed)
    temp_dir = tempfile.gettempdir()
    passed, error = _check_write_access(temp_dir, "temp directory")
    if not passed:
        errors.append(error)
    
    # 4. API key check
    passed, error = _check_api_key(settings)
    if not passed:
        errors.append(error)
    
    # 5. Rate limit config check
    passed, warning, error = _check_rate_limit_config(settings)
    if not passed:
        errors.append(error)
    elif warning:
        warnings.append(warning)
    
    # 6. Environment-specific warnings
    if settings.env == "prod" and not settings.api_key:
        warnings.append("Running in prod without API key authentication")
    
    return StartupCheckResult(
        ok=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )


def format_check_result(result: StartupCheckResult) -> str:
    """Format check result for console output.
    
    Returns a multi-line string with checkmarks and X marks suitable
    for Rich console output.
    """
    lines = ["Verifily Startup Checks"]
    lines.append("")
    
    # Build list of check items with their status
    checks: List[tuple[str, str]] = []
    
    # Add errors as failures
    for error in result.errors:
        checks.append(("error", error))
    
    # Add warnings
    for warning in result.warnings:
        checks.append(("warning", warning))
    
    # If no issues, show success
    if result.ok and not result.warnings:
        checks.append(("ok", "All startup checks passed"))
    
    # Format each check
    for status, message in checks:
        if status == "ok":
            lines.append(f"[green]✓[/green] {message}")
        elif status == "warning":
            lines.append(f"[yellow]⚠[/yellow] {message}")
        else:  # error
            lines.append(f"[red]✗[/red] {message}")
    
    return "\n".join(lines)
