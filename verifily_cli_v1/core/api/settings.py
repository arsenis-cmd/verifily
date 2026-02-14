"""Centralized server settings for Verifily API.

Reads environment variables with sensible defaults. Distinguishes dev vs prod
mode. Never exposes secrets in repr or serialization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bool_env(key: str, default: bool) -> bool:
    """Parse a 0/1 env var to bool."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip() in ("1", "true", "yes", "True", "TRUE")


def _int_env(key: str, default: int) -> int:
    """Parse an int env var with fallback."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _str_list_env(key: str) -> List[str]:
    """Parse comma-separated env var to list."""
    val = os.environ.get(key, "")
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    """Immutable server configuration. Safe to log — secrets are masked."""

    # ── Core ───────────────────────────────────────────────────────
    env: str = "dev"
    bind: str = "127.0.0.1"
    port: int = 8080
    allow_nonlocal: bool = False
    enable_docs: bool = True
    trust_proxy: bool = False
    
    # ── Security ───────────────────────────────────────────────────
    api_key: str = ""
    auth_mode: str = "simple"
    key_salt: str = ""
    
    # ── Rate Limiting ──────────────────────────────────────────────
    rate_limit_rpm: int = 0
    
    # ── Persistence ────────────────────────────────────────────────
    usage_persist: bool = False
    jobs_persist: bool = False
    data_dir: str = "/data"
    
    # ── Logging ────────────────────────────────────────────────────
    log_format: str = "text"
    
    # ── Features ───────────────────────────────────────────────────
    enable_admin: bool = False
    auth_persist: bool = False
    enable_billing: bool = False
    billing_persist: bool = False
    teams_enabled: bool = False
    teams_persist: bool = False
    
    # ── Quotas ─────────────────────────────────────────────────────
    quota_req_per_day: int = 2000
    quota_rows_per_day: int = 2_000_000
    quota_bytes_per_day: int = 200_000_000
    
    # ── Stripe ──────────────────────────────────────────────────────
    stripe_enabled: bool = False
    billing_enforce: bool = False
    subs_persist: bool = False

    # ── Billing Enforcement ──────────────────────────────────────
    default_plan: str = "FREE"

    # ── Teams/Admin ────────────────────────────────────────────────
    super_admin_key: str = ""

    # ── Enterprise Security ─────────────────────────────────────
    enterprise_security: bool = False
    token_secret: str = ""

    # ── Workspaces ───────────────────────────────────────────────
    workspaces_enabled: bool = False
    bootstrap_token: str = ""
    workspaces_store_path: str = ""

    def __repr__(self) -> str:
        masked_key = "***" if self.api_key else ""
        return (
            f"Settings(env={self.env!r}, bind={self.bind!r}, port={self.port}, "
            f"allow_nonlocal={self.allow_nonlocal}, enable_docs={self.enable_docs}, "
            f"api_key={'***' if self.api_key else ''!r}, "
            f"rate_limit_rpm={self.rate_limit_rpm}, "
            f"usage_persist={self.usage_persist}, jobs_persist={self.jobs_persist}, "
            f"data_dir={self.data_dir!r}, log_format={self.log_format!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict with api_key masked."""
        return {
            "env": self.env,
            "bind": self.bind,
            "port": self.port,
            "allow_nonlocal": self.allow_nonlocal,
            "enable_docs": self.enable_docs,
            "trust_proxy": self.trust_proxy,
            "api_key": "configured" if self.api_key else "not set",
            "rate_limit_rpm": self.rate_limit_rpm,
            "usage_persist": self.usage_persist,
            "jobs_persist": self.jobs_persist,
            "data_dir": self.data_dir,
            "log_format": self.log_format,
            "auth_mode": self.auth_mode,
            "key_salt": "configured" if self.key_salt else "not set",
            "enable_admin": self.enable_admin,
            "auth_persist": self.auth_persist,
            "quota_req_per_day": self.quota_req_per_day,
            "quota_rows_per_day": self.quota_rows_per_day,
            "quota_bytes_per_day": self.quota_bytes_per_day,
            "enable_billing": self.enable_billing,
            "billing_persist": self.billing_persist,
            "teams_enabled": self.teams_enabled,
            "super_admin_key": "configured" if self.super_admin_key else "not set",
            "teams_persist": self.teams_persist,
            "stripe_enabled": self.stripe_enabled,
            "billing_enforce": self.billing_enforce,
            "subs_persist": self.subs_persist,
            "default_plan": self.default_plan,
            "enterprise_security": self.enterprise_security,
            "token_secret": "configured" if self.token_secret else "not set",
            "workspaces_enabled": self.workspaces_enabled,
            "bootstrap_token": "configured" if self.bootstrap_token else "not set",
            "workspaces_store_path": self.workspaces_store_path or "not set",
        }


def load_settings(
    bind: Optional[str] = None,
    port: Optional[int] = None,
    allow_nonlocal: Optional[bool] = None,
    **overrides: Any,
) -> Settings:
    """Load settings from environment with optional overrides.

    Args:
        bind: Override bind host
        port: Override port
        allow_nonlocal: Override nonlocal binding check
        **overrides: Additional field overrides

    Returns:
        Settings instance
    """
    settings = Settings(
        env=os.environ.get("VERIFILY_ENV", "dev"),
        bind=os.environ.get("VERIFILY_BIND", "127.0.0.1"),
        port=_int_env("VERIFILY_PORT", 8080),
        allow_nonlocal=_bool_env("VERIFILY_ALLOW_NONLOCAL", False),
        enable_docs=_bool_env(
            "VERIFILY_ENABLE_DOCS",
            False if overrides.get("env", os.environ.get("VERIFILY_ENV", "dev")) == "prod" else True,
        ),
        trust_proxy=_bool_env("VERIFILY_TRUST_PROXY", False),
        api_key=os.environ.get("VERIFILY_API_KEY", ""),
        auth_mode=os.environ.get("VERIFILY_AUTH_MODE", "simple"),
        key_salt=os.environ.get("VERIFILY_KEY_SALT", ""),
        rate_limit_rpm=_int_env("VERIFILY_RATE_LIMIT_RPM", 0),
        usage_persist=_bool_env("VERIFILY_USAGE_PERSIST", False),
        jobs_persist=_bool_env("VERIFILY_JOBS_PERSIST", False),
        data_dir=os.environ.get("VERIFILY_DATA_DIR", "/data"),
        log_format=os.environ.get("VERIFILY_LOG_FORMAT", "text"),
        enable_admin=_bool_env("VERIFILY_ENABLE_ADMIN", False),
        auth_persist=_bool_env("VERIFILY_AUTH_PERSIST", False),
        enable_billing=_bool_env("VERIFILY_ENABLE_BILLING", False),
        billing_persist=_bool_env("VERIFILY_BILLING_PERSIST", False),
        teams_enabled=_bool_env("VERIFILY_TEAMS_ENABLED", False),
        teams_persist=_bool_env("VERIFILY_TEAMS_PERSIST", False),
        quota_req_per_day=_int_env("VERIFILY_QUOTA_REQ_PER_DAY", 2000),
        quota_rows_per_day=_int_env("VERIFILY_QUOTA_ROWS_PER_DAY", 2_000_000),
        quota_bytes_per_day=_int_env("VERIFILY_QUOTA_BYTES_PER_DAY", 200_000_000),
        super_admin_key=os.environ.get("VERIFILY_SUPER_ADMIN_KEY", ""),
        stripe_enabled=_bool_env("VERIFILY_STRIPE_ENABLED", False),
        billing_enforce=_bool_env("VERIFILY_BILLING_ENFORCE", False),
        subs_persist=_bool_env("VERIFILY_SUBS_PERSIST", False),
        default_plan=os.environ.get("VERIFILY_DEFAULT_PLAN", "FREE"),
        enterprise_security=_bool_env("VERIFILY_ENTERPRISE_SECURITY", False),
        token_secret=os.environ.get("VERIFILY_TOKEN_SECRET", ""),
        workspaces_enabled=_bool_env("VERIFILY_WORKSPACES_ENABLED", False),
        bootstrap_token=os.environ.get("VERIFILY_BOOTSTRAP_TOKEN", ""),
        workspaces_store_path=os.environ.get("VERIFILY_WORKSPACES_STORE_PATH", ""),
    )

    # Apply function overrides
    if bind is not None:
        settings = Settings(**{**settings.to_dict(), "bind": bind})
    if port is not None:
        settings = Settings(**{**settings.to_dict(), "port": port})
    if allow_nonlocal is not None:
        settings = Settings(**{**settings.to_dict(), "allow_nonlocal": allow_nonlocal})

    # Apply any additional overrides
    if overrides:
        current = settings.to_dict()
        current.update(overrides)
        settings = Settings(**current)

    return settings


def validate_host(host: str, allow_nonlocal: bool) -> None:
    """Refuse to bind to non-localhost unless explicitly allowed."""
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in local_hosts and not allow_nonlocal:
        raise ValueError(
            f"Refusing to bind to non-local host '{host}'. "
            f"Verifily API is designed for local use only. "
            f"Pass --allow-nonlocal to override this safety check."
        )


def print_startup_warnings(settings: Settings) -> None:
    """Print warnings about potentially unsafe settings."""
    warnings = []

    if not settings.api_key and not settings.enable_admin:
        warnings.append("No API key configured — server is open to all requests.")

    if settings.allow_nonlocal:
        warnings.append(
            "Non-local binding enabled — ensure you have proper firewall rules."
        )

    if settings.enable_admin and not settings.auth_persist:
        warnings.append("Admin endpoints enabled but auth persistence is off.")

    if settings.enable_billing:
        if not settings.billing_persist:
            warnings.append("Billing enabled but billing persistence is off.")

    if warnings:
        import click

        click.secho("\nWarnings:", fg="yellow")
        for w in warnings:
            click.secho(f"  • {w}", fg="yellow")
        click.echo()


# ── Production Settings (Pydantic-based) ───────────────────────────

class ProductionSettings:
    """Production-focused settings with workspace security.
    
    This class wraps the base Settings and adds production-specific
    configuration like workspace paths, path security, etc.
    """
    
    def __init__(self, base_settings: Optional[Settings] = None) -> None:
        self._base = base_settings or load_settings()
        
        # Production-specific settings
        self.workspace_root = Path(os.environ.get("VERIFILY_WORKSPACE_ROOT", "./workspace"))
        self.allow_abspath = _bool_env("VERIFILY_ALLOW_ABSPATH", False)
        self.prod_mode = _bool_env("VERIFILY_PROD", False)
        self.org_mode = _bool_env("VERIFILY_ORG_MODE", False)
        self.allowed_origins = _str_list_env("VERIFILY_ALLOWED_ORIGINS")
        
        # Persistence paths
        self.usage_log_path = os.environ.get("VERIFILY_USAGE_LOG_PATH")
        self.jobs_log_path = os.environ.get("VERIFILY_JOBS_LOG_PATH")
        self.org_log_path = os.environ.get("VERIFILY_ORG_LOG_PATH")
    
    @property
    def bind_host(self) -> str:
        return self._base.bind
    
    @property
    def port(self) -> int:
        return self._base.port
    
    @property
    def enable_docs(self) -> bool:
        return self._base.enable_docs
    
    def validate(self) -> None:
        """Validate production settings."""
        errors = []
        
        # Workspace must exist in prod mode
        if self.prod_mode and not self.workspace_root.exists():
            errors.append(
                f"PROD=1 requires workspace: {self.workspace_root}\n"
                f"  Create: mkdir -p {self.workspace_root}"
            )
        
        # Non-local binding requires allow_nonlocal
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if self.bind_host not in local_hosts and not self._base.allow_nonlocal:
            errors.append(
                f"Bind '{self.bind_host}' is not localhost\n"
                f"  Use ALLOW_NONLOCAL=1 or a reverse proxy"
            )
        
        # Prod mode security checks
        if self.prod_mode:
            if self.allow_abspath:
                errors.append("PROD=1 with ALLOW_ABSPATH=1 is insecure")
            if not self._base.api_key and not self.org_mode:
                errors.append("PROD=1 requires auth (API_KEY or ORG_MODE=1)")
        
        if errors:
            raise ValueError("\n".join(["Configuration errors:"] + 
                           [f"  {i+1}. {e}" for i, e in enumerate(errors)]))
    
    def as_safe_dict(self) -> Dict[str, Any]:
        """Return settings with secrets redacted."""
        return {
            "bind_host": self.bind_host,
            "port": self.port,
            "allow_nonlocal": self._base.allow_nonlocal,
            "allowed_origins": self.allowed_origins or "*none*",
            "workspace_root": str(self.workspace_root),
            "allow_abspath": self.allow_abspath,
            "prod_mode": self.prod_mode,
            "org_mode": self.org_mode,
            "auth_enabled": bool(self._base.api_key) or self.org_mode,
            "rate_limit_rpm": self._base.rate_limit_rpm or "unlimited",
            "persistence": {
                "usage": self._base.usage_persist,
                "jobs": self._base.jobs_persist,
            },
            "log_format": self._base.log_format,
            "features": {
                "billing": self._base.enable_billing,
                "admin": self._base.enable_admin,
                "docs": self._base.enable_docs,
            },
            "version": "1.0.0",
        }
    
    def print_startup_info(self) -> None:
        """Print human-readable startup information."""
        mode = "PRODUCTION" if self.prod_mode else "DEVELOPMENT"
        auth = "org_mode" if self.org_mode else ("api_key" if self._base.api_key else "disabled")
        
        print(f"""
┌─────────────────────────────────────────────────────────────┐
│  Verifily Server                                             │
│  Mode: {mode:15}                                 │
├─────────────────────────────────────────────────────────────┤
│  Network:                                                    │
│    Bind: {self.bind_host}:{self.port:5}                           │
├─────────────────────────────────────────────────────────────┤
│  Workspace:                                                  │
│    Root: {str(self.workspace_root):45}│
├─────────────────────────────────────────────────────────────┤
│  Security:                                                   │
│    Auth: {auth:15}                                 │
│    Rate limit: {self._base.rate_limit_rpm or 'unlimited':12}                         │
├─────────────────────────────────────────────────────────────┤
│  Persistence:                                                │
│    Usage: {str(self._base.usage_persist):5}  Jobs: {str(self._base.jobs_persist):5}                        │
├─────────────────────────────────────────────────────────────┤
│  Logs:                                                       │
│    Format: {self._base.log_format:10}                                 │
└─────────────────────────────────────────────────────────────┘
        """.strip())


def load_production_settings() -> ProductionSettings:
    """Load production settings from environment."""
    base = load_settings()
    prod = ProductionSettings(base)
    prod.validate()
    return prod
