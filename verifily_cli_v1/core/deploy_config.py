"""Unified deployment configuration for Verifily (Helm-lite).

Supports YAML config file with environment variable overrides.
Provides validation with helpful error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ConfigError(Exception):
    """Configuration error with helpful message."""
    pass


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8080
    allow_nonlocal: bool = False
    log_format: str = "text"
    enable_docs: bool = True
    
    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []
        
        if self.port < 1 or self.port > 65535:
            errors.append(f"server.port must be 1-65535, got {self.port}")
        
        if self.log_format not in ("text", "json"):
            errors.append(f"server.log_format must be 'text' or 'json', got {self.log_format}")
        
        # Safety check for non-local binding
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if self.host not in local_hosts and not self.allow_nonlocal:
            errors.append(
                f"server.host '{self.host}' is not localhost. "
                "Set server.allow_nonlocal=true or use a reverse proxy."
            )
        
        return errors


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enabled: bool = False
    api_key: Optional[str] = None
    workspaces_enabled: bool = False
    bootstrap_token: Optional[str] = None
    key_salt: Optional[str] = None
    
    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []
        
        if self.enabled:
            if not self.api_key and not self.workspaces_enabled:
                errors.append(
                    "auth.enabled=true requires auth.api_key or auth.workspaces_enabled=true"
                )
            
            if self.workspaces_enabled and not self.key_salt:
                errors.append(
                    "auth.workspaces_enabled=true requires auth.key_salt for secure hashing\n"
                    "  Generate: openssl rand -hex 32"
                )
        
        return errors
    
    def redacted_dict(self) -> Dict[str, Any]:
        """Return dict with secrets masked."""
        return {
            "enabled": self.enabled,
            "api_key": "***" if self.api_key else None,
            "workspaces_enabled": self.workspaces_enabled,
            "bootstrap_token": "***" if self.bootstrap_token else None,
            "key_salt": "***" if self.key_salt else None,
        }


@dataclass
class PersistenceConfig:
    """Persistence configuration."""
    usage: bool = False
    jobs: bool = False
    monitor: bool = False
    workspaces: bool = False
    
    def any_enabled(self) -> bool:
        """Check if any persistence is enabled."""
        return self.usage or self.jobs or self.monitor or self.workspaces


@dataclass
class LimitsConfig:
    """Resource limits configuration."""
    rate_limit_rpm: int = 0  # 0 = unlimited
    max_body_bytes: int = 10_000_000  # 10MB
    
    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []
        
        if self.rate_limit_rpm < 0:
            errors.append(f"limits.rate_limit_rpm must be >= 0, got {self.rate_limit_rpm}")
        
        if self.max_body_bytes < 1024:
            errors.append(f"limits.max_body_bytes must be >= 1024, got {self.max_body_bytes}")
        
        return errors


@dataclass
class DeployConfig:
    """Complete deployment configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployConfig":
        """Create config from dictionary."""
        server_data = data.get("server", {})
        auth_data = data.get("auth", {})
        persistence_data = data.get("persistence", {})
        limits_data = data.get("limits", {})
        
        return cls(
            server=ServerConfig(**server_data),
            auth=AuthConfig(**auth_data),
            persistence=PersistenceConfig(**persistence_data),
            limits=LimitsConfig(**limits_data),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "allow_nonlocal": self.server.allow_nonlocal,
                "log_format": self.server.log_format,
                "enable_docs": self.server.enable_docs,
            },
            "auth": self.auth.redacted_dict(),
            "persistence": {
                "usage": self.persistence.usage,
                "jobs": self.persistence.jobs,
                "monitor": self.persistence.monitor,
                "workspaces": self.persistence.workspaces,
            },
            "limits": {
                "rate_limit_rpm": self.limits.rate_limit_rpm,
                "max_body_bytes": self.limits.max_body_bytes,
            },
        }
    
    def validate(self) -> list[str]:
        """Validate entire configuration."""
        errors = []
        errors.extend(self.server.validate())
        errors.extend(self.auth.validate())
        errors.extend(self.limits.validate())
        return errors
    
    def is_production_like(self) -> bool:
        """Check if config appears production-like."""
        return (
            self.auth.enabled and
            not self.server.enable_docs and
            self.server.log_format == "json"
        )


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    if not HAS_YAML:
        raise ConfigError(
            "YAML support not available. Install with: pip install pyyaml\n"
            "Or use environment variables only."
        )
    
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    
    try:
        content = path.read_text()
        data = yaml.safe_load(content)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config file must contain a YAML object, got {type(data).__name__}")
        return data
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")


def apply_env_overrides(config: DeployConfig) -> DeployConfig:
    """Apply environment variable overrides to config.
    
    Env vars take precedence over file config.
    """
    # Server overrides
    if host := os.environ.get("VERIFILY_BIND"):
        config.server.host = host
    if port := os.environ.get("VERIFILY_PORT"):
        config.server.port = int(port)
    if os.environ.get("VERIFILY_ALLOW_NONLOCAL"):
        config.server.allow_nonlocal = os.environ["VERIFILY_ALLOW_NONLOCAL"] in ("1", "true", "True")
    if log_format := os.environ.get("VERIFILY_LOG_FORMAT"):
        config.server.log_format = log_format
    if os.environ.get("VERIFILY_ENABLE_DOCS"):
        config.server.enable_docs = os.environ["VERIFILY_ENABLE_DOCS"] in ("1", "true", "True")
    
    # Auth overrides
    if os.environ.get("VERIFILY_AUTH_ENABLED"):
        config.auth.enabled = os.environ["VERIFILY_AUTH_ENABLED"] in ("1", "true", "True")
    if api_key := os.environ.get("VERIFILY_API_KEY"):
        config.auth.api_key = api_key
        config.auth.enabled = True
    if os.environ.get("VERIFILY_WORKSPACES_ENABLED"):
        config.auth.workspaces_enabled = os.environ["VERIFILY_WORKSPACES_ENABLED"] in ("1", "true", "True")
    if bootstrap_token := os.environ.get("VERIFILY_BOOTSTRAP_TOKEN"):
        config.auth.bootstrap_token = bootstrap_token
    if key_salt := os.environ.get("VERIFILY_KEY_SALT"):
        config.auth.key_salt = key_salt
    
    # Persistence overrides
    if os.environ.get("VERIFILY_USAGE_PERSIST"):
        config.persistence.usage = os.environ["VERIFILY_USAGE_PERSIST"] in ("1", "true", "True")
    if os.environ.get("VERIFILY_JOBS_PERSIST"):
        config.persistence.jobs = os.environ["VERIFILY_JOBS_PERSIST"] in ("1", "true", "True")
    if os.environ.get("VERIFILY_MONITOR_PERSIST"):
        config.persistence.monitor = os.environ["VERIFILY_MONITOR_PERSIST"] in ("1", "true", "True")
    if os.environ.get("VERIFILY_WORKSPACES_PERSIST"):
        config.persistence.workspaces = os.environ["VERIFILY_WORKSPACES_PERSIST"] in ("1", "true", "True")
    
    # Limits overrides
    if rate_limit := os.environ.get("VERIFILY_RATE_LIMIT_RPM"):
        config.limits.rate_limit_rpm = int(rate_limit)
    if max_body := os.environ.get("VERIFILY_MAX_BODY_BYTES"):
        config.limits.max_body_bytes = int(max_body)
    
    return config


def load_deploy_config(config_path: Optional[Union[str, Path]] = None) -> DeployConfig:
    """Load deployment configuration.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file (YAML)
    3. Defaults
    
    Args:
        config_path: Path to YAML config file (default from VERIFILY_CONFIG_PATH env var)
        
    Returns:
        DeployConfig instance
        
    Raises:
        ConfigError: If config is invalid
    """
    # Determine config file path
    if config_path is None:
        config_path = os.environ.get("VERIFILY_CONFIG_PATH")
    
    # Load from file if specified
    if config_path:
        path = Path(config_path)
        data = load_yaml_config(path)
        config = DeployConfig.from_dict(data)
    else:
        config = DeployConfig()
    
    # Apply environment overrides
    config = apply_env_overrides(config)
    
    return config


def validate_deploy_config(config: DeployConfig) -> tuple[bool, list[str]]:
    """Validate deployment configuration.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = config.validate()
    
    # Additional validation
    if config.auth.workspaces_enabled and config.auth.key_salt:
        # Check key_salt length (should be hex, 32 bytes = 64 chars)
        if len(config.auth.key_salt) < 32:
            errors.append(
                "auth.key_salt should be at least 32 characters for security\n"
                "  Generate: openssl rand -hex 32"
            )
    
    return len(errors) == 0, errors


# Singleton instance
_deploy_config: Optional[DeployConfig] = None


def get_deploy_config() -> DeployConfig:
    """Get or load the deployment config singleton."""
    global _deploy_config
    if _deploy_config is None:
        _deploy_config = load_deploy_config()
    return _deploy_config


def reset_deploy_config() -> None:
    """Reset deploy config (for testing)."""
    global _deploy_config
    _deploy_config = None
