"""Runtime paths for Verifily enterprise deployment.

Determines all writable locations based on VERIFILY_HOME.
Provides standardized directory structure for persistence, logs, and runs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("verifily.runtime_paths")


class RuntimePaths:
    """Manages Verifily runtime directory structure.
    
    Directory layout under VERIFILY_HOME (default: /tmp/verifily):
    
    $VERIFILY_HOME/
    ├── store/          # Persistent data stores (JSONL files)
    ├── logs/           # Application logs
    ├── runs/           # Pipeline run artifacts (optional)
    └── temp/           # Temporary files
    
    Each component can override its specific path via env vars:
    - VERIFILY_WORKSPACES_STORE_PATH
    - VERIFILY_USAGE_LOG_PATH
    - VERIFILY_JOBS_LOG_PATH
    - VERIFILY_MONITOR_LOG_PATH
    """
    
    DEFAULT_HOME = "/tmp/verifily"
    
    def __init__(self, home: Optional[str] = None) -> None:
        """Initialize runtime paths.
        
        Args:
            home: Base directory (default from VERIFILY_HOME env var or /tmp/verifily)
        """
        self._home = Path(home or os.environ.get("VERIFILY_HOME", self.DEFAULT_HOME))
        self._store: Optional[Path] = None
        self._logs: Optional[Path] = None
        self._runs: Optional[Path] = None
        self._temp: Optional[Path] = None
    
    @property
    def home(self) -> Path:
        """Base Verifily home directory."""
        return self._home
    
    def ensure_directories(self) -> None:
        """Create all runtime directories if they don't exist."""
        # Create with appropriate permissions (user read/write/execute)
        self.home.mkdir(parents=True, exist_ok=True)
        self.store.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)
        logger.debug("Runtime directories ensured: %s", self.home)
    
    @property
    def store(self) -> Path:
        """Store directory for persistent data files."""
        if self._store is None:
            override = os.environ.get("VERIFILY_STORE_PATH")
            self._store = Path(override) if override else self.home / "store"
        return self._store
    
    @property
    def logs(self) -> Path:
        """Logs directory for application logs."""
        if self._logs is None:
            override = os.environ.get("VERIFILY_LOGS_PATH")
            self._logs = Path(override) if override else self.home / "logs"
        return self._logs
    
    @property
    def runs(self) -> Path:
        """Runs directory for pipeline artifacts."""
        if self._runs is None:
            override = os.environ.get("VERIFILY_RUNS_PATH")
            self._runs = Path(override) if override else self.home / "runs"
        return self._runs
    
    @property
    def temp(self) -> Path:
        """Temp directory for temporary files."""
        if self._temp is None:
            override = os.environ.get("VERIFILY_TEMP_PATH")
            self._temp = Path(override) if override else self.home / "temp"
        return self._temp
    
    # Component-specific paths
    
    def get_workspaces_store(self) -> Path:
        """Path for workspaces persistence file."""
        override = os.environ.get("VERIFILY_WORKSPACES_STORE_PATH")
        if override:
            return Path(override)
        return self.store / "workspaces.jsonl"
    
    def get_usage_log(self) -> Path:
        """Path for usage events log."""
        override = os.environ.get("VERIFILY_USAGE_LOG_PATH")
        if override:
            return Path(override)
        return self.store / "usage_events.jsonl"
    
    def get_jobs_log(self) -> Path:
        """Path for job events log."""
        override = os.environ.get("VERIFILY_JOBS_LOG_PATH")
        if override:
            return Path(override)
        return self.store / "jobs_events.jsonl"
    
    def get_monitor_log(self) -> Path:
        """Path for monitor events log."""
        override = os.environ.get("VERIFILY_MONITOR_LOG_PATH")
        if override:
            return Path(override)
        return self.store / "monitor_events.jsonl"
    
    def describe_paths(self) -> Dict[str, str]:
        """Return a dict describing all paths (for diagnostics)."""
        return {
            "home": str(self.home),
            "store": str(self.store),
            "logs": str(self.logs),
            "runs": str(self.runs),
            "temp": str(self.temp),
            "workspaces_store": str(self.get_workspaces_store()),
            "usage_log": str(self.get_usage_log()),
            "jobs_log": str(self.get_jobs_log()),
            "monitor_log": str(self.get_monitor_log()),
        }
    
    def validate_writable(self) -> list[str]:
        """Validate that all directories are writable.
        
        Returns:
            List of error messages (empty if all OK)
        """
        errors = []
        
        for name, path in [
            ("home", self.home),
            ("store", self.store),
            ("logs", self.logs),
            ("runs", self.runs),
            ("temp", self.temp),
        ]:
            try:
                # Try to create directory and write a test file
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except PermissionError as e:
                errors.append(f"{name} ({path}): Permission denied - {e}")
            except OSError as e:
                errors.append(f"{name} ({path}): {e}")
        
        return errors
    
    def is_default_home(self) -> bool:
        """Check if using default /tmp location."""
        return str(self.home) == self.DEFAULT_HOME


# Singleton instance
_runtime_paths: Optional[RuntimePaths] = None


def get_runtime_paths() -> RuntimePaths:
    """Get or create the runtime paths singleton."""
    global _runtime_paths
    if _runtime_paths is None:
        _runtime_paths = RuntimePaths()
    return _runtime_paths


def reset_runtime_paths() -> None:
    """Reset runtime paths (for testing)."""
    global _runtime_paths
    _runtime_paths = None


def get_verifily_home() -> Path:
    """Convenience function: get Verifily home directory."""
    return get_runtime_paths().home


def get_store_dir() -> Path:
    """Convenience function: get store directory."""
    return get_runtime_paths().store


def get_logs_dir() -> Path:
    """Convenience function: get logs directory."""
    return get_runtime_paths().logs


def get_runs_dir() -> Path:
    """Convenience function: get runs directory."""
    return get_runtime_paths().runs


def get_temp_dir() -> Path:
    """Convenience function: get temp directory."""
    return get_runtime_paths().temp
