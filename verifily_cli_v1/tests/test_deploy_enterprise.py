"""Tests for enterprise deployment features.

Target: ~15 tests, runtime <0.5s
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Runtime paths tests
from verifily_cli_v1.core.runtime_paths import (
    RuntimePaths,
    get_runtime_paths,
    get_verifily_home,
    get_store_dir,
    get_logs_dir,
    get_runs_dir,
    get_temp_dir,
    reset_runtime_paths,
)

# Deploy config tests
from verifily_cli_v1.core.deploy_config import (
    DeployConfig,
    ServerConfig,
    AuthConfig,
    PersistenceConfig,
    LimitsConfig,
    load_deploy_config,
    validate_deploy_config,
    apply_env_overrides,
    ConfigError,
    reset_deploy_config,
)

# Backup/restore tests
from verifily_cli_v1.core.backup_restore import (
    create_backup,
    restore_backup,
    verify_backup,
    BackupError,
    RestoreError,
)


@pytest.fixture(autouse=True)
def reset_state() -> Generator[None, None, None]:
    """Reset singletons before each test."""
    reset_runtime_paths()
    reset_deploy_config()
    yield
    reset_runtime_paths()
    reset_deploy_config()


@pytest.fixture
def temp_home(monkeypatch) -> Generator[Path, None, None]:
    """Create a temporary VERIFILY_HOME."""
    with tempfile.TemporaryDirectory() as tmpdir:
        home = Path(tmpdir)
        monkeypatch.setenv("VERIFILY_HOME", str(home))
        yield home


class TestRuntimePaths:
    """Test runtime paths module."""

    def test_default_home(self, monkeypatch) -> None:
        """Default home is /tmp/verifily."""
        monkeypatch.delenv("VERIFILY_HOME", raising=False)
        reset_runtime_paths()
        paths = get_runtime_paths()
        assert str(paths.home) == "/tmp/verifily"

    def test_custom_home_via_env(self, temp_home) -> None:
        """Custom home via VERIFILY_HOME env var."""
        paths = get_runtime_paths()
        assert paths.home == temp_home

    def test_ensure_directories_creates_all(self, temp_home) -> None:
        """ensure_directories creates all subdirectories."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        assert paths.store.exists()
        assert paths.logs.exists()
        assert paths.runs.exists()
        assert paths.temp.exists()

    def test_validate_writable_passes(self, temp_home) -> None:
        """validate_writable returns empty list when all writable."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        errors = paths.validate_writable()
        assert errors == []

    def test_component_specific_paths(self, temp_home) -> None:
        """Component-specific paths are under store."""
        paths = get_runtime_paths()
        
        assert paths.get_usage_log() == temp_home / "store" / "usage_events.jsonl"
        assert paths.get_jobs_log() == temp_home / "store" / "jobs_events.jsonl"
        assert paths.get_monitor_log() == temp_home / "store" / "monitor_events.jsonl"
        assert paths.get_workspaces_store() == temp_home / "store" / "workspaces.jsonl"

    def test_override_via_env(self, temp_home, monkeypatch) -> None:
        """Individual paths can be overridden via env vars."""
        custom_usage = temp_home / "custom" / "usage.jsonl"
        monkeypatch.setenv("VERIFILY_USAGE_LOG_PATH", str(custom_usage))
        
        paths = get_runtime_paths()
        assert paths.get_usage_log() == custom_usage

    def test_describe_paths(self, temp_home) -> None:
        """describe_paths returns all paths as strings."""
        paths = get_runtime_paths()
        desc = paths.describe_paths()
        
        assert "home" in desc
        assert "store" in desc
        assert "logs" in desc
        assert "usage_log" in desc
        assert all(isinstance(v, str) for v in desc.values())

    def test_is_default_home(self, monkeypatch) -> None:
        """is_default_home returns True for /tmp/verifily."""
        monkeypatch.setenv("VERIFILY_HOME", "/tmp/verifily")
        reset_runtime_paths()
        paths = get_runtime_paths()
        assert paths.is_default_home() is True

    def test_is_not_default_home(self, temp_home) -> None:
        """is_default_home returns False for custom home."""
        paths = get_runtime_paths()
        assert paths.is_default_home() is False


class TestDeployConfig:
    """Test deploy config loading and validation."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = DeployConfig()
        
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8080
        assert config.server.log_format == "text"
        assert config.auth.enabled is False
        assert config.persistence.usage is False

    def test_from_dict(self) -> None:
        """Config can be loaded from dictionary."""
        data = {
            "server": {"host": "0.0.0.0", "port": 9000},
            "auth": {"enabled": True, "api_key": "secret123"},
        }
        config = DeployConfig.from_dict(data)
        
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9000
        assert config.auth.enabled is True
        assert config.auth.api_key == "secret123"

    def test_validate_valid_config(self) -> None:
        """Valid config passes validation."""
        config = DeployConfig()
        is_valid, errors = validate_deploy_config(config)
        assert is_valid is True
        assert errors == []

    def test_validate_invalid_port(self) -> None:
        """Invalid port fails validation."""
        config = DeployConfig(server=ServerConfig(port=99999))
        is_valid, errors = validate_deploy_config(config)
        assert is_valid is False
        assert any("port" in e.lower() for e in errors)

    def test_validate_invalid_log_format(self) -> None:
        """Invalid log format fails validation."""
        config = DeployConfig(server=ServerConfig(log_format="xml"))
        is_valid, errors = validate_deploy_config(config)
        assert is_valid is False
        assert any("log_format" in e.lower() for e in errors)

    def test_validate_nonlocal_without_override(self) -> None:
        """Non-localhost binding without allow_nonlocal fails."""
        config = DeployConfig(server=ServerConfig(host="0.0.0.0", allow_nonlocal=False))
        is_valid, errors = validate_deploy_config(config)
        assert is_valid is False
        assert any("localhost" in e.lower() for e in errors)

    def test_validate_auth_without_key_or_workspaces(self) -> None:
        """Auth enabled without key or workspaces fails."""
        config = DeployConfig(auth=AuthConfig(enabled=True))
        is_valid, errors = validate_deploy_config(config)
        assert is_valid is False
        assert any("api_key" in e.lower() or "workspaces" in e.lower() for e in errors)

    def test_validate_workspaces_without_salt(self) -> None:
        """Workspaces enabled without key_salt fails."""
        config = DeployConfig(
            auth=AuthConfig(enabled=True, workspaces_enabled=True)
        )
        is_valid, errors = validate_deploy_config(config)
        assert is_valid is False
        assert any("key_salt" in e.lower() for e in errors)

    def test_env_override_server(self, monkeypatch) -> None:
        """Environment variables override config."""
        monkeypatch.setenv("VERIFILY_BIND", "0.0.0.0")
        monkeypatch.setenv("VERIFILY_PORT", "9090")
        monkeypatch.setenv("VERIFILY_LOG_FORMAT", "json")
        
        config = DeployConfig()
        config = apply_env_overrides(config)
        
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9090
        assert config.server.log_format == "json"

    def test_env_override_auth(self, monkeypatch) -> None:
        """Environment variables override auth config."""
        monkeypatch.setenv("VERIFILY_API_KEY", "test-key-123")
        
        config = DeployConfig()
        config = apply_env_overrides(config)
        
        assert config.auth.api_key == "test-key-123"
        assert config.auth.enabled is True

    def test_redacted_dict_masks_secrets(self) -> None:
        """redacted_dict masks secret values."""
        config = DeployConfig(
            auth=AuthConfig(
                enabled=True,
                api_key="secret",
                bootstrap_token="token",
                key_salt="salt",
            )
        )
        redacted = config.auth.redacted_dict()
        
        assert redacted["api_key"] == "***"
        assert redacted["bootstrap_token"] == "***"
        assert redacted["key_salt"] == "***"

    def test_is_production_like(self) -> None:
        """is_production_like detects prod configuration."""
        # Not prod-like
        dev_config = DeployConfig()
        assert dev_config.is_production_like() is False
        
        # Prod-like
        prod_config = DeployConfig(
            auth=AuthConfig(enabled=True, api_key="key"),
            server=ServerConfig(enable_docs=False, log_format="json"),
        )
        assert prod_config.is_production_like() is True


class TestBackupRestore:
    """Test backup and restore functionality."""

    def test_create_backup_empty(self, temp_home) -> None:
        """Backup can be created even with no data."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        backup_path = temp_home / "backup.tar.gz"
        result = create_backup(backup_path)
        
        assert result["success"] is True
        assert backup_path.exists()
        assert result["files_backed_up"] == 0  # No files exist yet

    def test_create_backup_with_data(self, temp_home) -> None:
        """Backup includes existing data files."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        # Create some data
        usage_log = paths.get_usage_log()
        usage_log.parent.mkdir(parents=True, exist_ok=True)
        usage_log.write_text('{"event": "test"}\n')
        
        backup_path = temp_home / "backup.tar.gz"
        result = create_backup(backup_path)
        
        assert result["success"] is True
        assert result["files_backed_up"] >= 1
        assert result["total_bytes"] > 0

    def test_verify_backup_valid(self, temp_home) -> None:
        """Valid backup passes verification."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        usage_log = paths.get_usage_log()
        usage_log.write_text('{"event": "test"}\n')
        
        backup_path = temp_home / "backup.tar.gz"
        create_backup(backup_path)
        
        is_valid, manifest = verify_backup(backup_path)
        assert is_valid is True
        assert manifest is not None
        assert "version" in manifest

    def test_verify_backup_invalid(self, temp_home) -> None:
        """Corrupted backup fails verification."""
        backup_path = temp_home / "invalid.tar.gz"
        backup_path.write_bytes(b"not a valid tar.gz")
        
        is_valid, manifest = verify_backup(backup_path)
        assert is_valid is False

    def test_restore_requires_force(self, temp_home) -> None:
        """Restore requires --force if files exist."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        # Create backup
        usage_log = paths.get_usage_log()
        usage_log.write_text('{"event": "original"}\n')
        backup_path = temp_home / "backup.tar.gz"
        create_backup(backup_path)
        
        # Modify file
        usage_log.write_text('{"event": "modified"}\n')
        
        # Restore without force should fail
        with pytest.raises(RestoreError) as exc_info:
            restore_backup(backup_path, force=False)
        assert "--force" in str(exc_info.value) or "exists" in str(exc_info.value).lower()

    def test_restore_with_force(self, temp_home) -> None:
        """Restore with --force overwrites files."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        # Create backup
        usage_log = paths.get_usage_log()
        usage_log.write_text('{"event": "original"}\n')
        backup_path = temp_home / "backup.tar.gz"
        create_backup(backup_path)
        
        # Modify file
        usage_log.write_text('{"event": "modified"}\n')
        
        # Restore with force
        result = restore_backup(backup_path, force=True)
        
        assert result["success"] is True
        assert result["files_restored"] >= 1
        
        # Verify content restored
        content = usage_log.read_text()
        assert "original" in content

    def test_backup_manifest_structure(self, temp_home) -> None:
        """Backup manifest has correct structure."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        usage_log = paths.get_usage_log()
        usage_log.write_text('{"event": "test"}\n')
        
        backup_path = temp_home / "backup.tar.gz"
        result = create_backup(backup_path)
        manifest = result["manifest"]
        
        assert "version" in manifest
        assert "created_at" in manifest
        assert "verifily_version" in manifest
        assert "files" in manifest
        
        for file_info in manifest["files"]:
            assert "path" in file_info
            assert "sha256" in file_info
            assert "size" in file_info


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_backup_restore_cycle(self, temp_home) -> None:
        """Complete backup and restore cycle works."""
        paths = get_runtime_paths()
        paths.ensure_directories()
        
        # Create data
        usage_log = paths.get_usage_log()
        usage_log.write_text('{"event": "test1"}\n{"event": "test2"}\n')
        
        jobs_log = paths.get_jobs_log()
        jobs_log.write_text('{"job": "test"}\n')
        
        # Backup
        backup_path = temp_home / "full_backup.tar.gz"
        create_backup(backup_path)
        
        # Wipe
        usage_log.unlink()
        jobs_log.unlink()
        
        # Restore
        result = restore_backup(backup_path, force=True)
        
        assert result["success"] is True
        assert result["files_restored"] >= 2
        assert usage_log.exists()
        assert jobs_log.exists()
        
        # Verify content
        usage_content = usage_log.read_text()
        assert "test1" in usage_content
        assert "test2" in usage_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
