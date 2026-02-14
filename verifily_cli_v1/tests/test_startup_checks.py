"""Tests for startup environment validation.

These tests verify that the startup checks correctly identify
environment issues that would prevent safe server operation.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from verifily_cli_v1.core.api.settings import Settings
from verifily_cli_v1.core.api.startup_checks import (
    StartupCheckResult,
    _check_api_key,
    _check_python_version,
    _check_rate_limit_config,
    _check_write_access,
    format_check_result,
    run_startup_checks,
)


class TestPythonVersionCheck:
    """Tests for Python version validation."""

    def test_python_version_ok(self):
        """Should pass when Python >= 3.9."""
        # Current Python should pass
        passed, error = _check_python_version()
        assert passed is True
        assert error is None

    def test_python_version_too_old(self):
        """Should fail when Python < 3.9."""
        # Create a mock version_info that behaves like the real one
        class MockVersionInfo:
            major = 3
            minor = 8
            micro = 0
            releaselevel = "final"
            serial = 0
            def __getitem__(self, i):
                return (3, 8, 0, "final", 0)[i]
            def __ge__(self, other):
                return (3, 8) >= other
        
        with patch.object(sys, "version_info", MockVersionInfo()):
            passed, error = _check_python_version()
            assert passed is False
            assert "3.8" in error
            assert "3.9" in error

    def test_python_version_3_9_boundary(self):
        """Should pass at exactly Python 3.9."""
        # Create a mock version_info that behaves like the real one
        class MockVersionInfo:
            major = 3
            minor = 9
            micro = 0
            releaselevel = "final"
            serial = 0
            def __getitem__(self, i):
                return (3, 9, 0, "final", 0)[i]
            def __ge__(self, other):
                return (3, 9) >= other
        
        with patch.object(sys, "version_info", MockVersionInfo()):
            passed, error = _check_python_version()
            assert passed is True
            assert error is None


class TestWriteAccessCheck:
    """Tests for write permission validation."""

    def test_write_access_ok(self, tmp_path: Path):
        """Should pass for writable directory."""
        passed, error = _check_write_access(str(tmp_path), "test directory")
        assert passed is True
        assert error is None

    def test_write_access_creates_directory(self, tmp_path: Path):
        """Should create directory if it doesn't exist."""
        nested = tmp_path / "nested" / "path"
        assert not nested.exists()
        
        passed, error = _check_write_access(str(nested), "nested directory")
        assert passed is True
        assert nested.exists()

    def test_write_access_no_permission(self, tmp_path: Path):
        """Should use temp fallback for read-only directory."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # Remove write permission
        
        try:
            passed, error = _check_write_access(str(readonly_dir), "readonly directory")
            # With fallback, should pass but with temp path
            assert passed is True
            # Error now contains warning about using temp fallback
            assert "temp fallback" in error.lower() or "fallback" in error.lower()
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(stat.S_IRWXU)

    def test_write_access_nonexistent_parent(self, tmp_path: Path):
        """Should use temp fallback when path cannot be created."""
        # Try to create a path that's impossible (file exists where dir should be)
        blocking_file = tmp_path / "blocking_file"
        blocking_file.write_text("block")
        
        # Try to create directory inside the file path
        impossible_path = blocking_file / "subdirectory"
        passed, error = _check_write_access(str(impossible_path), "impossible path")
        
        # With fallback, should pass but use temp directory
        assert passed is True
        assert "fallback" in error.lower() or "temp" in error.lower()


class TestApiKeyCheck:
    """Tests for API key validation."""

    def test_api_key_not_set(self):
        """Should pass when VERIFILY_API_KEY is not set."""
        settings = Settings(api_key="")
        with patch.dict(os.environ, {}, clear=True):
            passed, error = _check_api_key(settings)
            assert passed is True
            assert error is None

    def test_api_key_valid(self):
        """Should pass when VERIFILY_API_KEY is set and non-empty."""
        settings = Settings(api_key="secret123")
        with patch.dict(os.environ, {"VERIFILY_API_KEY": "secret123"}):
            passed, error = _check_api_key(settings)
            assert passed is True
            assert error is None

    def test_api_key_empty_string(self):
        """Should fail when VERIFILY_API_KEY is set but empty."""
        settings = Settings(api_key="")
        with patch.dict(os.environ, {"VERIFILY_API_KEY": ""}):
            passed, error = _check_api_key(settings)
            assert passed is False
            assert "empty" in error.lower()

    def test_api_key_whitespace_only(self):
        """Should fail when VERIFILY_API_KEY is whitespace only."""
        settings = Settings(api_key="")
        with patch.dict(os.environ, {"VERIFILY_API_KEY": "   \t\n  "}):
            passed, error = _check_api_key(settings)
            assert passed is False
            assert "empty" in error.lower()


class TestRateLimitConfigCheck:
    """Tests for rate limit configuration validation."""

    def test_rate_limit_zero_means_disabled(self):
        """Should warn but pass when RPM is 0 (disabled)."""
        settings = Settings(rate_limit_rpm=0)
        passed, warning, error = _check_rate_limit_config(settings)
        assert passed is True
        assert warning is not None
        assert "not configured" in warning.lower()
        assert error is None

    def test_rate_limit_positive(self):
        """Should pass for positive RPM values."""
        settings = Settings(rate_limit_rpm=100)
        passed, warning, error = _check_rate_limit_config(settings)
        assert passed is True
        assert warning is None
        assert error is None

    def test_rate_limit_negative(self):
        """Should fail for negative RPM values."""
        settings = Settings(rate_limit_rpm=-1)
        passed, warning, error = _check_rate_limit_config(settings)
        assert passed is False
        assert warning is None
        assert "must be >= 0" in error


class TestRunStartupChecks:
    """Integration tests for the full startup checks."""

    def test_valid_config_passes(self, tmp_path: Path):
        """Should pass with valid configuration."""
        settings = Settings(
            data_dir=str(tmp_path),
            rate_limit_rpm=100,
        )
        
        with patch.dict(os.environ, {}, clear=True):
            result = run_startup_checks(settings)
            
        assert result.ok is True
        assert len(result.errors) == 0

    def test_missing_persistence_path_uses_fallback(self, tmp_path: Path):
        """Should use temp fallback when persistence path is not writable."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # Remove write permission
        
        settings = Settings(
            data_dir=str(readonly_dir),
            usage_persist=True,
        )
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                result = run_startup_checks(settings)
            
            # With fallback, should pass but may have warnings
            assert result.ok is True
        finally:
            readonly_dir.chmod(stat.S_IRWXU)

    def test_empty_api_key_fails(self, tmp_path: Path):
        """Should fail when VERIFILY_API_KEY is set but empty."""
        settings = Settings(
            data_dir=str(tmp_path),
        )
        
        with patch.dict(os.environ, {"VERIFILY_API_KEY": ""}):
            result = run_startup_checks(settings)
        
        assert result.ok is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_prod_without_api_key_warns(self, tmp_path: Path):
        """Should warn when running in prod without API key."""
        settings = Settings(
            data_dir=str(tmp_path),
            env="prod",
            api_key="",
        )
        
        with patch.dict(os.environ, {}, clear=True):
            result = run_startup_checks(settings)
        
        assert result.ok is True  # Not an error
        assert any("prod without api key" in w.lower() for w in result.warnings)

    def test_persistence_paths_checked(self, tmp_path: Path):
        """Should check all persistence paths when enabled."""
        settings = Settings(
            data_dir=str(tmp_path),
            usage_persist=True,
            jobs_persist=True,
        )
        
        with patch.dict(os.environ, {}, clear=True):
            result = run_startup_checks(settings)
        
        assert result.ok is True


class TestFormatCheckResult:
    """Tests for result formatting."""

    def test_format_success(self):
        """Should format successful result."""
        result = StartupCheckResult(ok=True, warnings=[], errors=[])
        output = format_check_result(result)
        assert "Startup Checks" in output
        assert "✓" in output or "All startup checks passed" in output

    def test_format_with_warnings(self):
        """Should format result with warnings."""
        result = StartupCheckResult(
            ok=True,
            warnings=["Warning 1", "Warning 2"],
            errors=[]
        )
        output = format_check_result(result)
        assert "Startup Checks" in output
        assert "Warning 1" in output
        assert "Warning 2" in output

    def test_format_with_errors(self):
        """Should format result with errors."""
        result = StartupCheckResult(
            ok=False,
            warnings=[],
            errors=["Error 1", "Error 2"]
        )
        output = format_check_result(result)
        assert "Startup Checks" in output
        assert "Error 1" in output
        assert "Error 2" in output
        assert "✗" in output or "error" in output.lower()


class TestStartupCheckResultModel:
    """Tests for the Pydantic model."""

    def test_model_creation(self):
        """Should create valid model instance."""
        result = StartupCheckResult(ok=True, warnings=[], errors=[])
        assert result.ok is True
        assert result.warnings == []
        assert result.errors == []

    def test_model_with_data(self):
        """Should handle warnings and errors."""
        result = StartupCheckResult(
            ok=False,
            warnings=["warning"],
            errors=["error"]
        )
        assert result.ok is False
        assert "warning" in result.warnings
        assert "error" in result.errors

    def test_model_serialization(self):
        """Should serialize to dict correctly."""
        result = StartupCheckResult(ok=True, warnings=["w"], errors=[])
        data = result.model_dump()
        assert data["ok"] is True
        assert data["warnings"] == ["w"]
        assert data["errors"] == []
