"""Tests for workspace security model.

Target: ~20 tests, runtime <0.3s
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from verifily_cli_v1.core.api.workspace import (
    AbsolutePathError,
    OutsideWorkspaceError,
    PathTraversalError,
    WorkspacePath,
    WorkspaceSecurityError,
    _contains_traversal,
    _is_absolute_path,
    dev_mode_path_warning,
    get_datasets_directory,
    get_runs_directory,
    get_temp_directory,
    resolve_workspace_path,
    safe_path_for_api,
    validate_workspace_exists,
)


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


class TestPathValidation:
    """Test low-level path validation functions."""

    def test_contains_traversal_detects_dotdot(self) -> None:
        """Detects .. in paths."""
        assert _contains_traversal("../etc/passwd") is True
        assert _contains_traversal("foo/../../../bar") is True
        assert _contains_traversal("foo/bar/baz") is False
        assert _contains_traversal("foo/..bar") is False  # .. not as component

    def test_contains_traversal_windows_style(self) -> None:
        """Detects Windows-style traversal."""
        assert _contains_traversal("..\\windows\\system32") is True
        assert _contains_traversal("foo\\..\\bar") is True

    def test_is_absolute_unix(self) -> None:
        """Detects Unix absolute paths."""
        assert _is_absolute_path("/etc/passwd") is True
        assert _is_absolute_path("/home/user/file.txt") is True
        assert _is_absolute_path("relative/path") is False
        assert _is_absolute_path("./relative") is False

    def test_is_absolute_windows(self) -> None:
        """Detects Windows absolute paths."""
        assert _is_absolute_path("C:\\Windows\\file.txt") is True
        assert _is_absolute_path("D:\\data") is True
        assert _is_absolute_path("\\\\server\\share") is True
        assert _is_absolute_path("relative\\path") is False


class TestWorkspacePathResolution:
    """Test workspace path resolution."""

    def test_resolve_simple_relative_path(self, temp_workspace: Path) -> None:
        """Can resolve simple relative path."""
        wp = resolve_workspace_path(
            "datasets/v1/data.jsonl",
            workspace_root=temp_workspace,
            allow_absolute=False,
            prod_mode=False,
        )
        assert wp.path == temp_workspace / "datasets" / "v1" / "data.jsonl"
        assert wp.workspace_root == temp_workspace

    def test_resolve_strips_leading_dot_slash(self, temp_workspace: Path) -> None:
        """Strips leading ./ from paths."""
        wp = resolve_workspace_path(
            "./datasets/data.jsonl",
            workspace_root=temp_workspace,
        )
        assert wp.path == temp_workspace / "datasets" / "data.jsonl"

    def test_resolve_blocks_leading_slash_without_flag(self, temp_workspace: Path) -> None:
        """Blocks leading / from paths when allow_absolute=False."""
        with pytest.raises(AbsolutePathError):
            resolve_workspace_path(
                "/datasets/data.jsonl",
                workspace_root=temp_workspace,
                allow_absolute=False,
                prod_mode=False,
            )

    def test_blocks_path_traversal(self, temp_workspace: Path) -> None:
        """Blocks .. path traversal attempts."""
        with pytest.raises(PathTraversalError) as exc_info:
            resolve_workspace_path(
                "../etc/passwd",
                workspace_root=temp_workspace,
            )
        assert "Path traversal" in str(exc_info.value)

    def test_blocks_deep_traversal(self, temp_workspace: Path) -> None:
        """Blocks deep path traversal."""
        with pytest.raises(PathTraversalError):
            resolve_workspace_path(
                "foo/../../../etc/passwd",
                workspace_root=temp_workspace,
            )

    def test_blocks_absolute_in_prod(self, temp_workspace: Path) -> None:
        """Blocks absolute paths in production mode."""
        with pytest.raises(AbsolutePathError) as exc_info:
            resolve_workspace_path(
                "/etc/passwd",
                workspace_root=temp_workspace,
                allow_absolute=False,
                prod_mode=True,
            )
        assert "Absolute paths not allowed" in str(exc_info.value)

    def test_allows_absolute_in_dev_with_flag(self, temp_workspace: Path) -> None:
        """Allows absolute paths in dev mode with flag."""
        # This should strip the leading / and resolve relative
        wp = resolve_workspace_path(
            "/datasets/data.jsonl",
            workspace_root=temp_workspace,
            allow_absolute=True,
            prod_mode=False,
        )
        assert wp.path == temp_workspace / "datasets" / "data.jsonl"

    def test_blocks_escape_via_symlink(self, temp_workspace: Path) -> None:
        """Blocks paths that escape workspace via traversal."""
        # Create a file outside workspace
        outside_file = temp_workspace.parent / "outside.txt"
        outside_file.write_text("secret")

        # Traversal attempt is blocked by rule 1 (PathTraversalError)
        with pytest.raises(PathTraversalError):
            resolve_workspace_path(
                f"../{outside_file.name}",
                workspace_root=temp_workspace,
            )

    def test_workspace_path_methods(self, temp_workspace: Path) -> None:
        """WorkspacePath helper methods work."""
        wp = resolve_workspace_path("test.txt", workspace_root=temp_workspace)
        
        assert wp.exists() is False
        assert wp.relative_to_workspace() == Path("test.txt")
        
        # Create file
        wp.path.parent.mkdir(parents=True, exist_ok=True)
        wp.path.write_text("test")
        
        assert wp.exists() is True


class TestWorkspaceDirectories:
    """Test workspace directory helpers."""

    def test_get_runs_directory(self, temp_workspace: Path, monkeypatch) -> None:
        """Returns runs directory under workspace."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        
        runs_dir = get_runs_directory()
        assert runs_dir == temp_workspace / "runs"
        assert runs_dir.exists() is True

    def test_get_datasets_directory(self, temp_workspace: Path, monkeypatch) -> None:
        """Returns datasets directory under workspace."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        
        datasets_dir = get_datasets_directory()
        assert datasets_dir == temp_workspace / "datasets"
        assert datasets_dir.exists() is True

    def test_get_temp_directory(self, temp_workspace: Path, monkeypatch) -> None:
        """Returns temp directory under workspace."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        
        temp_dir = get_temp_directory()
        assert temp_dir == temp_workspace / "temp"
        assert temp_dir.exists() is True


class TestSafePathForApi:
    """Test API-safe path resolution."""

    def test_safe_path_resolves(self, temp_workspace: Path, monkeypatch) -> None:
        """Returns string path for API use."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        
        path = safe_path_for_api("datasets/test.jsonl")
        assert path == str(temp_workspace / "datasets" / "test.jsonl")

    def test_safe_path_raises_on_error(self, temp_workspace: Path, monkeypatch) -> None:
        """Raises WorkspaceSecurityError on violation."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        
        with pytest.raises(WorkspaceSecurityError) as exc_info:
            safe_path_for_api("../etc/passwd", operation="read")
        assert "Cannot read path" in str(exc_info.value)


class TestValidateWorkspace:
    """Test workspace validation."""

    def test_validates_existing_workspace(self, temp_workspace: Path, monkeypatch) -> None:
        """Passes if workspace exists."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        monkeypatch.setenv("VERIFILY_PROD", "1")
        
        # Should not raise
        validate_workspace_exists()

    def test_creates_workspace_in_dev(self, temp_workspace: Path, monkeypatch) -> None:
        """Auto-creates workspace in dev mode."""
        new_workspace = temp_workspace / "new_workspace"
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(new_workspace))
        monkeypatch.setenv("VERIFILY_PROD", "0")
        
        validate_workspace_exists()
        assert new_workspace.exists() is True

    def test_raises_if_missing_in_prod(self, temp_workspace: Path, monkeypatch) -> None:
        """Raises error if workspace missing in production."""
        missing_workspace = temp_workspace / "missing"
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(missing_workspace))
        monkeypatch.setenv("VERIFILY_PROD", "1")
        
        with pytest.raises(WorkspaceSecurityError) as exc_info:
            validate_workspace_exists()
        assert "Workspace root does not exist" in str(exc_info.value)


class TestDevModeWarnings:
    """Test development mode warnings."""

    def test_warns_on_absolute_in_dev(self, temp_workspace: Path, monkeypatch, caplog) -> None:
        """Logs warning when using absolute paths in dev."""
        monkeypatch.setenv("VERIFILY_WORKSPACE_ROOT", str(temp_workspace))
        monkeypatch.setenv("VERIFILY_PROD", "0")
        
        import logging
        with caplog.at_level(logging.WARNING):
            dev_mode_path_warning("/absolute/path")
        
        assert "DEV MODE" in caplog.text
        assert "absolute path" in caplog.text


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_empty_path(self, temp_workspace: Path) -> None:
        """Handles empty path."""
        wp = resolve_workspace_path("", workspace_root=temp_workspace)
        assert wp.path == temp_workspace

    def test_single_dot(self, temp_workspace: Path) -> None:
        """Handles single dot."""
        wp = resolve_workspace_path(".", workspace_root=temp_workspace)
        assert wp.path == temp_workspace

    def test_very_deep_path(self, temp_workspace: Path) -> None:
        """Handles very deep nested paths."""
        deep_path = "/".join(["level"] * 20) + "/file.txt"
        wp = resolve_workspace_path(deep_path, workspace_root=temp_workspace)
        assert wp.path == temp_workspace / deep_path.replace("/", os.sep)

    def test_special_characters_in_filename(self, temp_workspace: Path) -> None:
        """Handles special characters in filenames."""
        wp = resolve_workspace_path(
            "datasets/file-with_special.chars.jsonl",
            workspace_root=temp_workspace,
        )
        assert "file-with_special.chars.jsonl" in str(wp.path)

    def test_unicode_paths(self, temp_workspace: Path) -> None:
        """Handles unicode in paths."""
        wp = resolve_workspace_path(
            "datasets/日本語/データ.jsonl",
            workspace_root=temp_workspace,
        )
        assert "日本語" in str(wp.path)
        assert "データ.jsonl" in str(wp.path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
