"""Tests for verifily ci-init command."""

from __future__ import annotations

from pathlib import Path

import pytest

from verifily_cli_v1.commands.ci_init import generate_ci


class TestCIInitGitHub:
    def test_creates_workflow_file(self, tmp_path):
        result = generate_ci("github", str(tmp_path))
        path = Path(result["file_path"])
        assert path.exists()
        assert path.name == "verifily.yml"
        assert ".github/workflows" in str(path)

    def test_workflow_contains_key_lines(self, tmp_path):
        result = generate_ci("github", str(tmp_path))
        content = Path(result["file_path"]).read_text()
        assert "pipeline --ci" in content
        assert "tee verifily_result.json" in content
        assert "upload-artifact" in content
        assert "if: failure()" in content

    def test_force_overwrites(self, tmp_path):
        generate_ci("github", str(tmp_path))
        # Without force: should fail
        with pytest.raises(FileExistsError):
            generate_ci("github", str(tmp_path), force=False)
        # With force: should succeed
        result = generate_ci("github", str(tmp_path), force=True)
        assert Path(result["file_path"]).exists()


class TestCIInitGitLab:
    def test_creates_gitlab_file(self, tmp_path):
        result = generate_ci("gitlab", str(tmp_path))
        path = Path(result["file_path"])
        assert path.exists()
        assert path.name == ".gitlab-ci.yml"

    def test_gitlab_contains_key_lines(self, tmp_path):
        result = generate_ci("gitlab", str(tmp_path))
        content = Path(result["file_path"]).read_text()
        assert "pipeline --ci" in content
        assert "tee verifily_result.json" in content
        assert "when: always" in content
        assert "artifacts" in content
        assert "expire_in" in content


class TestCIInitErrors:
    def test_unknown_provider_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown CI provider"):
            generate_ci("jenkins", str(tmp_path))

    def test_result_has_expected_keys(self, tmp_path):
        result = generate_ci("github", str(tmp_path))
        assert "provider" in result
        assert "file_path" in result
        assert "relative_path" in result
        assert "project_root" in result
        assert result["provider"] == "github"
