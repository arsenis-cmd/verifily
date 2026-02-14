"""Tests for CI action templates.

Verifies that CI action templates are valid YAML and have required fields.
Target: ~6 tests, runtime <0.3s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestGitHubActionTemplate:
    """Test GitHub Action template validity."""

    def test_action_yml_exists(self) -> None:
        """action.yml file exists."""
        action_path = PROJECT_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        assert action_path.exists()

    def test_action_yml_is_valid_yaml(self) -> None:
        """action.yml is valid YAML."""
        action_path = PROJECT_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        content = action_path.read_text()
        data = yaml.safe_load(content)
        assert data is not None

    def test_action_has_required_fields(self) -> None:
        """action.yml has all required fields."""
        action_path = PROJECT_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        data = yaml.safe_load(action_path.read_text())
        
        assert "name" in data
        assert "description" in data
        assert "inputs" in data
        assert "outputs" in data
        assert "runs" in data

    def test_action_has_expected_inputs(self) -> None:
        """action.yml has expected input fields."""
        action_path = PROJECT_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        data = yaml.safe_load(action_path.read_text())
        
        inputs = data.get("inputs", {})
        expected_inputs = ["config-path", "fail-on", "timeout"]
        for inp in expected_inputs:
            assert inp in inputs, f"Missing input: {inp}"

    def test_action_has_expected_outputs(self) -> None:
        """action.yml has expected output fields."""
        action_path = PROJECT_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        data = yaml.safe_load(action_path.read_text())
        
        outputs = data.get("outputs", {})
        expected_outputs = ["decision", "exit-code"]
        for outp in expected_outputs:
            assert outp in outputs, f"Missing output: {outp}"


class TestCIDockerfile:
    """Test CI Dockerfile validity."""

    def test_dockerfile_exists(self) -> None:
        """Dockerfile exists."""
        dockerfile_path = PROJECT_ROOT / "docker" / "verifily-ci.Dockerfile"
        assert dockerfile_path.exists()

    def test_dockerfile_has_required_elements(self) -> None:
        """Dockerfile has required elements."""
        dockerfile_path = PROJECT_ROOT / "docker" / "verifily-ci.Dockerfile"
        content = dockerfile_path.read_text()
        
        assert "FROM" in content
        assert "verifily" in content.lower()
        assert "LABEL" in content


class TestCIExamples:
    """Test CI example files validity."""

    def test_github_actions_example_exists(self) -> None:
        """GitHub Actions example exists."""
        example_path = PROJECT_ROOT / "examples" / "ci" / "github-actions.yml"
        assert example_path.exists()

    def test_gitlab_ci_example_exists(self) -> None:
        """GitLab CI example exists."""
        example_path = PROJECT_ROOT / "examples" / "ci" / "gitlab-ci.yml"
        assert example_path.exists()

    def test_azure_pipelines_example_exists(self) -> None:
        """Azure Pipelines example exists."""
        example_path = PROJECT_ROOT / "examples" / "ci" / "azure-pipelines.yml"
        assert example_path.exists()

    def test_github_actions_example_is_valid_yaml(self) -> None:
        """GitHub Actions example is valid YAML."""
        example_path = PROJECT_ROOT / "examples" / "ci" / "github-actions.yml"
        content = example_path.read_text()
        data = yaml.safe_load(content)
        assert data is not None
        assert "jobs" in data or "name" in data

    def test_gitlab_ci_example_is_valid_yaml(self) -> None:
        """GitLab CI example is valid YAML."""
        example_path = PROJECT_ROOT / "examples" / "ci" / "gitlab-ci.yml"
        content = example_path.read_text()
        data = yaml.safe_load(content)
        assert data is not None

    def test_azure_pipelines_example_is_valid_yaml(self) -> None:
        """Azure Pipelines example is valid YAML."""
        example_path = PROJECT_ROOT / "examples" / "ci" / "azure-pipelines.yml"
        content = example_path.read_text()
        data = yaml.safe_load(content)
        assert data is not None


class TestExitCodes:
    """Test exit code documentation."""

    def test_exit_codes_documented(self) -> None:
        """Exit codes are documented in README or docs."""
        readme_path = PROJECT_ROOT / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            # Should mention exit codes
            assert "exit code" in content.lower() or "exit_code" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
