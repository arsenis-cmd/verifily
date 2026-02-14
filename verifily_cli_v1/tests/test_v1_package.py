"""Tests for v1 packaging: version alignment, doctor command, and install smoke."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from verifily_cli_v1 import __version__ as cli_version
from verifily_cli_v1.cli import app
from verifily_cli_v1.core.env import (
    EnvCheck,
    check_binary,
    check_package,
    check_platform,
    check_python_version,
    check_ssl,
    check_tmp_write,
    check_verifily_version,
    check_write_permission,
)

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Version alignment ────────────────────────────────────────────


class TestVersionAlignment:
    def test_cli_version_matches_version_file(self):
        version_file = REPO_ROOT / "VERSION"
        if version_file.exists():
            assert cli_version == version_file.read_text().strip()

    def test_cli_version_matches_pyproject(self):
        pyproject = REPO_ROOT / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text()
            for line in text.splitlines():
                if line.strip().startswith("version") and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val[0].isdigit():
                        assert cli_version == val
                        break

    def test_sdk_version_matches_cli(self):
        # SDK may be importable as verifily_sdk or via nested path
        try:
            from verifily_sdk import __version__ as sdk_version
        except (ImportError, AttributeError):
            # Fallback: read version from source file directly
            sdk_init = REPO_ROOT / "verifily_sdk" / "verifily_sdk" / "__init__.py"
            sdk_version = None
            for line in sdk_init.read_text().splitlines():
                if line.startswith("__version__"):
                    sdk_version = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
            assert sdk_version is not None, "Could not find SDK version"
        assert sdk_version == cli_version

    def test_version_is_semver(self):
        parts = cli_version.split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()

    def test_cli_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert cli_version in result.output

    def test_version_command(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert cli_version in result.output

    def test_health_includes_version(self):
        from fastapi.testclient import TestClient
        from verifily_cli_v1.core.api.server import create_app
        client = TestClient(create_app())
        resp = client.get("/health")
        assert resp.json()["version"] == cli_version


# ── Doctor: env checks ───────────────────────────────────────────


class TestEnvChecks:
    def test_python_version_pass(self):
        c = check_python_version("3.9")
        assert c.status == "PASS"

    def test_python_version_fail(self):
        c = check_python_version("99.0")
        assert c.status == "FAIL"

    def test_platform_always_pass(self):
        c = check_platform()
        assert c.status == "PASS"

    def test_verifily_version_reports(self):
        c = check_verifily_version()
        assert c.status == "PASS"
        assert cli_version in c.detail

    def test_check_package_installed(self):
        c = check_package("rich", required=True)
        assert c.status == "PASS"

    def test_check_package_missing_required(self):
        c = check_package("nonexistent_pkg_xyz", required=True)
        assert c.status == "FAIL"

    def test_check_package_missing_optional(self):
        c = check_package("nonexistent_pkg_xyz", required=False)
        assert c.status == "WARN"

    def test_check_package_with_import_name(self):
        c = check_package("pyyaml", import_name="yaml", required=True)
        assert c.status == "PASS"

    def test_tmp_write_pass(self):
        c = check_tmp_write()
        assert c.status == "PASS"

    def test_write_permission_fail(self):
        c = check_write_permission("/nonexistent_root_path_xyz", "test")
        assert c.status == "FAIL"

    def test_check_binary_missing(self):
        c = check_binary("nonexistent_binary_xyz_123")
        assert c.status == "WARN"

    def test_ssl_check(self):
        c = check_ssl()
        assert c.status in ("PASS", "WARN")


# ── Doctor: CLI command ──────────────────────────────────────────


class TestDoctorCommand:
    def test_doctor_exits_success(self):
        """Doctor should exit 0 or 2 in normal test environment."""
        result = runner.invoke(app, ["doctor"])
        # 0=healthy, 2=warnings (e.g. optional binaries missing)
        assert result.exit_code in (0, 2)

    def test_doctor_shows_verifily_version(self):
        result = runner.invoke(app, ["doctor"])
        assert cli_version in result.output

    def test_doctor_shows_python(self):
        result = runner.invoke(app, ["doctor"])
        assert "Python" in result.output

    def test_doctor_shows_platform(self):
        result = runner.invoke(app, ["doctor"])
        assert "Platform" in result.output

    def test_doctor_shows_packages(self):
        result = runner.invoke(app, ["doctor"])
        assert "typer" in result.output
        assert "rich" in result.output

    def test_doctor_with_valid_config(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("key: value\n")
        result = runner.invoke(app, ["doctor", "--config", str(cfg)])
        assert result.exit_code in (0, 2)
        assert "test.yaml" in result.output

    def test_doctor_with_missing_config(self):
        result = runner.invoke(app, ["doctor", "--config", "/tmp/nonexistent_config_xyz.yaml"])
        assert result.exit_code == 3
        assert "FAIL" in result.output

    def test_doctor_fatal_on_unwritable_dir(self):
        """Simulate an unwritable directory causing FAIL."""
        from verifily_cli_v1.commands import doctor as doctor_mod

        def fake_checks():
            return [
                EnvCheck(name="Test", status="PASS", detail="ok"),
                EnvCheck(name="Write: test", status="FAIL", detail="Permission denied"),
            ]

        with mock.patch.object(doctor_mod, "run_all_checks", fake_checks):
            exit_code = doctor_mod.run()
            assert exit_code == 3

    def test_doctor_warnings_exit_2(self):
        """Doctor with only warnings should exit 2."""
        from verifily_cli_v1.commands import doctor as doctor_mod

        def only_pass_and_warn():
            return [
                EnvCheck(name="Test", status="PASS", detail="ok"),
                EnvCheck(name="Optional", status="WARN", detail="missing"),
            ]

        with mock.patch.object(doctor_mod, "run_all_checks", only_pass_and_warn):
            exit_code = doctor_mod.run()
            assert exit_code == 2


# ── Docker files ─────────────────────────────────────────────────


class TestDockerFiles:
    def test_dockerfile_exists(self):
        assert (REPO_ROOT / "docker" / "Dockerfile").is_file()

    def test_compose_exists(self):
        assert (REPO_ROOT / "docker-compose.yml").is_file()

    def test_dockerfile_exposes_8080(self):
        text = (REPO_ROOT / "docker" / "Dockerfile").read_text()
        assert "EXPOSE 8080" in text

    def test_dockerfile_has_healthcheck(self):
        text = (REPO_ROOT / "docker" / "Dockerfile").read_text()
        assert "HEALTHCHECK" in text

    def test_compose_maps_8080(self):
        text = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "8080:8080" in text


# ── Release docs ─────────────────────────────────────────────────


class TestReleaseDocs:
    def test_release_checklist_exists(self):
        assert (REPO_ROOT / "docs" / "release_checklist.md").is_file()

    def test_stability_contract_exists(self):
        assert (REPO_ROOT / "docs" / "stability_contract.md").is_file()

    def test_docker_production_exists(self):
        assert (REPO_ROOT / "docs" / "docker_production.md").is_file()

    def test_stability_contract_lists_exit_codes(self):
        text = (REPO_ROOT / "docs" / "stability_contract.md").read_text()
        assert "SHIP" in text
        assert "DONT_SHIP" in text
        assert "TOOL_ERROR" in text
