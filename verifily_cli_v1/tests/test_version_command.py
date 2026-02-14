"""Tests for the verifily version command."""

from __future__ import annotations

import platform

from typer.testing import CliRunner

from verifily_cli_v1 import __version__
from verifily_cli_v1.cli import app

runner = CliRunner()


class TestVersionCommand:
    def test_version_prints_version_string(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_prints_python_version(self):
        result = runner.invoke(app, ["version"])
        assert platform.python_version() in result.output

    def test_version_prints_platform(self):
        result = runner.invoke(app, ["version"])
        assert platform.system() in result.output

    def test_version_shows_extras(self):
        result = runner.invoke(app, ["version"])
        # fastapi and httpx are installed in dev, so "api" and "sdk" should show
        assert "api" in result.output
        assert "sdk" in result.output

    def test_version_flag_still_works(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_is_semver(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()

    def test_version_matches_version_file(self):
        from pathlib import Path
        version_file = Path(__file__).resolve().parent.parent.parent / "VERSION"
        if version_file.exists():
            file_version = version_file.read_text().strip()
            assert __version__ == file_version
