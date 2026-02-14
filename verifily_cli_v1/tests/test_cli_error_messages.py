"""Tests for CLI error messages — user-facing UX polish."""

from __future__ import annotations

from typer.testing import CliRunner

from verifily_cli_v1.cli import app

runner = CliRunner()


class TestPipelineErrors:
    def test_missing_config_shows_hint(self):
        result = runner.invoke(app, ["pipeline", "--config", "/tmp/nonexistent_verifily.yaml"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output
        assert "quickstart" in result.output.lower()

    def test_missing_config_suggests_quickstart(self):
        result = runner.invoke(app, ["pipeline", "--config", "/tmp/no_such_file.yaml"])
        assert "quickstart" in result.output


class TestReportErrors:
    def test_missing_dataset_shows_hint(self):
        result = runner.invoke(app, ["report", "--dataset", "/tmp/nonexistent_data.jsonl"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Error" in result.output
        assert "ingest" in result.output.lower()


class TestNoArgsHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "verifily" in result.output.lower()
        # Should show quick start or commands
        assert "pipeline" in result.output or "quickstart" in result.output
