"""Tests for adoption hooks: badges, bundles, pre-commit, GitHub Action."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from verifily_cli_v1.cli import app
from verifily_cli_v1.commands.badge import (
    BADGE_COLORS,
    badge_from_run,
    generate_badge,
)
from verifily_cli_v1.commands.bundle import create_bundle

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Badge mapping ────────────────────────────────────────────────


class TestBadgeMapping:
    def test_ship_badge(self):
        b = generate_badge("SHIP")
        assert b["color"] == "brightgreen"
        assert b["message"] == "SHIP"

    def test_dont_ship_badge(self):
        b = generate_badge("DONT_SHIP")
        assert b["color"] == "red"

    def test_investigate_badge(self):
        b = generate_badge("INVESTIGATE")
        assert b["color"] == "yellow"

    def test_contract_fail_badge(self):
        b = generate_badge("CONTRACT_FAIL")
        assert b["color"] == "orange"

    def test_tool_error_badge(self):
        b = generate_badge("TOOL_ERROR")
        assert b["color"] == "orange"

    def test_unknown_badge(self):
        b = generate_badge("SOMETHING_ELSE")
        assert b["color"] == "lightgrey"

    def test_badge_schema_version(self):
        b = generate_badge("SHIP")
        assert b["schemaVersion"] == 1

    def test_badge_label_default(self):
        b = generate_badge("SHIP")
        assert b["label"] == "verifily"

    def test_badge_custom_label(self):
        b = generate_badge("SHIP", label="my-project")
        assert b["label"] == "my-project"


# ── Badge from run ───────────────────────────────────────────────


class TestBadgeFromRun:
    def test_reads_decision_summary(self, tmp_path):
        (tmp_path / "decision_summary.json").write_text(
            json.dumps({"recommendation": "SHIP", "exit_code": 0})
        )
        b = badge_from_run(str(tmp_path))
        assert b["message"] == "SHIP"
        assert b["color"] == "brightgreen"

    def test_reads_pipeline_result_fallback(self, tmp_path):
        (tmp_path / "pipeline_result.json").write_text(
            json.dumps({"decision": {"recommendation": "DONT_SHIP", "exit_code": 1}})
        )
        b = badge_from_run(str(tmp_path))
        assert b["message"] == "DONT_SHIP"
        assert b["color"] == "red"

    def test_prefers_decision_summary(self, tmp_path):
        (tmp_path / "decision_summary.json").write_text(
            json.dumps({"recommendation": "SHIP"})
        )
        (tmp_path / "pipeline_result.json").write_text(
            json.dumps({"decision": {"recommendation": "DONT_SHIP"}})
        )
        b = badge_from_run(str(tmp_path))
        assert b["message"] == "SHIP"

    def test_missing_artifacts_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            badge_from_run(str(tmp_path))


# ── Badge CLI ────────────────────────────────────────────────────


class TestBadgeCLI:
    def test_badge_decision_flag(self):
        result = runner.invoke(app, ["badge", "--decision", "SHIP", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["message"] == "SHIP"
        assert data["color"] == "brightgreen"

    def test_badge_writes_file(self, tmp_path):
        out = str(tmp_path / "badge.json")
        result = runner.invoke(app, ["badge", "--decision", "INVESTIGATE", "--out", out, "--json"])
        assert result.exit_code == 0
        data = json.loads((tmp_path / "badge.json").read_text())
        assert data["message"] == "INVESTIGATE"

    def test_badge_from_run(self, tmp_path):
        (tmp_path / "decision_summary.json").write_text(
            json.dumps({"recommendation": "DONT_SHIP"})
        )
        result = runner.invoke(app, ["badge", "--from-run", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["message"] == "DONT_SHIP"

    def test_badge_no_args_errors(self):
        result = runner.invoke(app, ["badge"])
        assert result.exit_code != 0

    def test_badge_both_args_errors(self):
        result = runner.invoke(app, ["badge", "--decision", "SHIP", "--from-run", "/tmp"])
        assert result.exit_code != 0


# ── Bundle ───────────────────────────────────────────────────────


class TestBundle:
    def _make_run(self, tmp_path):
        """Create a minimal run directory."""
        run = tmp_path / "run"
        run.mkdir()
        (run / "decision_summary.json").write_text(
            json.dumps({"recommendation": "SHIP", "exit_code": 0})
        )
        (run / "hashes.json").write_text(json.dumps({"files": {}, "chain_hash": "abc"}))
        (run / "config.yaml").write_text("task: sft\n")
        return run

    def test_excludes_dataset_jsonl(self, tmp_path):
        run = self._make_run(tmp_path)
        (run / "dataset.jsonl").write_text('{"text": "secret data"}\n')
        out = str(tmp_path / "bundle.zip")
        result = create_bundle(str(run), out)
        assert "dataset.jsonl" not in result["files_included"]
        assert "dataset.jsonl" in result["files_excluded"]
        with zipfile.ZipFile(out) as zf:
            assert "dataset.jsonl" not in zf.namelist()

    def test_excludes_csv(self, tmp_path):
        run = self._make_run(tmp_path)
        (run / "data.csv").write_text("a,b\n1,2\n")
        out = str(tmp_path / "bundle.zip")
        result = create_bundle(str(run), out)
        assert "data.csv" in result["files_excluded"]

    def test_excludes_model_weights(self, tmp_path):
        run = self._make_run(tmp_path)
        (run / "model.pt").write_bytes(b"\x00" * 10)
        (run / "model.safetensors").write_bytes(b"\x00" * 10)
        out = str(tmp_path / "bundle.zip")
        result = create_bundle(str(run), out)
        assert "model.pt" in result["files_excluded"]
        assert "model.safetensors" in result["files_excluded"]

    def test_includes_manifest(self, tmp_path):
        run = self._make_run(tmp_path)
        out = str(tmp_path / "bundle.zip")
        create_bundle(str(run), out)
        with zipfile.ZipFile(out) as zf:
            assert "bundle_manifest.json" in zf.namelist()

    def test_manifest_has_sha256(self, tmp_path):
        run = self._make_run(tmp_path)
        out = str(tmp_path / "bundle.zip")
        result = create_bundle(str(run), out)
        manifest = result["manifest"]
        assert "files" in manifest
        for name, sha in manifest["files"].items():
            assert len(sha) == 64  # SHA-256 hex length

    def test_includes_eval_results(self, tmp_path):
        run = self._make_run(tmp_path)
        eval_dir = run / "eval"
        eval_dir.mkdir()
        (eval_dir / "eval_results.json").write_text(json.dumps({"f1": 0.72}))
        out = str(tmp_path / "bundle.zip")
        result = create_bundle(str(run), out)
        assert "eval/eval_results.json" in result["files_included"]

    def test_missing_run_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            create_bundle(str(tmp_path / "nonexistent"), str(tmp_path / "out.zip"))


# ── Bundle CLI ───────────────────────────────────────────────────


class TestBundleCLI:
    def test_bundle_creates_zip(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "decision_summary.json").write_text(json.dumps({"recommendation": "SHIP"}))
        out = str(tmp_path / "out.zip")
        result = runner.invoke(app, ["bundle", "--run", str(run), "--out", out])
        assert result.exit_code == 0
        assert Path(out).exists()

    def test_bundle_json_output(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "hashes.json").write_text(json.dumps({"files": {}}))
        out = str(tmp_path / "out.zip")
        result = runner.invoke(app, ["bundle", "--run", str(run), "--out", out, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "files" in data


# ── Pre-commit script ────────────────────────────────────────────


class TestPrecommitScript:
    def test_script_exists(self):
        assert (REPO_ROOT / "scripts" / "precommit_verifily.sh").is_file()

    def test_script_has_shebang(self):
        text = (REPO_ROOT / "scripts" / "precommit_verifily.sh").read_text()
        assert text.startswith("#!/usr/bin/env bash")

    def test_script_references_verifily(self):
        text = (REPO_ROOT / "scripts" / "precommit_verifily.sh").read_text()
        assert "verifily pipeline" in text


# ── GitHub Action ────────────────────────────────────────────────


class TestGitHubAction:
    def test_action_yml_exists(self):
        assert (REPO_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml").is_file()

    def test_action_has_inputs(self):
        path = REPO_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        data = yaml.safe_load(path.read_text())
        inputs = data["inputs"]
        assert "config-path" in inputs
        assert "project-path" in inputs
        assert "fail-on" in inputs
        assert "api-key" in inputs
        assert "plan" in inputs

    def test_action_has_outputs(self):
        path = REPO_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        data = yaml.safe_load(path.read_text())
        outputs = data["outputs"]
        assert "exit-code" in outputs
        assert "decision" in outputs

    def test_action_is_composite(self):
        path = REPO_ROOT / ".github" / "actions" / "verifily-gate" / "action.yml"
        data = yaml.safe_load(path.read_text())
        assert data["runs"]["using"] == "composite"
