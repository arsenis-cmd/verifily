"""Tests for verifily quickstart command."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from verifily_cli_v1.commands.quickstart import scaffold


class TestQuickstartScaffold:
    def test_creates_all_files(self, tmp_path):
        result = scaffold(str(tmp_path / "qs"), force=False)
        project = Path(result["project_dir"])

        assert (project / "verifily.yaml").exists()
        assert (project / "data" / "raw" / "sample.csv").exists()
        assert (project / "data" / "raw" / "eval.jsonl").exists()
        assert (project / "runs" / "baseline" / "config.yaml").exists()
        assert (project / "runs" / "baseline" / "environment.json").exists()
        assert (project / "runs" / "baseline" / "run_meta.json").exists()
        assert (project / "runs" / "baseline" / "eval" / "eval_results.json").exists()
        assert (project / "runs" / "baseline" / "hashes.json").exists()
        assert (project / "scripts" / "run_demo.sh").exists()
        assert (project / "README.md").exists()

    def test_verifily_yaml_valid(self, tmp_path):
        result = scaffold(str(tmp_path / "qs"))
        project = Path(result["project_dir"])

        cfg = yaml.safe_load((project / "verifily.yaml").read_text())
        assert "ship_if" in cfg
        assert cfg["ship_if"]["min_f1"] >= 0
        assert cfg["schema"] == "sft"

    def test_baseline_passes_contract(self, tmp_path):
        from verifily_cli_v1.commands.contract_check import validate_run_contract

        result = scaffold(str(tmp_path / "qs"))
        project = Path(result["project_dir"])

        contract = validate_run_contract(str(project / "runs" / "baseline"))
        assert contract["valid"] is True
        assert contract["has_eval"] is True

    def test_force_overwrites(self, tmp_path):
        target = str(tmp_path / "qs")
        scaffold(target)
        # Second call without force should fail
        with pytest.raises(FileExistsError):
            scaffold(target, force=False)
        # With force should succeed
        result = scaffold(target, force=True)
        assert Path(result["project_dir"]).exists()

    def test_result_has_expected_keys(self, tmp_path):
        result = scaffold(str(tmp_path / "qs"))
        assert "project_dir" in result
        assert "created_paths" in result
        assert "next_steps" in result
        assert len(result["created_paths"]) >= 8
        assert len(result["next_steps"]) == 3

    def test_sample_csv_no_pii(self, tmp_path):
        result = scaffold(str(tmp_path / "qs"))
        project = Path(result["project_dir"])
        csv_text = (project / "data" / "raw" / "sample.csv").read_text()

        # No email, phone, SSN patterns
        assert not re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", csv_text)
        assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", csv_text)

    def test_run_demo_script_is_executable(self, tmp_path):
        import stat
        result = scaffold(str(tmp_path / "qs"))
        project = Path(result["project_dir"])
        script = project / "scripts" / "run_demo.sh"
        assert script.stat().st_mode & stat.S_IEXEC

    def test_run_demo_script_has_key_commands(self, tmp_path):
        result = scaffold(str(tmp_path / "qs"))
        project = Path(result["project_dir"])
        script_text = (project / "scripts" / "run_demo.sh").read_text()
        assert "ingest" in script_text
        assert "pipeline" in script_text
        assert "--ci" in script_text
        assert "ALL PASSED" in script_text
