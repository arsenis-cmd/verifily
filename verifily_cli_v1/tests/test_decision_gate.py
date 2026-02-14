"""Tests for decision gate: precedence, confidence, threshold validation, CI output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifily_cli_v1.commands.pipeline import (
    EXIT_CONTRACT_FAIL,
    EXIT_DONT_SHIP,
    EXIT_INVESTIGATE,
    EXIT_SHIP,
    EXIT_TOOL_ERROR,
    make_decision,
    validate_thresholds,
)

# ── Helpers ────────────────────────────────────────────────────────

VALID_CONTRACT = {"valid": True, "checks": [], "has_eval": True}
INVALID_CONTRACT = {
    "valid": False,
    "checks": [
        {"file": "config.yaml", "status": "FAIL", "detail": "missing"},
        {"file": "hashes.json", "status": "FAIL", "detail": "missing"},
    ],
    "has_eval": False,
}

CLEAN_CONTAMINATION = {"status": "PASS", "exact_overlaps": 0, "near_duplicates": 0}
FAIL_CONTAMINATION = {"status": "FAIL", "exact_overlaps": 5, "near_duplicates": 3}
WARN_CONTAMINATION = {"status": "WARN", "exact_overlaps": 0, "near_duplicates": 2}

GOOD_EVAL = {"overall": {"f1": 0.85, "exact_match": 0.75}}
GOOD_BASELINE = {"overall": {"f1": 0.80, "exact_match": 0.70}}

PII_CLEAN_REPORT = {"pii_total_hits": 0, "pii_clean": True, "row_count": 10}
PII_DIRTY_REPORT = {"pii_total_hits": 5, "pii_clean": False, "row_count": 10}

STANDARD_CRITERIA = {"min_f1": 0.65, "min_exact_match": 0.50, "max_pii_hits": 0}


def _decide(**overrides):
    """Call make_decision with sensible defaults, overridable."""
    defaults = dict(
        contract_result=VALID_CONTRACT,
        contamination_result=CLEAN_CONTAMINATION,
        report_result=PII_CLEAN_REPORT,
        eval_results=GOOD_EVAL,
        baseline_results=GOOD_BASELINE,
        ship_criteria=STANDARD_CRITERIA,
    )
    defaults.update(overrides)
    return make_decision(**defaults)


# ── Blocker precedence ─────────────────────────────────────────────

class TestBlockerPrecedence:
    def test_contract_fail_overrides_all(self):
        """Contract invalid → DONT_SHIP exit 3, even with good metrics."""
        d = _decide(
            contract_result=INVALID_CONTRACT,
            contamination_result=FAIL_CONTAMINATION,
        )
        assert d["recommendation"] == "DONT_SHIP"
        assert d["exit_code"] == EXIT_CONTRACT_FAIL
        assert "contract_fail" in d["risk_flags"]
        assert "CONTRACT_INVALID" in d["blocker_precedence"]
        assert "missing_files" in d  # contract fail includes missing files

    def test_pii_violation_blocks(self):
        """PII hits > max → DONT_SHIP even with great metrics."""
        d = _decide(report_result=PII_DIRTY_REPORT)
        assert d["recommendation"] == "DONT_SHIP"
        assert d["exit_code"] == EXIT_DONT_SHIP
        assert "pii_violation" in d["risk_flags"]

    def test_contamination_fail_blocks_despite_good_metrics(self):
        """Contamination FAIL → DONT_SHIP even if F1 is excellent."""
        d = _decide(
            contamination_result=FAIL_CONTAMINATION,
            eval_results={"overall": {"f1": 0.99, "exact_match": 0.95}},
        )
        assert d["recommendation"] == "DONT_SHIP"
        assert d["exit_code"] == EXIT_DONT_SHIP
        assert "contamination_fail" in d["risk_flags"]

    def test_eval_missing_blocks_when_thresholds_set(self):
        """Eval missing with metric thresholds → DONT_SHIP."""
        d = _decide(eval_results=None)
        assert d["recommendation"] == "DONT_SHIP"
        assert "eval_missing" in d["risk_flags"]

    def test_eval_missing_ok_when_no_thresholds(self):
        """Eval missing but no thresholds → SHIP (no blocker)."""
        d = _decide(eval_results=None, ship_criteria={})
        assert d["recommendation"] == "SHIP"
        assert d["exit_code"] == EXIT_SHIP

    def test_f1_below_threshold_blocks(self):
        d = _decide(eval_results={"overall": {"f1": 0.30, "exact_match": 0.80}})
        assert d["recommendation"] == "DONT_SHIP"
        assert "f1_below_threshold" in d["risk_flags"]

    def test_multiple_blockers_all_recorded(self):
        """Multiple blockers produce multiple entries in blocker_precedence."""
        d = _decide(
            contamination_result=FAIL_CONTAMINATION,
            report_result=PII_DIRTY_REPORT,
            eval_results={"overall": {"f1": 0.10, "exact_match": 0.10}},
        )
        assert d["recommendation"] == "DONT_SHIP"
        assert len(d["blocker_precedence"]) >= 3  # PII + contamination + F1

    def test_blocker_precedence_field_present(self):
        """blocker_precedence is always in decision output."""
        d = _decide()
        assert "blocker_precedence" in d


# ── INVESTIGATE semantics ──────────────────────────────────────────

class TestInvestigateSemantics:
    def test_contamination_warn_with_good_metrics_is_investigate(self):
        d = _decide(contamination_result=WARN_CONTAMINATION)
        assert d["recommendation"] == "INVESTIGATE"
        assert d["exit_code"] == EXIT_INVESTIGATE
        assert "contamination_warn" in d["risk_flags"]

    def test_no_risk_flags_is_ship(self):
        d = _decide()
        assert d["recommendation"] == "SHIP"
        assert d["exit_code"] == EXIT_SHIP
        assert len(d["risk_flags"]) == 0


# ── Confidence scoring ─────────────────────────────────────────────

class TestConfidence:
    def test_ship_confidence_above_0_9(self):
        d = _decide()
        assert d["confidence"] >= 0.9

    def test_ship_confidence_at_most_1_0(self):
        d = _decide()
        assert d["confidence"] <= 1.0

    def test_investigate_confidence_0_5_to_0_8(self):
        d = _decide(contamination_result=WARN_CONTAMINATION)
        assert 0.5 <= d["confidence"] <= 0.8

    def test_dont_ship_confidence_0_0_to_0_3(self):
        d = _decide(contamination_result=FAIL_CONTAMINATION)
        assert 0.0 < d["confidence"] <= 0.3

    def test_contract_fail_confidence_1_0(self):
        d = _decide(contract_result=INVALID_CONTRACT)
        assert d["confidence"] == 1.0

    def test_confidence_is_deterministic(self):
        """Same inputs → same confidence every time."""
        d1 = _decide()
        d2 = _decide()
        assert d1["confidence"] == d2["confidence"]

    def test_more_blockers_higher_dont_ship_confidence(self):
        """More blockers → higher confidence in DONT_SHIP."""
        d_one = _decide(contamination_result=FAIL_CONTAMINATION)
        d_many = _decide(
            contamination_result=FAIL_CONTAMINATION,
            report_result=PII_DIRTY_REPORT,
            eval_results={"overall": {"f1": 0.10, "exact_match": 0.10}},
        )
        assert d_many["confidence"] >= d_one["confidence"]


# ── Threshold validation ───────────────────────────────────────────

class TestThresholdValidation:
    def test_negative_min_f1_rejected(self):
        with pytest.raises(ValueError, match="min_f1"):
            validate_thresholds({"min_f1": -0.5})

    def test_min_f1_above_1_rejected(self):
        with pytest.raises(ValueError, match="min_f1"):
            validate_thresholds({"min_f1": 1.5})

    def test_negative_regression_rejected(self):
        with pytest.raises(ValueError, match="max_f1_regression"):
            validate_thresholds({"max_f1_regression": -0.1})

    def test_negative_pii_hits_rejected(self):
        with pytest.raises(ValueError, match="max_pii_hits"):
            validate_thresholds({"max_pii_hits": -1})

    def test_valid_thresholds_pass(self):
        # Should not raise
        validate_thresholds({"min_f1": 0.65, "min_exact_match": 0.50, "max_pii_hits": 0})

    def test_empty_criteria_pass(self):
        validate_thresholds({})

    def test_string_value_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_thresholds({"min_f1": "high"})


# ── CI output stability ───────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RC_DIR = REPO_ROOT / "examples" / "real_conditions"


class TestCIOutputStability:
    def test_ci_final_line_format(self, capsys):
        """Pipeline CI mode prints 'Decision: X (exit N)' to stderr."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        run_pipeline(RC_DIR / "verifily_clean.yaml", ci=True)
        captured = capsys.readouterr()
        assert "Decision: SHIP (exit 0)" in captured.err

    def test_ci_json_is_valid_json(self, capsys):
        """CI mode outputs valid JSON to stdout."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        run_pipeline(RC_DIR / "verifily_clean.yaml", ci=True)
        captured = capsys.readouterr()
        decision = json.loads(captured.out)
        assert decision["recommendation"] == "SHIP"
        assert decision["exit_code"] == 0

    def test_ci_json_no_ansi(self, capsys):
        """CI mode JSON output contains no ANSI escape sequences."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        run_pipeline(RC_DIR / "verifily_clean.yaml", ci=True)
        captured = capsys.readouterr()
        assert "\x1b[" not in captured.out

    def test_dont_ship_final_line(self, capsys):
        """DONT_SHIP pipeline also prints deterministic final line."""
        from verifily_cli_v1.commands.pipeline import run_pipeline

        run_pipeline(RC_DIR / "verifily.yaml", ci=True)
        captured = capsys.readouterr()
        assert "Decision: DONT_SHIP (exit 1)" in captured.err

    def test_decision_artifacts_on_contract_fail(self, tmp_path):
        """Contract fail still writes decision_summary.json when output_dir set."""
        from verifily_cli_v1.commands.pipeline import run_pipeline
        from verifily_cli_v1.core.io import write_yaml

        # Create a config pointing to a nonexistent run dir
        cfg = {
            "run_dir": str(tmp_path / "nonexistent_run"),
            "train_data": str(RC_DIR / "data" / "train.jsonl"),
            "eval_data": str(RC_DIR / "data" / "eval_clean.jsonl"),
            "ship_if": {"min_f1": 0.65},
        }
        config_path = tmp_path / "bad_pipeline.yaml"
        write_yaml(config_path, cfg)

        out_dir = tmp_path / "output"
        result = run_pipeline(config_path, ci=True, output_dir=str(out_dir))

        assert result["decision"]["exit_code"] == EXIT_CONTRACT_FAIL
        assert (out_dir / "decision_summary.json").exists()

        # Verify the artifact content
        artifact = json.loads((out_dir / "decision_summary.json").read_text())
        assert artifact["recommendation"] == "DONT_SHIP"
        assert artifact["exit_code"] == EXIT_CONTRACT_FAIL
        assert "missing_files" in artifact

    def test_threshold_error_returns_tool_error(self, tmp_path):
        """Invalid thresholds → EXIT_TOOL_ERROR in decision."""
        from verifily_cli_v1.commands.pipeline import run_pipeline
        from verifily_cli_v1.core.io import write_yaml

        cfg = {
            "run_dir": str(RC_DIR / "runs" / "run_01_good"),
            "train_data": str(RC_DIR / "data" / "train.jsonl"),
            "eval_data": str(RC_DIR / "data" / "eval_clean.jsonl"),
            "ship_if": {"min_f1": -0.5},  # invalid
        }
        config_path = tmp_path / "bad_thresholds.yaml"
        write_yaml(config_path, cfg)

        result = run_pipeline(config_path, ci=True)
        assert result["decision"]["exit_code"] == EXIT_TOOL_ERROR
        assert "invalid_thresholds" in result["decision"]["risk_flags"]
