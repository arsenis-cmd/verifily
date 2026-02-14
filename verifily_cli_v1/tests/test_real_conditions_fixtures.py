"""Tests validating the real_conditions fixture data and Verifily harness.

All tests are deterministic, require no network or GPU, and run in < 5 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Paths ───────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RC_DIR = REPO_ROOT / "examples" / "real_conditions"
DATA_DIR = RC_DIR / "data"
RUNS_DIR = RC_DIR / "runs"


# ── Helpers ─────────────────────────────────────────────────────

def read_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Fixture data shape tests ────────────────────────────────────

class TestFixtureDataShape:
    """Verify the fixture datasets exist and have expected properties."""

    def test_train_row_count(self):
        rows = read_jsonl(DATA_DIR / "train.jsonl")
        assert len(rows) == 60

    def test_train_schema(self):
        rows = read_jsonl(DATA_DIR / "train.jsonl")
        for i, row in enumerate(rows):
            assert "instruction" in row, f"row {i} missing instruction"
            assert "output" in row, f"row {i} missing output"
            assert row["instruction"], f"row {i} has empty instruction"
            assert row["output"], f"row {i} has empty output"

    def test_eval_clean_row_count(self):
        rows = read_jsonl(DATA_DIR / "eval_clean.jsonl")
        assert len(rows) == 20

    def test_eval_leaked_exact_row_count(self):
        rows = read_jsonl(DATA_DIR / "eval_leaked_exact.jsonl")
        assert len(rows) == 20

    def test_eval_leaked_near_row_count(self):
        rows = read_jsonl(DATA_DIR / "eval_leaked_near.jsonl")
        assert len(rows) == 20


# ── Overlap expectation tests ───────────────────────────────────

class TestOverlapExpectations:
    """Verify contamination overlap guarantees on fixture data."""

    @staticmethod
    def _row_text(row):
        parts = []
        for key in ("instruction", "output", "input"):
            if key in row and row[key]:
                parts.append(str(row[key]).strip())
        return " ".join(parts)

    @staticmethod
    def _sha(text):
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()

    def _compute_exact_overlaps(self, train_rows, eval_rows):
        train_hashes = {self._sha(self._row_text(r)) for r in train_rows}
        overlaps = 0
        for r in eval_rows:
            if self._sha(self._row_text(r)) in train_hashes:
                overlaps += 1
        return overlaps

    def test_clean_eval_zero_exact_overlap(self):
        train = read_jsonl(DATA_DIR / "train.jsonl")
        clean = read_jsonl(DATA_DIR / "eval_clean.jsonl")
        overlaps = self._compute_exact_overlaps(train, clean)
        assert overlaps == 0, f"Clean eval should have 0 exact overlaps, got {overlaps}"

    def test_exact_leak_has_sufficient_overlap(self):
        train = read_jsonl(DATA_DIR / "train.jsonl")
        leaked = read_jsonl(DATA_DIR / "eval_leaked_exact.jsonl")
        overlaps = self._compute_exact_overlaps(train, leaked)
        fraction = overlaps / len(leaked)
        assert fraction >= 0.15, (
            f"Exact leak eval should have >= 15% overlap, got {fraction:.1%} ({overlaps}/{len(leaked)})"
        )

    def test_near_leak_has_sufficient_near_overlap(self):
        """Check that near-leak eval has >= 30% near-duplicate rows."""
        from verifily_cli_v1.commands.contamination import _row_text, _ngrams, _jaccard

        train = read_jsonl(DATA_DIR / "train.jsonl")
        near = read_jsonl(DATA_DIR / "eval_leaked_near.jsonl")

        train_ngrams = [_ngrams(_row_text(r), 3) for r in train]
        near_count = 0

        for eval_row in near:
            eval_ng = _ngrams(_row_text(eval_row), 3)
            best_sim = max(_jaccard(eval_ng, tng) for tng in train_ngrams)
            if best_sim >= 0.70:
                near_count += 1

        fraction = near_count / len(near)
        assert fraction >= 0.30, (
            f"Near leak eval should have >= 30% near-duplicates, got {fraction:.1%} ({near_count}/{len(near)})"
        )

    def test_clean_eval_no_near_duplicates(self):
        """Clean eval should have no near-duplicates above threshold."""
        from verifily_cli_v1.commands.contamination import _row_text, _ngrams, _jaccard

        train = read_jsonl(DATA_DIR / "train.jsonl")
        clean = read_jsonl(DATA_DIR / "eval_clean.jsonl")

        train_ngrams = [_ngrams(_row_text(r), 3) for r in train]
        near_count = 0

        for eval_row in clean:
            eval_ng = _ngrams(_row_text(eval_row), 3)
            best_sim = max(_jaccard(eval_ng, tng) for tng in train_ngrams)
            if best_sim >= 0.70:
                near_count += 1

        assert near_count == 0, (
            f"Clean eval should have 0 near-duplicates, got {near_count}"
        )


# ── Run contract tests ──────────────────────────────────────────

class TestRunContract:
    """Verify each fixture run directory satisfies the Verifily run contract."""

    RUN_NAMES = ["run_01_good", "run_02_good", "run_03_regression", "run_04_recovery"]

    @pytest.fixture(params=RUN_NAMES)
    def run_dir(self, request):
        return RUNS_DIR / request.param

    def test_run_contract_valid(self, run_dir):
        from verifily_cli_v1.commands.contract_check import validate_run_contract
        result = validate_run_contract(run_dir)
        assert result["valid"], f"{run_dir.name} failed contract: {result['checks']}"

    def test_run_has_config_yaml(self, run_dir):
        assert (run_dir / "config.yaml").exists()

    def test_run_has_hashes_json(self, run_dir):
        assert (run_dir / "hashes.json").exists()

    def test_run_has_environment_json(self, run_dir):
        assert (run_dir / "environment.json").exists()

    def test_run_has_eval_results(self, run_dir):
        assert (run_dir / "eval" / "eval_results.json").exists()

    def test_run_meta_status_completed(self, run_dir):
        meta = json.loads((run_dir / "run_meta.json").read_text())
        assert meta["status"] == "completed"

    def test_eval_results_has_f1(self, run_dir):
        results = json.loads((run_dir / "eval" / "eval_results.json").read_text())
        assert "overall" in results
        assert "f1" in results["overall"]
        assert isinstance(results["overall"]["f1"], float)


# ── Regression detection tests ──────────────────────────────────

class TestRegressionDetection:
    """Verify the history command detects regressions correctly."""

    def test_history_detects_regression(self):
        from verifily_cli_v1.commands.history import analyze_history
        result = analyze_history(RUNS_DIR, metric="f1", regression_threshold=0.02)

        assert result["has_regression"] is True
        assert len(result["regressions"]) == 1
        assert result["regressions"][0]["run"] == "run_03_regression"
        assert result["exit_code"] == 2

    def test_history_regression_delta(self):
        from verifily_cli_v1.commands.history import analyze_history
        result = analyze_history(RUNS_DIR, metric="f1", regression_threshold=0.02)

        reg = result["regressions"][0]
        assert reg["delta"] < -0.02, f"Delta {reg['delta']} should be < -0.02"

    def test_history_no_regression_with_high_threshold(self):
        from verifily_cli_v1.commands.history import analyze_history
        result = analyze_history(RUNS_DIR, metric="f1", regression_threshold=0.10)

        assert result["has_regression"] is False
        assert result["exit_code"] == 0

    def test_history_run_ordering(self):
        from verifily_cli_v1.commands.history import analyze_history
        result = analyze_history(RUNS_DIR, metric="f1")

        names = [r["name"] for r in result["runs"]]
        assert names == ["run_01_good", "run_02_good", "run_03_regression", "run_04_recovery"]

    def test_history_metric_trend(self):
        from verifily_cli_v1.commands.history import analyze_history
        result = analyze_history(RUNS_DIR, metric="f1")

        values = [r["metric"] for r in result["runs"]]
        # run_01 < run_02 (improvement), run_02 > run_03 (regression), run_03 < run_04 (recovery)
        assert values[0] < values[1], "run_02 should improve over run_01"
        assert values[1] > values[2], "run_03 should regress from run_02"
        assert values[2] < values[3], "run_04 should recover from run_03"


# ── CI exit code tests ──────────────────────────────────────────

class TestCIExitCodes:
    """Verify pipeline exit codes match expected values."""

    def test_contamination_clean_exit_0(self):
        from verifily_cli_v1.commands.contamination import check_contamination
        result = check_contamination(
            DATA_DIR / "train.jsonl",
            DATA_DIR / "eval_clean.jsonl",
        )
        assert result["exit_code"] == 0
        assert result["status"] == "PASS"

    def test_contamination_exact_exit_1(self):
        from verifily_cli_v1.commands.contamination import check_contamination
        result = check_contamination(
            DATA_DIR / "train.jsonl",
            DATA_DIR / "eval_leaked_exact.jsonl",
        )
        assert result["exit_code"] == 1
        assert result["status"] == "FAIL"

    def test_contamination_near_exit_2(self):
        from verifily_cli_v1.commands.contamination import check_contamination
        result = check_contamination(
            DATA_DIR / "train.jsonl",
            DATA_DIR / "eval_leaked_near.jsonl",
        )
        assert result["exit_code"] == 2
        assert result["status"] == "WARN"

    def test_pipeline_leaked_returns_dont_ship(self):
        from verifily_cli_v1.commands.pipeline import run_pipeline
        result = run_pipeline(RC_DIR / "verifily.yaml")
        decision = result["decision"]
        assert decision["recommendation"] == "DONT_SHIP"
        assert decision["exit_code"] == 1

    def test_pipeline_clean_returns_ship(self):
        from verifily_cli_v1.commands.pipeline import run_pipeline
        result = run_pipeline(RC_DIR / "verifily_clean.yaml")
        decision = result["decision"]
        assert decision["recommendation"] == "SHIP"
        assert decision["exit_code"] == 0

    def test_pipeline_decision_has_required_fields(self):
        from verifily_cli_v1.commands.pipeline import run_pipeline
        result = run_pipeline(RC_DIR / "verifily_clean.yaml")
        decision = result["decision"]
        for field in ["recommendation", "exit_code", "confidence", "reasons",
                      "risk_flags", "metrics", "baseline_metrics", "deltas"]:
            assert field in decision, f"Decision missing field: {field}"

    def test_pipeline_contamination_fail_is_risk_flag(self):
        from verifily_cli_v1.commands.pipeline import run_pipeline
        result = run_pipeline(RC_DIR / "verifily.yaml")
        assert "contamination_fail" in result["decision"]["risk_flags"]


# ── Contract check command tests ────────────────────────────────

class TestContractCheck:
    """Verify contract-check behavior."""

    def test_valid_run_passes(self):
        from verifily_cli_v1.commands.contract_check import validate_run_contract
        result = validate_run_contract(RUNS_DIR / "run_01_good")
        assert result["valid"] is True

    def test_missing_dir_fails(self):
        from verifily_cli_v1.commands.contract_check import validate_run_contract
        result = validate_run_contract(RUNS_DIR / "nonexistent_run")
        assert result["valid"] is False

    def test_dataset_schema_validation_passes(self):
        from verifily_cli_v1.commands.contract_check import validate_dataset_schema
        result = validate_dataset_schema(DATA_DIR / "train.jsonl", "sft")
        assert result["valid"] is True
        assert result["total_rows"] == 60

    def test_dataset_schema_wrong_schema_fails(self):
        from verifily_cli_v1.commands.contract_check import validate_dataset_schema
        # SFT data validated against classification schema should fail
        result = validate_dataset_schema(DATA_DIR / "train.jsonl", "classification")
        assert result["valid"] is False


# ── Report command tests ────────────────────────────────────────

class TestReport:
    """Verify report output."""

    def test_report_row_count(self):
        from verifily_cli_v1.commands.report import dataset_report
        report = dataset_report(DATA_DIR / "train.jsonl")
        assert report["row_count"] == 60

    def test_report_pii_clean(self):
        from verifily_cli_v1.commands.report import dataset_report
        report = dataset_report(DATA_DIR / "train.jsonl")
        assert report["pii_clean"] is True

    def test_report_has_tag_distribution(self):
        from verifily_cli_v1.commands.report import dataset_report
        report = dataset_report(DATA_DIR / "train.jsonl")
        assert "source" in report["tag_distribution"]
        assert "human" in report["tag_distribution"]["source"]
        assert "synthetic" in report["tag_distribution"]["source"]
