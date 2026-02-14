"""Tests for the customer drill demo.

Validates end-to-end: ingest → contract-check → contamination → pipeline.
All tests are deterministic, require no network or GPU, and run in < 2 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"
RAW_DIR = DRILL_DIR / "raw"
RUNS_DIR = DRILL_DIR / "runs"


# ── Helpers ──────────────────────────────────────────────────────

def _ingest_tickets(tmp_path: Path) -> dict:
    """Ingest support_tickets.csv into tmp_path, return ingest result."""
    from verifily_cli_v1.commands.ingest import ingest

    return ingest(
        RAW_DIR / "support_tickets.csv",
        tmp_path / "artifact",
        schema="sft",
        mapping={"question": "subject", "answer": "resolution", "context": "body"},
        extra_tags={"source": "customer_drill"},
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Ingest ───────────────────────────────────────────────────────

class TestCustomerDrillIngest:
    """Ingest of messy support_tickets.csv."""

    def test_row_counts(self, tmp_path):
        result = _ingest_tickets(tmp_path)
        assert result["rows_in"] == 30
        assert result["rows_out"] == 28
        assert result["rows_dropped"] == 2

    def test_artifacts_written(self, tmp_path):
        _ingest_tickets(tmp_path)
        art = tmp_path / "artifact"
        assert (art / "dataset.jsonl").exists()
        assert (art / "manifest.json").exists()
        assert (art / "hashes.json").exists()
        assert (art / "report.json").exists()

    def test_deterministic_hash(self, tmp_path):
        r1 = _ingest_tickets(tmp_path / "run1")
        r2 = _ingest_tickets(tmp_path / "run2")
        assert r1["dataset_hash"] == r2["dataset_hash"]
        assert r1["dataset_hash"] != ""

    def test_pii_detected_in_report(self, tmp_path):
        _ingest_tickets(tmp_path)
        report = json.loads((tmp_path / "artifact" / "report.json").read_text())
        assert report["pii_total_hits"] > 0
        assert report["pii_scan"]["email"] >= 1
        assert report["pii_scan"]["phone"] >= 1

    def test_exact_duplicates_in_report(self, tmp_path):
        _ingest_tickets(tmp_path)
        report = json.loads((tmp_path / "artifact" / "report.json").read_text())
        assert report["exact_duplicates"] >= 2

    def test_tags_applied(self, tmp_path):
        _ingest_tickets(tmp_path)
        rows = _read_jsonl(tmp_path / "artifact" / "dataset.jsonl")
        for row in rows:
            assert row["tags"]["source"] == "customer_drill"


# ── Contract Check ───────────────────────────────────────────────

class TestCustomerDrillContract:
    """Contract-check on ingested dataset and run directories."""

    def test_dataset_schema_passes(self, tmp_path):
        from verifily_cli_v1.commands.contract_check import validate_dataset_schema

        _ingest_tickets(tmp_path)
        result = validate_dataset_schema(
            tmp_path / "artifact" / "dataset.jsonl", "sft"
        )
        assert result["valid"], f"Schema errors: {result['errors']}"
        assert result["total_rows"] == 28

    def test_run_clean_contract_passes(self):
        from verifily_cli_v1.commands.contract_check import validate_run_contract

        result = validate_run_contract(RUNS_DIR / "run_clean")
        assert result["valid"]
        assert result["has_eval"]

    def test_run_leaked_contract_passes(self):
        from verifily_cli_v1.commands.contract_check import validate_run_contract

        result = validate_run_contract(RUNS_DIR / "run_leaked")
        assert result["valid"]
        assert result["has_eval"]


# ── Contamination ────────────────────────────────────────────────

class TestCustomerDrillContamination:
    """Contamination checks with different eval sets."""

    def test_clean_eval_passes(self, tmp_path):
        from verifily_cli_v1.commands.contamination import check_contamination

        _ingest_tickets(tmp_path)
        train = tmp_path / "artifact" / "dataset.jsonl"
        result = check_contamination(train, RAW_DIR / "eval_clean.jsonl")
        assert result["status"] == "PASS"
        assert result["exit_code"] == 0
        assert result["exact_overlaps"] == 0
        assert result["near_duplicates"] == 0

    def test_exact_leak_fails(self, tmp_path):
        from verifily_cli_v1.commands.contamination import check_contamination

        _ingest_tickets(tmp_path)
        train = tmp_path / "artifact" / "dataset.jsonl"
        result = check_contamination(train, RAW_DIR / "eval_leaked_exact.jsonl")
        assert result["status"] == "FAIL"
        assert result["exit_code"] == 1
        assert result["exact_overlaps"] == 4
        assert result["exact_overlap_fraction"] == 0.4

    def test_near_leak_warns(self, tmp_path):
        from verifily_cli_v1.commands.contamination import check_contamination

        _ingest_tickets(tmp_path)
        train = tmp_path / "artifact" / "dataset.jsonl"
        result = check_contamination(train, RAW_DIR / "eval_leaked_near.jsonl")
        assert result["status"] == "WARN"
        assert result["exit_code"] == 2
        assert result["near_duplicates"] == 3
        assert result["exact_overlaps"] == 0


# ── Pipeline ─────────────────────────────────────────────────────

class TestCustomerDrillPipeline:
    """Pipeline CI gate for clean and leaked scenarios."""

    def _write_pipeline_config(self, tmp_path, train_path, eval_file, run_name):
        """Write a pipeline config and return its path."""
        from verifily_cli_v1.core.io import write_yaml

        cfg = {
            "run_dir": str(RUNS_DIR / run_name),
            "train_data": str(train_path),
            "eval_data": str(RAW_DIR / eval_file),
            "baseline_run": str(RUNS_DIR / run_name),
            "ship_if": {
                "min_f1": 0.65,
                "min_exact_match": 0.50,
                "max_f1_regression": 0.03,
                "max_pii_hits": 10,
            },
        }
        config_path = tmp_path / "pipeline.yaml"
        write_yaml(config_path, cfg)
        return config_path

    def test_clean_pipeline_ships(self, tmp_path):
        from verifily_cli_v1.commands.pipeline import run_pipeline

        _ingest_tickets(tmp_path)
        train = tmp_path / "artifact" / "dataset.jsonl"
        config = self._write_pipeline_config(
            tmp_path, train, "eval_clean.jsonl", "run_clean"
        )
        result = run_pipeline(config, ci=True)
        decision = result["decision"]
        assert decision["recommendation"] == "SHIP"
        assert decision["exit_code"] == 0

    def test_leaked_pipeline_blocks(self, tmp_path):
        from verifily_cli_v1.commands.pipeline import run_pipeline

        _ingest_tickets(tmp_path)
        train = tmp_path / "artifact" / "dataset.jsonl"
        config = self._write_pipeline_config(
            tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked"
        )
        result = run_pipeline(config, ci=True)
        decision = result["decision"]
        assert decision["recommendation"] == "DONT_SHIP"
        assert decision["exit_code"] == 1
        assert "contamination_fail" in decision["risk_flags"]

    def test_leaked_pipeline_has_metrics(self, tmp_path):
        from verifily_cli_v1.commands.pipeline import run_pipeline

        _ingest_tickets(tmp_path)
        train = tmp_path / "artifact" / "dataset.jsonl"
        config = self._write_pipeline_config(
            tmp_path, train, "eval_leaked_exact.jsonl", "run_leaked"
        )
        result = run_pipeline(config, ci=True)
        decision = result["decision"]
        assert "f1" in decision["metrics"]
        assert decision["metrics"]["f1"] == 0.72


# ── Fixture data shape ───────────────────────────────────────────

class TestCustomerDrillFixtures:
    """Validate the static fixture files exist and have correct shapes."""

    def test_csv_has_30_rows(self):
        import csv

        with open(RAW_DIR / "support_tickets.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 30

    def test_csv_has_expected_columns(self):
        import csv

        with open(RAW_DIR / "support_tickets.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        expected = {"ticket_id", "subject", "body", "resolution", "category", "created_at", "extra_noise"}
        assert set(row.keys()) == expected

    def test_eval_clean_has_10_rows(self):
        rows = _read_jsonl(RAW_DIR / "eval_clean.jsonl")
        assert len(rows) == 10

    def test_eval_leaked_exact_has_10_rows(self):
        rows = _read_jsonl(RAW_DIR / "eval_leaked_exact.jsonl")
        assert len(rows) == 10

    def test_eval_leaked_near_has_10_rows(self):
        rows = _read_jsonl(RAW_DIR / "eval_leaked_near.jsonl")
        assert len(rows) == 10

    def test_expected_ship_json(self):
        data = json.loads((DRILL_DIR / "expected" / "expected_ship.json").read_text())
        assert data["recommendation"] == "SHIP"
        assert data["exit_code"] == 0

    def test_expected_dont_ship_json(self):
        data = json.loads((DRILL_DIR / "expected" / "expected_dont_ship.json").read_text())
        assert data["recommendation"] == "DONT_SHIP"
        assert data["exit_code"] == 1
