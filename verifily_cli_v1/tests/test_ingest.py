"""Tests for verifily ingest command.

All tests are deterministic, require no network or GPU, and run in < 2 seconds.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from verifily_cli_v1.commands.ingest import (
    apply_mapping,
    canonicalize_row,
    detect_schema,
    ingest,
    parse_map_args,
    parse_tag_args,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "examples" / "ingest_demo"


# ── Helpers ─────────────────────────────────────────────────────

def _write_csv(path: Path, header: list[str], rows: list[list[str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── test_ingest_csv_sft_mapping ─────────────────────────────────

class TestIngestCSVSFTMapping:
    """CSV with columns prompt, completion, passage → canonical sft rows."""

    def test_maps_question_answer_context(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.csv",
            tmp_path / "out",
            schema="sft",
            mapping={"question": "prompt", "answer": "completion", "context": "passage"},
        )
        assert result["schema"] == "sft"
        assert result["rows_out"] == 12  # 13 rows minus 1 empty
        assert result["rows_dropped"] == 1

        # Check first row is canonicalized correctly
        rows = _read_output(tmp_path / "out")
        first = rows[0]
        assert "input" in first
        assert "output" in first
        assert "id" in first
        assert first["id"].startswith("row_")
        assert "Context:" in first["input"]
        assert "Question:" in first["input"]

    def test_artifacts_written(self, tmp_path):
        ingest(
            DEMO_DIR / "data_sft.csv",
            tmp_path / "out",
            schema="sft",
            mapping={"question": "prompt", "answer": "completion", "context": "passage"},
        )
        out = tmp_path / "out"
        assert (out / "dataset.jsonl").exists()
        assert (out / "manifest.json").exists()
        assert (out / "hashes.json").exists()
        assert (out / "report.json").exists()

    def test_manifest_has_required_fields(self, tmp_path):
        ingest(
            DEMO_DIR / "data_sft.csv",
            tmp_path / "out",
            schema="sft",
            mapping={"question": "prompt", "answer": "completion", "context": "passage"},
        )
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        for key in ["schema", "input_path", "row_count_in", "row_count_out",
                     "dropped_rows_count", "top_drop_reasons", "tag_distribution",
                     "created_at", "verifily_version"]:
            assert key in manifest, f"manifest missing key: {key}"

    def test_hashes_valid(self, tmp_path):
        ingest(
            DEMO_DIR / "data_sft.csv",
            tmp_path / "out",
            schema="sft",
            mapping={"question": "prompt", "answer": "completion", "context": "passage"},
        )
        from verifily_cli_v1.core.hashing import sha256_file
        hashes = json.loads((tmp_path / "out" / "hashes.json").read_text())
        actual = sha256_file(tmp_path / "out" / "dataset.jsonl")
        assert hashes["dataset_hash"] == actual
        assert hashes["files"]["dataset.jsonl"] == actual


# ── test_ingest_jsonl_sft_identity ──────────────────────────────

class TestIngestJSONLSFTIdentity:
    """JSONL already has input/output — identity mapping."""

    def test_identity_ingest(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            tmp_path / "out",
            schema="sft",
        )
        assert result["rows_in"] == 8
        assert result["rows_out"] == 8
        assert result["rows_dropped"] == 0

    def test_preserves_existing_tags(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            tmp_path / "out",
            schema="sft",
        )
        rows = _read_output(tmp_path / "out")
        first = rows[0]
        assert first["tags"]["source"] == "human"
        assert first["tags"]["domain"] == "geography"

    def test_extra_tags_merged(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            tmp_path / "out",
            schema="sft",
            extra_tags={"project": "demo"},
        )
        rows = _read_output(tmp_path / "out")
        # extra tag present
        assert rows[0]["tags"]["project"] == "demo"
        # existing tag NOT overridden
        assert rows[0]["tags"]["source"] == "human"


# ── test_ingest_classification_csv ──────────────────────────────

class TestIngestClassificationCSV:
    """CSV with body,category → canonical classification rows."""

    def test_classification_mapping(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_cls.csv",
            tmp_path / "out",
            schema="classification",
            mapping={"text": "body", "label": "category"},
        )
        assert result["schema"] == "classification"
        assert result["rows_out"] == 10

        rows = _read_output(tmp_path / "out")
        assert "text" in rows[0]
        assert "label" in rows[0]
        assert rows[0]["label"] in ("positive", "negative", "neutral")


# ── test_ingest_auto_schema ─────────────────────────────────────

class TestIngestAutoSchema:
    """Auto-detect sft vs classification."""

    def test_auto_detects_sft_from_input_output(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            tmp_path / "out",
            schema="auto",
        )
        assert result["schema"] == "sft"

    def test_auto_detects_classification(self, tmp_path):
        # Create a JSONL with text+label
        inp = tmp_path / "cls.jsonl"
        _write_jsonl(inp, [
            {"text": "Great product", "label": "positive"},
            {"text": "Terrible", "label": "negative"},
        ])
        result = ingest(inp, tmp_path / "out", schema="auto")
        assert result["schema"] == "classification"

    def test_auto_detects_sft_from_question_answer(self, tmp_path):
        inp = tmp_path / "qa.jsonl"
        _write_jsonl(inp, [
            {"question": "What is AI?", "answer": "Artificial intelligence."},
        ])
        result = ingest(inp, tmp_path / "out", schema="auto")
        assert result["schema"] == "sft"

    def test_auto_fails_on_unknown_columns(self, tmp_path):
        inp = tmp_path / "bad.jsonl"
        _write_jsonl(inp, [{"foo": "bar", "baz": "qux"}])
        with pytest.raises(ValueError, match="Cannot auto-detect schema"):
            ingest(inp, tmp_path / "out", schema="auto")


# ── test_ingest_deterministic_ids ───────────────────────────────

class TestIngestDeterministicIDs:
    """Two ingest runs → same IDs and hashes."""

    def test_deterministic_ids_and_hashes(self, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        r1 = ingest(DEMO_DIR / "data_sft.jsonl", out1, schema="sft")
        r2 = ingest(DEMO_DIR / "data_sft.jsonl", out2, schema="sft")

        rows1 = _read_output(out1)
        rows2 = _read_output(out2)

        # Same IDs
        ids1 = [r["id"] for r in rows1]
        ids2 = [r["id"] for r in rows2]
        assert ids1 == ids2

        # Same dataset hash
        assert r1["dataset_hash"] == r2["dataset_hash"]
        assert r1["dataset_hash"] != ""

    def test_deterministic_with_mapping(self, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        mapping = {"question": "prompt", "answer": "completion", "context": "passage"}

        r1 = ingest(DEMO_DIR / "data_sft.csv", out1, schema="sft", mapping=mapping)
        r2 = ingest(DEMO_DIR / "data_sft.csv", out2, schema="sft", mapping=mapping)

        assert r1["dataset_hash"] == r2["dataset_hash"]


# ── test_ingest_strict_vs_non_strict ────────────────────────────

class TestIngestStrictVsNonStrict:
    """Bad rows: drop in non-strict, fail in strict."""

    def _make_mixed_input(self, tmp_path):
        inp = tmp_path / "mixed.jsonl"
        _write_jsonl(inp, [
            {"input": "Good row", "output": "Valid answer"},
            {"input": "", "output": ""},  # empty required fields
            {"input": "Another good", "output": "Another answer"},
            {"foo": "bar"},  # missing required fields entirely
        ])
        return inp

    def test_non_strict_drops_bad_rows(self, tmp_path):
        inp = self._make_mixed_input(tmp_path)
        result = ingest(inp, tmp_path / "out", schema="sft", strict=False)
        assert result["rows_in"] == 4
        assert result["rows_out"] == 2
        assert result["rows_dropped"] == 2

    def test_strict_raises_on_bad_row(self, tmp_path):
        inp = self._make_mixed_input(tmp_path)
        with pytest.raises(ValueError, match="Row 1"):
            ingest(inp, tmp_path / "out", schema="sft", strict=True)


# ── test_ingest_dry_run ─────────────────────────────────────────

class TestIngestDryRun:
    """Dry run: validate + preview but write nothing."""

    def test_dry_run_writes_nothing(self, tmp_path):
        out = tmp_path / "should_not_exist"
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            out,
            schema="sft",
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["rows_out"] == 8
        assert not out.exists()

    def test_dry_run_has_preview(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            tmp_path / "out",
            schema="sft",
            dry_run=True,
        )
        assert "preview" in result
        assert len(result["preview"]) == 3


# ── Additional edge case tests ──────────────────────────────────

class TestIngestEdgeCases:
    """Extra coverage for limit, id_col, tags, report."""

    def test_limit(self, tmp_path):
        result = ingest(
            DEMO_DIR / "data_sft.jsonl",
            tmp_path / "out",
            schema="sft",
            limit=3,
        )
        assert result["rows_in"] == 3
        assert result["rows_out"] == 3

    def test_id_col_from_csv(self, tmp_path):
        inp = tmp_path / "with_id.csv"
        _write_csv(inp, ["my_id", "input", "output"], [
            ["ID001", "Hello", "World"],
            ["ID002", "Foo", "Bar"],
        ])
        result = ingest(inp, tmp_path / "out", schema="sft", id_col="my_id")
        rows = _read_output(tmp_path / "out")
        assert rows[0]["id"] == "ID001"
        assert rows[1]["id"] == "ID002"

    def test_report_json_written(self, tmp_path):
        ingest(DEMO_DIR / "data_sft.jsonl", tmp_path / "out", schema="sft")
        report = json.loads((tmp_path / "out" / "report.json").read_text())
        assert "field_stats" in report
        assert "input" in report["field_stats"]
        assert "pii_scan" in report
        assert "exact_duplicate_rate" in report

    def test_parse_map_args(self):
        m = parse_map_args(["question:prompt", "answer:completion"])
        assert m == {"question": "prompt", "answer": "completion"}

    def test_parse_map_args_invalid(self):
        with pytest.raises(ValueError, match="Invalid --map"):
            parse_map_args(["badformat"])

    def test_parse_tag_args(self):
        t = parse_tag_args(["source:customer", "project:demo"])
        assert t == {"source": "customer", "project": "demo"}

    def test_detect_schema_sft(self):
        assert detect_schema([{"input": "x", "output": "y"}]) == "sft"
        assert detect_schema([{"question": "x", "answer": "y"}]) == "sft"

    def test_detect_schema_classification(self):
        assert detect_schema([{"text": "x", "label": "y"}]) == "classification"


# ── Utility ─────────────────────────────────────────────────────

def _read_output(out_dir: Path) -> list[dict]:
    rows = []
    with open(out_dir / "dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
