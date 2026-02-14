"""Tests for Verifily v1.1 — Input Pipeline Widening.

Covers: readers, schemas, flatten, minhash_lsh, pii, contamination scaling, integration.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


def _has_pyarrow():
    try:
        import pyarrow
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════
# Readers
# ═══════════════════════════════════════════════════════════════════


class TestReaders:
    def test_csv_reader_roundtrip(self, tmp_path):
        from verifily_cli_v1.core.readers import CsvReader

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("input,output\nhello,world\nfoo,bar\n")

        reader = CsvReader()
        assert reader.can_read(csv_file)
        result = reader.read(csv_file)
        assert result.format == "csv"
        assert result.row_count == 2
        assert result.rows[0] == {"input": "hello", "output": "world"}
        assert "input" in result.columns
        assert "output" in result.columns

    def test_jsonl_reader_roundtrip(self, tmp_path):
        from verifily_cli_v1.core.readers import JsonlReader

        jsonl_file = tmp_path / "data.jsonl"
        rows = [{"input": "hi", "output": "hello"}, {"input": "bye", "output": "goodbye"}]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        reader = JsonlReader()
        assert reader.can_read(jsonl_file)
        result = reader.read(jsonl_file)
        assert result.format == "jsonl"
        assert result.row_count == 2
        assert result.rows[0]["input"] == "hi"

    @pytest.mark.skipif(
        not _has_pyarrow(), reason="pyarrow not installed"
    )
    def test_parquet_reader_roundtrip(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from verifily_cli_v1.core.readers import ParquetReader

        table = pa.table({"input": ["hello", "foo"], "output": ["world", "bar"]})
        pq_file = tmp_path / "data.parquet"
        pq.write_table(table, pq_file)

        reader = ParquetReader()
        assert reader.can_read(pq_file)
        result = reader.read(pq_file)
        assert result.format == "parquet"
        assert result.row_count == 2
        assert result.rows[0]["input"] == "hello"
        assert "num_row_groups" in result.provenance

    def test_read_dataset_dispatches_by_extension(self, tmp_path):
        from verifily_cli_v1.core.readers import read_dataset

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text,label\ngreat,positive\n")
        result = read_dataset(csv_file)
        assert result.format == "csv"

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"text":"great","label":"positive"}\n')
        result = read_dataset(jsonl_file)
        assert result.format == "jsonl"


# ═══════════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════════


class TestSchemas:
    def test_all_schemas_registered(self):
        from verifily_cli_v1.core.schemas import SCHEMA_REGISTRY

        expected = {"sft", "qa", "classification", "chat", "summarization", "translation", "rm_pairwise"}
        assert set(SCHEMA_REGISTRY.keys()) == expected

    def test_detect_schema_from_fields(self):
        from verifily_cli_v1.core.schemas import detect_schema_from_fields

        assert detect_schema_from_fields({"input", "output"}) == "sft"
        assert detect_schema_from_fields({"text", "label"}) == "classification"
        assert detect_schema_from_fields({"question", "context", "answer"}) == "qa"
        assert detect_schema_from_fields({"messages"}) == "chat"
        assert detect_schema_from_fields({"prompt", "chosen", "rejected"}) == "rm_pairwise"
        assert detect_schema_from_fields({"document", "summary"}) == "summarization"
        # QA should win over SFT when context is present
        assert detect_schema_from_fields({"question", "answer", "context"}) == "qa"

    def test_schema_content_fields(self):
        from verifily_cli_v1.core.schemas import get_content_fields

        sft_fields = get_content_fields("sft")
        assert "input" in sft_fields
        assert "output" in sft_fields

        chat_fields = get_content_fields("chat")
        assert "messages" in chat_fields


# ═══════════════════════════════════════════════════════════════════
# Flatten
# ═══════════════════════════════════════════════════════════════════


class TestFlatten:
    def test_flatten_nested_dict(self):
        from verifily_cli_v1.core.flatten import flatten_row

        row = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        flat = flatten_row(row)
        assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}

    def test_flatten_preserves_flat_row(self):
        from verifily_cli_v1.core.flatten import flatten_row

        row = {"input": "hello", "output": "world"}
        flat = flatten_row(row)
        assert flat == row

    def test_expand_chat_turns(self):
        from verifily_cli_v1.core.flatten import expand_chat_turns

        row = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        pairs = expand_chat_turns(row)
        assert len(pairs) == 1
        assert pairs[0]["input"] == "Hi"
        assert pairs[0]["output"] == "Hello!"
        assert pairs[0]["tags"]["source_schema"] == "chat"
        assert pairs[0]["tags"]["system_prompt"] == "Be helpful."

    def test_expand_chat_multiturn(self):
        from verifily_cli_v1.core.flatten import expand_chat_turns

        row = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye!"},
            ]
        }
        pairs = expand_chat_turns(row)
        assert len(pairs) == 2
        assert pairs[0]["input"] == "Hi"
        assert pairs[0]["output"] == "Hello!"
        assert pairs[1]["input"] == "Bye"
        assert pairs[1]["output"] == "Goodbye!"
        assert pairs[0]["tags"]["turn"] == 0
        assert pairs[1]["tags"]["turn"] == 1


# ═══════════════════════════════════════════════════════════════════
# MinHash LSH
# ═══════════════════════════════════════════════════════════════════


class TestMinHashLSH:
    def test_identical_docs_match(self):
        from verifily_cli_v1.core.minhash_lsh import MinHashLSH, minhash_jaccard

        lsh = MinHashLSH(num_perm=128, threshold=0.70)
        shingles = {"the cat", "cat sat", "sat on", "on the", "the mat"}
        lsh.insert(0, shingles)

        candidates = lsh.query(shingles)
        assert 0 in candidates

        sig = lsh.signatures[0]
        assert minhash_jaccard(sig, sig) == 1.0

    def test_disjoint_docs_no_match(self):
        from verifily_cli_v1.core.minhash_lsh import MinHashLSH

        lsh = MinHashLSH(num_perm=128, threshold=0.70)
        lsh.insert(0, {"aaa bbb", "bbb ccc", "ccc ddd"})

        # Completely different shingles
        candidates = lsh.query({"xxx yyy", "yyy zzz", "zzz www"})
        # Candidate set should not include 0 (or be empty)
        # Note: LSH can have false positives, so we verify with Jaccard
        from verifily_cli_v1.core.minhash_lsh import compute_minhash_signature, minhash_jaccard
        sig_a = lsh.signatures[0]
        sig_b = compute_minhash_signature({"xxx yyy", "yyy zzz", "zzz www"}, lsh.params)
        assert minhash_jaccard(sig_a, sig_b) < 0.3

    def test_lsh_finds_near_duplicates(self):
        from verifily_cli_v1.core.minhash_lsh import MinHashLSH

        lsh = MinHashLSH(num_perm=128, threshold=0.50)
        base = {"the quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"}
        lsh.insert(0, base)

        # Similar text (6 of 8 shingles shared)
        similar = {"the quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over a", "a sleeping", "sleeping cat"}
        candidates = lsh.query(similar)
        assert 0 in candidates

    def test_deterministic_signatures(self):
        from verifily_cli_v1.core.minhash_lsh import MinHashLSH

        lsh1 = MinHashLSH(num_perm=64, threshold=0.70, seed=42)
        lsh2 = MinHashLSH(num_perm=64, threshold=0.70, seed=42)

        shingles = {"hello world", "world is", "is great"}
        lsh1.insert(0, shingles)
        lsh2.insert(0, shingles)

        assert lsh1.signatures[0] == lsh2.signatures[0]


# ═══════════════════════════════════════════════════════════════════
# Contamination Scaling
# ═══════════════════════════════════════════════════════════════════


class TestContaminationScaling:
    def test_lsh_contamination_matches_bruteforce(self, tmp_path):
        """LSH and brute-force should find the same exact overlaps on a small dataset."""
        from verifily_cli_v1.commands.contamination import check_contamination

        train = [{"input": f"train text {i}", "output": f"out {i}"} for i in range(20)]
        # Some eval rows match train exactly
        eval_rows = [
            {"input": "train text 0", "output": "out 0"},  # exact match
            {"input": "train text 5", "output": "out 5"},  # exact match
            {"input": "unique eval text", "output": "unique out"},  # no match
        ]

        train_file = tmp_path / "train.jsonl"
        eval_file = tmp_path / "eval.jsonl"
        train_file.write_text("\n".join(json.dumps(r) for r in train) + "\n")
        eval_file.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")

        result_lsh = check_contamination(train_file, eval_file, use_lsh=True)
        result_bf = check_contamination(train_file, eval_file, use_lsh=False)

        assert result_lsh["exact_overlaps"] == result_bf["exact_overlaps"] == 2
        assert result_lsh["method"] == "lsh"
        assert result_bf["method"] == "brute_force"

    def test_sampling_reduces_rows(self, tmp_path):
        from verifily_cli_v1.commands.contamination import check_contamination

        train = [{"input": f"train {i}", "output": f"out {i}"} for i in range(100)]
        eval_rows = [{"input": f"eval {i}", "output": f"eout {i}"} for i in range(50)]

        train_file = tmp_path / "train.jsonl"
        eval_file = tmp_path / "eval.jsonl"
        train_file.write_text("\n".join(json.dumps(r) for r in train) + "\n")
        eval_file.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")

        result = check_contamination(
            train_file, eval_file, sample_train=20, sample_eval=10,
        )
        assert result["train_rows"] == 20
        assert result["eval_rows"] == 10
        assert result["sampled"] is True

    @pytest.mark.slow
    def test_large_dataset_completes(self, tmp_path):
        """50K synthetic rows should complete without OOM in reasonable time."""
        import time
        from verifily_cli_v1.commands.contamination import check_contamination

        train = [{"input": f"training example number {i} with some text", "output": f"response {i}"} for i in range(50_000)]
        eval_rows = [{"input": f"evaluation example number {i} with some text", "output": f"eval response {i}"} for i in range(1000)]

        train_file = tmp_path / "train.jsonl"
        eval_file = tmp_path / "eval.jsonl"
        train_file.write_text("\n".join(json.dumps(r) for r in train) + "\n")
        eval_file.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")

        t0 = time.monotonic()
        result = check_contamination(train_file, eval_file, use_lsh=True)
        elapsed = time.monotonic() - t0

        assert result["status"] in ("PASS", "WARN", "FAIL")
        assert elapsed < 60  # Should complete in under 60s


# ═══════════════════════════════════════════════════════════════════
# PII
# ═══════════════════════════════════════════════════════════════════


class TestPII:
    def test_regex_detects_email_phone_ssn(self):
        from verifily_cli_v1.core.pii import scan_text_regex

        result = scan_text_regex("Contact john@example.com or 555-123-4567, SSN 123-45-6789")
        types_found = result.pii_types_found
        assert "email" in types_found
        assert "phone" in types_found
        assert "ssn" in types_found

    def test_scan_dataset_format_compatible(self):
        from verifily_cli_v1.core.pii import scan_dataset

        rows = [
            {"text": "Email me at test@test.com", "label": "positive"},
            {"text": "No PII here", "label": "neutral"},
        ]
        result = scan_dataset(rows)
        assert "pii_scan" in result
        assert "pii_clean" in result
        assert "pii_total_hits" in result
        assert result["pii_total_hits"] >= 1
        assert result["pii_clean"] is False

    def test_ner_fallback_when_spacy_missing(self):
        from verifily_cli_v1.core.pii import scan_text_ner

        # Even if spaCy isn't installed, scan_text_ner should not crash
        result = scan_text_ner("Contact john@example.com", nlp=None)
        assert result.has_pii
        assert "email" in result.pii_types_found


# ═══════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════


class TestIntegration:
    @pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
    def test_ingest_parquet_end_to_end(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from verifily_cli_v1.commands.ingest import ingest

        table = pa.table({"input": ["hello", "foo"], "output": ["world", "bar"]})
        pq_file = tmp_path / "data.parquet"
        pq.write_table(table, pq_file)

        out_dir = tmp_path / "output"
        result = ingest(pq_file, out_dir, schema="sft")

        assert result["rows_out"] == 2
        assert result["schema"] == "sft"
        assert (out_dir / "dataset.jsonl").exists()
        assert (out_dir / "manifest.json").exists()
        assert (out_dir / "hashes.json").exists()

    def test_ingest_chat_schema(self, tmp_path):
        from verifily_cli_v1.commands.ingest import ingest

        rows = [
            {"messages": [
                {"role": "user", "content": "Hi there"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Great!"},
            ]},
            {"messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]},
        ]
        jsonl_file = tmp_path / "chat.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        out_dir = tmp_path / "output"
        result = ingest(jsonl_file, out_dir, schema="auto", expand_chat=True)

        # First row expands to 2 SFT pairs, second to 1
        assert result["rows_out"] == 3
        assert result["schema"] == "sft"

        # Verify output file
        with open(out_dir / "dataset.jsonl") as f:
            output_rows = [json.loads(line) for line in f if line.strip()]
        assert len(output_rows) == 3
        assert output_rows[0]["input"] == "Hi there"
        assert output_rows[0]["output"] == "Hello!"
