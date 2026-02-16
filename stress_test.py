#!/usr/bin/env python3
"""Verifily v1.1 — Comprehensive Stress Test.

Tests every major feature: all 7 schemas, Parquet, CSV, JSONL,
chat expansion, flatten, LSH contamination, PII, large datasets,
fingerprint, classify, report, contract-check, pipeline.
"""

import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────

PASS = 0
FAIL = 0
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


def run_all():
    global PASS, FAIL
    print("=" * 70)
    print("VERIFILY v1.1 — COMPREHENSIVE STRESS TEST")
    print("=" * 70)
    t0 = time.monotonic()

    for name, fn in TESTS:
        try:
            fn()
            PASS += 1
            print(f"  PASS  {name}")
        except Exception as e:
            FAIL += 1
            print(f"  FAIL  {name}")
            print(f"        {e}")

    elapsed = time.monotonic() - t0
    print("=" * 70)
    print(f"Results: {PASS} passed, {FAIL} failed ({elapsed:.1f}s)")
    print("=" * 70)
    return FAIL == 0


# ── Test data generators ────────────────────────────────────────

def make_sft_data(n=100):
    return [{"input": f"Question {i}: What is {i}?", "output": f"Answer: {i} is a number.",
             "tags": {"source": "synthetic", "idx": i}} for i in range(n)]

def make_qa_data(n=100):
    return [{"question": f"What is {i}?", "context": f"The number {i} is an integer.", "answer": f"{i}"}
            for i in range(n)]

def make_classification_data(n=100):
    labels = ["positive", "negative", "neutral"]
    return [{"text": f"This is review number {i}. It is great!", "label": labels[i % 3]}
            for i in range(n)]

def make_chat_data(n=50):
    rows = []
    for i in range(n):
        rows.append({"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": f"Answer {i}"},
            {"role": "user", "content": f"Follow-up {i}"},
            {"role": "assistant", "content": f"Follow-up answer {i}"},
        ]})
    return rows

def make_summarization_data(n=100):
    return [{"document": f"This is a long document number {i}. " * 10, "summary": f"Summary of doc {i}."}
            for i in range(n)]

def make_translation_data(n=100):
    return [{"source": f"Hello world {i}", "target": f"Bonjour le monde {i}",
             "source_lang": "en", "target_lang": "fr"} for i in range(n)]

def make_rm_pairwise_data(n=100):
    return [{"prompt": f"Write about {i}", "chosen": f"Good response {i}", "rejected": f"Bad response {i}"}
            for i in range(n)]

def make_nested_data(n=50):
    return [{"input": f"q {i}", "output": f"a {i}", "meta": {"author": f"user_{i}", "source": {"origin": "web", "page": i}}}
            for i in range(n)]

def make_pii_data(n=30):
    rows = []
    for i in range(n):
        if i % 3 == 0:
            rows.append({"text": f"Email john{i}@example.com for info", "label": "pii"})
        elif i % 3 == 1:
            rows.append({"text": f"Call 555-{i:03d}-1234 today", "label": "pii"})
        else:
            rows.append({"text": f"No personal info here, just item {i}", "label": "clean"})
    return rows

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path, rows):
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════

# ── 1. Schema Registry ──────────────────────────────────────────

@test("Schema registry: all 7 schemas registered")
def _():
    from verifily_cli_v1.core.schemas import SCHEMA_REGISTRY, schema_names
    names = schema_names()
    assert len(names) == 7, f"Expected 7 schemas, got {len(names)}: {names}"
    for expected in ["sft", "qa", "classification", "chat", "summarization", "translation", "rm_pairwise"]:
        assert expected in SCHEMA_REGISTRY, f"Missing schema: {expected}"

@test("Schema detection: all 7 schemas auto-detected correctly")
def _():
    from verifily_cli_v1.core.schemas import detect_schema_from_fields
    assert detect_schema_from_fields({"input", "output"}) == "sft"
    assert detect_schema_from_fields({"question", "context", "answer"}) == "qa"
    assert detect_schema_from_fields({"text", "label"}) == "classification"
    assert detect_schema_from_fields({"messages"}) == "chat"
    assert detect_schema_from_fields({"document", "summary"}) == "summarization"
    assert detect_schema_from_fields({"source", "target", "source_lang"}) == "translation"
    assert detect_schema_from_fields({"prompt", "chosen", "rejected"}) == "rm_pairwise"

# ── 2. Readers ──────────────────────────────────────────────────

@test("Reader: CSV roundtrip (100 rows)")
def _():
    from verifily_cli_v1.core.readers import read_dataset
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.csv"
        write_csv(path, make_classification_data(100))
        result = read_dataset(path)
        assert result.format == "csv"
        assert result.row_count == 100

@test("Reader: JSONL roundtrip (500 rows)")
def _():
    from verifily_cli_v1.core.readers import read_dataset
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.jsonl"
        write_jsonl(path, make_sft_data(500))
        result = read_dataset(path)
        assert result.format == "jsonl"
        assert result.row_count == 500

@test("Reader: Parquet roundtrip (1000 rows)")
def _():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from verifily_cli_v1.core.readers import read_dataset
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.parquet"
        data = make_sft_data(1000)
        table = pa.table({
            "input": [r["input"] for r in data],
            "output": [r["output"] for r in data],
        })
        pq.write_table(table, path)
        result = read_dataset(path)
        assert result.format == "parquet"
        assert result.row_count == 1000
        assert "num_row_groups" in result.provenance

@test("Reader: Parquet with limit=50")
def _():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from verifily_cli_v1.core.readers import read_dataset
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.parquet"
        table = pa.table({"text": [f"row {i}" for i in range(200)], "label": ["pos"] * 200})
        pq.write_table(table, path)
        result = read_dataset(path, limit=50)
        assert result.row_count == 50

@test("Reader: TSV support")
def _():
    from verifily_cli_v1.core.readers import read_dataset
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.tsv"
        path.write_text("input\toutput\nhello\tworld\nfoo\tbar\n")
        result = read_dataset(path)
        assert result.format == "csv"
        assert result.row_count == 2

@test("Reader: unsupported extension raises ValueError")
def _():
    from verifily_cli_v1.core.readers import read_dataset
    try:
        read_dataset("/fake/path.xlsx")
        assert False, "Should have raised"
    except ValueError as e:
        assert "Unsupported" in str(e)

# ── 3. Flatten ──────────────────────────────────────────────────

@test("Flatten: nested dict 3 levels deep")
def _():
    from verifily_cli_v1.core.flatten import flatten_row
    row = {"a": {"b": {"c": 42}}, "x": 1}
    flat = flatten_row(row)
    assert flat == {"a.b.c": 42, "x": 1}

@test("Flatten: preserves tags and messages")
def _():
    from verifily_cli_v1.core.flatten import flatten_row
    row = {"input": "hi", "tags": {"source": "human", "lang": "en"}, "messages": [{"role": "user"}]}
    flat = flatten_row(row)
    assert flat["tags"] == {"source": "human", "lang": "en"}
    assert flat["messages"] == [{"role": "user"}]

@test("Flatten: custom separator")
def _():
    from verifily_cli_v1.core.flatten import flatten_row
    row = {"meta": {"author": "Alice"}}
    flat = flatten_row(row, separator="__")
    assert flat == {"meta__author": "Alice"}

@test("Chat expansion: single turn")
def _():
    from verifily_cli_v1.core.flatten import expand_chat_turns
    row = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    pairs = expand_chat_turns(row)
    assert len(pairs) == 1
    assert pairs[0]["input"] == "Hi"
    assert pairs[0]["output"] == "Hello"

@test("Chat expansion: multi-turn with system prompt")
def _():
    from verifily_cli_v1.core.flatten import expand_chat_turns
    row = {"messages": [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"}, {"role": "assistant", "content": "A3"},
    ]}
    pairs = expand_chat_turns(row)
    assert len(pairs) == 3
    assert pairs[0]["tags"]["system_prompt"] == "Be concise"
    assert pairs[2]["tags"]["turn"] == 2

@test("Chat expansion: 50 conversations → 100 SFT pairs")
def _():
    from verifily_cli_v1.core.flatten import flatten_rows
    data = make_chat_data(50)
    expanded = flatten_rows(data, expand_chat=True)
    assert len(expanded) == 100  # each has 2 user/assistant pairs

@test("Flatten rows: expand_chat=False preserves messages")
def _():
    from verifily_cli_v1.core.flatten import flatten_rows
    data = [{"messages": [{"role": "user", "content": "Hi"}]}]
    result = flatten_rows(data, expand_chat=False)
    assert len(result) == 1
    assert "messages" in result[0]

# ── 4. PII Detection ────────────────────────────────────────────

@test("PII regex: detects all 5 types")
def _():
    from verifily_cli_v1.core.pii import scan_text_regex
    text = "Email: test@example.com, Phone: 555-123-4567, SSN: 123-45-6789, IP: 192.168.1.1, CC: 4111-1111-1111-1111"
    result = scan_text_regex(text)
    assert len(result.pii_types_found) == 5

@test("PII scan_dataset: format compatible with report.py")
def _():
    from verifily_cli_v1.core.pii import scan_dataset
    rows = make_pii_data(30)
    result = scan_dataset(rows)
    assert "pii_scan" in result
    assert "pii_clean" in result
    assert "pii_total_hits" in result
    assert "pii_confidence" in result
    assert result["pii_clean"] is False
    assert result["pii_total_hits"] > 0

@test("PII confidence filtering: min_confidence=0.9 filters low-confidence")
def _():
    from verifily_cli_v1.core.pii import scan_dataset
    rows = [{"text": "Call 555-123-4567", "label": "x"}]  # phone has confidence 0.60
    result_all = scan_dataset(rows, min_confidence=0.0)
    result_high = scan_dataset(rows, min_confidence=0.90)
    assert result_all["pii_total_hits"] >= 1
    assert result_high["pii_total_hits"] == 0  # phone filtered out

@test("PII NER fallback: graceful when spaCy missing")
def _():
    from verifily_cli_v1.core.pii import scan_text_ner
    result = scan_text_ner("Contact john@example.com", nlp=None)
    assert result.has_pii  # still finds email via regex

@test("PII scan_row: handles chat messages")
def _():
    from verifily_cli_v1.core.pii import scan_row
    row = {"messages": [{"role": "user", "content": "My email is test@test.com"}]}
    result = scan_row(row)
    assert "email" in result.pii_types_found

# ── 5. MinHash LSH ──────────────────────────────────────────────

@test("LSH: identical documents → similarity 1.0")
def _():
    from verifily_cli_v1.core.minhash_lsh import MinHashLSH, minhash_jaccard
    lsh = MinHashLSH(num_perm=128, threshold=0.70)
    shingles = {"the cat", "cat sat", "sat on", "on the", "the mat"}
    lsh.insert(0, shingles)
    sig = lsh.signatures[0]
    assert minhash_jaccard(sig, sig) == 1.0

@test("LSH: finds near-duplicate with threshold 0.5")
def _():
    from verifily_cli_v1.core.minhash_lsh import MinHashLSH, minhash_jaccard
    lsh = MinHashLSH(num_perm=128, threshold=0.50)
    base = {"the quick brown", "quick brown fox", "brown fox jumps", "fox jumps over",
            "jumps over the", "over the lazy", "the lazy dog", "lazy dog today"}
    lsh.insert(0, base)
    # 6 of 8 shingles shared → Jaccard ~ 0.6
    similar = {"the quick brown", "quick brown fox", "brown fox jumps", "fox jumps over",
               "jumps over the", "over the lazy", "the sleeping cat", "sleeping cat now"}
    candidates = lsh.query(similar)
    assert 0 in candidates

@test("LSH: deterministic across instances")
def _():
    from verifily_cli_v1.core.minhash_lsh import MinHashLSH
    lsh1 = MinHashLSH(num_perm=64, seed=42)
    lsh2 = MinHashLSH(num_perm=64, seed=42)
    s = {"hello world", "world is", "is great"}
    lsh1.insert(0, s)
    lsh2.insert(0, s)
    assert lsh1.signatures[0] == lsh2.signatures[0]

@test("LSH: 10K documents indexed without error")
def _():
    from verifily_cli_v1.core.minhash_lsh import MinHashLSH
    lsh = MinHashLSH(num_perm=64, threshold=0.70)
    for i in range(10_000):
        shingles = {f"word{i}_{j}" for j in range(5)}
        lsh.insert(i, shingles)
    assert len(lsh.signatures) == 10_000

# ── 6. Ingest — All Schemas ─────────────────────────────────────

@test("Ingest: SFT from JSONL (200 rows)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_sft_data(200))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="auto")
        assert result["schema"] == "sft"
        assert result["rows_out"] == 200

@test("Ingest: QA from JSONL (100 rows)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_qa_data(100))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="auto")
        assert result["schema"] == "qa"
        assert result["rows_out"] == 100

@test("Ingest: Classification from CSV (100 rows)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_csv(Path(d) / "in.csv", make_classification_data(100))
        result = ingest(Path(d) / "in.csv", Path(d) / "out", schema="auto")
        assert result["schema"] == "classification"
        assert result["rows_out"] == 100

@test("Ingest: Chat → expanded SFT pairs (50 convos → 100 pairs)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_chat_data(50))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="auto", expand_chat=True)
        assert result["schema"] == "sft"
        assert result["rows_out"] == 100

@test("Ingest: Summarization from JSONL (100 rows)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_summarization_data(100))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="auto")
        assert result["schema"] == "summarization"
        assert result["rows_out"] == 100

@test("Ingest: Translation from JSONL (100 rows)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_translation_data(100))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="auto")
        assert result["schema"] == "translation"
        assert result["rows_out"] == 100

@test("Ingest: RM Pairwise from JSONL (100 rows)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_rm_pairwise_data(100))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="auto")
        assert result["schema"] == "rm_pairwise"
        assert result["rows_out"] == 100

@test("Ingest: Parquet SFT (500 rows)")
def _():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        data = make_sft_data(500)
        table = pa.table({"input": [r["input"] for r in data], "output": [r["output"] for r in data]})
        pq.write_table(table, Path(d) / "in.parquet")
        result = ingest(Path(d) / "in.parquet", Path(d) / "out", schema="auto")
        assert result["schema"] == "sft"
        assert result["rows_out"] == 500

@test("Ingest: nested fields get flattened")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_nested_data(50))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="sft")
        assert result["rows_out"] == 50

@test("Ingest: artifacts all present (dataset.jsonl, manifest.json, hashes.json, report.json)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_sft_data(10))
        ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="sft")
        out = Path(d) / "out"
        for f in ["dataset.jsonl", "manifest.json", "hashes.json", "report.json"]:
            assert (out / f).exists(), f"Missing artifact: {f}"

@test("Ingest: dry-run does not write files")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_sft_data(10))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="sft", dry_run=True)
        assert result["dry_run"] is True
        assert not (Path(d) / "out").exists()

@test("Ingest: --limit truncates input")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "in.jsonl", make_sft_data(100))
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="sft", limit=25)
        assert result["rows_in"] == 25
        assert result["rows_out"] == 25

@test("Ingest: column mapping works")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        rows = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(20)]
        write_jsonl(Path(d) / "in.jsonl", rows)
        result = ingest(Path(d) / "in.jsonl", Path(d) / "out", schema="sft",
                       mapping={"input": "prompt", "output": "completion"})
        assert result["rows_out"] == 20

# ── 7. Contamination ────────────────────────────────────────────

@test("Contamination LSH: exact overlaps detected (20 train, 5 eval with 2 leaks)")
def _():
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        train = make_sft_data(20)
        eval_rows = [train[0], train[5], {"input": "unique1", "output": "unique1"}, {"input": "unique2", "output": "unique2"}, {"input": "unique3", "output": "unique3"}]
        write_jsonl(Path(d) / "train.jsonl", train)
        write_jsonl(Path(d) / "eval.jsonl", eval_rows)
        result = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl", use_lsh=True)
        assert result["exact_overlaps"] == 2
        assert result["status"] == "FAIL"
        assert result["method"] == "lsh"

@test("Contamination brute-force: same results as LSH on small data")
def _():
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        train = make_sft_data(30)
        eval_rows = [train[0], train[10], {"input": "novel", "output": "novel"}]
        write_jsonl(Path(d) / "train.jsonl", train)
        write_jsonl(Path(d) / "eval.jsonl", eval_rows)
        r_lsh = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl", use_lsh=True)
        r_bf = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl", use_lsh=False)
        assert r_lsh["exact_overlaps"] == r_bf["exact_overlaps"]

@test("Contamination: sampling reduces dataset size")
def _():
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "train.jsonl", make_sft_data(500))
        write_jsonl(Path(d) / "eval.jsonl", make_sft_data(200))
        result = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl",
                                     sample_train=50, sample_eval=20)
        assert result["train_rows"] == 50
        assert result["eval_rows"] == 20
        assert result["sampled"] is True

@test("Contamination: near-duplicate detection via LSH")
def _():
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        train = [{"input": "The quick brown fox jumps over the lazy dog", "output": "response"}]
        eval_rows = [{"input": "The quick brown fox jumps over a lazy dog", "output": "response"}]  # near-dup
        write_jsonl(Path(d) / "train.jsonl", train)
        write_jsonl(Path(d) / "eval.jsonl", eval_rows)
        result = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl",
                                     use_lsh=True, jaccard_cutoff=0.5)
        assert result["near_duplicates"] >= 1 or result["exact_overlaps"] >= 1

@test("Contamination: PASS on clean data")
def _():
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        train = [{"input": f"train unique {i}", "output": f"tout {i}"} for i in range(50)]
        eval_rows = [{"input": f"eval unique {i}", "output": f"eout {i}"} for i in range(20)]
        write_jsonl(Path(d) / "train.jsonl", train)
        write_jsonl(Path(d) / "eval.jsonl", eval_rows)
        result = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl")
        assert result["status"] == "PASS"
        assert result["exit_code"] == 0

@test("Contamination: 10K train × 1K eval with LSH (performance)")
def _():
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        train = [{"input": f"train text number {i} about topic {i % 100}", "output": f"response {i}"} for i in range(10_000)]
        eval_rows = [{"input": f"eval question number {i} on subject {i % 50}", "output": f"answer {i}"} for i in range(1_000)]
        write_jsonl(Path(d) / "train.jsonl", train)
        write_jsonl(Path(d) / "eval.jsonl", eval_rows)
        t0 = time.monotonic()
        result = check_contamination(Path(d) / "train.jsonl", Path(d) / "eval.jsonl", use_lsh=True)
        elapsed = time.monotonic() - t0
        assert result["status"] in ("PASS", "WARN", "FAIL")
        assert elapsed < 30, f"Too slow: {elapsed:.1f}s (expected <30s)"

# ── 8. Report ───────────────────────────────────────────────────

@test("Report: field stats and PII scan")
def _():
    from verifily_cli_v1.commands.report import dataset_report
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "data.jsonl", make_pii_data(30))
        report = dataset_report(Path(d) / "data.jsonl", schema="classification")
        assert report["row_count"] == 30
        assert report["pii_clean"] is False
        assert "email" in report["pii_scan"]

@test("Report: PII confidence filtering")
def _():
    from verifily_cli_v1.commands.report import dataset_report
    with tempfile.TemporaryDirectory() as d:
        rows = [{"text": "Call 555-123-4567", "label": "x"}]
        write_jsonl(Path(d) / "data.jsonl", rows)
        report_all = dataset_report(Path(d) / "data.jsonl", schema="classification", min_confidence=0.0)
        report_high = dataset_report(Path(d) / "data.jsonl", schema="classification", min_confidence=0.9)
        assert report_all["pii_total_hits"] >= 1
        assert report_high["pii_total_hits"] == 0

# ── 9. Classify ─────────────────────────────────────────────────

@test("Classify: detects SFT schema")
def _():
    from verifily_cli_v1.core.classify import classify_dataset
    result = classify_dataset(make_sft_data(50))
    assert result["suggested_schema"] == "sft"

@test("Classify: detects chat schema")
def _():
    from verifily_cli_v1.core.classify import classify_dataset
    result = classify_dataset(make_chat_data(20))
    assert result["suggested_schema"] == "chat"

@test("Classify: detects QA schema")
def _():
    from verifily_cli_v1.core.classify import classify_dataset
    result = classify_dataset(make_qa_data(20))
    assert result["suggested_schema"] == "qa"

@test("Classify: PII risk flagged")
def _():
    from verifily_cli_v1.core.classify import classify_dataset
    result = classify_dataset(make_pii_data(30))
    assert result["tags_summary"]["pii_risk"]["rows_with_pii"] > 0

# ── 10. Fingerprint ─────────────────────────────────────────────

@test("Fingerprint: compute and diff")
def _():
    from verifily_cli_v1.core.fingerprint import compute_fingerprint, diff_fingerprints
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "a.jsonl", make_sft_data(100))
        write_jsonl(Path(d) / "b.jsonl", make_sft_data(100) + make_sft_data(50))
        fp_a = compute_fingerprint(Path(d) / "a.jsonl")
        fp_b = compute_fingerprint(Path(d) / "b.jsonl")
        assert fp_a.rows == 100
        assert fp_b.rows == 150
        diff = diff_fingerprints(fp_a, fp_b)
        assert diff["row_delta"] == 50
        assert 0.0 <= diff["similarity_score"] <= 1.0

# ── 11. Contract Check ──────────────────────────────────────────

@test("Contract check: validates SFT dataset (instruction/output)")
def _():
    from verifily_cli_v1.commands.contract_check import validate_dataset_schema
    with tempfile.TemporaryDirectory() as d:
        rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]
        write_jsonl(Path(d) / "data.jsonl", rows)
        result = validate_dataset_schema(Path(d) / "data.jsonl", "sft")
        assert result["valid"] is True

@test("Contract check: validates classification dataset")
def _():
    from verifily_cli_v1.commands.contract_check import validate_dataset_schema
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "data.jsonl", make_classification_data(10))
        result = validate_dataset_schema(Path(d) / "data.jsonl", "classification")
        assert result["valid"] is True

@test("Contract check: rejects wrong schema")
def _():
    from verifily_cli_v1.commands.contract_check import validate_dataset_schema
    with tempfile.TemporaryDirectory() as d:
        write_jsonl(Path(d) / "data.jsonl", make_sft_data(10))
        result = validate_dataset_schema(Path(d) / "data.jsonl", "classification")
        assert result["valid"] is False

# ── 12. End-to-End Pipeline ─────────────────────────────────────

@test("E2E: ingest → report → contamination (full pipeline)")
def _():
    from verifily_cli_v1.commands.ingest import ingest
    from verifily_cli_v1.commands.report import dataset_report
    from verifily_cli_v1.commands.contamination import check_contamination
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Ingest train
        write_jsonl(d / "train_raw.jsonl", make_sft_data(200))
        ingest(d / "train_raw.jsonl", d / "train_out", schema="sft")
        # Ingest eval
        eval_data = make_sft_data(50)  # some overlap with train
        eval_data.extend([{"input": f"unique eval {i}", "output": f"uout {i}"} for i in range(50)])
        write_jsonl(d / "eval_raw.jsonl", eval_data)
        ingest(d / "eval_raw.jsonl", d / "eval_out", schema="sft")
        # Report
        report = dataset_report(d / "train_out" / "dataset.jsonl", schema="sft")
        assert report["row_count"] == 200
        # Contamination
        result = check_contamination(
            d / "train_out" / "dataset.jsonl",
            d / "eval_out" / "dataset.jsonl",
            use_lsh=True,
        )
        assert result["status"] in ("PASS", "WARN", "FAIL")

@test("E2E: chat ingest → Parquet round trip")
def _():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from verifily_cli_v1.commands.ingest import ingest
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Create chat data, ingest to SFT
        write_jsonl(d / "chat.jsonl", make_chat_data(20))
        result = ingest(d / "chat.jsonl", d / "chat_out", schema="auto", expand_chat=True)
        assert result["rows_out"] == 40  # 20 convos × 2 turns each
        # Read output and write to Parquet
        with open(d / "chat_out" / "dataset.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        table = pa.table({
            "input": [r["input"] for r in rows],
            "output": [r["output"] for r in rows],
        })
        pq.write_table(table, d / "reexport.parquet")
        # Re-ingest from Parquet
        result2 = ingest(d / "reexport.parquet", d / "pq_out", schema="sft")
        assert result2["rows_out"] == 40


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
