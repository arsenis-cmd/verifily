"""Tests for dataset fingerprinting and diff-datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifily_cli_v1.core.fingerprint import (
    DatasetFingerprint,
    compute_fingerprint,
    diff_fingerprints,
    load_fingerprint,
    write_fingerprint,
)
from verifily_cli_v1.commands.fingerprint_cmd import run_diff, run_fingerprint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FP_DEMO = REPO_ROOT / "examples" / "fingerprint_demo"

# ── Module-level cache to avoid recomputing fingerprints ──────────

_fp_cache: dict = {}


def _get_fp(name: str) -> DatasetFingerprint:
    """Compute fingerprint once per dataset name, cache for the module."""
    if name not in _fp_cache:
        _fp_cache[name] = compute_fingerprint(FP_DEMO / name, seed=42)
    return _fp_cache[name]


class TestFingerprintDeterminism:
    def test_same_dataset_same_fingerprint(self):
        fp1 = _get_fp("ds_a.jsonl")
        fp2 = compute_fingerprint(FP_DEMO / "ds_a.jsonl", seed=42)
        assert fp1.minhash_signature == fp2.minhash_signature
        assert fp1.exact_hash_sketch == fp2.exact_hash_sketch
        assert fp1.rows == fp2.rows
        assert fp1.exact_dup_rate == fp2.exact_dup_rate
        assert fp1.length_stats == fp2.length_stats

    def test_different_seed_different_minhash(self):
        fp1 = _get_fp("ds_a.jsonl")
        fp2 = compute_fingerprint(FP_DEMO / "ds_a.jsonl", seed=99)
        assert fp1.minhash_signature != fp2.minhash_signature


class TestFingerprintSimilarity:
    def test_neardup_higher_than_disjoint(self):
        fp_a = _get_fp("ds_a.jsonl")
        fp_b = _get_fp("ds_b_neardup.jsonl")
        fp_c = _get_fp("ds_c_disjoint.jsonl")

        diff_ab = diff_fingerprints(fp_a, fp_b)
        diff_ac = diff_fingerprints(fp_a, fp_c)

        assert diff_ab["similarity_score"] > diff_ac["similarity_score"]
        assert diff_ab["similarity_score"] > 0.5
        assert diff_ac["similarity_score"] < 0.3

    def test_exact_hash_overlap_neardup_higher(self):
        fp_a = _get_fp("ds_a.jsonl")
        fp_b = _get_fp("ds_b_neardup.jsonl")
        fp_c = _get_fp("ds_c_disjoint.jsonl")

        diff_ab = diff_fingerprints(fp_a, fp_b)
        diff_ac = diff_fingerprints(fp_a, fp_c)

        assert diff_ab["exact_dup_overlap_estimate"] > diff_ac["exact_dup_overlap_estimate"]


class TestFingerprintPrivacy:
    def test_no_raw_text_in_fingerprint(self):
        fp = _get_fp("ds_a.jsonl")
        fp_json = json.dumps(fp.to_dict())

        # Must not contain any recognizable input text
        assert "billing portal" not in fp_json.lower()
        assert "export button" not in fp_json.lower()
        assert "password reset" not in fp_json.lower()
        assert "Settings > Billing" not in fp_json

    def test_fingerprint_contains_no_raw_text_exhaustive(self):
        """Scan fingerprint JSON for ALL known text snippets from fixtures."""
        fp = _get_fp("ds_a.jsonl")
        fp_json = json.dumps(fp.to_dict()).lower()

        # Load actual row texts and verify none appear in fingerprint
        from verifily_cli_v1.core.io import read_jsonl
        rows = read_jsonl(FP_DEMO / "ds_a.jsonl")
        for row in rows:
            for key in ("input", "output"):
                if key in row and row[key]:
                    # Check multi-word phrases (3+ words) from the data
                    words = row[key].split()
                    for i in range(len(words) - 2):
                        phrase = " ".join(words[i:i + 3]).lower()
                        if len(phrase) > 10:
                            assert phrase not in fp_json, f"Raw text leaked: {phrase!r}"

    def test_diff_contains_no_raw_text(self):
        """Diff output must not contain any raw text from datasets."""
        fp_a = _get_fp("ds_a.jsonl")
        fp_b = _get_fp("ds_b_neardup.jsonl")
        diff = diff_fingerprints(fp_a, fp_b)
        diff_json = json.dumps(diff).lower()

        from verifily_cli_v1.core.io import read_jsonl
        for ds_name in ("ds_a.jsonl", "ds_b_neardup.jsonl"):
            rows = read_jsonl(FP_DEMO / ds_name)
            for row in rows:
                for key in ("input", "output"):
                    if key in row and row[key]:
                        words = row[key].split()
                        for i in range(len(words) - 2):
                            phrase = " ".join(words[i:i + 3]).lower()
                            if len(phrase) > 10:
                                assert phrase not in diff_json, f"Raw text leaked in diff: {phrase!r}"


class TestFingerprintFields:
    def test_schema_detection(self):
        assert _get_fp("ds_a.jsonl").schema == "sft"

    def test_row_count(self):
        assert _get_fp("ds_a.jsonl").rows == 15

    def test_length_stats_present(self):
        fp = _get_fp("ds_a.jsonl")
        assert "min" in fp.length_stats
        assert "median" in fp.length_stats
        assert "p95" in fp.length_stats
        assert "max" in fp.length_stats
        assert fp.length_stats["min"] > 0
        assert fp.length_stats["max"] >= fp.length_stats["min"]

    def test_minhash_has_128_values(self):
        assert len(_get_fp("ds_a.jsonl").minhash_signature) == 128

    def test_tag_distribution(self):
        fp = _get_fp("ds_a.jsonl")
        assert "source" in fp.tag_distribution
        assert "priority" in fp.tag_distribution

    def test_env_perm_override(self, monkeypatch):
        """VERIFILY_FINGERPRINT_PERMS env var controls permutation count."""
        monkeypatch.setenv("VERIFILY_FINGERPRINT_PERMS", "32")
        fp = compute_fingerprint(FP_DEMO / "ds_a.jsonl")
        assert len(fp.minhash_signature) == 32


class TestFingerprintIO:
    def test_write_and_load_roundtrip(self, tmp_path):
        fp = _get_fp("ds_a.jsonl")
        write_fingerprint(fp, tmp_path)

        loaded = load_fingerprint(tmp_path)
        assert loaded.rows == fp.rows
        assert loaded.minhash_signature == fp.minhash_signature
        assert loaded.schema == fp.schema

    def test_artifact_dir_input(self, tmp_path):
        """Test that compute_fingerprint works with artifact dirs containing dataset.jsonl."""
        from verifily_cli_v1.core.io import read_jsonl, write_jsonl

        art_dir = tmp_path / "artifact"
        art_dir.mkdir()
        rows = read_jsonl(FP_DEMO / "ds_a.jsonl")
        write_jsonl(art_dir / "dataset.jsonl", rows)

        fp = compute_fingerprint(str(art_dir))
        assert fp.rows == 15
        assert fp.schema == "sft"


class TestDiffDatasets:
    def test_diff_has_expected_keys(self):
        diff = diff_fingerprints(_get_fp("ds_a.jsonl"), _get_fp("ds_b_neardup.jsonl"))

        assert "similarity_score" in diff
        assert "exact_dup_overlap_estimate" in diff
        assert "row_delta" in diff
        assert "dup_rate_delta" in diff
        assert "length_deltas" in diff
        assert "tag_deltas" in diff

    def test_scores_in_range(self):
        diff = diff_fingerprints(_get_fp("ds_a.jsonl"), _get_fp("ds_c_disjoint.jsonl"))

        assert 0.0 <= diff["similarity_score"] <= 1.0
        assert 0.0 <= diff["exact_dup_overlap_estimate"] <= 1.0

    def test_run_diff_writes_output(self, tmp_path):
        result = run_diff(
            str(FP_DEMO / "ds_a.jsonl"),
            str(FP_DEMO / "ds_b_neardup.jsonl"),
            out=str(tmp_path),
        )
        assert (tmp_path / "diff.json").exists()
        assert (tmp_path / "diff.txt").exists()
        assert "diff_path" in result


class TestRunFingerprint:
    def test_writes_fingerprint_json(self, tmp_path):
        result = run_fingerprint(
            str(FP_DEMO / "ds_a.jsonl"),
            out=str(tmp_path),
        )
        assert (tmp_path / "fingerprint.json").exists()
        assert "fingerprint_path" in result
        assert result["rows"] == 15
