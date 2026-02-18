"""Tests for verifily_cli_v1.core.quality — text quality analysis."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.quality import (
    QualityIssue,
    QualityReport,
    analyze_quality,
    _check_empty_rows,
    _check_encoding,
    _check_length_outliers,
    _check_near_duplicates,
    _check_repetition,
    _check_class_imbalance,
    _row_text,
    _vocab_stats,
)


class TestRowText:
    def test_sft_fields(self):
        row = {"input": "hello", "output": "world"}
        assert "hello" in _row_text(row)
        assert "world" in _row_text(row)

    def test_fallback_to_string_fields(self):
        row = {"subject": "my ticket", "resolution": "fixed it"}
        text = _row_text(row)
        assert "my ticket" in text
        assert "fixed it" in text

    def test_empty_row(self):
        assert _row_text({}) == ""

    def test_ignores_id(self):
        row = {"id": "row_abc123", "input": "test"}
        text = _row_text(row)
        assert "row_abc" not in text
        assert "test" in text


class TestEmptyRows:
    def test_no_empty(self):
        texts = ["this is normal text", "another good row"]
        assert _check_empty_rows(texts) is None

    def test_detects_empty(self):
        texts = ["this is a good row with enough text", "", "  ", "ok"]
        issue = _check_empty_rows(texts)
        assert issue is not None
        assert issue.category == "empty_or_low_info"
        assert issue.count == 3

    def test_short_rows(self):
        texts = ["ok", "a", "hello world this is fine"]
        issue = _check_empty_rows(texts)
        assert issue is not None
        assert issue.count == 2


class TestLengthOutliers:
    def test_uniform_lengths(self):
        texts = ["word " * 20 for _ in range(10)]
        assert _check_length_outliers(texts) is None

    def test_detects_outlier(self):
        texts = ["word " * 20 for _ in range(20)]
        texts.append("word " * 200)
        issue = _check_length_outliers(texts)
        assert issue is not None
        assert issue.category == "length_outlier"

    def test_too_few_rows(self):
        assert _check_length_outliers(["short", "also"]) is None


class TestEncoding:
    def test_clean_text(self):
        texts = ["normal text", "another clean row"]
        assert _check_encoding(texts) is None

    def test_detects_mojibake(self):
        texts = ["clean", "Ã¢â€ broken encoding"]
        issue = _check_encoding(texts)
        assert issue is not None
        assert issue.category == "encoding_issue"
        assert issue.count == 1

    def test_detects_control_chars(self):
        texts = ["clean", "has \x00 null byte"]
        issue = _check_encoding(texts)
        assert issue is not None


class TestRepetition:
    def test_no_repetition(self):
        texts = ["the quick brown fox jumps over the lazy dog"]
        assert _check_repetition(texts) is None

    def test_detects_repeated_phrases(self):
        phrase = "buy now limited offer "
        texts = [phrase * 10]
        issue = _check_repetition(texts)
        assert issue is not None
        assert issue.category == "repetition"


class TestNearDuplicates:
    def test_no_duplicates(self):
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning is a subset of artificial intelligence",
        ]
        issue, count = _check_near_duplicates(texts)
        assert count == 0

    def test_detects_near_duplicates(self):
        base = "explain what a neural network is and how it works in modern deep learning applications"
        texts = [
            base,
            base + " using simple terms and examples",
            "completely different topic about cooking Italian pasta with tomato sauce and fresh basil",
        ]
        issue, count = _check_near_duplicates(texts, threshold=0.60)
        assert count >= 2

    def test_single_row(self):
        issue, count = _check_near_duplicates(["only one"])
        assert count == 0


class TestClassImbalance:
    def test_balanced(self):
        rows = [{"label": "A"}] * 50 + [{"label": "B"}] * 50
        assert _check_class_imbalance(rows) is None

    def test_imbalanced(self):
        rows = [{"label": "A"}] * 95 + [{"label": "B"}] * 3 + [{"label": "C"}] * 2
        issue = _check_class_imbalance(rows)
        assert issue is not None
        assert issue.category == "class_imbalance"

    def test_not_classification(self):
        rows = [{"input": "hello", "output": "world"}] * 10
        assert _check_class_imbalance(rows) is None


class TestVocabStats:
    def test_basic_stats(self):
        texts = ["the cat sat", "the dog ran", "a bird flew"]
        stats = _vocab_stats(texts)
        assert stats["total_tokens"] == 9
        assert stats["unique_tokens"] > 0
        assert 0 < stats["type_token_ratio"] <= 1.0
        assert stats["length_mean"] == 3.0

    def test_empty(self):
        stats = _vocab_stats([])
        assert stats["total_tokens"] == 0


class TestAnalyzeQuality:
    def test_clean_dataset(self):
        rows = [
            {"input": f"question number {i} about machine learning", "output": f"answer {i} about the topic"}
            for i in range(20)
        ]
        report = analyze_quality(rows)
        assert isinstance(report, QualityReport)
        assert report.quality_score >= 50
        assert report.total_rows == 20

    def test_empty_dataset(self):
        report = analyze_quality([])
        assert report.total_rows == 0
        assert report.quality_score == 0

    def test_bad_dataset_low_score(self):
        rows = [{"input": "", "output": ""}] * 10
        report = analyze_quality(rows)
        assert report.quality_score < 80  # penalized for all-empty rows
        assert any(i.category == "empty_or_low_info" for i in report.issues)

    def test_to_dict(self):
        rows = [{"input": "test", "output": "test"}] * 5
        report = analyze_quality(rows)
        d = report.to_dict()
        assert "quality_score" in d
        assert "issues" in d
        assert "stats" in d

    def test_issues_sorted_by_severity(self):
        # Mix of empty (error) and outlier (warning) rows
        rows = [{"input": "", "output": ""}] * 5
        rows += [{"input": "word " * 100, "output": "word " * 100}]
        rows += [{"input": "normal text here", "output": "normal text here"}] * 10
        report = analyze_quality(rows)
        if len(report.issues) >= 2:
            severities = [i.severity for i in report.issues]
            error_idx = [i for i, s in enumerate(severities) if s == "error"]
            warning_idx = [i for i, s in enumerate(severities) if s == "warning"]
            if error_idx and warning_idx:
                assert max(error_idx) < min(warning_idx)
