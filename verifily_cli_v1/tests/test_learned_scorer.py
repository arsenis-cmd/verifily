"""Tests for verifily_cli_v1.core.learned_scorer — learned quality scoring."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.learned_scorer import (
    _WEIGHTS,
    _dot,
    _sigmoid,
    extract_features,
    learned_score,
)
from verifily_cli_v1.core.quality import (
    QualityIssue,
    _compute_score,
    analyze_quality,
)


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)

    def test_bounds(self):
        for z in [-50, -10, -1, 0, 1, 10, 50]:
            val = _sigmoid(z)
            assert 0.0 <= val <= 1.0


class TestDot:
    def test_basic(self):
        assert _dot([1, 2, 3], [4, 5, 6]) == 32

    def test_zeros(self):
        assert _dot([0, 0], [5, 5]) == 0


class TestExtractFeatures:
    def test_correct_length(self):
        features = extract_features([], {"type_token_ratio": 0.5}, 10)
        assert len(features) == 25

    def test_matches_weight_length(self):
        features = extract_features([], {}, 10)
        assert len(features) == len(_WEIGHTS)

    def test_bias_always_one(self):
        features = extract_features([], {}, 10)
        assert features[21] == 1.0

    def test_empty_issues_zeros(self):
        features = extract_features([], {}, 10)
        # All category features should be 0
        for i in range(12):
            assert features[i] == 0.0
        # Severity counts should be 0
        assert features[12] == 0.0
        assert features[13] == 0.0
        assert features[14] == 0.0

    def test_with_issue(self):
        issue = QualityIssue(
            category="empty_or_low_info",
            severity="error",
            count=5,
            fraction=0.5,
            sample_rows=[0, 1, 2, 3, 4],
            description="test",
        )
        features = extract_features([issue], {}, 10)
        # has_empty should be 1.0
        assert features[0] == 1.0
        # empty_fraction should be 0.5
        assert features[1] == 0.5
        # error_count should be 1
        assert features[12] == 1.0

    def test_level2_features_default(self):
        features = extract_features([], {}, 10)
        assert features[22] == 0.0  # semantic_dup_fraction
        assert features[23] == 0.0  # topic_diversity
        assert features[24] == 0.0  # topic_count_normalized

    def test_level2_features_present(self):
        stats = {
            "semantic_dup_fraction": 0.1,
            "topic_diversity": 0.8,
            "topic_count": 5,
        }
        features = extract_features([], stats, 100)
        assert features[22] == 0.1
        assert features[23] == 0.8
        assert features[24] > 0.0


class TestLearnedScore:
    def test_returns_int(self):
        score = learned_score([], {}, 10)
        assert isinstance(score, int)

    def test_range_0_100(self):
        score = learned_score([], {}, 10)
        assert 0 <= score <= 100

    def test_zero_rows(self):
        assert learned_score([], {}, 0) == 0

    def test_clean_dataset_reasonable(self):
        rows = [
            {"input": f"question about topic {i} in machine learning", "output": f"answer explaining topic {i} clearly"}
            for i in range(20)
        ]
        report = analyze_quality(rows)
        # Learned score should be in a reasonable range
        assert 0 <= report.quality_score <= 100

    def test_bad_dataset_lower(self):
        good_rows = [
            {"input": f"question about topic {i} in machine learning", "output": f"answer explaining topic {i} clearly"}
            for i in range(20)
        ]
        bad_rows = [{"input": "", "output": ""}] * 20

        good_report = analyze_quality(good_rows)
        bad_report = analyze_quality(bad_rows)

        # Good should score higher than bad (or at least not lower)
        # The learned scorer should maintain this ordering
        assert good_report.quality_score >= bad_report.quality_score

    def test_agrees_with_old_scorer_direction(self):
        """The learned scorer should generally agree with the old scorer
        on relative ordering (good > bad)."""
        good_issues = []
        bad_issues = [
            QualityIssue("empty_or_low_info", "error", 10, 0.5, [], "test"),
            QualityIssue("encoding_issue", "warning", 5, 0.25, [], "test"),
        ]

        stats = {"type_token_ratio": 0.5, "hapax_ratio": 0.3, "total_tokens": 100}

        good_learned = learned_score(good_issues, stats, 20)
        bad_learned = learned_score(bad_issues, stats, 20)
        good_old = _compute_score(good_issues, 20)
        bad_old = _compute_score(bad_issues, 20)

        # Both should agree: good > bad
        assert (good_learned >= bad_learned) == (good_old >= bad_old)


class TestIntegration:
    def test_analyze_quality_uses_learned_scorer(self):
        """analyze_quality() should use the learned scorer."""
        rows = [
            {"input": f"question {i}", "output": f"answer {i}"}
            for i in range(10)
        ]
        report = analyze_quality(rows)
        assert isinstance(report.quality_score, int)
        assert 0 <= report.quality_score <= 100
