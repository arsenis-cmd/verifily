"""Tests for verifily_cli_v1.core.model_judge — ML quality judge."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.model_judge import (
    QualityJudge,
    _has_model_judge,
    judge_quality,
)


class TestAvailability:
    def test_has_model_judge_returns_bool(self):
        assert isinstance(_has_model_judge(), bool)


class TestGracefulDegradation:
    def test_judge_quality_returns_dict_or_none(self):
        """judge_quality should always return a dict or None, never raise."""
        texts = ["hello world this is a test", "another test sentence here"]
        result = judge_quality(texts)
        assert result is None or isinstance(result, dict)

    def test_judge_quality_with_tfidf_fallback(self):
        """Force TF-IDF backend — should always work."""
        texts = [
            "machine learning model training optimization",
            "deep neural network architecture layers",
            "cooking pasta tomato sauce italian food",
        ]
        result = judge_quality(texts, prefer_backend="tfidf")
        assert result is not None
        assert "model_quality_score" in result
        assert 0 <= result["model_quality_score"] <= 100
        assert result["model_backend"] == "tfidf"

    def test_empty_texts(self):
        result = judge_quality([], prefer_backend="tfidf")
        assert result is not None
        assert result["model_quality_score"] == 0


class TestQualityJudgeTfidf:
    """Tests using TF-IDF backend (always available)."""

    def test_judge_rows_returns_scores(self):
        judge = QualityJudge(prefer_backend="tfidf")
        texts = [
            "Gradient descent iteratively adjusts parameters to minimize error",
            "A neural network consists of layers of interconnected nodes",
            "",
        ]
        scores = judge.judge_rows(texts)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_judge_dataset_has_required_keys(self):
        judge = QualityJudge(prefer_backend="tfidf")
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with many layers",
        ]
        result = judge.judge_dataset(texts)
        assert "model_quality_score" in result
        assert "per_row_mean" in result
        assert "low_quality_count" in result
        assert "high_quality_fraction" in result
        assert "model_backend" in result

    def test_score_in_range(self):
        judge = QualityJudge(prefer_backend="tfidf")
        texts = ["hello world test sentence number one"] * 5
        result = judge.judge_dataset(texts)
        assert 0 <= result["model_quality_score"] <= 100
        assert 0.0 <= result["per_row_mean"] <= 1.0
        assert 0.0 <= result["high_quality_fraction"] <= 1.0
