"""Tests for verifily_cli_v1.core.eval_metrics — NLP evaluation metrics."""

from __future__ import annotations

import math

import pytest

from verifily_cli_v1.core.eval_metrics import (
    EvalResult,
    _lcs_length,
    _normalize,
    _tokenize,
    bleu_score,
    compute_metrics,
    exact_match_score,
    length_ratio,
    rouge_l_score,
    token_f1_score,
)


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalize("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("  hello   world  ") == "hello world"

    def test_empty(self):
        assert _normalize("") == ""


class TestTokenize:
    def test_basic(self):
        assert _tokenize("The cat sat.") == ["the", "cat", "sat"]

    def test_empty(self):
        assert _tokenize("") == []


class TestExactMatch:
    def test_identical(self):
        assert exact_match_score("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        assert exact_match_score("Hello World", "hello world") == 1.0

    def test_punctuation_ignored(self):
        assert exact_match_score("hello, world!", "hello world") == 1.0

    def test_different(self):
        assert exact_match_score("hello", "goodbye") == 0.0

    def test_empty_both(self):
        assert exact_match_score("", "") == 1.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1_score("the cat sat", "the cat sat") == 1.0

    def test_partial_overlap(self):
        score = token_f1_score("the cat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert token_f1_score("hello", "goodbye") == 0.0

    def test_both_empty(self):
        assert token_f1_score("", "") == 1.0

    def test_one_empty(self):
        assert token_f1_score("", "hello") == 0.0
        assert token_f1_score("hello", "") == 0.0

    def test_symmetry(self):
        a, b = "the quick brown fox", "the slow brown dog"
        assert token_f1_score(a, b) == token_f1_score(b, a)


class TestBLEU:
    def test_identical(self):
        text = "the cat sat on the mat"
        score = bleu_score(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_empty_prediction(self):
        assert bleu_score("", "the cat sat") == 0.0

    def test_no_overlap(self):
        assert bleu_score("hello world", "goodbye moon") == 0.0

    def test_partial_overlap(self):
        score = bleu_score(
            "the cat sat on the mat",
            "the cat sat on a mat near the door",
        )
        assert 0.0 < score < 1.0

    def test_brevity_penalty(self):
        short = bleu_score("cat", "the cat sat on the mat")
        long_ = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert short < long_


class TestROUGEL:
    def test_identical(self):
        text = "the cat sat on the mat"
        assert rouge_l_score(text, text) == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        assert rouge_l_score("hello", "goodbye") == 0.0

    def test_partial(self):
        score = rouge_l_score("the cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_both_empty(self):
        assert rouge_l_score("", "") == 1.0

    def test_one_empty(self):
        assert rouge_l_score("", "hello") == 0.0
        assert rouge_l_score("hello", "") == 0.0


class TestLCSLength:
    def test_identical(self):
        tokens = ["a", "b", "c"]
        assert _lcs_length(tokens, tokens) == 3

    def test_no_common(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial(self):
        assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_empty(self):
        assert _lcs_length([], ["a", "b"]) == 0
        assert _lcs_length(["a"], []) == 0


class TestLengthRatio:
    def test_equal(self):
        assert length_ratio("hello world", "hello world") == pytest.approx(1.0)

    def test_shorter(self):
        ratio = length_ratio("hello", "hello world foo bar")
        assert ratio < 1.0

    def test_longer(self):
        ratio = length_ratio("hello world foo bar", "hello")
        assert ratio > 1.0

    def test_empty_reference(self):
        assert length_ratio("hello", "") == float("inf")

    def test_both_empty(self):
        assert length_ratio("", "") == 0.0


class TestComputeMetrics:
    def test_identical_pairs(self):
        preds = ["the cat sat", "the dog ran"]
        refs = ["the cat sat", "the dog ran"]
        result = compute_metrics(preds, refs)
        assert isinstance(result, EvalResult)
        assert result.exact_match == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.num_examples == 2

    def test_empty_lists(self):
        result = compute_metrics([], [])
        assert result.num_examples == 0
        assert result.exact_match == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_metrics(["a"], ["b", "c"])

    def test_per_example(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "baz qux"]
        result = compute_metrics(preds, refs, include_per_example=True)
        assert len(result.per_example) == 2
        assert result.per_example[0]["exact_match"] == 1.0
        assert result.per_example[1]["exact_match"] == 0.0

    def test_to_dict(self):
        result = compute_metrics(["hello"], ["hello"])
        d = result.to_dict()
        assert "overall" in d
        assert "num_examples" in d
        assert d["overall"]["exact_match"] == 1.0

    def test_realistic_scenario(self):
        preds = [
            "A neural network is a computer system inspired by the brain",
            "Gradient descent optimizes model weights step by step",
        ]
        refs = [
            "A neural network is a computer system inspired by the human brain. It has layers of nodes.",
            "Gradient descent is an optimization algorithm that adjusts model parameters step by step.",
        ]
        result = compute_metrics(preds, refs)
        assert result.f1 > 0.5
        assert result.rouge_l > 0.4
        assert result.num_examples == 2
