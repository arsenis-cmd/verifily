"""Tests for verifily_cli_v1.core.tfidf — TF-IDF vectorizer and cosine similarity."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.tfidf import (
    TfidfVectorizer,
    cosine_similarity,
    pairwise_cosine,
    tokenize,
)


class TestTokenize:
    def test_lowercases(self):
        tokens = tokenize("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stopwords(self):
        tokens = tokenize("the cat is on the mat")
        assert "the" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens

    def test_removes_short_tokens(self):
        tokens = tokenize("I a am good")
        assert "i" not in tokens
        assert "a" not in tokens

    def test_splits_punctuation(self):
        tokens = tokenize("hello, world! foo-bar")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty(self):
        assert tokenize("") == []


class TestTfidfVectorizer:
    def test_fit_sets_idf(self):
        vec = TfidfVectorizer()
        vec.fit(["cat dog", "dog bird", "bird fish"])
        assert len(vec._idf) > 0

    def test_transform_returns_sparse_vectors(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform(["cat dog", "dog bird"])
        assert len(vectors) == 2
        assert isinstance(vectors[0], dict)

    def test_l2_normalized(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform(["the quick brown fox jumps over lazy dog"])
        import math
        for v in vectors:
            if v:
                norm = math.sqrt(sum(x * x for x in v.values()))
                assert norm == pytest.approx(1.0, abs=0.01)

    def test_sublinear_tf(self):
        """Repeated terms should not dominate linearly."""
        vec = TfidfVectorizer()
        vectors = vec.fit_transform([
            "cat cat cat cat cat dog",
            "cat dog",
        ])
        # Both vectors should have 'cat' but the ratio shouldn't be 5:1
        if "cat" in vectors[0] and "cat" in vectors[1]:
            ratio = vectors[0].get("cat", 0) / vectors[1].get("cat", 1)
            assert ratio < 3.0  # Sublinear keeps it bounded

    def test_max_features(self):
        vec = TfidfVectorizer(max_features=3)
        vec.fit(["alpha beta gamma delta epsilon zeta"])
        assert len(vec._idf) <= 3

    def test_min_df(self):
        vec = TfidfVectorizer(min_df=2)
        vec.fit(["cat dog", "dog bird", "bird fish"])
        # "cat" and "fish" appear only once, should be filtered
        assert "cat" not in vec._idf
        assert "fish" not in vec._idf
        assert "dog" in vec._idf
        assert "bird" in vec._idf

    def test_empty_corpus(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform([])
        assert vectors == []

    def test_single_document(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform(["hello world machine learning"])
        assert len(vectors) == 1

    def test_get_top_terms(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform(["machine learning deep neural network"])
        top = vec.get_top_terms(vectors[0], n=3)
        assert len(top) <= 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = {"cat": 0.5, "dog": 0.5}
        sim = cosine_similarity(a, a)
        assert sim == pytest.approx(0.5, abs=0.01)  # dot product of unnormalized

    def test_orthogonal_vectors(self):
        a = {"cat": 1.0}
        b = {"dog": 1.0}
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vector(self):
        assert cosine_similarity({}, {"cat": 1.0}) == 0.0
        assert cosine_similarity({}, {}) == 0.0

    def test_partial_overlap(self):
        a = {"cat": 0.6, "dog": 0.8}
        b = {"cat": 0.6, "bird": 0.8}
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_normalized_identical(self):
        """Two identical normalized vectors should have cosine ~1.0."""
        vec = TfidfVectorizer()
        # Need a third doc so identical terms don't hit max_df_ratio filter
        vectors = vec.fit_transform([
            "machine learning model training",
            "machine learning model training",
            "cooking pasta tomato sauce",
        ])
        sim = cosine_similarity(vectors[0], vectors[1])
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similar_texts(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform([
            "machine learning model training optimization",
            "deep learning model training gradient descent",
            "cooking pasta tomato sauce basil oregano",
        ])
        sim_related = cosine_similarity(vectors[0], vectors[1])
        sim_unrelated = cosine_similarity(vectors[0], vectors[2])
        assert sim_related > sim_unrelated


class TestPairwiseCosine:
    def test_finds_similar_pairs(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform([
            "machine learning model training",
            "machine learning model optimization",
            "cooking italian pasta dinner",
        ])
        pairs = pairwise_cosine(vectors, threshold=0.3)
        # ML texts should be similar, cooking should not match
        pair_indices = {(p[0], p[1]) for p in pairs}
        assert (0, 1) in pair_indices

    def test_respects_threshold(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform([
            "alpha beta gamma",
            "delta epsilon zeta",
        ])
        # Very different texts, high threshold should find nothing
        pairs = pairwise_cosine(vectors, threshold=0.99)
        assert len(pairs) == 0

    def test_sorted_by_similarity(self):
        vec = TfidfVectorizer()
        vectors = vec.fit_transform([
            "cat dog animal pet",
            "cat dog animal pet friend",
            "car truck vehicle transport",
        ])
        pairs = pairwise_cosine(vectors, threshold=0.1)
        if len(pairs) >= 2:
            assert pairs[0][2] >= pairs[1][2]
