"""Tests for verifily_cli_v1.core.embeddings — embedding backends."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.embeddings import (
    TfidfEmbedder,
    _has_torch,
    _has_transformers,
    cosine_similarity_dense,
    get_embedder,
    pairwise_cosine_dense,
)


def _skip_no_torch():
    return pytest.mark.skipif(not _has_torch(), reason="torch not installed")


def _skip_no_transformers():
    return pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")


class TestAvailabilityChecks:
    def test_has_torch_returns_bool(self):
        assert isinstance(_has_torch(), bool)

    def test_has_transformers_returns_bool(self):
        assert isinstance(_has_transformers(), bool)


class TestTfidfEmbedder:
    def test_embed_returns_correct_shape(self):
        embedder = TfidfEmbedder(max_dim=100)
        texts = ["hello world machine learning", "deep neural network training"]
        vectors = embedder.embed(texts)
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(x, float) for x in vectors[0])

    def test_backend_is_tfidf(self):
        embedder = TfidfEmbedder()
        assert embedder.backend == "tfidf"

    def test_dim_matches_vectors(self):
        embedder = TfidfEmbedder(max_dim=50)
        texts = ["hello world foo bar", "baz qux machine learning"]
        vectors = embedder.embed(texts)
        # After embedding, dim should match the actual vector length
        assert len(vectors[0]) == embedder.dim

    def test_cosine_identical_texts(self):
        embedder = TfidfEmbedder()
        texts = [
            "machine learning model training",
            "machine learning model training",
            "cooking pasta dinner recipe",
        ]
        vectors = embedder.embed(texts)
        sim = cosine_similarity_dense(vectors[0], vectors[1])
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similar_vs_dissimilar(self):
        embedder = TfidfEmbedder()
        texts = [
            "machine learning neural network deep learning",
            "deep learning model training optimization",
            "cooking pasta tomato sauce italian food",
        ]
        vectors = embedder.embed(texts)
        sim_ml = cosine_similarity_dense(vectors[0], vectors[1])
        sim_diff = cosine_similarity_dense(vectors[0], vectors[2])
        assert sim_ml > sim_diff

    def test_empty_text(self):
        embedder = TfidfEmbedder()
        vectors = embedder.embed(["", "hello world machine"])
        assert len(vectors) == 2


class TestGetEmbedder:
    def test_tfidf_always_available(self):
        embedder = get_embedder(prefer="tfidf")
        assert embedder.backend == "tfidf"

    def test_auto_returns_something(self):
        embedder = get_embedder(prefer="auto")
        assert embedder.backend in ("tfidf", "transformer")

    def test_transformer_raises_if_unavailable(self):
        if not _has_torch() or not _has_transformers():
            with pytest.raises(ImportError):
                get_embedder(prefer="transformer")


class TestCosineSimilarityDense:
    def test_identical(self):
        a = [0.6, 0.8]
        assert cosine_similarity_dense(a, a) == pytest.approx(1.0, abs=0.01)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity_dense(a, b) == pytest.approx(0.0, abs=0.01)

    def test_zero_vector(self):
        assert cosine_similarity_dense([0, 0], [1, 1]) == 0.0

    def test_negative_correlation(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        sim = cosine_similarity_dense(a, b)
        assert sim == 0.0  # clamped to 0


class TestPairwiseCosineDense:
    def test_finds_similar(self):
        vectors = [
            [0.9, 0.1, 0.0],
            [0.85, 0.15, 0.0],
            [0.0, 0.0, 1.0],
        ]
        pairs = pairwise_cosine_dense(vectors, threshold=0.9)
        pair_indices = {(p[0], p[1]) for p in pairs}
        assert (0, 1) in pair_indices

    def test_respects_threshold(self):
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        pairs = pairwise_cosine_dense(vectors, threshold=0.5)
        assert len(pairs) == 0
