"""Tests for verifily_cli_v1.core.topic_cluster — K-means topic clustering."""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.tfidf import TfidfVectorizer
from verifily_cli_v1.core.topic_cluster import (
    auto_k,
    extract_topics,
    kmeans_sparse,
    topic_diversity_score,
)


def _make_vectors(texts):
    vec = TfidfVectorizer()
    return vec.fit_transform(texts)


class TestKmeansSparse:
    def test_two_clear_clusters(self):
        texts = [
            "machine learning neural network deep learning",
            "neural network training optimization gradient",
            "deep learning model architecture layers",
            "cooking pasta tomato sauce italian",
            "recipe dinner meal preparation food",
            "kitchen cooking baking dessert cake",
        ]
        vectors = _make_vectors(texts)
        assignments, centroids = kmeans_sparse(vectors, k=2, seed=42)
        assert len(assignments) == 6
        assert len(centroids) == 2
        # ML texts should cluster together, cooking texts together
        ml_clusters = {assignments[i] for i in range(3)}
        food_clusters = {assignments[i] for i in range(3, 6)}
        assert len(ml_clusters) == 1  # all ML in same cluster
        assert len(food_clusters) == 1  # all food in same cluster
        assert ml_clusters != food_clusters

    def test_k_larger_than_n(self):
        vectors = _make_vectors(["hello world", "foo bar"])
        assignments, centroids = kmeans_sparse(vectors, k=10)
        assert len(assignments) == 2
        assert len(centroids) <= 2

    def test_single_vector(self):
        vectors = _make_vectors(["only one document"])
        assignments, centroids = kmeans_sparse(vectors, k=3)
        assert assignments == [0]
        assert len(centroids) == 1

    def test_empty(self):
        assignments, centroids = kmeans_sparse([], k=3)
        assert assignments == []
        assert centroids == []

    def test_deterministic_with_seed(self):
        texts = [f"document about topic {i} in field {i % 3}" for i in range(20)]
        vectors = _make_vectors(texts)
        a1, c1 = kmeans_sparse(vectors, k=3, seed=123)
        a2, c2 = kmeans_sparse(vectors, k=3, seed=123)
        assert a1 == a2


class TestExtractTopics:
    def test_returns_top_terms(self):
        texts = [
            "machine learning algorithm model",
            "machine learning neural network",
            "cooking recipe food dinner",
            "cooking kitchen meal preparation",
        ]
        vectors = _make_vectors(texts)
        assignments, centroids = kmeans_sparse(vectors, k=2, seed=42)
        topics = extract_topics(vectors, assignments, centroids, top_n=3)
        assert len(topics) == 2
        for topic in topics:
            assert "top_terms" in topic
            assert len(topic["top_terms"]) <= 3
            assert "size" in topic
            assert topic["size"] > 0

    def test_cluster_sizes_sum_to_n(self):
        texts = [f"document {i} about various topics" for i in range(10)]
        vectors = _make_vectors(texts)
        assignments, centroids = kmeans_sparse(vectors, k=3, seed=42)
        topics = extract_topics(vectors, assignments, centroids)
        total = sum(t["size"] for t in topics)
        assert total == 10

    def test_coherence_bounded(self):
        texts = [f"word{i} word{i+1}" for i in range(10)]
        vectors = _make_vectors(texts)
        assignments, centroids = kmeans_sparse(vectors, k=2, seed=42)
        topics = extract_topics(vectors, assignments, centroids)
        for topic in topics:
            assert 0.0 <= topic["coherence"] <= 1.0


class TestTopicDiversity:
    def test_identical_topics_low_diversity(self):
        topics = [
            {"top_terms": ["machine", "learning", "model"]},
            {"top_terms": ["machine", "learning", "model"]},
        ]
        score = topic_diversity_score(topics)
        assert score == 0.0

    def test_distinct_topics_high_diversity(self):
        topics = [
            {"top_terms": ["machine", "learning", "neural"]},
            {"top_terms": ["cooking", "recipe", "food"]},
        ]
        score = topic_diversity_score(topics)
        assert score == 1.0

    def test_partial_overlap(self):
        topics = [
            {"top_terms": ["machine", "learning", "model"]},
            {"top_terms": ["machine", "deep", "network"]},
        ]
        score = topic_diversity_score(topics)
        assert 0.0 < score < 1.0

    def test_single_topic(self):
        topics = [{"top_terms": ["machine", "learning"]}]
        assert topic_diversity_score(topics) == 0.0


class TestAutoK:
    def test_small_dataset(self):
        assert auto_k(3) == 2

    def test_medium_dataset(self):
        k = auto_k(100)
        assert 2 <= k <= 20

    def test_large_dataset(self):
        k = auto_k(10000)
        assert k == 20  # capped at 20

    def test_minimum(self):
        assert auto_k(1) == 1
        assert auto_k(5) == 2
