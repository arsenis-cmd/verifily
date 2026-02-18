"""Pure-Python K-means clustering on TF-IDF sparse vectors.

Provides topic extraction and diversity metrics for dataset
quality analysis.  Zero external dependencies.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Set, Tuple

from verifily_cli_v1.core.tfidf import SparseVector, cosine_similarity


# ---------------------------------------------------------------------------
# Sparse vector math
# ---------------------------------------------------------------------------

def _sparse_add(a: SparseVector, b: SparseVector) -> SparseVector:
    """Element-wise addition of two sparse vectors."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0.0) + v
    return result


def _sparse_scale(vec: SparseVector, s: float) -> SparseVector:
    """Scale a sparse vector by a scalar."""
    return {k: v * s for k, v in vec.items()}


def _sparse_centroid(vectors: List[SparseVector]) -> SparseVector:
    """Compute the mean of a list of sparse vectors."""
    if not vectors:
        return {}
    acc: SparseVector = {}
    for vec in vectors:
        for k, v in vec.items():
            acc[k] = acc.get(k, 0.0) + v
    n = len(vectors)
    return {k: v / n for k, v in acc.items()}


def _cosine_distance(a: SparseVector, b: SparseVector) -> float:
    """1 - cosine_similarity (cosine distance)."""
    return 1.0 - cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

def _kmeans_pp_init(
    vectors: List[SparseVector],
    k: int,
    rng: random.Random,
) -> List[SparseVector]:
    """K-means++ initialization: pick diverse centroids."""
    n = len(vectors)
    centroids: List[SparseVector] = []

    # First centroid: random
    idx = rng.randint(0, n - 1)
    centroids.append(dict(vectors[idx]))

    for _ in range(1, k):
        # Distance to nearest centroid for each point
        dists = []
        for vec in vectors:
            min_d = min(_cosine_distance(vec, c) for c in centroids)
            dists.append(min_d * min_d)  # squared distance

        total = sum(dists)
        if total == 0:
            # All points identical to existing centroids
            idx = rng.randint(0, n - 1)
        else:
            # Weighted random choice
            threshold = rng.random() * total
            cumulative = 0.0
            idx = 0
            for i, d in enumerate(dists):
                cumulative += d
                if cumulative >= threshold:
                    idx = i
                    break

        centroids.append(dict(vectors[idx]))

    return centroids


def kmeans_sparse(
    vectors: List[SparseVector],
    k: int = 5,
    max_iter: int = 50,
    seed: int = 42,
) -> Tuple[List[int], List[SparseVector]]:
    """K-means clustering on sparse TF-IDF vectors using cosine distance.

    Args:
        vectors: List of sparse TF-IDF vectors.
        k: Number of clusters (auto-reduced if fewer vectors).
        max_iter: Maximum iterations.
        seed: For deterministic initialization.

    Returns:
        (assignments, centroids) — assignments[i] is the cluster for vectors[i].
    """
    n = len(vectors)
    if n == 0:
        return [], []

    k = min(k, n)
    if k <= 1:
        return [0] * n, [_sparse_centroid(vectors)]

    rng = random.Random(seed)
    centroids = _kmeans_pp_init(vectors, k, rng)
    assignments = [0] * n

    for _ in range(max_iter):
        # Assign each vector to nearest centroid
        changed = False
        for i, vec in enumerate(vectors):
            best_cluster = 0
            best_dist = float("inf")
            for c_idx, centroid in enumerate(centroids):
                d = _cosine_distance(vec, centroid)
                if d < best_dist:
                    best_dist = d
                    best_cluster = c_idx
            if assignments[i] != best_cluster:
                assignments[i] = best_cluster
                changed = True

        if not changed:
            break

        # Recompute centroids
        for c_idx in range(k):
            cluster_vecs = [vectors[i] for i in range(n) if assignments[i] == c_idx]
            if cluster_vecs:
                centroids[c_idx] = _sparse_centroid(cluster_vecs)

    return assignments, centroids


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def extract_topics(
    vectors: List[SparseVector],
    assignments: List[int],
    centroids: List[SparseVector],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Extract top terms per cluster as 'topics'.

    Returns:
        List of topic dicts with cluster id, size, top terms, and coherence.
    """
    k = len(centroids)
    topics: List[Dict[str, Any]] = []

    for c_idx in range(k):
        cluster_indices = [i for i, a in enumerate(assignments) if a == c_idx]
        size = len(cluster_indices)
        if size == 0:
            continue

        # Top terms from centroid
        sorted_terms = sorted(centroids[c_idx].items(), key=lambda x: -x[1])
        top_terms = [t for t, _ in sorted_terms[:top_n]]

        # Coherence: average pairwise cosine within cluster
        coherence = 0.0
        if size >= 2:
            cluster_vecs = [vectors[i] for i in cluster_indices]
            pair_count = 0
            total_sim = 0.0
            for i in range(min(size, 50)):  # cap for performance
                for j in range(i + 1, min(size, 50)):
                    total_sim += cosine_similarity(cluster_vecs[i], cluster_vecs[j])
                    pair_count += 1
            coherence = total_sim / pair_count if pair_count > 0 else 0.0

        topics.append({
            "cluster": c_idx,
            "size": size,
            "top_terms": top_terms,
            "coherence": round(coherence, 4),
        })

    return topics


def topic_diversity_score(topics: List[Dict[str, Any]]) -> float:
    """Compute topic diversity: average pairwise Jaccard distance between
    cluster term sets.

    Returns 0.0-1.0: higher = more diverse topics.
    """
    if len(topics) < 2:
        return 0.0

    term_sets = [set(t["top_terms"]) for t in topics]
    total_dist = 0.0
    pair_count = 0

    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            a, b = term_sets[i], term_sets[j]
            union = a | b
            if not union:
                continue
            intersection = a & b
            jaccard = len(intersection) / len(union)
            total_dist += 1.0 - jaccard  # distance
            pair_count += 1

    return round(total_dist / pair_count, 4) if pair_count > 0 else 0.0


def auto_k(n_docs: int) -> int:
    """Heuristic for number of clusters."""
    if n_docs <= 5:
        return min(2, n_docs)
    return min(max(2, int(math.sqrt(n_docs / 10))), 20)
