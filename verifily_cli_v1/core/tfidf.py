"""Pure-Python TF-IDF vectorizer and cosine similarity.

Zero external dependencies. Uses dict-based sparse vectors for
memory efficiency. Implements sublinear TF scaling and smooth IDF.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

# Sparse vector: term -> weight
SparseVector = Dict[str, float]

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+", re.UNICODE)

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "not", "no", "if", "so", "as", "up", "out",
    "about", "into", "than", "then", "also", "just", "more",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    """TF-IDF vectorizer with sublinear TF scaling.

    Usage::

        vec = TfidfVectorizer()
        vectors = vec.fit_transform(texts)
        sim = cosine_similarity(vectors[0], vectors[1])
    """

    def __init__(
        self,
        max_features: int = 10_000,
        min_df: int = 1,
        max_df_ratio: float = 0.95,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self._idf: Dict[str, float] = {}
        self._vocab: Set[str] = set()
        self._n_docs: int = 0

    def fit(self, texts: List[str]) -> "TfidfVectorizer":
        """Learn vocabulary and IDF weights from texts."""
        self._n_docs = len(texts)
        if self._n_docs == 0:
            self._idf = {}
            self._vocab = set()
            return self

        # Count document frequency
        df: Counter = Counter()
        for text in texts:
            tokens = set(tokenize(text))
            df.update(tokens)

        # Filter by min_df and max_df_ratio
        max_df = int(self._n_docs * self.max_df_ratio)
        filtered = {
            term: count for term, count in df.items()
            if count >= self.min_df and count <= max_df
        }

        # Limit to top features by document frequency
        if len(filtered) > self.max_features:
            top = sorted(filtered.items(), key=lambda x: -x[1])[:self.max_features]
            filtered = dict(top)

        # Compute IDF: log(N / df) + 1 (smooth IDF)
        self._idf = {
            term: math.log(self._n_docs / count) + 1.0
            for term, count in filtered.items()
        }
        self._vocab = set(self._idf.keys())
        return self

    def transform(self, texts: List[str]) -> List[SparseVector]:
        """Transform texts to TF-IDF sparse vectors (L2 normalized)."""
        vectors: List[SparseVector] = []
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            vec: SparseVector = {}
            for term, count in tf.items():
                if term in self._idf:
                    # Sublinear TF: 1 + log(tf)
                    tf_val = 1.0 + math.log(count) if count > 0 else 0.0
                    vec[term] = tf_val * self._idf[term]

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
            if norm > 0:
                vec = {k: v / norm for k, v in vec.items()}

            vectors.append(vec)
        return vectors

    def fit_transform(self, texts: List[str]) -> List[SparseVector]:
        """Fit and transform in one pass."""
        self.fit(texts)
        return self.transform(texts)

    def get_top_terms(
        self, vector: SparseVector, n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return top-n terms by TF-IDF weight from a vector."""
        return sorted(vector.items(), key=lambda x: -x[1])[:n]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: SparseVector, b: SparseVector) -> float:
    """Cosine similarity between two L2-normalized sparse vectors.

    Returns 0.0-1.0. Since vectors are already L2-normalized,
    this is just the dot product.
    """
    if not a or not b:
        return 0.0

    # Use the smaller dict for iteration
    if len(a) > len(b):
        a, b = b, a

    dot = sum(a[k] * b[k] for k in a if k in b)
    return max(0.0, min(1.0, dot))


def pairwise_cosine(
    vectors: List[SparseVector],
    threshold: float = 0.85,
    max_comparisons: int = 500_000,
) -> List[Tuple[int, int, float]]:
    """Find all pairs with cosine similarity >= threshold.

    If N*(N-1)/2 exceeds max_comparisons, samples random pairs.

    Returns:
        List of (i, j, similarity) tuples, sorted by descending similarity.
    """
    n = len(vectors)
    total_pairs = n * (n - 1) // 2

    pairs: List[Tuple[int, int, float]] = []

    if total_pairs <= max_comparisons:
        # Exhaustive search
        for i in range(n):
            if not vectors[i]:
                continue
            for j in range(i + 1, n):
                if not vectors[j]:
                    continue
                sim = cosine_similarity(vectors[i], vectors[j])
                if sim >= threshold:
                    pairs.append((i, j, sim))
    else:
        # Random sampling
        rng = random.Random(42)
        sampled: Set[Tuple[int, int]] = set()
        attempts = 0
        max_attempts = max_comparisons * 3

        while len(sampled) < max_comparisons and attempts < max_attempts:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                attempts += 1
                continue
            pair = (min(i, j), max(i, j))
            if pair in sampled:
                attempts += 1
                continue
            sampled.add(pair)
            attempts += 1

            if vectors[pair[0]] and vectors[pair[1]]:
                sim = cosine_similarity(vectors[pair[0]], vectors[pair[1]])
                if sim >= threshold:
                    pairs.append((pair[0], pair[1], sim))

    pairs.sort(key=lambda x: -x[2])
    return pairs
