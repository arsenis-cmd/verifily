"""ML-based quality judge using embeddings.

Uses a simple linear classifier head on top of text embeddings
to produce per-row and aggregate quality scores.

Optional dependency: requires ``torch`` + ``transformers`` for
transformer-based embeddings. Falls back to TF-IDF if unavailable.

Usage::

    result = judge_quality(texts)
    if result:
        print(f"Model quality score: {result['model_quality_score']}")
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from verifily_cli_v1.core.embeddings import _has_torch, _has_transformers


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def _has_model_judge() -> bool:
    """Check if full model judge dependencies are available."""
    return _has_torch() and _has_transformers()


# ---------------------------------------------------------------------------
# Quality Judge
# ---------------------------------------------------------------------------

class QualityJudge:
    """Quality classifier using embeddings + linear head.

    Architecture:
        text -> embedder -> embedding (dim-D) -> linear (D -> 1) -> sigmoid -> score

    The linear head weights can be:
    - Loaded from a file (trained via scripts/train_judge.py)
    - Auto-initialized with heuristic weights (feature-based)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedder_name: str = "distilbert-base-uncased",
        prefer_backend: str = "auto",
    ):
        self._model_path = model_path
        self._embedder_name = embedder_name
        self._prefer_backend = prefer_backend
        self._embedder = None
        self._head_weights: Optional[List[float]] = None
        self._head_bias: float = 0.0

    def _load(self) -> None:
        """Load embedder and classifier head."""
        from verifily_cli_v1.core.embeddings import get_embedder

        self._embedder = get_embedder(
            prefer=self._prefer_backend,
            model_name=self._embedder_name,
        )

        if self._model_path:
            self._load_head(self._model_path)
        else:
            # Heuristic head: use embedding norm variance as quality proxy
            self._head_weights = None  # Will use heuristic scoring

    def _load_head(self, path: str) -> None:
        """Load trained linear head weights from a file."""
        import json
        with open(path) as f:
            data = json.load(f)
        self._head_weights = data["weights"]
        self._head_bias = data.get("bias", 0.0)

    def _heuristic_score(self, embedding: List[float]) -> float:
        """Score a row based on embedding properties.

        Uses the magnitude and entropy of the embedding vector as
        a proxy for text quality. Well-formed, diverse text tends to
        produce embeddings with moderate norm and higher entropy.
        """
        if not embedding:
            return 0.0

        # Embedding norm (already L2 normalized, but check)
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm < 0.01:
            return 0.1  # Very low quality signal

        # Entropy of absolute values (diversity of features)
        abs_vals = [abs(x) for x in embedding]
        total = sum(abs_vals) or 1.0
        probs = [v / total for v in abs_vals if v > 0]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # Normalize entropy to 0-1 range (max entropy = log(dim))
        max_entropy = math.log(max(len(embedding), 1))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Combine: higher entropy = more diverse = generally better quality
        return max(0.0, min(1.0, normalized_entropy))

    def _linear_score(self, embedding: List[float]) -> float:
        """Score using trained linear head."""
        if self._head_weights is None:
            return self._heuristic_score(embedding)

        z = sum(w * x for w, x in zip(self._head_weights, embedding))
        z += self._head_bias

        # Sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def judge_rows(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Score individual rows for quality (0.0-1.0).

        Returns:
            List of quality scores, one per text.
        """
        if self._embedder is None:
            self._load()

        embeddings = self._embedder.embed(texts, batch_size=batch_size)

        scores = []
        for emb in embeddings:
            score = self._linear_score(emb)
            scores.append(round(score, 4))

        return scores

    def judge_dataset(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Aggregate quality assessment for a dataset.

        Returns dict with:
            - model_quality_score: 0-100 aggregate score
            - per_row_mean: mean per-row score
            - low_quality_count: rows scoring < 0.3
            - high_quality_fraction: fraction scoring >= 0.7
            - model_backend: embedder backend used
        """
        if self._embedder is None:
            self._load()

        per_row = self.judge_rows(texts, batch_size=batch_size)

        if not per_row:
            return {
                "model_quality_score": 0,
                "per_row_mean": 0.0,
                "low_quality_count": 0,
                "high_quality_fraction": 0.0,
                "model_backend": self._embedder.backend,
            }

        mean_score = sum(per_row) / len(per_row)
        low_quality = [i for i, s in enumerate(per_row) if s < 0.3]
        high_quality = sum(1 for s in per_row if s >= 0.7)

        return {
            "model_quality_score": max(0, min(100, int(mean_score * 100))),
            "per_row_mean": round(mean_score, 4),
            "low_quality_count": len(low_quality),
            "high_quality_fraction": round(high_quality / len(per_row), 4),
            "model_backend": self._embedder.backend,
        }


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def judge_quality(
    texts: List[str],
    model_path: Optional[str] = None,
    prefer_backend: str = "auto",
) -> Optional[Dict[str, Any]]:
    """Convenience function for quality judging.

    Returns None if no suitable backend is available,
    otherwise returns a dict with quality assessment results.
    """
    try:
        judge = QualityJudge(
            model_path=model_path,
            prefer_backend=prefer_backend,
        )
        return judge.judge_dataset(texts)
    except Exception:
        return None
