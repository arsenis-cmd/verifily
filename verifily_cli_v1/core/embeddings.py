"""Transformer-based text embeddings for Verifily.

Optional dependency: requires ``torch`` + ``transformers``.
Falls back to TF-IDF dense vectors if unavailable.

Usage::

    embedder = get_embedder()       # auto-selects best backend
    vectors = embedder.embed(texts)  # List[List[float]]

Install with::

    pip install verifily[ml]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    """Check if PyTorch is importable."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_transformers() -> bool:
    """Check if HuggingFace transformers is importable."""
    try:
        import transformers
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TF-IDF dense embedder (always available)
# ---------------------------------------------------------------------------

class TfidfEmbedder:
    """TF-IDF based embedder — wraps TfidfVectorizer into fixed-dim dense vectors.

    Always available (zero deps). Used as fallback when torch is not installed.
    """

    def __init__(self, max_dim: int = 512):
        self._max_dim = max_dim
        self._vocab_order: List[str] = []

    @property
    def dim(self) -> int:
        return len(self._vocab_order) if self._vocab_order else self._max_dim

    @property
    def backend(self) -> str:
        return "tfidf"

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Fit TF-IDF on texts and return dense vectors."""
        from verifily_cli_v1.core.tfidf import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=self._max_dim)
        sparse_vecs = vectorizer.fit_transform(texts)

        # Build stable vocabulary ordering
        all_terms: set = set()
        for vec in sparse_vecs:
            all_terms.update(vec.keys())

        self._vocab_order = sorted(all_terms)[:self._max_dim]
        term_to_idx = {t: i for i, t in enumerate(self._vocab_order)}

        # Project sparse → dense
        dim = len(self._vocab_order)
        dense: List[List[float]] = []
        for vec in sparse_vecs:
            row = [0.0] * dim
            for term, weight in vec.items():
                idx = term_to_idx.get(term)
                if idx is not None:
                    row[idx] = weight
            dense.append(row)

        return dense


# ---------------------------------------------------------------------------
# Transformer embedder (optional)
# ---------------------------------------------------------------------------

class TransformerEmbedder:
    """Transformer-based embedder using HuggingFace models.

    Loads model lazily on first ``embed()`` call.
    Uses mean pooling over the last hidden state.
    """

    DEFAULT_MODEL = "distilbert-base-uncased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
    ):
        self._model_name = model_name
        self._device_pref = device
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dim: Optional[int] = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()

        # Determine device
        if self._device_pref == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self._device_pref)

        self._model = self._model.to(self._device)
        self._dim = self._model.config.hidden_size

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore

    @property
    def backend(self) -> str:
        return "transformer"

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings via mean pooling of last hidden state."""
        import torch

        if self._model is None:
            self._load()

        all_embeddings: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self._model(**encoded)

            # Mean pooling (ignoring padding tokens)
            hidden = outputs.last_hidden_state  # (batch, seq, dim)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts

            # L2 normalize
            norms = mean_pooled.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = mean_pooled / norms

            all_embeddings.extend(normalized.cpu().tolist())

        return all_embeddings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(
    prefer: str = "auto",
    model_name: str = TransformerEmbedder.DEFAULT_MODEL,
    max_tfidf_dim: int = 512,
) -> Any:
    """Get the best available embedder.

    Args:
        prefer: ``"transformer"``, ``"tfidf"``, or ``"auto"``
            (auto = transformer if available, else tfidf).
        model_name: HuggingFace model name (transformer backend only).
        max_tfidf_dim: Maximum TF-IDF vector dimensionality.

    Returns:
        Embedder instance with ``.embed()``, ``.dim``, ``.backend``.
    """
    if prefer == "tfidf":
        return TfidfEmbedder(max_dim=max_tfidf_dim)

    if prefer == "transformer" or prefer == "auto":
        if _has_torch() and _has_transformers():
            return TransformerEmbedder(model_name=model_name)
        if prefer == "transformer":
            raise ImportError(
                "Transformer backend requires torch + transformers. "
                "Install with: pip install verifily[ml]"
            )

    return TfidfEmbedder(max_dim=max_tfidf_dim)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def cosine_similarity_dense(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def pairwise_cosine_dense(
    embeddings: List[List[float]],
    threshold: float = 0.90,
) -> List[Tuple[int, int, float]]:
    """Find all pairs above threshold in dense embeddings.

    Returns list of (i, j, similarity) sorted by descending similarity.
    """
    n = len(embeddings)
    pairs: List[Tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity_dense(embeddings[i], embeddings[j])
            if sim >= threshold:
                pairs.append((i, j, sim))

    pairs.sort(key=lambda x: -x[2])
    return pairs
