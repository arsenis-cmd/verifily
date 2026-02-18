"""Pure Python evaluation metrics for Verifily.

Implements standard NLP metrics without external dependencies:
exact match, token F1, BLEU (1-4), ROUGE-L, and length ratio.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Aggregated evaluation result."""
    exact_match: float
    f1: float
    bleu: float
    rouge_l: float
    length_ratio: float
    num_examples: int
    per_example: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": {
                "exact_match": round(self.exact_match, 4),
                "f1": round(self.f1, 4),
                "bleu": round(self.bleu, 4),
                "rouge_l": round(self.rouge_l, 4),
                "length_ratio": round(self.length_ratio, 4),
            },
            "num_examples": self.num_examples,
        }


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization on normalized text."""
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

def exact_match_score(prediction: str, reference: str) -> float:
    """1.0 if normalized prediction equals normalized reference, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def token_f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score (harmonic mean of precision and recall)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    # Number of common tokens (with multiplicity)
    common = sum((pred_counts & ref_counts).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _count_ngrams(tokens: List[str], n: int) -> Counter:
    """Count n-grams in a token list."""
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
) -> float:
    """Sentence-level BLEU score with brevity penalty.

    Computes modified n-gram precision for n=1..max_n,
    applies uniform weights, and brevity penalty.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))

    # Modified precision for each n
    log_precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _count_ngrams(pred_tokens, n)
        ref_ngrams = _count_ngrams(ref_tokens, n)

        if not pred_ngrams:
            return 0.0

        clipped = sum(
            min(count, ref_ngrams.get(ng, 0))
            for ng, count in pred_ngrams.items()
        )
        total = sum(pred_ngrams.values())

        if clipped == 0:
            return 0.0

        log_precisions.append(math.log(clipped / total))

    # Geometric mean with uniform weights
    avg_log = sum(log_precisions) / len(log_precisions)
    return bp * math.exp(avg_log)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP (two rows)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_score(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Length ratio
# ---------------------------------------------------------------------------

def length_ratio(prediction: str, reference: str) -> float:
    """Ratio of prediction length to reference length (in tokens)."""
    pred_len = len(_tokenize(prediction))
    ref_len = len(_tokenize(reference))
    if ref_len == 0:
        return 0.0 if pred_len == 0 else float("inf")
    return pred_len / ref_len


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[str],
    references: List[str],
    *,
    max_bleu_n: int = 4,
    include_per_example: bool = False,
) -> EvalResult:
    """Compute all evaluation metrics.

    Args:
        predictions: List of predicted texts.
        references: List of reference texts (same length).
        max_bleu_n: Maximum n-gram order for BLEU (default 4).
        include_per_example: If True, include per-example scores.

    Returns:
        EvalResult with aggregated scores.

    Raises:
        ValueError: If predictions and references have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            f"must have the same length"
        )

    n = len(predictions)
    if n == 0:
        return EvalResult(
            exact_match=0.0, f1=0.0, bleu=0.0,
            rouge_l=0.0, length_ratio=0.0, num_examples=0,
        )

    em_scores = []
    f1_scores = []
    bleu_scores = []
    rouge_scores = []
    lr_scores = []
    per_example = []

    for pred, ref in zip(predictions, references):
        em = exact_match_score(pred, ref)
        f1 = token_f1_score(pred, ref)
        bl = bleu_score(pred, ref, max_n=max_bleu_n)
        rl = rouge_l_score(pred, ref)
        lr = length_ratio(pred, ref)

        em_scores.append(em)
        f1_scores.append(f1)
        bleu_scores.append(bl)
        rouge_scores.append(rl)
        lr_scores.append(lr)

        if include_per_example:
            per_example.append({
                "exact_match": round(em, 4),
                "f1": round(f1, 4),
                "bleu": round(bl, 4),
                "rouge_l": round(rl, 4),
                "length_ratio": round(lr, 4),
            })

    return EvalResult(
        exact_match=sum(em_scores) / n,
        f1=sum(f1_scores) / n,
        bleu=sum(bleu_scores) / n,
        rouge_l=sum(rouge_scores) / n,
        length_ratio=sum(lr_scores) / n,
        num_examples=n,
        per_example=per_example,
    )
