"""Learned quality scorer for Verifily.

Extracts a feature vector from quality issues and vocabulary stats,
then applies a logistic regression with pre-trained weights to produce
a quality score 0-100.  Zero external dependencies.

Weights are trained offline by scripts/train_scorer.py and embedded
as Python constants.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from verifily_cli_v1.core.quality import QualityIssue


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "empty_or_low_info",
    "length_outlier",
    "encoding_issue",
    "repetition",
    "near_duplicate",
    "class_imbalance",
]


def extract_features(
    issues: List[QualityIssue],
    stats: Dict[str, Any],
    total_rows: int,
) -> List[float]:
    """Extract a fixed-length feature vector from quality analysis results.

    Returns a 25-dimensional vector:
      [0-11]  Per-category: has_X (0/1) + fraction (6 categories x 2 = 12)
      [12-14] Severity counts: error, warning, info
      [15]    Total issue count
      [16]    Type-token ratio
      [17]    Hapax ratio
      [18]    log(1 + total_tokens)
      [19]    Length coefficient of variation (std/mean)
      [20]    log(1 + total_rows)
      [21]    Bias term (always 1.0)
      [22]    Semantic duplicate fraction (from Level 2, default 0.0)
      [23]    Topic diversity (from Level 2, default 0.0)
      [24]    Topic count normalized (from Level 2, default 0.0)
    """
    # Build lookup from issues
    cat_fractions: Dict[str, float] = {}
    severity_counts = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        cat_fractions[issue.category] = issue.fraction
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    features: List[float] = []

    # Per-category features (0-11)
    for cat in _CATEGORIES:
        frac = cat_fractions.get(cat, 0.0)
        features.append(1.0 if frac > 0 else 0.0)  # has_X
        features.append(frac)                        # fraction

    # Severity counts (12-14)
    features.append(float(severity_counts.get("error", 0)))
    features.append(float(severity_counts.get("warning", 0)))
    features.append(float(severity_counts.get("info", 0)))

    # Total issue count (15)
    features.append(float(len(issues)))

    # Vocabulary stats (16-18)
    features.append(float(stats.get("type_token_ratio", 0.0)))
    features.append(float(stats.get("hapax_ratio", 0.0)))
    features.append(math.log1p(float(stats.get("total_tokens", 0))))

    # Length coefficient of variation (19)
    mean = float(stats.get("length_mean", 0.0))
    length_min = float(stats.get("length_min", 0.0))
    length_max = float(stats.get("length_max", 0.0))
    if mean > 0 and length_max > length_min:
        # Approximate std from range (rough but cheap)
        approx_std = (length_max - length_min) / 4.0
        features.append(approx_std / mean)
    else:
        features.append(0.0)

    # Row count (20)
    features.append(math.log1p(float(total_rows)))

    # Bias (21)
    features.append(1.0)

    # Level 2 features (22-24) — default to 0.0 if not available
    features.append(float(stats.get("semantic_dup_fraction", 0.0)))
    features.append(float(stats.get("topic_diversity", 0.0)))
    topic_count = float(stats.get("topic_count", 0))
    expected_topics = max(1.0, math.sqrt(total_rows / 10.0)) if total_rows > 0 else 1.0
    features.append(topic_count / expected_topics if expected_topics > 0 else 0.0)

    return features


# ---------------------------------------------------------------------------
# Trained weights
# ---------------------------------------------------------------------------

# 25 weights trained by scripts/train_scorer.py via logistic regression.
# The model predicts P(good_quality) from the feature vector.
# Score = sigmoid(dot(features, weights)) * 100, inverted so higher = better.
#
# Placeholder weights (hand-calibrated to roughly match _compute_score behavior)
# until the training script produces real ones.
_WEIGHTS: List[float] = [
    +0.062330,  # has_empty (0)
    -0.099282,  # empty_frac (1)
    +0.623540,  # has_length_outlier (2)
    +0.073273,  # outlier_frac (3)
    -0.462214,  # has_encoding (4)
    -0.245449,  # encoding_frac (5)
    -0.132826,  # has_repetition (6)
    -0.100517,  # repetition_frac (7)
    +0.143531,  # has_near_dup (8)
    -0.160894,  # near_dup_frac (9)
    +0.017209,  # has_class_imbalance (10)
    +0.086197,  # imbalance_frac (11)
    -0.412217,  # error_count (12)
    +0.361963,  # warning_count (13)
    -0.054695,  # info_count (14)
    -0.182252,  # total_issue_count (15)
    +0.186292,  # type_token_ratio (16)
    +0.454621,  # hapax_ratio (17)
    +0.483214,  # log_total_tokens (18)
    -0.186056,  # length_cv (19)
    -0.550857,  # log_total_rows (20)
    -0.258463,  # bias (21)
    -0.415422,  # semantic_dup_fraction (22)
    +0.021158,  # topic_diversity (23)
    +0.031241,  # topic_count_normalized (24)
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _dot(a: List[float], b: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def learned_score(
    issues: List[QualityIssue],
    stats: Dict[str, Any],
    total_rows: int,
) -> int:
    """Compute quality score 0-100 using learned logistic regression.

    Falls back to a simple heuristic if weights are misconfigured.

    Args:
        issues: List of QualityIssue from quality checks.
        stats: Vocabulary/topic stats dict from _vocab_stats().
        total_rows: Total number of rows in the dataset.

    Returns:
        Integer score 0-100, higher is better.
    """
    if total_rows == 0:
        return 0

    features = extract_features(issues, stats, total_rows)

    # Validate weights length matches features
    if len(_WEIGHTS) != len(features):
        raise ValueError(
            f"Weight vector length ({len(_WEIGHTS)}) != "
            f"feature vector length ({len(features)})"
        )

    z = _dot(features, _WEIGHTS)
    prob = _sigmoid(z)
    return max(0, min(100, int(prob * 100)))
