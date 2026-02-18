"""Text quality analysis for Verifily.

Analyzes dataset rows for quality issues: length outliers, repetition,
near-duplicates, encoding problems, empty rows, vocabulary stats, and
class imbalance.  Pure Python — zero external dependencies.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from verifily_cli_v1.core.minhash_lsh import MinHashLSH, minhash_jaccard


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class QualityIssue:
    """A single quality issue found in the dataset."""
    category: str       # "length_outlier", "near_duplicate", etc.
    severity: str       # "error", "warning", "info"
    count: int
    fraction: float     # count / total_rows
    sample_rows: List[int]  # up to 5 row indices
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    """Result of dataset quality analysis."""
    total_rows: int
    quality_score: int  # 0-100, higher = better
    issues: List[QualityIssue]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "quality_score": self.quality_score,
            "issues": [i.to_dict() for i in self.issues],
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_CONTENT_KEYS = {
    "input", "output", "instruction", "response", "question", "answer",
    "context", "text", "content", "prompt", "completion", "document",
    "summary", "source", "target", "chosen", "rejected", "sql",
    "subject", "body", "resolution", "message", "description", "title",
    "sentence", "paragraph", "article", "highlights",
}


def _row_text(row: Dict[str, Any]) -> str:
    """Extract all text content from a row."""
    parts = []
    for k in sorted(row.keys()):
        if k in _CONTENT_KEYS:
            v = row[k]
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        c = item.get("content", "")
                        if c:
                            parts.append(str(c))
                    elif isinstance(item, str):
                        parts.append(item)
    # Fallback: if no known content keys matched, use all string fields
    if not parts:
        for k in sorted(row.keys()):
            v = row.get(k)
            if isinstance(v, str) and len(v) > 5 and k not in ("id", "tags"):
                parts.append(v)
    return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_empty_rows(texts: List[str]) -> QualityIssue | None:
    """Find rows with no meaningful content."""
    bad = [i for i, t in enumerate(texts) if len(t.strip()) < 10]
    if not bad:
        return None
    return QualityIssue(
        category="empty_or_low_info",
        severity="error" if len(bad) / len(texts) > 0.05 else "warning",
        count=len(bad),
        fraction=len(bad) / len(texts),
        sample_rows=bad[:5],
        description=f"{len(bad)} rows have less than 10 characters of content",
    )


def _check_length_outliers(texts: List[str]) -> QualityIssue | None:
    """Flag rows whose length is >2 standard deviations from the mean."""
    if len(texts) < 5:
        return None
    lengths = [len(t.split()) for t in texts]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std = math.sqrt(variance) if variance > 0 else 0
    if std == 0:
        return None
    outliers = [i for i, n in enumerate(lengths) if abs(n - mean) > 2 * std]
    if not outliers:
        return None
    return QualityIssue(
        category="length_outlier",
        severity="warning",
        count=len(outliers),
        fraction=len(outliers) / len(texts),
        sample_rows=outliers[:5],
        description=(
            f"{len(outliers)} rows have abnormal length "
            f"(mean={mean:.0f} words, std={std:.0f})"
        ),
    )


_MOJIBAKE = re.compile(r"[Ã¢â€ï¿½Â]+|[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _check_encoding(texts: List[str]) -> QualityIssue | None:
    """Detect mojibake and control characters."""
    bad = [i for i, t in enumerate(texts) if _MOJIBAKE.search(t)]
    if not bad:
        return None
    return QualityIssue(
        category="encoding_issue",
        severity="warning",
        count=len(bad),
        fraction=len(bad) / len(texts),
        sample_rows=bad[:5],
        description=f"{len(bad)} rows contain encoding artifacts or control characters",
    )


def _check_repetition(texts: List[str], ngram_size: int = 4) -> QualityIssue | None:
    """Find rows with repeated phrase patterns."""
    bad = []
    for i, text in enumerate(texts):
        words = text.lower().split()
        if len(words) < ngram_size * 3:
            continue
        ngrams = [
            tuple(words[j : j + ngram_size])
            for j in range(len(words) - ngram_size + 1)
        ]
        counts = Counter(ngrams)
        max_repeat = max(counts.values()) if counts else 0
        if max_repeat >= 3:
            bad.append(i)
    if not bad:
        return None
    return QualityIssue(
        category="repetition",
        severity="warning",
        count=len(bad),
        fraction=len(bad) / len(texts),
        sample_rows=bad[:5],
        description=f"{len(bad)} rows contain repeated phrases (3+ times)",
    )


def _check_near_duplicates(
    texts: List[str],
    threshold: float = 0.80,
    num_perm: int = 64,
) -> tuple[QualityIssue | None, int]:
    """Find near-duplicate rows within the dataset using MinHash LSH."""
    if len(texts) < 2:
        return None, 0

    def _word_shingles(text: str, n: int = 3) -> Set[str]:
        words = text.lower().split()
        if len(words) < n:
            return set(words)
        return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

    lsh = MinHashLSH(num_perm=num_perm, threshold=threshold, seed=42)
    for i, text in enumerate(texts):
        shingles = _word_shingles(text)
        if shingles:
            lsh.insert(i, shingles)

    dup_rows: Set[int] = set()
    seen_pairs: Set[tuple] = set()
    for i in range(len(texts)):
        if i not in lsh.signatures:
            continue
        candidates = lsh.query_signature(lsh.signatures[i])
        for j in candidates:
            if j == i:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            sim = minhash_jaccard(lsh.signatures[i], lsh.signatures[j])
            if sim >= threshold:
                dup_rows.add(i)
                dup_rows.add(j)

    count = len(dup_rows)
    if count == 0:
        return None, 0
    return QualityIssue(
        category="near_duplicate",
        severity="warning" if count / len(texts) < 0.10 else "error",
        count=count,
        fraction=count / len(texts),
        sample_rows=sorted(dup_rows)[:5],
        description=f"{count} rows are near-duplicates of other rows (Jaccard >= {threshold})",
    ), count


def _check_semantic_duplicates(
    texts: List[str],
    threshold: float = 0.90,
    max_pairs: int = 100_000,
) -> tuple[QualityIssue | None, Dict[str, Any]]:
    """Find semantic duplicates using TF-IDF cosine similarity.

    Returns:
        (QualityIssue or None, stats dict with semantic_dup_fraction).
    """
    if len(texts) < 2:
        return None, {"semantic_dup_count": 0, "semantic_dup_fraction": 0.0}

    from verifily_cli_v1.core.tfidf import TfidfVectorizer, pairwise_cosine

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    pairs = pairwise_cosine(vectors, threshold=threshold, max_comparisons=max_pairs)

    dup_rows: Set[int] = set()
    for i, j, _ in pairs:
        dup_rows.add(i)
        dup_rows.add(j)

    count = len(dup_rows)
    fraction = count / len(texts) if texts else 0.0
    stats = {"semantic_dup_count": count, "semantic_dup_fraction": fraction}

    if count == 0:
        return None, stats

    return QualityIssue(
        category="semantic_duplicate",
        severity="warning" if fraction < 0.15 else "error",
        count=count,
        fraction=fraction,
        sample_rows=sorted(dup_rows)[:5],
        description=(
            f"{count} rows are semantically similar (TF-IDF cosine >= {threshold})"
        ),
    ), stats


def _compute_topic_stats(
    texts: List[str],
    max_docs: int = 5000,
) -> Dict[str, Any]:
    """Compute topic diversity stats via TF-IDF + K-means.

    Returns dict with topic_count, topic_diversity, and topics list.
    """
    if len(texts) < 5:
        return {"topic_count": 0, "topic_diversity": 0.0, "topics": []}

    from verifily_cli_v1.core.tfidf import TfidfVectorizer
    from verifily_cli_v1.core.topic_cluster import (
        auto_k, extract_topics, kmeans_sparse, topic_diversity_score,
    )

    # Sample if too many docs
    sample = texts[:max_docs]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sample)

    # Filter out empty vectors
    non_empty = [(i, v) for i, v in enumerate(vectors) if v]
    if len(non_empty) < 2:
        return {"topic_count": 0, "topic_diversity": 0.0, "topics": []}

    non_empty_vecs = [v for _, v in non_empty]
    k = auto_k(len(non_empty_vecs))
    assignments, centroids = kmeans_sparse(non_empty_vecs, k=k)
    topics = extract_topics(non_empty_vecs, assignments, centroids)
    diversity = topic_diversity_score(topics)

    return {
        "topic_count": len(topics),
        "topic_diversity": diversity,
        "topics": topics,
    }


def _check_class_imbalance(
    rows: List[Dict[str, Any]],
) -> QualityIssue | None:
    """Detect class imbalance in classification datasets."""
    labels = []
    for row in rows:
        label = row.get("label") or row.get("category") or row.get("class")
        if label is not None:
            labels.append(str(label))
    if len(labels) < len(rows) * 0.5:
        return None  # Not a classification dataset
    counts = Counter(labels)
    if len(counts) < 2:
        return None
    total = sum(counts.values())
    minority_classes = [
        cls for cls, c in counts.items() if c / total < 0.05
    ]
    if not minority_classes:
        return None
    return QualityIssue(
        category="class_imbalance",
        severity="warning",
        count=len(minority_classes),
        fraction=len(minority_classes) / len(counts),
        sample_rows=[],
        description=(
            f"{len(minority_classes)} class(es) have <5% representation: "
            + ", ".join(minority_classes[:3])
        ),
    )


# ---------------------------------------------------------------------------
# Vocabulary stats
# ---------------------------------------------------------------------------

def _vocab_stats(texts: List[str]) -> Dict[str, Any]:
    """Compute vocabulary statistics."""
    all_words: List[str] = []
    row_word_counts: List[int] = []
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        row_word_counts.append(len(words))

    total_tokens = len(all_words)
    if total_tokens == 0:
        return {"total_tokens": 0, "unique_tokens": 0, "type_token_ratio": 0}

    word_counts = Counter(all_words)
    unique = len(word_counts)
    hapax = sum(1 for c in word_counts.values() if c == 1)

    sorted_lens = sorted(row_word_counts)
    n = len(sorted_lens)

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique,
        "type_token_ratio": round(unique / total_tokens, 4),
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax / unique, 4) if unique else 0,
        "length_mean": round(sum(row_word_counts) / n, 1),
        "length_median": sorted_lens[n // 2],
        "length_p95": sorted_lens[int(n * 0.95)] if n >= 20 else sorted_lens[-1],
        "length_min": sorted_lens[0],
        "length_max": sorted_lens[-1],
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHTS = {"error": 15, "warning": 5, "info": 1}


def _compute_score(issues: List[QualityIssue], total_rows: int) -> int:
    """Compute quality score 0-100 from issues."""
    if total_rows == 0:
        return 0
    penalty = 0.0
    for issue in issues:
        base = _SEVERITY_WEIGHTS.get(issue.severity, 1)
        # Scale penalty by fraction of affected rows (capped at 25 per issue)
        penalty += min(25, base * issue.fraction * 100)
    return max(0, min(100, int(100 - penalty)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_quality(
    rows: List[Dict[str, Any]],
    schema: str = "auto",
) -> QualityReport:
    """Run all quality checks on a dataset.

    Args:
        rows: List of row dicts (raw or canonicalized).
        schema: Schema hint (unused currently, reserved for future).

    Returns:
        QualityReport with score, issues, and stats.
    """
    if not rows:
        return QualityReport(
            total_rows=0,
            quality_score=0,
            issues=[],
            stats={},
        )

    texts = [_row_text(row) for row in rows]
    issues: List[QualityIssue] = []

    # Run checks
    empty = _check_empty_rows(texts)
    if empty:
        issues.append(empty)

    length = _check_length_outliers(texts)
    if length:
        issues.append(length)

    encoding = _check_encoding(texts)
    if encoding:
        issues.append(encoding)

    repetition = _check_repetition(texts)
    if repetition:
        issues.append(repetition)

    near_dup, near_dup_count = _check_near_duplicates(texts)
    if near_dup:
        issues.append(near_dup)

    imbalance = _check_class_imbalance(rows)
    if imbalance:
        issues.append(imbalance)

    # Semantic duplicate check (TF-IDF cosine similarity)
    sem_dup, sem_stats = _check_semantic_duplicates(texts)
    if sem_dup:
        issues.append(sem_dup)

    # Sort by severity
    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: severity_order.get(i.severity, 9))

    # Stats
    stats = _vocab_stats(texts)
    stats["near_duplicate_count"] = near_dup_count
    stats.update(sem_stats)

    # Topic diversity
    topic_stats = _compute_topic_stats(texts)
    stats.update(topic_stats)

    try:
        from verifily_cli_v1.core.learned_scorer import learned_score
        score = learned_score(issues, stats, len(rows))
    except Exception:
        score = _compute_score(issues, len(rows))

    # Optional: ML model judge (Level 3 — requires torch + transformers)
    try:
        from verifily_cli_v1.core.model_judge import judge_quality
        model_result = judge_quality(texts)
        if model_result:
            stats["model_quality"] = model_result
            model_score = model_result["model_quality_score"]
            score = int(0.7 * score + 0.3 * model_score)
    except Exception:
        pass  # graceful degradation

    return QualityReport(
        total_rows=len(rows),
        quality_score=score,
        issues=issues,
        stats=stats,
    )
