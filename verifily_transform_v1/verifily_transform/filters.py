"""Verifily Transform filters: quality, length, leakage, toxicity."""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from verifily_transform.utils import detect_pii

logger = logging.getLogger(__name__)


def apply_filters(
    rows: List[Dict[str, Any]],
    task: str,
    seed_rows: Optional[List[Dict[str, Any]]] = None,
    min_length: int = 20,
    max_length: int = 512,
    leakage_check: bool = True,
    pii_removal: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply quality filters to rows.

    Args:
        rows: rows to filter.
        task: "sft" or "classification".
        seed_rows: original seed data for leakage detection.
        min_length: minimum text length.
        max_length: maximum text length.
        leakage_check: check for leakage from seed data.
        pii_removal: flag PII-containing rows.

    Returns:
        (filtered_rows, rejection_counts) tuple.
    """
    filters: List[Tuple[str, Callable]] = [
        ("empty", lambda r: _check_empty(r, task)),
        ("too_short", lambda r: _check_length(r, task, min_length, "min")),
        ("too_long", lambda r: _check_length(r, task, max_length, "max")),
        ("bad_chars", lambda r: _check_bad_characters(r, task)),
        ("repetitive", lambda r: _check_repetitive(r, task)),
    ]

    if leakage_check and seed_rows:
        seed_set = _build_seed_set(seed_rows, task)
        seed_id_set = {id(r) for r in seed_rows}
        filters.append(("leakage", lambda r: _check_leakage(r, task, seed_set, seed_id_set)))

    if pii_removal:
        filters.append(("pii", lambda r: _check_pii(r, task)))

    passed = []
    rejection_counts: Dict[str, int] = {}

    for row in rows:
        rejected = False
        for filter_name, filter_fn in filters:
            if not filter_fn(row):
                rejection_counts[filter_name] = rejection_counts.get(filter_name, 0) + 1
                rejected = True
                break
        if not rejected:
            passed.append(row)

    total_rejected = len(rows) - len(passed)
    logger.info(
        "Filters: %d/%d passed (%.1f%%). Rejections: %s",
        len(passed), len(rows),
        100 * len(passed) / max(len(rows), 1),
        rejection_counts,
    )
    return passed, rejection_counts


def _get_text(row: Dict[str, Any], task: str) -> str:
    """Extract the primary text content from a row."""
    if task == "sft":
        return f"{row.get('instruction', '')} {row.get('output', '')}"
    return row.get("text", "")


def _check_empty(row: Dict[str, Any], task: str) -> bool:
    """Reject if primary fields are empty."""
    if task == "sft":
        return bool(row.get("instruction", "").strip() and row.get("output", "").strip())
    return bool(row.get("text", "").strip())


def _check_length(row: Dict[str, Any], task: str, limit: int, mode: str) -> bool:
    """Check text length against min/max."""
    text = _get_text(row, task)
    length = len(text)
    if mode == "min":
        return length >= limit
    return length <= limit


def _check_bad_characters(row: Dict[str, Any], task: str) -> bool:
    """Reject rows with excessive special characters or encoding artifacts."""
    text = _get_text(row, task)
    if not text:
        return False

    # Reject if > 30% non-alphanumeric (excluding spaces and common punctuation)
    clean = re.sub(r'[a-zA-Z0-9\s.,!?;:\'"()\-/]', '', text)
    ratio = len(clean) / max(len(text), 1)
    return ratio < 0.3


def _check_repetitive(row: Dict[str, Any], task: str) -> bool:
    """Reject rows with excessive repetition."""
    text = _get_text(row, task)
    words = text.lower().split()
    if len(words) < 5:
        return True  # too short to judge

    # Check if any single word makes up > 40% of the text
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(words) > 0.4:
        return False

    # Check for repeated phrases (3-gram)
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    if trigram_counts and trigram_counts.most_common(1)[0][1] > max(3, len(trigrams) * 0.2):
        return False

    return True


def _build_seed_set(seed_rows: List[Dict[str, Any]], task: str) -> set:
    """Build a set of normalized seed content for leakage detection."""
    seeds = set()
    for row in seed_rows:
        if task == "sft":
            content = row.get("instruction", "").lower().strip()
        else:
            content = row.get("text", "").lower().strip()
        if content:
            seeds.add(content)
    return seeds


def _check_leakage(row: Dict[str, Any], task: str, seed_set: set, seed_id_set: set) -> bool:
    """Reject synthetic rows that are too similar to seed data (potential copies).

    Original seed rows (identified by object identity) are always allowed through.
    """
    # Skip leakage check for original seed rows — they are by definition in the seed set
    if id(row) in seed_id_set:
        return True

    if task == "sft":
        content = row.get("instruction", "").lower().strip()
    else:
        content = row.get("text", "").lower().strip()

    # Exact match
    if content in seed_set:
        return False

    # High token overlap (> 90%)
    content_tokens = set(content.split())
    for seed in seed_set:
        seed_tokens = set(seed.split())
        if not content_tokens or not seed_tokens:
            continue
        overlap = len(content_tokens & seed_tokens) / max(len(content_tokens), 1)
        if overlap > 0.9:
            return False

    return True


def _check_pii(row: Dict[str, Any], task: str) -> bool:
    """Reject rows containing PII."""
    text = _get_text(row, task)
    findings = detect_pii(text)
    return len(findings) == 0
