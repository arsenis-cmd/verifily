"""Verifily Transform deduplication: exact and fuzzy."""

import hashlib
import logging
from typing import Any, Dict, List, Set

from verifily_transform.config import DedupeConfig

logger = logging.getLogger(__name__)


def deduplicate(
    rows: List[Dict[str, Any]],
    config: DedupeConfig,
    task: str,
) -> List[Dict[str, Any]]:
    """Remove duplicate rows via exact hash and/or fuzzy matching.

    Args:
        rows: normalized rows.
        config: deduplication configuration.
        task: "sft" or "classification".

    Returns:
        Deduplicated rows.
    """
    original_count = len(rows)

    if config.exact:
        rows = _exact_dedup(rows, task)
        logger.info("After exact dedup: %d rows (removed %d)", len(rows), original_count - len(rows))

    if config.fuzzy:
        pre_fuzzy = len(rows)
        rows = _fuzzy_dedup(rows, task, config.fuzzy_threshold, config.num_perm)
        logger.info("After fuzzy dedup: %d rows (removed %d)", len(rows), pre_fuzzy - len(rows))

    logger.info("Dedup total: %d -> %d rows", original_count, len(rows))
    return rows


def _content_key(row: Dict[str, Any], task: str) -> str:
    """Extract the primary content string for dedup comparison."""
    if task == "sft":
        return f"{row.get('instruction', '')}|||{row.get('output', '')}"
    else:
        return row.get("text", "")


def _exact_dedup(rows: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """Remove rows with identical content hashes."""
    seen: Set[str] = set()
    unique = []
    for row in rows:
        content = _content_key(row, task)
        h = hashlib.sha256(content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(row)
    return unique


def _fuzzy_dedup(
    rows: List[Dict[str, Any]],
    task: str,
    threshold: float,
    num_perm: int,
) -> List[Dict[str, Any]]:
    """Remove near-duplicate rows using MinHash LSH.

    Falls back to n-gram Jaccard if datasketch is not installed.
    """
    try:
        return _fuzzy_dedup_minhash(rows, task, threshold, num_perm)
    except ImportError:
        logger.info("datasketch not installed, using brute-force Jaccard (slower)")
        return _fuzzy_dedup_jaccard(rows, task, threshold)


def _fuzzy_dedup_minhash(
    rows: List[Dict[str, Any]],
    task: str,
    threshold: float,
    num_perm: int,
) -> List[Dict[str, Any]]:
    """MinHash LSH deduplication via datasketch."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    for i, row in enumerate(rows):
        content = _content_key(row, task)
        tokens = set(content.lower().split())
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(token.encode())
        minhashes.append((i, m))

    keep_indices = set()
    for i, m in minhashes:
        key = f"row_{i}"
        result = lsh.query(m)
        if not result:
            lsh.insert(key, m)
            keep_indices.add(i)

    return [rows[i] for i in sorted(keep_indices)]


def _fuzzy_dedup_jaccard(
    rows: List[Dict[str, Any]],
    task: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    """Brute-force Jaccard similarity dedup (O(n^2), for small datasets)."""
    tokenized = []
    for row in rows:
        content = _content_key(row, task)
        tokens = set(content.lower().split())
        tokenized.append(tokens)

    keep = []
    for i in range(len(rows)):
        is_dup = False
        for j in keep:
            sim = _jaccard(tokenized[i], tokenized[j])
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    return [rows[i] for i in keep]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
