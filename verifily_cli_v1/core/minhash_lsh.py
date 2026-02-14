"""MinHash LSH — locality-sensitive hashing for scalable near-duplicate detection.

Pure Python implementation reusing hash primitives from ``fingerprint.py``.
Replaces O(N^2) brute-force Jaccard with O(N) index + O(candidates) verify.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from verifily_cli_v1.core.fingerprint import (
    MAX_UINT64,
    _MERSENNE_P,
    _make_hash_params,
    _shingle_hash,
)


def compute_minhash_signature(
    shingles: Set[str],
    params: List[tuple],
) -> List[int]:
    """Compute MinHash signature for a set of shingles.

    Args:
        shingles: Set of text shingles (n-grams).
        params: Pre-computed (a, b) hash parameters from ``_make_hash_params``.

    Returns:
        List of minimum hash values, one per permutation.
    """
    num_perm = len(params)
    signature = [MAX_UINT64] * num_perm

    for shingle in shingles:
        sh = _shingle_hash(shingle)
        for idx, (a, b) in enumerate(params):
            val = (a * sh + b) % _MERSENNE_P
            if val < signature[idx]:
                signature[idx] = val

    return signature


def optimal_bands(num_perm: int, threshold: float) -> Tuple[int, int]:
    """Compute optimal (bands, rows_per_band) for a target Jaccard threshold.

    Uses the approximation: threshold ~ (1/bands)^(1/rows_per_band).
    Finds the split that minimises the difference from the target threshold.
    """
    best_bands = 1
    best_rows = num_perm
    best_diff = float("inf")

    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        # Approximate threshold for this configuration
        approx_t = (1.0 / b) ** (1.0 / r)
        diff = abs(approx_t - threshold)
        if diff < best_diff:
            best_diff = diff
            best_bands = b
            best_rows = r

    return best_bands, best_rows


def minhash_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    n = min(len(sig_a), len(sig_b))
    matching = sum(1 for i in range(n) if sig_a[i] == sig_b[i])
    return matching / n


class MinHashLSH:
    """Locality-Sensitive Hashing index for near-duplicate detection.

    Usage::

        lsh = MinHashLSH(num_perm=128, threshold=0.70)
        for doc_id, shingles in documents:
            lsh.insert(doc_id, shingles)

        candidates = lsh.query(query_shingles)
        for cand_id in candidates:
            sim = minhash_jaccard(lsh.signatures[query_id], lsh.signatures[cand_id])
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.70,
        seed: int = 42,
    ) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.seed = seed
        self.params = _make_hash_params(num_perm, seed)
        self.bands, self.rows_per_band = optimal_bands(num_perm, threshold)
        # band_index -> {band_hash: set of doc_ids}
        self._buckets: List[Dict[int, List[int]]] = [
            defaultdict(list) for _ in range(self.bands)
        ]
        # doc_id -> signature (for later verification)
        self.signatures: Dict[int, List[int]] = {}

    def insert(self, doc_id: int, shingles: Set[str]) -> None:
        """Index a document by its shingles."""
        sig = compute_minhash_signature(shingles, self.params)
        self.signatures[doc_id] = sig
        self._index_signature(doc_id, sig)

    def insert_signature(self, doc_id: int, sig: List[int]) -> None:
        """Index a document by a pre-computed signature."""
        self.signatures[doc_id] = sig
        self._index_signature(doc_id, sig)

    def _index_signature(self, doc_id: int, sig: List[int]) -> None:
        """Add signature to LSH band buckets."""
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(sig[start:end]))
            self._buckets[band_idx][band_hash].append(doc_id)

    def query(self, shingles: Set[str]) -> Set[int]:
        """Find candidate doc_ids that may be near-duplicates of the query."""
        sig = compute_minhash_signature(shingles, self.params)
        return self.query_signature(sig)

    def query_signature(self, sig: List[int]) -> Set[int]:
        """Find candidates from a pre-computed signature."""
        candidates: Set[int] = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(sig[start:end]))
            bucket = self._buckets[band_idx].get(band_hash)
            if bucket:
                candidates.update(bucket)
        return candidates
