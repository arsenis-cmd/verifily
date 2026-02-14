"""Dataset fingerprinting — privacy-safe comparison without raw data.

Produces a DatasetFingerprint containing:
- Length statistics (whitespace token proxy)
- Exact duplicate rate
- MinHash signature (deterministic, configurable permutations)
- Top-K exact hash sketch for overlap estimation
- Tag distribution

No raw text is stored in the fingerprint.

Privacy guarantees:
- No raw text is stored or logged.
- Only hashes and statistical sketches are produced.
- Deterministic with fixed seed — identical inputs always produce identical outputs.
- Text is normalized (lowercased, whitespace-split) before hashing.

Environment variables:
- VERIFILY_FINGERPRINT_PERMS: Override default permutation count (default: 128).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import statistics
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from verifily_cli_v1.core.hashing import sha256_string
from verifily_cli_v1.core.io import read_json, read_jsonl, write_json

# ── Constants ──────────────────────────────────────────────────────

_DEFAULT_NUM_PERM = 128
SKETCH_SIZE = 200       # Top-K hash sketch for overlap estimation
HASH_SEED_PREFIX = "verifily_minhash_v1"
MAX_UINT64 = (1 << 64) - 1
# Mersenne prime for universal hashing
_MERSENNE_P = (1 << 61) - 1


def _get_num_perm() -> int:
    """Get number of MinHash permutations (env-overridable)."""
    env_val = os.environ.get("VERIFILY_FINGERPRINT_PERMS")
    if env_val:
        return max(8, int(env_val))
    return _DEFAULT_NUM_PERM


# ── Data classes ───────────────────────────────────────────────────

@dataclass
class DatasetFingerprint:
    version: str = "v1"
    schema: str = "unknown"
    rows: int = 0
    tag_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    length_stats: Dict[str, float] = field(default_factory=dict)
    exact_dup_rate: float = 0.0
    minhash_signature: List[int] = field(default_factory=list)
    exact_hash_sketch: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetFingerprint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Text extraction (same pattern as contamination.py) ─────────────

def _row_text(row: Dict[str, Any]) -> str:
    """Extract searchable text from a row (excludes tags/id)."""
    parts = []
    for key in ("instruction", "output", "input", "question", "answer", "context", "text"):
        if key in row and row[key]:
            parts.append(str(row[key]).strip())
    return " ".join(parts) if parts else json.dumps(row, sort_keys=True)


# ── Schema detection ───────────────────────────────────────────────

def _detect_schema(rows: List[Dict[str, Any]]) -> str:
    """Detect schema from field names in the first few rows."""
    if not rows:
        return "unknown"
    sample = rows[:5]
    keys: Set[str] = set()
    for r in sample:
        keys.update(r.keys())
    if {"input", "output"}.issubset(keys) or {"question", "answer"}.issubset(keys):
        return "sft"
    if {"text", "label"}.issubset(keys):
        return "classification"
    if "messages" in keys:
        return "chat"
    return "unknown"


# ── MinHash (deterministic, fast universal hashing) ───────────────

def _make_hash_params(num_perm: int, seed: int) -> List[tuple]:
    """Pre-generate (a, b) pairs for universal hash family h(x) = (a*x + b) mod p.

    Deterministic: derived from seed via SHA-256 chain.
    """
    params = []
    for i in range(num_perm):
        raw = f"{HASH_SEED_PREFIX}:{seed}:{i}".encode("utf-8")
        h = hashlib.sha256(raw).digest()
        a = int.from_bytes(h[:8], "big") % _MERSENNE_P
        b = int.from_bytes(h[8:16], "big") % _MERSENNE_P
        if a == 0:
            a = 1  # a must be non-zero
        params.append((a, b))
    return params


def _word_shingles(text: str, n: int = 3) -> Set[str]:
    """Extract word-level n-gram shingles from normalized text."""
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else {""}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _shingle_hash(shingle: str) -> int:
    """Hash a shingle to a uint64. Deterministic."""
    h = hashlib.sha256(shingle.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _compute_minhash(texts: List[str], seed: int = 42, num_perm: int = 0) -> List[int]:
    """Compute MinHash signature over a collection of texts.

    Uses universal hashing: h_i(x) = (a_i * hash(x) + b_i) mod p
    This requires only ONE sha256 per shingle (not num_perm).
    """
    if num_perm <= 0:
        num_perm = _get_num_perm()

    params = _make_hash_params(num_perm, seed)
    signature = [MAX_UINT64] * num_perm

    for text in texts:
        shingles = _word_shingles(text)
        for shingle in shingles:
            sh = _shingle_hash(shingle)
            for idx, (a, b) in enumerate(params):
                val = (a * sh + b) % _MERSENNE_P
                if val < signature[idx]:
                    signature[idx] = val

    return signature


# ── Public API ─────────────────────────────────────────────────────

def compute_fingerprint(
    dataset_path: Union[str, Path],
    schema: str = "auto",
    seed: int = 42,
    num_perm: int = 0,
) -> DatasetFingerprint:
    """Compute a privacy-safe fingerprint for a dataset.

    Args:
        dataset_path: Path to a JSONL file or an artifact directory containing dataset.jsonl.
        schema: Schema type or "auto" for detection.
        seed: Random seed for deterministic MinHash.
        num_perm: Number of MinHash permutations (0 = use default/env).

    Returns:
        DatasetFingerprint with no raw text stored.
    """
    if num_perm <= 0:
        num_perm = _get_num_perm()

    path = Path(dataset_path)

    # Resolve artifact dir vs direct JSONL
    if path.is_dir():
        jsonl_path = path / "dataset.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"dataset.jsonl not found in {path}")
        path = jsonl_path

    rows = read_jsonl(path)
    if not rows:
        return DatasetFingerprint(
            schema=schema if schema != "auto" else "unknown",
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
        )

    # Schema
    detected_schema = schema if schema != "auto" else _detect_schema(rows)

    # Extract texts
    texts = [_row_text(r) for r in rows]

    # Length stats (whitespace token count proxy)
    token_counts = [len(t.split()) for t in texts]
    token_counts_sorted = sorted(token_counts)
    n = len(token_counts_sorted)
    p95_idx = min(int(n * 0.95), n - 1)
    length_stats = {
        "min": float(token_counts_sorted[0]),
        "median": float(statistics.median(token_counts)),
        "p95": float(token_counts_sorted[p95_idx]),
        "max": float(token_counts_sorted[-1]),
    }

    # Exact duplicates + hash sketch
    text_hashes = [sha256_string(t) for t in texts]
    hash_counts = Counter(text_hashes)
    total_dup_rows = sum(c for c in hash_counts.values() if c > 1)
    exact_dup_rate = total_dup_rows / len(rows) if rows else 0.0

    # Top-K hash sketch (truncated to 16 hex chars = 8 bytes for privacy)
    unique_hashes = sorted(set(h[:16] for h in text_hashes))
    exact_hash_sketch = unique_hashes[:SKETCH_SIZE]

    # Tag distribution
    tag_dist: Dict[str, Dict[str, int]] = {}
    for row in rows:
        tags = row.get("tags", {})
        if isinstance(tags, dict):
            for k, v in tags.items():
                if k not in tag_dist:
                    tag_dist[k] = {}
                sv = str(v)
                tag_dist[k][sv] = tag_dist[k].get(sv, 0) + 1

    # MinHash
    minhash_sig = _compute_minhash(texts, seed=seed, num_perm=num_perm)

    return DatasetFingerprint(
        version="v1",
        schema=detected_schema,
        rows=len(rows),
        tag_distribution=tag_dist,
        length_stats=length_stats,
        exact_dup_rate=round(exact_dup_rate, 6),
        minhash_signature=minhash_sig,
        exact_hash_sketch=exact_hash_sketch,
        created_at=datetime.datetime.utcnow().isoformat() + "Z",
    )


def diff_fingerprints(
    a: DatasetFingerprint,
    b: DatasetFingerprint,
) -> Dict[str, Any]:
    """Compare two fingerprints and return a diff summary.

    Returns:
        Dict with similarity_score, exact_dup_overlap_estimate, deltas, etc.
        No raw text is included in the output.
    """
    # MinHash Jaccard estimate
    if a.minhash_signature and b.minhash_signature:
        num_perm = min(len(a.minhash_signature), len(b.minhash_signature))
        matching = sum(
            1 for x, y in zip(a.minhash_signature, b.minhash_signature) if x == y
        )
        similarity_score = matching / num_perm
    else:
        similarity_score = 0.0

    # Exact hash sketch overlap (Jaccard on sketch sets)
    set_a = set(a.exact_hash_sketch)
    set_b = set(b.exact_hash_sketch)
    if set_a or set_b:
        exact_dup_overlap_estimate = len(set_a & set_b) / len(set_a | set_b)
    else:
        exact_dup_overlap_estimate = 0.0

    # Length stats deltas
    length_deltas = {}
    for key in ("min", "median", "p95", "max"):
        va = a.length_stats.get(key, 0.0)
        vb = b.length_stats.get(key, 0.0)
        length_deltas[key] = round(vb - va, 4)

    # Tag distribution deltas
    all_tag_keys = set(a.tag_distribution.keys()) | set(b.tag_distribution.keys())
    tag_deltas: Dict[str, Dict[str, int]] = {}
    for tk in sorted(all_tag_keys):
        da = a.tag_distribution.get(tk, {})
        db = b.tag_distribution.get(tk, {})
        all_vals = set(da.keys()) | set(db.keys())
        tag_deltas[tk] = {v: db.get(v, 0) - da.get(v, 0) for v in sorted(all_vals)}

    return {
        "similarity_score": round(similarity_score, 6),
        "exact_dup_overlap_estimate": round(exact_dup_overlap_estimate, 6),
        "row_delta": b.rows - a.rows,
        "dup_rate_delta": round(b.exact_dup_rate - a.exact_dup_rate, 6),
        "length_deltas": length_deltas,
        "tag_deltas": tag_deltas,
        "schema_a": a.schema,
        "schema_b": b.schema,
    }


def write_fingerprint(fp: DatasetFingerprint, output_dir: Union[str, Path]) -> Path:
    """Write fingerprint.json to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return write_json(out / "fingerprint.json", fp.to_dict())


def load_fingerprint(path: Union[str, Path]) -> DatasetFingerprint:
    """Load a fingerprint from a JSON file or directory containing fingerprint.json."""
    p = Path(path)
    if p.is_dir():
        p = p / "fingerprint.json"
    data = read_json(p)
    return DatasetFingerprint.from_dict(data)
