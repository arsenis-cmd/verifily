"""SHA-256 hashing for files and content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def sha256_file(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_string(s: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_dict(d: Dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of a dict (canonical JSON)."""
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return sha256_string(canonical)


def compute_file_hashes(directory: Union[str, Path]) -> Dict[str, str]:
    """Compute SHA-256 hashes for all files in a directory (non-recursive)."""
    d = Path(directory)
    hashes: Dict[str, str] = {}
    if not d.is_dir():
        return hashes
    for f in sorted(d.iterdir()):
        if f.is_file():
            hashes[f.name] = sha256_file(f)
    return hashes


def compute_chain_hash(hashes: Dict[str, str]) -> str:
    """Compute a chain hash from ordered file hashes."""
    combined = "|".join(f"{k}={v}" for k, v in sorted(hashes.items()))
    return sha256_string(combined)


def verify_hashes(
    directory: Union[str, Path], expected: Dict[str, str]
) -> List[Dict[str, str]]:
    """Verify file hashes against expected values.

    Returns list of dicts with keys: file, status, expected, actual.
    """
    d = Path(directory)
    results = []
    for filename, expected_hash in sorted(expected.items()):
        filepath = d / filename
        if not filepath.exists():
            results.append(
                {
                    "file": filename,
                    "status": "MISSING",
                    "expected": expected_hash,
                    "actual": "",
                }
            )
            continue
        actual_hash = sha256_file(filepath)
        status = "PASS" if actual_hash == expected_hash else "FAIL"
        results.append(
            {
                "file": filename,
                "status": status,
                "expected": expected_hash,
                "actual": actual_hash,
            }
        )
    return results
