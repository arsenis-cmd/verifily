"""Manifest schema, creation, and verification."""

from __future__ import annotations

import datetime
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from verifily_cli_v1.core.hashing import compute_chain_hash, compute_file_hashes
from verifily_cli_v1.core.io import read_json, write_json


def create_manifest(
    output_dir: Union[str, Path],
    *,
    name: str,
    task: str = "sft",
    schema: str = "sft",
    format: str = "jsonl",
    row_count: int = 0,
    source_files: Optional[List[str]] = None,
    tags: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a manifest.json for a dataset artifact directory.

    Computes file hashes and writes manifest.json + hashes.json.
    Returns the manifest dict.
    """
    out = Path(output_dir)
    file_hashes = compute_file_hashes(out)
    # Exclude manifest.json and hashes.json from the hash set (they're written after)
    file_hashes.pop("manifest.json", None)
    file_hashes.pop("hashes.json", None)

    chain_hash = compute_chain_hash(file_hashes)

    manifest: Dict[str, Any] = {
        "name": name,
        "version": "1.0",
        "task": task,
        "schema": schema,
        "format": format,
        "row_count": row_count,
        "source_files": source_files or [],
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "platform": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "os": platform.platform(),
            "arch": platform.machine(),
        },
        "file_hashes": file_hashes,
        "chain_hash": chain_hash,
        "tags": tags or {},
    }

    if extra:
        manifest.update(extra)

    write_json(out / "manifest.json", manifest)

    # Write hashes.json separately for quick verification
    hashes_data = {
        "files": file_hashes,
        "chain_hash": chain_hash,
        "manifest_hash": "",  # placeholder, filled below
    }
    write_json(out / "hashes.json", hashes_data)

    # Now compute manifest hash and update hashes.json
    from verifily_cli_v1.core.hashing import sha256_file

    manifest_hash = sha256_file(out / "manifest.json")
    hashes_data["manifest_hash"] = manifest_hash
    write_json(out / "hashes.json", hashes_data)

    return manifest


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a manifest.json file."""
    return read_json(path)


def verify_manifest(directory: Union[str, Path]) -> Dict[str, Any]:
    """Verify a dataset directory against its manifest.

    Returns a dict with:
      - valid: bool
      - checks: list of {file, status, detail}
      - chain_hash_match: bool
    """
    d = Path(directory)
    manifest_path = d / "manifest.json"
    hashes_path = d / "hashes.json"

    if not manifest_path.exists():
        return {
            "valid": False,
            "checks": [],
            "error": "manifest.json not found",
        }

    manifest = read_json(manifest_path)
    expected_hashes = manifest.get("file_hashes", {})

    from verifily_cli_v1.core.hashing import sha256_file, verify_hashes

    checks = verify_hashes(d, expected_hashes)

    all_pass = all(c["status"] == "PASS" for c in checks)

    # Verify chain hash
    actual_chain = compute_chain_hash(
        {c["file"]: c["actual"] for c in checks if c["actual"]}
    )
    expected_chain = manifest.get("chain_hash", "")
    chain_match = actual_chain == expected_chain

    return {
        "valid": all_pass and chain_match,
        "checks": checks,
        "chain_hash_match": chain_match,
        "expected_chain": expected_chain,
        "actual_chain": actual_chain,
    }
