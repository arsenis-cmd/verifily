"""Verifily Transform manifest: read and validate dataset manifests."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from verifily_transform.errors import ArtifactError
from verifily_transform.utils import sha256_file

logger = logging.getLogger(__name__)


def load_manifest(path: str) -> Dict[str, Any]:
    """Load and validate a manifest.json file."""
    p = Path(path)
    if not p.exists():
        raise ArtifactError(f"Manifest not found: {path}")

    with open(p) as f:
        manifest = json.load(f)

    required = ["dataset_id", "format", "task", "num_rows", "content_hash", "files"]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ArtifactError(f"Manifest missing required fields: {missing}")

    return manifest


def verify_manifest(manifest_dir: str) -> Dict[str, str]:
    """Verify dataset integrity against its manifest.

    Returns dict of {check_name: "PASS" or "FAIL"}.
    """
    base = Path(manifest_dir)
    manifest_path = base / "manifest.json"

    if not manifest_path.exists():
        return {"manifest_exists": "FAIL"}

    manifest = load_manifest(str(manifest_path))
    results = {"manifest_exists": "PASS"}

    # Check each file entry
    for file_entry in manifest.get("files", []):
        rel_path = file_entry["path"]
        expected_hash = file_entry.get("sha256", "")
        file_path = base / rel_path

        if not file_path.exists():
            results[f"file_{rel_path}"] = "FAIL (missing)"
            continue

        actual_hash = sha256_file(str(file_path))
        if actual_hash == expected_hash:
            results[f"file_{rel_path}"] = "PASS"
        else:
            results[f"file_{rel_path}"] = f"FAIL (expected {expected_hash[:20]}... got {actual_hash[:20]}...)"

    # Check hashes.json
    hashes_path = base / "hashes.json"
    if hashes_path.exists():
        with open(hashes_path) as f:
            hashes = json.load(f)
        if hashes.get("dataset_hash") == manifest.get("content_hash"):
            results["hash_consistency"] = "PASS"
        else:
            results["hash_consistency"] = "FAIL"
    else:
        results["hash_consistency"] = "SKIP (no hashes.json)"

    return results


def format_verification(results: Dict[str, str]) -> str:
    """Format verification results as a human-readable string."""
    lines = ["Dataset integrity check:"]
    all_pass = True
    for check, status in results.items():
        icon = "PASS" if "PASS" in status else "FAIL"
        if "FAIL" in status:
            all_pass = False
        lines.append(f"  {check}: {status}")
    lines.append("")
    lines.append(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return "\n".join(lines)
