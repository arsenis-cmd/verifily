"""Shareable report bundle — zip of safe artifacts (no raw data)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

# Files to include if they exist in the run directory.
INCLUDE_FILES: List[str] = [
    "decision_summary.json",
    "pipeline_result.json",
    "audit_log.jsonl",
    "audit.jsonl",
    "usage.json",
    "eval/eval_results.json",
    "eval_results.json",
    "hashes.json",
    "environment.json",
    "config.yaml",
    "run_meta.json",
    "report.json",
]

# File patterns to never include (raw data, model weights).
EXCLUDE_SUFFIXES: List[str] = [
    ".csv",
    ".parquet",
    ".bin",
    ".pt",
    ".safetensors",
]

EXCLUDE_NAMES: List[str] = [
    "dataset.jsonl",
]


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_excluded(rel: str, path: Path) -> bool:
    """Check if a file should be excluded from the bundle."""
    name = path.name
    if name in EXCLUDE_NAMES:
        return True
    for suffix in EXCLUDE_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def create_bundle(run_dir: str, out_path: str) -> Dict[str, Any]:
    """Create a zip bundle of shareable artifacts from a run directory.

    Args:
        run_dir: Path to a run or pipeline output directory.
        out_path: Path to write the output zip file.

    Returns:
        Dict with bundle_path, files_included, files_excluded, manifest.

    Raises:
        FileNotFoundError: If run_dir does not exist.
    """
    d = Path(run_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    included: List[str] = []
    excluded: List[str] = []
    manifest_files: Dict[str, str] = {}

    # Collect files to include
    for rel_name in INCLUDE_FILES:
        p = d / rel_name
        if p.exists() and p.is_file():
            included.append(rel_name)
            manifest_files[rel_name] = _sha256(p)

    # Also scan for any files we should explicitly note as excluded
    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(d))
        if rel in included:
            continue
        if _is_excluded(rel, p):
            excluded.append(rel)

    manifest = {
        "bundle_version": 1,
        "source_dir": str(d.resolve()),
        "files": manifest_files,
    }

    # Write zip
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_name in included:
            zf.write(d / rel_name, rel_name)
        # Write manifest last
        zf.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )

    return {
        "bundle_path": str(out.resolve()),
        "files_included": included,
        "files_excluded": excluded,
        "manifest": manifest,
    }
