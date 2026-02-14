"""verifily fingerprint / diff-datasets — CLI wrappers for dataset fingerprinting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from verifily_cli_v1.core.fingerprint import (
    DatasetFingerprint,
    compute_fingerprint,
    diff_fingerprints,
    load_fingerprint,
    write_fingerprint,
)
from verifily_cli_v1.core.io import write_json


def run_fingerprint(
    dataset: str,
    schema: str = "auto",
    out: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute fingerprint and optionally write to disk.

    Returns:
        Fingerprint dict.
    """
    fp = compute_fingerprint(dataset, schema=schema, seed=seed)

    # Determine output directory
    if out:
        out_dir = Path(out)
    else:
        ds_path = Path(dataset)
        out_dir = ds_path if ds_path.is_dir() else ds_path.parent

    write_fingerprint(fp, out_dir)
    result = fp.to_dict()
    result["fingerprint_path"] = str(out_dir / "fingerprint.json")
    return result


def run_diff(
    path_a: str,
    path_b: str,
    out: Optional[str] = None,
) -> Dict[str, Any]:
    """Load or compute fingerprints for two datasets and diff them.

    Returns:
        Diff dict with similarity_score, overlap estimate, deltas.
    """
    fp_a = _resolve_fingerprint(path_a)
    fp_b = _resolve_fingerprint(path_b)

    diff = diff_fingerprints(fp_a, fp_b)

    if out:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "diff.json", diff)

        # Human-readable summary
        lines = [
            "Dataset Diff Summary",
            "=" * 40,
            f"Similarity (MinHash Jaccard): {diff['similarity_score']:.4f}",
            f"Exact hash overlap estimate:  {diff['exact_dup_overlap_estimate']:.4f}",
            f"Row delta:                    {diff['row_delta']:+d}",
            f"Dup rate delta:               {diff['dup_rate_delta']:+.4f}",
            f"Schema A: {diff['schema_a']}  |  Schema B: {diff['schema_b']}",
            "",
            "Length deltas:",
        ]
        for k, v in diff.get("length_deltas", {}).items():
            lines.append(f"  {k}: {v:+.1f}")
        lines.append("")
        (out_dir / "diff.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        diff["diff_path"] = str(out_dir / "diff.json")
        diff["diff_txt_path"] = str(out_dir / "diff.txt")

    return diff


def _resolve_fingerprint(path: str) -> DatasetFingerprint:
    """Load fingerprint from path — tries fingerprint.json first, then computes."""
    p = Path(path)

    # Direct fingerprint.json
    if p.is_file() and p.name == "fingerprint.json":
        return load_fingerprint(p)

    # Directory with fingerprint.json
    if p.is_dir() and (p / "fingerprint.json").exists():
        return load_fingerprint(p / "fingerprint.json")

    # Compute on the fly
    return compute_fingerprint(path)
