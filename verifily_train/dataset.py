"""Verifily Train dataset primitives: loading, validation, manifest, tagging."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from verifily_train.errors import DataError
from verifily_train.utils import sha256_file, utcnow_iso

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

SFT_REQUIRED = {"instruction", "output"}
CLS_REQUIRED = {"text", "label"}


def _validate_row(row: Dict[str, Any], task: str, idx: int) -> None:
    required = SFT_REQUIRED if task == "sft" else CLS_REQUIRED
    missing = required - set(row.keys())
    if missing:
        raise DataError(f"Row {idx}: missing fields {missing} for task '{task}'")


# ---------------------------------------------------------------------------
# DatasetVersion
# ---------------------------------------------------------------------------

@dataclass
class DatasetVersion:
    """Immutable snapshot of a Verifily dataset."""

    dataset_id: str = ""
    version: str = ""
    task: str = "sft"
    num_rows: int = 0
    tag_keys: List[str] = field(default_factory=list)
    tag_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    train_path: str = ""
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    content_hash: str = ""
    created_at: str = ""

    @classmethod
    def from_local_paths(
        cls,
        train_path: str,
        task: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        dataset_id: str = "local",
        version: str = "v0",
    ) -> "DatasetVersion":
        """Build a DatasetVersion from explicit local file paths."""
        tp = Path(train_path)
        if not tp.exists():
            raise DataError(f"Training data not found: {train_path}")

        # Quick scan for row count + tag keys + tag distribution
        rows = load_jsonl(str(tp))
        tag_keys: List[str] = []
        tag_dist: Dict[str, Dict[str, int]] = {}
        if rows and "tags" in rows[0]:
            tag_keys = sorted(rows[0]["tags"].keys())
        for row in rows:
            tags = row.get("tags", {})
            for k, v in tags.items():
                tag_dist.setdefault(k, {})
                sv = str(v)
                tag_dist[k][sv] = tag_dist[k].get(sv, 0) + 1

        content_hash = sha256_file(str(tp))

        return cls(
            dataset_id=dataset_id,
            version=version,
            task=task,
            num_rows=len(rows),
            tag_keys=tag_keys,
            tag_distribution=tag_dist,
            train_path=str(tp),
            val_path=val_path if val_path and Path(val_path).exists() else None,
            test_path=test_path if test_path and Path(test_path).exists() else None,
            content_hash=content_hash,
            created_at=utcnow_iso(),
        )

    @classmethod
    def from_manifest(cls, manifest_path: str) -> "DatasetVersion":
        """Load from a manifest.json produced by the Verifily platform."""
        mp = Path(manifest_path)
        if not mp.exists():
            raise DataError(f"Manifest not found: {manifest_path}")

        with open(mp) as f:
            m = json.load(f)

        base = mp.parent

        # Resolve file paths relative to the manifest directory
        train_path = ""
        val_path = None
        test_path = None
        for file_entry in m.get("files", []):
            rel = file_entry["path"]
            resolved = str(base / rel)
            split = file_entry.get("split", "")
            if split == "train":
                train_path = resolved
            elif split == "val":
                val_path = resolved
            elif split == "test":
                test_path = resolved

        if not train_path:
            raise DataError("Manifest must include a file with split='train'")

        return cls(
            dataset_id=m.get("dataset_id", "unknown"),
            version=m.get("version", "v0"),
            task=m.get("task", "sft"),
            num_rows=m.get("num_rows", 0),
            tag_keys=m.get("tag_keys", []),
            tag_distribution=m.get("tag_distribution", {}),
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            content_hash=m.get("content_hash", ""),
            created_at=m.get("created_at", utcnow_iso()),
        )

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Serialise to a manifest-like dict."""
        files = [{"path": Path(self.train_path).name, "split": "train",
                   "sha256": self.content_hash}]
        if self.val_path:
            files.append({"path": Path(self.val_path).name, "split": "val",
                           "sha256": sha256_file(self.val_path)})
        if self.test_path:
            files.append({"path": Path(self.test_path).name, "split": "test",
                           "sha256": sha256_file(self.test_path)})
        result = {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "task": self.task,
            "num_rows": self.num_rows,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "tag_keys": self.tag_keys,
            "files": files,
        }
        if self.tag_distribution:
            result["tag_distribution"] = self.tag_distribution
        return result


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"File not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"Invalid JSON on line {i + 1} of {path}: {e}")
    return rows


def load_multi_jsonl(
    paths: List[str],
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Load and merge multiple JSONL files with optional weighted sampling.

    If weights are provided, samples are drawn proportionally from each dataset.
    If no weights, all datasets are simply concatenated.
    Each row gets a '_source_file' tag added for attribution.
    """
    import random as _rand

    all_datasets = []
    for path in paths:
        rows = load_jsonl(path)
        source = Path(path).stem
        for row in rows:
            row.setdefault("tags", {})["_source_dataset"] = source
        all_datasets.append(rows)

    if weights is None:
        # Simple concatenation
        merged = []
        for ds in all_datasets:
            merged.extend(ds)
        _rand.shuffle(merged)
        logger.info("Multi-dataset: concatenated %d rows from %d files", len(merged), len(paths))
        return merged

    # Weighted sampling: normalise weights, sample proportionally
    total_weight = sum(weights)
    normalised = [w / total_weight for w in weights]
    total_rows = sum(len(ds) for ds in all_datasets)

    merged = []
    for ds, w in zip(all_datasets, normalised):
        n_sample = int(total_rows * w)
        if n_sample >= len(ds):
            merged.extend(ds)
        else:
            merged.extend(_rand.sample(ds, n_sample))

    _rand.shuffle(merged)
    logger.info("Multi-dataset: sampled %d rows from %d files (weighted)", len(merged), len(paths))
    return merged


def validate_dataset(rows: List[Dict[str, Any]], task: str) -> None:
    """Validate every row has the required fields for the given task."""
    for i, row in enumerate(rows):
        _validate_row(row, task, i)


# ---------------------------------------------------------------------------
# Tag slicing
# ---------------------------------------------------------------------------

def slice_by_tag(
    rows: List[Dict[str, Any]], tag_key: str
) -> Dict[str, List[int]]:
    """Group row indices by tag value. Missing tags go to '_untagged'."""
    slices: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        val = row.get("tags", {}).get(tag_key, "_untagged")
        slices.setdefault(str(val), []).append(i)
    return slices
