"""Verifily Transform artifacts: dataset packaging, versioning, hashing."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from verifily_transform.config import TransformConfig
from verifily_transform.errors import ArtifactError
from verifily_transform.utils import ensure_dir, sha256_file, sha256_dict, utcnow_iso, write_jsonl

logger = logging.getLogger(__name__)


def package_artifact(
    rows: List[Dict[str, Any]],
    config: TransformConfig,
    transform_id: str,
    stats: Dict[str, Any],
) -> str:
    """Package the final dataset as an immutable artifact.

    Creates:
      - dataset.jsonl (the final training data)
      - manifest.json (dataset metadata + hashes)
      - hashes.json (integrity chain)
      - transform_meta.json (full run metadata)
      - transform_config.yaml (frozen config)

    Args:
        rows: final processed rows.
        config: the transform config used.
        transform_id: unique run identifier.
        stats: pipeline statistics dict.

    Returns:
        Path to the output artifact directory.
    """
    output_dir = str(Path(config.output.dir) / config.output.name)
    ensure_dir(output_dir)

    # 1. Write dataset.jsonl
    dataset_path = str(Path(output_dir) / "dataset.jsonl")
    write_jsonl(rows, dataset_path)
    dataset_hash = sha256_file(dataset_path)
    logger.info("Wrote dataset.jsonl: %d rows, hash=%s", len(rows), dataset_hash)

    # 2. Write transform_config.yaml (frozen)
    config_path = str(Path(output_dir) / "transform_config.yaml")
    config.to_yaml(config_path)
    config_hash = sha256_dict(config.to_dict())

    # 3. Build tag distribution
    tag_distribution: Dict[str, Dict[str, int]] = {}
    tag_keys: list = []
    for row in rows:
        tags = row.get("tags", {})
        for k, v in tags.items():
            tag_distribution.setdefault(k, {})
            sv = str(v)
            tag_distribution[k][sv] = tag_distribution[k].get(sv, 0) + 1
    if tag_distribution:
        tag_keys = sorted(tag_distribution.keys())

    # 4. Write manifest.json
    manifest = {
        "dataset_id": config.output.name,
        "version": "v1",
        "format": "jsonl",
        "task": config.labeling.task,
        "num_rows": len(rows),
        "content_hash": dataset_hash,
        "created_at": utcnow_iso(),
        "tag_keys": tag_keys,
        "tag_distribution": tag_distribution,
        "files": [
            {
                "path": "dataset.jsonl",
                "split": "train",
                "num_rows": len(rows),
                "sha256": dataset_hash,
            }
        ],
        "row_schema": _infer_schema(rows, config.labeling.task),
    }
    manifest_path = str(Path(output_dir) / "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest.json")

    # 5. Write hashes.json
    hashes = {
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "manifest_hash": sha256_file(manifest_path),
        "hash_algorithm": "sha256",
    }
    hashes_path = str(Path(output_dir) / "hashes.json")
    with open(hashes_path, "w") as f:
        json.dump(hashes, f, indent=2)
    logger.info("Wrote hashes.json")

    # 6. Write transform_meta.json
    meta = {
        "transform_id": transform_id,
        "status": "completed",
        "created_at": utcnow_iso(),
        "config": config.to_dict(),
        "output_dir": output_dir,
        "dataset_name": config.output.name,
        "task": config.labeling.task,
        "num_rows_final": len(rows),
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "stats": stats,
    }
    meta_path = str(Path(output_dir) / "transform_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Wrote transform_meta.json")

    return output_dir


def _infer_schema(rows: List[Dict[str, Any]], task: str) -> Dict[str, List[str]]:
    """Infer the row schema from the data."""
    if not rows:
        return {}
    sample = rows[0]
    fields = [k for k in sample.keys() if k != "tags"]
    return {task: fields}
