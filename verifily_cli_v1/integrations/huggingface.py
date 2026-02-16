"""HuggingFace datasets reader — load datasets via hf:// URIs.

URI format::

    hf://dataset_name[/config][?split=train&limit=1000]

Examples::

    hf://squad
    hf://squad/plain_text?split=train
    hf://tatsu-lab/alpaca?split=train&limit=5000

Requires ``datasets`` library: ``pip install verifily[huggingface]``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from verifily_cli_v1.core.readers import DatasetReader, ReadResult


def _import_datasets():
    """Lazy-import the HuggingFace datasets library."""
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace datasets support requires the datasets library. "
            "Install with: pip install verifily[huggingface]"
        )


def _parse_hf_uri(uri: str) -> Tuple[str, Optional[str], str, Optional[int]]:
    """Parse hf://name[/config][?split=X&limit=N] → (name, config, split, limit).

    Examples:
        hf://squad              → ("squad", None, "train", None)
        hf://squad/plain_text   → ("squad", "plain_text", "train", None)
        hf://squad?split=validation&limit=100
                                → ("squad", None, "validation", 100)
        hf://tatsu-lab/alpaca   → ("tatsu-lab/alpaca", None, "train", None)
    """
    # Strip hf:// or hf:/ prefix (Path normalizes hf:// to hf:/)
    raw = uri
    if raw.startswith("hf://"):
        raw = raw[5:]
    elif raw.startswith("hf:/"):
        raw = raw[4:]

    # Parse query params
    split = "train"
    limit: Optional[int] = None
    if "?" in raw:
        path_part, query_part = raw.split("?", 1)
        params = parse_qs(query_part)
        split = params.get("split", ["train"])[0]
        if "limit" in params:
            limit = int(params["limit"][0])
    else:
        path_part = raw

    # Parse name and optional config
    # Handle org/name format: tatsu-lab/alpaca/config → name=tatsu-lab/alpaca, config=config
    parts = path_part.strip("/").split("/")
    if len(parts) >= 3:
        # org/name/config
        name = f"{parts[0]}/{parts[1]}"
        config = parts[2]
    elif len(parts) == 2:
        # Could be org/name or name/config — heuristic: if first part contains
        # a dash or underscore and second part looks like a config, treat as org/name
        # Default: treat as org/name (most common HF convention)
        name = f"{parts[0]}/{parts[1]}"
        config = None
    elif len(parts) == 1:
        name = parts[0]
        config = None
    else:
        name = path_part
        config = None

    return name, config, split, limit


class HuggingFaceReader(DatasetReader):
    """Read datasets from HuggingFace Hub via hf:// URIs."""

    extensions = ()  # No file extensions — uses URI scheme

    # Store original string URI since Path normalizes hf:// to hf:/
    _original_uri: str = ""

    def can_read(self, path: Path) -> bool:
        # Path("hf://x") becomes "hf:/x" — check both forms
        s = str(path)
        return s.startswith("hf://") or s.startswith("hf:/")

    def read(
        self,
        path: Path,
        *,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> ReadResult:
        ds_lib = _import_datasets()
        name, config, split, uri_limit = _parse_hf_uri(str(path))
        effective_limit = limit or uri_limit

        ds = ds_lib.load_dataset(name, config, split=split)

        if effective_limit:
            ds = ds.select(range(min(effective_limit, len(ds))))

        if columns:
            ds = ds.select_columns(columns)

        rows = [dict(row) for row in ds]
        all_columns = ds.column_names if hasattr(ds, "column_names") else list(rows[0].keys()) if rows else []

        return ReadResult(
            rows=rows,
            format="huggingface",
            source_path=str(path),
            row_count=len(rows),
            columns=list(all_columns) if isinstance(all_columns, list) else [all_columns],
            provenance={
                "dataset_name": name,
                "config": config,
                "split": split,
                "hf_features": str(ds.features) if hasattr(ds, "features") else "",
            },
        )

    def read_batched(
        self,
        path: Path,
        *,
        batch_size: int = 10_000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        ds_lib = _import_datasets()
        name, config, split, _ = _parse_hf_uri(str(path))

        ds = ds_lib.load_dataset(name, config, split=split, streaming=True)

        batch: List[Dict[str, Any]] = []
        for row in ds:
            record = dict(row)
            if columns:
                record = {k: v for k, v in record.items() if k in columns}
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
