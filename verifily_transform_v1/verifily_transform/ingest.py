"""Verifily Transform ingestion: read raw data from JSONL, CSV, or TXT."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from verifily_transform.config import InputConfig
from verifily_transform.errors import IngestError

logger = logging.getLogger(__name__)


def ingest(config: InputConfig) -> List[Dict[str, Any]]:
    """Read raw data files from the configured input path.

    Supports:
      - JSONL: one JSON object per line
      - CSV: first row is header, each row becomes a dict
      - TXT: each line becomes {"text": <line>}

    If path is a directory, reads all matching files inside it.
    If path is a single file, reads that file.

    Returns a list of raw row dicts.
    """
    p = Path(config.path)
    if not p.exists():
        raise IngestError(f"Input path not found: {config.path}")

    if p.is_file():
        files = [p]
    elif p.is_dir():
        ext_map = {"jsonl": "*.jsonl", "csv": "*.csv", "txt": "*.txt"}
        pattern = ext_map.get(config.format, f"*.{config.format}")
        files = sorted(p.glob(pattern))
        if not files:
            raise IngestError(f"No {config.format} files found in {config.path}")
    else:
        raise IngestError(f"Input path is neither file nor directory: {config.path}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        logger.info("Ingesting %s", fp)
        try:
            if config.format == "jsonl":
                rows.extend(_read_jsonl(str(fp)))
            elif config.format == "csv":
                rows.extend(_read_csv(str(fp), config.encoding))
            elif config.format == "txt":
                rows.extend(_read_txt(str(fp), config.encoding))
        except Exception as e:
            raise IngestError(f"Failed to read {fp}: {e}") from e

    logger.info("Ingested %d raw rows from %d file(s)", len(rows), len(files))
    return rows


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise IngestError(f"Invalid JSON on line {i+1} of {path}: {e}")
    return rows


def _read_csv(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_txt(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append({"text": line})
    return rows
