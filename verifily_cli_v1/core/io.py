"""Safe file I/O, JSON/YAML helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory and parents if needed, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file and return parsed contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Union[str, Path], data: Any, indent: int = 2) -> Path:
    """Write data to a JSON file."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return p


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL file, return list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]]) -> Path:
    """Write a list of dicts to a JSONL file."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML file and return parsed contents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Union[str, Path], data: Dict[str, Any]) -> Path:
    """Write data to a YAML file."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return p


def resolve_input_files(
    path: Union[str, Path], extensions: Optional[List[str]] = None
) -> List[Path]:
    """Resolve a path to a list of files. If directory, discover files by extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if p.is_file():
        return [p]
    if extensions is None:
        extensions = [".jsonl", ".json", ".csv", ".txt"]
    files = []
    for ext in extensions:
        files.extend(sorted(p.glob(f"*{ext}")))
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {extensions} found in: {p}"
        )
    return files


def count_lines(path: Union[str, Path]) -> int:
    """Count non-empty lines in a file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def file_size_human(path: Union[str, Path]) -> str:
    """Return human-readable file size."""
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
