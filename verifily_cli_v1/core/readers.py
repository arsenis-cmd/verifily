"""DatasetReader abstraction for CSV, JSONL, and Parquet.

Dispatches by file extension. Parquet requires optional ``pyarrow`` dependency
(install via ``pip install verifily[arrow]``).
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class ReadResult:
    """Result of reading a dataset file."""

    rows: List[Dict[str, Any]]
    format: str  # "csv", "jsonl", "parquet"
    source_path: str
    row_count: int
    columns: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


# ── Abstract reader ───────────────────────────────────────────────


class DatasetReader(ABC):
    """Base class for dataset readers."""

    extensions: tuple  # file extensions this reader handles

    @abstractmethod
    def can_read(self, path: Path) -> bool: ...

    @abstractmethod
    def read(
        self,
        path: Path,
        *,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> ReadResult: ...

    @abstractmethod
    def read_batched(
        self,
        path: Path,
        *,
        batch_size: int = 10_000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[List[Dict[str, Any]]]: ...


# ── CSV reader ────────────────────────────────────────────────────


class CsvReader(DatasetReader):
    extensions = (".csv", ".tsv")

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def read(
        self,
        path: Path,
        *,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> ReadResult:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows: List[Dict[str, Any]] = []
        all_columns: List[str] = []
        byte_size = path.stat().st_size

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            all_columns = reader.fieldnames or []
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                if columns:
                    row = {k: v for k, v in row.items() if k in columns}
                rows.append(dict(row))

        return ReadResult(
            rows=rows,
            format="csv",
            source_path=str(path),
            row_count=len(rows),
            columns=list(all_columns),
            provenance={"byte_size": byte_size, "delimiter": delimiter},
        )

    def read_batched(
        self,
        path: Path,
        *,
        batch_size: int = 10_000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        batch: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if columns:
                    row = {k: v for k, v in row.items() if k in columns}
                batch.append(dict(row))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


# ── JSONL reader ──────────────────────────────────────────────────


class JsonlReader(DatasetReader):
    extensions = (".jsonl", ".jsonlines")

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def read(
        self,
        path: Path,
        *,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> ReadResult:
        rows: List[Dict[str, Any]] = []
        all_columns: set = set()
        byte_size = path.stat().st_size

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                all_columns.update(row.keys())
                if columns:
                    row = {k: v for k, v in row.items() if k in columns}
                rows.append(row)

        return ReadResult(
            rows=rows,
            format="jsonl",
            source_path=str(path),
            row_count=len(rows),
            columns=sorted(all_columns),
            provenance={"byte_size": byte_size},
        )

    def read_batched(
        self,
        path: Path,
        *,
        batch_size: int = 10_000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if columns:
                    row = {k: v for k, v in row.items() if k in columns}
                batch.append(row)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


# ── Parquet reader ────────────────────────────────────────────────


class ParquetReader(DatasetReader):
    extensions = (".parquet", ".pq")

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def _import_pyarrow(self):
        try:
            import pyarrow.parquet as pq
            return pq
        except ImportError:
            raise ImportError(
                "Parquet support requires pyarrow. "
                "Install it with: pip install verifily[arrow]"
            )

    def read(
        self,
        path: Path,
        *,
        limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> ReadResult:
        pq = self._import_pyarrow()
        byte_size = path.stat().st_size

        pf = pq.ParquetFile(path)
        metadata = pf.metadata
        all_columns = pf.schema_arrow.names

        if limit is not None:
            # Read row groups until we have enough rows
            collected: List[Dict[str, Any]] = []
            for i in range(metadata.num_row_groups):
                table = pf.read_row_group(i, columns=columns)
                collected.extend(table.to_pydict_rows() if hasattr(table, 'to_pydict_rows') else
                                 [dict(zip(table.column_names, vals)) for vals in zip(*[col.to_pylist() for col in table.columns])])
                if len(collected) >= limit:
                    collected = collected[:limit]
                    break
            rows = collected
        else:
            table = pq.read_table(path, columns=columns)
            # Convert to list of dicts
            col_names = table.column_names
            col_data = [table.column(c).to_pylist() for c in col_names]
            rows = [dict(zip(col_names, vals)) for vals in zip(*col_data)] if col_data else []

        return ReadResult(
            rows=rows,
            format="parquet",
            source_path=str(path),
            row_count=len(rows),
            columns=list(all_columns),
            provenance={
                "byte_size": byte_size,
                "num_row_groups": metadata.num_row_groups,
                "num_rows_metadata": metadata.num_rows,
                "created_by": metadata.created_by or "",
            },
        )

    def read_batched(
        self,
        path: Path,
        *,
        batch_size: int = 10_000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        pq = self._import_pyarrow()
        pf = pq.ParquetFile(path)

        for i in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(i, columns=columns)
            col_names = table.column_names
            col_data = [table.column(c).to_pylist() for c in col_names]
            group_rows = [dict(zip(col_names, vals)) for vals in zip(*col_data)] if col_data else []

            # Yield in batch_size chunks
            for start in range(0, len(group_rows), batch_size):
                yield group_rows[start : start + batch_size]


# ── Dispatch ──────────────────────────────────────────────────────

_READERS: List[DatasetReader] = [CsvReader(), JsonlReader(), ParquetReader()]


def get_reader(path: Union[str, Path]) -> DatasetReader:
    """Return the appropriate reader for a file path based on extension."""
    p = Path(path)
    for reader in _READERS:
        if reader.can_read(p):
            return reader
    raise ValueError(
        f"Unsupported file extension '{p.suffix}'. "
        f"Supported: .csv, .tsv, .jsonl, .jsonlines, .parquet, .pq"
    )


def read_dataset(
    path: Union[str, Path],
    *,
    limit: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> ReadResult:
    """Read a dataset file, auto-detecting format by extension."""
    p = Path(path)
    reader = get_reader(p)
    return reader.read(p, limit=limit, columns=columns)
