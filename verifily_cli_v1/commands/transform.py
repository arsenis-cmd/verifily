"""verifily transform — validate, normalize, and package raw data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from verifily_cli_v1.core.hashing import sha256_dict, sha256_file, sha256_string
from verifily_cli_v1.core.io import (
    count_lines,
    ensure_dir,
    file_size_human,
    read_jsonl,
    resolve_input_files,
    write_jsonl,
)
from verifily_cli_v1.core.manifest import create_manifest

console = Console()

# ── Field mapping candidates ──────────────────────────────────────

_INSTRUCTION_FIELDS = {"instruction", "question", "prompt", "query", "input_text"}
_OUTPUT_FIELDS = {"output", "answer", "response", "completion", "target"}
_INPUT_FIELDS = {"input", "context", "passage", "document"}
_TEXT_FIELDS = {"text", "content", "body", "sentence"}
_LABEL_FIELDS = {"label", "class", "category", "sentiment", "tag"}


def _detect_mapping(keys: set, schema: str) -> Dict[str, str]:
    """Auto-detect field mapping from raw keys to canonical schema."""
    mapping: Dict[str, str] = {}
    lower_keys = {k.lower(): k for k in keys}

    if schema in ("sft", "qa"):
        for candidate in _INSTRUCTION_FIELDS:
            if candidate in lower_keys:
                mapping[lower_keys[candidate]] = "instruction"
                break
        for candidate in _OUTPUT_FIELDS:
            if candidate in lower_keys:
                mapping[lower_keys[candidate]] = "output"
                break
        for candidate in _INPUT_FIELDS:
            if candidate in lower_keys:
                mapping[lower_keys[candidate]] = "input"
                break
    elif schema == "classification":
        for candidate in _TEXT_FIELDS:
            if candidate in lower_keys:
                mapping[lower_keys[candidate]] = "text"
                break
        for candidate in _LABEL_FIELDS:
            if candidate in lower_keys:
                mapping[lower_keys[candidate]] = "label"
                break

    return mapping


def _read_csv_as_dicts(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV file as a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_txt_as_dicts(path: Path) -> List[Dict[str, Any]]:
    """Read a TXT file (one sample per line) as a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append({"text": line})
    return rows


def _ingest_file(path: Path) -> List[Dict[str, Any]]:
    """Ingest a single file based on extension."""
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return read_jsonl(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]
    elif ext == ".csv":
        return _read_csv_as_dicts(path)
    elif ext == ".txt":
        return _read_txt_as_dicts(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _normalize_row(
    row: Dict[str, Any], mapping: Dict[str, str], schema: str
) -> Optional[Dict[str, Any]]:
    """Normalize a single row using the field mapping."""
    normalized: Dict[str, Any] = {}

    for src_key, dst_key in mapping.items():
        if src_key in row and row[src_key]:
            normalized[dst_key] = str(row[src_key]).strip()

    # Validate required fields
    if schema in ("sft", "qa"):
        if "instruction" not in normalized or "output" not in normalized:
            return None
        normalized.setdefault("input", "")
        normalized["tags"] = row.get("tags", {})
        if not isinstance(normalized["tags"], dict):
            normalized["tags"] = {}
    elif schema == "classification":
        if "text" not in normalized or "label" not in normalized:
            return None
        normalized["tags"] = row.get("tags", {})
        if not isinstance(normalized["tags"], dict):
            normalized["tags"] = {}

    return normalized


def _apply_privacy(row: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Apply privacy handling to a row."""
    if mode == "none":
        return row

    import re

    patterns = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    def redact_value(val: str) -> str:
        if not isinstance(val, str):
            return val
        for pii_type, pattern in patterns.items():
            if mode == "redact":
                val = re.sub(pattern, f"[{pii_type}_REDACTED]", val)
            elif mode == "hash":
                val = re.sub(
                    pattern,
                    lambda m: sha256_string(m.group(0))[:12],
                    val,
                )
        return val

    result = {}
    for key, value in row.items():
        if isinstance(value, str):
            result[key] = redact_value(value)
        else:
            result[key] = value
    return result


def run(
    input_path: str,
    output_path: str,
    format: str = "jsonl",
    schema: str = "sft",
    privacy: str = "redact",
    expand: Optional[int] = None,
    plan: bool = False,
    verbose: bool = False,
) -> None:
    """Validate, normalize, and package raw data into a training-ready dataset."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    console.print(
        Panel(
            "[bold]Verifily Transform[/bold]\n"
            f"Input:  {in_path}\n"
            f"Output: {out_path}\n"
            f"Schema: {schema}  Format: {format}  Privacy: {privacy}",
            border_style="blue",
        )
    )

    # ── Discover input files ──
    try:
        files = resolve_input_files(in_path)
    except FileNotFoundError as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise SystemExit(1)

    console.print(f"  Found [cyan]{len(files)}[/cyan] input file(s)")
    if verbose:
        for f in files:
            console.print(f"    {f.name}  ({file_size_human(f)})")

    # ── Plan mode: show what would happen and exit ──
    if plan:
        console.print("\n[yellow bold]--plan mode:[/yellow bold] dry run, no files written\n")

        # Count total rows
        total = 0
        for f in files:
            try:
                rows = _ingest_file(f)
                total += len(rows)
            except Exception:
                total += count_lines(f)

        plan_table = Table(title="Transform Plan", show_header=True, header_style="bold")
        plan_table.add_column("Step", style="cyan")
        plan_table.add_column("Detail")
        plan_table.add_row("Ingest", f"{len(files)} file(s), ~{total} rows")
        plan_table.add_row("Normalize", f"Schema: {schema}")
        plan_table.add_row("Privacy", privacy)
        plan_table.add_row("Deduplicate", "Exact hash")
        if expand:
            plan_table.add_row("Expand", f"{expand}x (recorded in manifest)")
        plan_table.add_row("Output", str(out_path))
        console.print(plan_table)
        console.print(
            f"\n[dim]Next:[/dim] verifily transform --in {in_path} --out {out_path} --schema {schema}"
        )
        return

    # ── Ingest ──
    all_rows: List[Dict[str, Any]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(files))
        for f in files:
            try:
                rows = _ingest_file(f)
                all_rows.extend(rows)
            except Exception as e:
                if verbose:
                    console.print(f"  [yellow]Skipping {f.name}: {e}[/yellow]")
            progress.advance(task)

    ingested_count = len(all_rows)
    console.print(f"  Ingested [cyan]{ingested_count}[/cyan] rows")

    if not all_rows:
        console.print("[red bold]Error:[/red bold] No rows ingested. Check input path and format.")
        raise SystemExit(1)

    # ── Detect mapping ──
    sample_keys = set(all_rows[0].keys())
    mapping = _detect_mapping(sample_keys, schema)

    if verbose:
        console.print(f"  Field mapping: {mapping}")

    # If no mapping found, try identity (fields already named correctly)
    if not mapping:
        if schema in ("sft", "qa"):
            for key in ("instruction", "output", "input"):
                if key in sample_keys:
                    mapping[key] = key
        elif schema == "classification":
            for key in ("text", "label"):
                if key in sample_keys:
                    mapping[key] = key

    # ── Normalize ──
    normalized: List[Dict[str, Any]] = []
    dropped = 0
    for row in all_rows:
        result = _normalize_row(row, mapping, schema)
        if result is not None:
            normalized.append(result)
        else:
            dropped += 1

    console.print(f"  Normalized [cyan]{len(normalized)}[/cyan] rows ({dropped} dropped)")

    # ── Privacy ──
    if privacy != "none":
        normalized = [_apply_privacy(r, privacy) for r in normalized]
        console.print(f"  Privacy applied: [cyan]{privacy}[/cyan]")

    # ── Deduplicate (exact) ──
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    dupes = 0
    for row in normalized:
        key = sha256_dict({k: v for k, v in row.items() if k != "tags"})
        if key not in seen:
            seen.add(key)
            deduped.append(row)
        else:
            dupes += 1

    console.print(f"  Deduplicated: [cyan]{len(deduped)}[/cyan] rows ({dupes} duplicates removed)")

    # ── Write output ──
    out_dir = ensure_dir(out_path)
    dataset_path = out_dir / "dataset.jsonl"
    write_jsonl(dataset_path, deduped)

    source_names = [f.name for f in files]

    extra: Dict[str, Any] = {
        "pipeline": {
            "ingested": ingested_count,
            "normalized": len(normalized),
            "dropped": dropped,
            "duplicates_removed": dupes,
            "final": len(deduped),
            "privacy_mode": privacy,
        }
    }
    if expand:
        extra["expand_factor"] = expand
        extra["expand_note"] = "Recorded for future synthetic generation"

    manifest = create_manifest(
        out_dir,
        name=out_path.name,
        task="sft" if schema in ("sft", "qa") else "classification",
        schema=schema,
        format=format,
        row_count=len(deduped),
        source_files=source_names,
        extra=extra,
    )

    # ── Summary ──
    console.print()
    summary = Table(title="Transform Complete", show_header=True, header_style="bold")
    summary.add_column("Artifact", style="cyan")
    summary.add_column("Detail")
    summary.add_row("dataset.jsonl", f"{len(deduped)} rows, {file_size_human(dataset_path)}")
    summary.add_row("manifest.json", f"chain_hash: {manifest['chain_hash'][:16]}...")
    summary.add_row("hashes.json", "Integrity hashes for all artifacts")
    summary.add_row("Output dir", str(out_dir))
    console.print(summary)

    console.print(
        f"\n[dim]Next:[/dim] verifily train --config train.yaml --run-dir runs/my_run"
    )
