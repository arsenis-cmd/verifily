"""verifily ingest — convert messy CSV/JSONL into canonical Verifily dataset artifacts."""

from __future__ import annotations

import csv
import datetime
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.table import Table

from verifily_cli_v1.core.hashing import sha256_file, sha256_string
from verifily_cli_v1.core.io import ensure_dir, read_jsonl, write_json, write_jsonl

console = Console(stderr=True)

# ── PII patterns (reused from report.py) ────────────────────────

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}

# ── Schema definitions ──────────────────────────────────────────

# Canonical required fields per schema.
SCHEMA_REQUIRED = {
    "sft": [["input", "output"], ["question", "answer"]],  # either group
    "classification": [["text", "label"]],                   # only one group
}


# ── Reading helpers ─────────────────────────────────────────────

def read_csv_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read a CSV file into list of dicts.  All values are strings."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def read_input_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read CSV or JSONL, auto-detected by extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        return read_csv_rows(p)
    if ext in (".jsonl", ".jsonlines"):
        return read_jsonl(p)
    raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .jsonl.")


# ── Column mapping ──────────────────────────────────────────────

def apply_mapping(
    rows: List[Dict[str, Any]], mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Rename columns according to mapping {canonical_field: source_column}.

    Only the mapped columns are renamed; others are carried through as-is.
    """
    if not mapping:
        return rows
    # Invert: source_column -> canonical_field
    inv = {v: k for k, v in mapping.items()}
    out: List[Dict[str, Any]] = []
    for row in rows:
        new: Dict[str, Any] = {}
        for k, v in row.items():
            new[inv.get(k, k)] = v
        out.append(new)
    return out


def parse_map_args(map_args: Optional[List[str]]) -> Dict[str, str]:
    """Parse --map canonical:source pairs into {canonical: source} dict."""
    if not map_args:
        return {}
    mapping: Dict[str, str] = {}
    for arg in map_args:
        if ":" not in arg:
            raise ValueError(
                f"Invalid --map value '{arg}'. Expected canonical_field:source_column "
                f"(e.g. question:prompt)"
            )
        canonical, source = arg.split(":", 1)
        mapping[canonical.strip()] = source.strip()
    return mapping


def parse_tag_args(tag_args: Optional[List[str]]) -> Dict[str, str]:
    """Parse --tag key:value pairs into dict."""
    if not tag_args:
        return {}
    tags: Dict[str, str] = {}
    for arg in tag_args:
        if ":" not in arg:
            raise ValueError(f"Invalid --tag value '{arg}'. Expected key:value.")
        k, v = arg.split(":", 1)
        tags[k.strip()] = v.strip()
    return tags


# ── Schema detection ────────────────────────────────────────────

def detect_schema(rows: List[Dict[str, Any]]) -> str:
    """Auto-detect schema from column names of the first row."""
    if not rows:
        raise ValueError("Cannot auto-detect schema: no rows in input.")
    keys = set(rows[0].keys())
    # SFT patterns
    if ("input" in keys and "output" in keys) or (
        "question" in keys and "answer" in keys
    ):
        return "sft"
    # Classification pattern
    if "text" in keys and "label" in keys:
        return "classification"
    raise ValueError(
        f"Cannot auto-detect schema from columns: {sorted(keys)}.\n"
        "Expected one of:\n"
        "  sft:            input + output  OR  question + answer\n"
        "  classification: text + label\n"
        "Hint: use --map to rename your columns, e.g. "
        "--map question:prompt --map answer:completion"
    )


# ── Row validation ──────────────────────────────────────────────

def _has_required(row: Dict[str, Any], schema: str) -> Tuple[bool, str]:
    """Check if row has required fields for schema.

    Returns (ok, reason).
    """
    groups = SCHEMA_REQUIRED[schema]
    for group in groups:
        if all(k in row and str(row[k]).strip() for k in group):
            return True, ""
    expected = " OR ".join(str(g) for g in groups)
    return False, f"missing required fields {expected}"


# ── Canonicalization ────────────────────────────────────────────

def canonicalize_row(
    row: Dict[str, Any],
    schema: str,
    extra_tags: Dict[str, str],
) -> Dict[str, Any]:
    """Convert a raw row into canonical schema.

    Returns the canonical dict (with id, tags, and schema-specific fields).
    """
    canonical: Dict[str, Any] = {}

    if schema == "sft":
        if "question" in row and "answer" in row:
            context = str(row.get("context", "")).strip()
            question = str(row["question"]).strip()
            if context:
                canonical["input"] = f"Context:\n{context}\n\nQuestion:\n{question}"
            else:
                canonical["input"] = f"Question:\n{question}"
            canonical["output"] = str(row["answer"]).strip()
        else:
            canonical["input"] = str(row.get("input", "")).strip()
            canonical["output"] = str(row.get("output", "")).strip()

    elif schema == "classification":
        canonical["text"] = str(row.get("text", "")).strip()
        canonical["label"] = str(row.get("label", "")).strip()

    # Tags: merge extra_tags (don't override existing keys)
    existing_tags = row.get("tags", {})
    if isinstance(existing_tags, str):
        try:
            existing_tags = json.loads(existing_tags)
        except (json.JSONDecodeError, TypeError):
            existing_tags = {}
    if not isinstance(existing_tags, dict):
        existing_tags = {}
    merged_tags = {**extra_tags}
    merged_tags.update(existing_tags)  # existing keys win
    canonical["tags"] = merged_tags

    # Deterministic ID: sha256 of canonical row (excluding id itself)
    row_json = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    canonical["id"] = "row_" + sha256_string(row_json)[:16]

    return canonical


# ── Report generation ───────────────────────────────────────────

def _token_count(text: str) -> int:
    """Approximate token count by whitespace split."""
    return len(text.split())


def _percentile(data: List[int], p: float) -> int:
    """Simple percentile (nearest rank)."""
    if not data:
        return 0
    k = int(len(data) * p / 100)
    k = min(k, len(data) - 1)
    return sorted(data)[k]


def generate_report(
    rows: List[Dict[str, Any]], schema: str
) -> Dict[str, Any]:
    """Generate a lightweight report for the ingested dataset."""
    # Field length stats
    field_stats: Dict[str, Dict[str, Any]] = {}
    text_fields = ["input", "output"] if schema == "sft" else ["text"]

    for field in text_fields:
        lengths = [_token_count(str(r.get(field, ""))) for r in rows]
        char_lengths = [len(str(r.get(field, ""))) for r in rows]
        if lengths:
            field_stats[field] = {
                "token_min": min(lengths),
                "token_median": int(statistics.median(lengths)),
                "token_p95": _percentile(lengths, 95),
                "token_max": max(lengths),
                "char_min": min(char_lengths),
                "char_median": int(statistics.median(char_lengths)),
                "char_p95": _percentile(char_lengths, 95),
                "char_max": max(char_lengths),
            }

    # Exact duplicate rate (by full-line hash)
    line_hashes = Counter()
    for r in rows:
        h = sha256_string(json.dumps(r, sort_keys=True, ensure_ascii=False))
        line_hashes[h] += 1
    total = len(rows)
    unique = len(line_hashes)
    duplicate_count = total - unique
    duplicate_rate = duplicate_count / total if total else 0.0

    # PII scan (counts only)
    pii_counts: Dict[str, int] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        hits = 0
        for r in rows:
            text = " ".join(str(v) for k, v in r.items() if k != "tags")
            if pattern.search(text):
                hits += 1
        pii_counts[pii_type] = hits
    pii_total = sum(pii_counts.values())

    return {
        "row_count": total,
        "schema": schema,
        "field_stats": field_stats,
        "exact_duplicates": duplicate_count,
        "exact_duplicate_rate": round(duplicate_rate, 4),
        "unique_rows": unique,
        "pii_scan": pii_counts,
        "pii_total_hits": pii_total,
        "pii_clean": pii_total == 0,
    }


# ── Artifact writing ───────────────────────────────────────────

def write_artifacts(
    rows: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    *,
    schema: str,
    input_path: str,
    row_count_in: int,
    dropped: List[Dict[str, Any]],
    extra_tags: Dict[str, str],
) -> Dict[str, Any]:
    """Write dataset.jsonl, manifest.json, hashes.json, report.json.

    Returns manifest dict.
    """
    out = ensure_dir(output_dir)

    # 1. dataset.jsonl
    write_jsonl(out / "dataset.jsonl", rows)

    # 2. report.json
    report = generate_report(rows, schema)
    write_json(out / "report.json", report)

    # 3. Tag distribution for manifest
    tag_dist: Dict[str, Dict[str, int]] = {}
    for r in rows:
        tags = r.get("tags", {})
        if isinstance(tags, dict):
            for k, v in tags.items():
                tag_dist.setdefault(k, Counter())
                tag_dist[k][str(v)] += 1
    tag_summary = {k: dict(v) for k, v in tag_dist.items()}

    # Collect drop reasons
    drop_reasons: Counter = Counter()
    for d in dropped:
        drop_reasons[d.get("reason", "unknown")] += 1
    top_drop_reasons = dict(drop_reasons.most_common(10))

    # 4. manifest.json (write before hashes since hashes includes manifest hash)
    try:
        from verifily_cli_v1 import __version__
        ver = __version__
    except Exception:
        ver = "unknown"

    manifest = {
        "schema": schema,
        "input_path": input_path,
        "row_count_in": row_count_in,
        "row_count_out": len(rows),
        "dropped_rows_count": len(dropped),
        "top_drop_reasons": top_drop_reasons,
        "tag_distribution": tag_summary,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "verifily_version": ver,
    }
    write_json(out / "manifest.json", manifest)

    # 5. hashes.json — hash all files we just wrote
    file_hashes: Dict[str, str] = {}
    for fname in sorted(["dataset.jsonl", "report.json", "manifest.json"]):
        fp = out / fname
        if fp.exists():
            file_hashes[fname] = sha256_file(fp)

    hashes_data = {
        "files": file_hashes,
        "dataset_hash": file_hashes.get("dataset.jsonl", ""),
    }
    write_json(out / "hashes.json", hashes_data)

    return manifest


# ── Core ingest logic ───────────────────────────────────────────

def ingest(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    schema: str = "auto",
    mapping: Optional[Dict[str, str]] = None,
    extra_tags: Optional[Dict[str, str]] = None,
    id_col: Optional[str] = None,
    limit: Optional[int] = None,
    strict: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the full ingest pipeline.

    Returns a result dict with keys:
        rows_in, rows_out, rows_dropped, schema, output_dir,
        dataset_hash, manifest, dropped_details
    """
    extra_tags = extra_tags or {}
    mapping = mapping or {}

    # 1. Read input
    raw_rows = read_input_file(input_path)
    if limit is not None and limit > 0:
        raw_rows = raw_rows[:limit]
    row_count_in = len(raw_rows)

    if row_count_in == 0:
        raise ValueError(f"No rows found in {input_path}")

    # 2. Apply column mapping
    mapped_rows = apply_mapping(raw_rows, mapping)

    # 3. Detect or validate schema
    if schema == "auto":
        schema = detect_schema(mapped_rows)

    if schema not in SCHEMA_REQUIRED:
        raise ValueError(f"Unknown schema '{schema}'. Expected: sft, classification.")

    # 4. Validate and canonicalize
    canonical: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for i, row in enumerate(mapped_rows):
        ok, reason = _has_required(row, schema)
        if not ok:
            if strict:
                raise ValueError(f"Row {i}: {reason}")
            dropped.append({"row_index": i, "reason": reason})
            continue
        c = canonicalize_row(row, schema, extra_tags)
        # Override id if user supplied id_col and source row had it
        if id_col and id_col in raw_rows[i] and str(raw_rows[i][id_col]).strip():
            c["id"] = str(raw_rows[i][id_col]).strip()
        canonical.append(c)

    if not canonical:
        raise ValueError(
            f"All {row_count_in} rows were invalid for schema '{schema}'. "
            f"Check your column names or use --map."
        )

    # 5. Dry run: report but don't write
    if dry_run:
        return {
            "dry_run": True,
            "rows_in": row_count_in,
            "rows_out": len(canonical),
            "rows_dropped": len(dropped),
            "schema": schema,
            "output_dir": str(output_dir),
            "dataset_hash": None,
            "manifest": None,
            "dropped_details": dropped,
            "preview": canonical[:3],
        }

    # 6. Write artifacts
    manifest = write_artifacts(
        canonical,
        output_dir,
        schema=schema,
        input_path=str(input_path),
        row_count_in=row_count_in,
        dropped=dropped,
        extra_tags=extra_tags,
    )

    # Read back dataset hash
    out = Path(output_dir)
    hashes = {}
    if (out / "hashes.json").exists():
        import json as _json
        with open(out / "hashes.json") as f:
            hashes = _json.load(f)

    return {
        "dry_run": False,
        "rows_in": row_count_in,
        "rows_out": len(canonical),
        "rows_dropped": len(dropped),
        "schema": schema,
        "output_dir": str(output_dir),
        "dataset_hash": hashes.get("dataset_hash", ""),
        "manifest": manifest,
        "dropped_details": dropped,
    }


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    input_path: str,
    output_path: str,
    schema: str = "auto",
    map_args: Optional[List[str]] = None,
    tag_args: Optional[List[str]] = None,
    id_col: Optional[str] = None,
    limit: Optional[int] = None,
    strict: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """CLI entry point for verifily ingest."""
    mapping = parse_map_args(map_args)
    tags = parse_tag_args(tag_args)

    result = ingest(
        input_path,
        output_path,
        schema=schema,
        mapping=mapping,
        extra_tags=tags,
        id_col=id_col,
        limit=limit,
        strict=strict,
        dry_run=dry_run,
    )

    # Display results
    if result["dry_run"]:
        console.print("\n[bold yellow]DRY RUN[/bold yellow] — no files written\n")
        console.print(f"  Input:   {input_path}")
        console.print(f"  Schema:  [cyan]{result['schema']}[/cyan]")
        console.print(f"  Rows in: {result['rows_in']}")
        console.print(f"  Rows out: [green]{result['rows_out']}[/green]")
        if result["rows_dropped"]:
            console.print(f"  Dropped: [yellow]{result['rows_dropped']}[/yellow]")
        console.print("\n  [dim]Preview (first 3 rows):[/dim]")
        for row in result.get("preview", []):
            console.print(f"    {json.dumps(row, ensure_ascii=False)[:120]}")
        console.print()
    else:
        console.print(f"\n[bold green]Ingest complete[/bold green]\n")
        console.print(f"  Output:  {result['output_dir']}")
        console.print(f"  Schema:  [cyan]{result['schema']}[/cyan]")
        console.print(f"  Rows in: {result['rows_in']}")
        console.print(f"  Rows out: [green]{result['rows_out']}[/green]")
        if result["rows_dropped"]:
            console.print(f"  Dropped: [yellow]{result['rows_dropped']}[/yellow]")
        console.print(f"  Hash:    [dim]{result['dataset_hash'][:24]}…[/dim]")
        console.print()

        if verbose and result["rows_dropped"] and result.get("dropped_details"):
            console.print("[dim]Dropped rows:[/dim]")
            for d in result["dropped_details"][:10]:
                console.print(f"  row {d['row_index']}: {d['reason']}")
            console.print()

    return result
