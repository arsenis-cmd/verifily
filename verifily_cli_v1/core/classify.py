"""Dataset classification engine for Verifily.

Heuristic-only, deterministic, local.  No model calls, no external APIs.

Detects:
  - Schema type: sft, qa, classification, chat, unknown
  - Buckets by inferred row type + optional category tag
  - Tags: language (en/non-en), pii_risk (email/phone patterns), duplicate_rate
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── PII patterns (from centralized pii module) ──────────────────

from verifily_cli_v1.core.pii import PII_PATTERNS as _PII_PATTERNS

_EMAIL_RE = _PII_PATTERNS["email"]
_PHONE_RE = _PII_PATTERNS["phone"]

# ── Schema detection (from centralized schemas module) ────────────

from verifily_cli_v1.core.schemas import SCHEMA_REGISTRY, detect_schema_from_fields


def _detect_schema(fields: set) -> str:
    """Detect the most likely schema from a row's field names."""
    try:
        return detect_schema_from_fields(fields)
    except ValueError:
        return "unknown"


def _detect_row_schema(row: Dict[str, Any]) -> str:
    """Detect schema for a single row, including chat content inspection."""
    fields = set(row.keys())
    lower_fields = {f.lower() for f in fields}

    # Chat: check messages field contains role/content dicts
    for key in ("messages", "conversations"):
        if key in lower_fields:
            val = row.get(key) or row.get(key.capitalize())
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, dict) and {"role", "content"}.issubset(
                    {k.lower() for k in first.keys()}
                ):
                    return "chat"

    return _detect_schema(fields)


def _is_mostly_ascii(text: str, threshold: float = 0.90) -> bool:
    """Return True if >= threshold of characters are basic ASCII/Latin."""
    if not text:
        return True
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold


def _text_from_row(row: Dict[str, Any]) -> str:
    """Extract a representative text blob from a row for scanning."""
    parts = []
    for v in row.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for vv in item.values():
                        if isinstance(vv, str):
                            parts.append(vv)
                elif isinstance(item, str):
                    parts.append(item)
    return " ".join(parts)


def _row_hash(row: Dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a row's sorted key-value pairs."""
    import json
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _infer_category(row: Dict[str, Any]) -> str:
    """Infer a coarse category tag from row fields."""
    # Check for explicit tags/category fields
    for key in ("category", "tag", "source", "type", "topic"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    # Check inside a tags dict
    tags = row.get("tags")
    if isinstance(tags, dict):
        for key in ("source", "category", "type"):
            val = tags.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
    return "uncategorized"


# ── Main entry point ─────────────────────────────────────────────

def classify_dataset(
    rows: List[Dict[str, Any]],
    *,
    max_rows_scan: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Classify a dataset and return a deterministic result dict.

    Args:
        rows: List of row dicts (JSONL-like).
        max_rows_scan: Max rows to scan for heuristics (0=all).
        seed: Unused for now but reserved for deterministic sampling.

    Returns:
        Dict with keys: row_count, suggested_schema, dataset_buckets,
        tags_summary, warnings, exit_code.
    """
    total = len(rows)
    scan_rows = rows[:max_rows_scan] if max_rows_scan > 0 else rows
    scanned = len(scan_rows)

    if total == 0:
        return {
            "row_count": 0,
            "rows_scanned": 0,
            "suggested_schema": "unknown",
            "dataset_buckets": [],
            "tags_summary": {
                "language": {"en": 0, "non_en": 0},
                "pii_risk": {"emails": 0, "phones": 0, "rows_with_pii": 0},
                "duplicate_rate": 0.0,
                "categories": {},
            },
            "warnings": ["Empty dataset."],
            "exit_code": 0,
        }

    # ── Schema detection (vote across scanned rows) ──────────
    schema_votes: Counter = Counter()
    for row in scan_rows:
        schema_votes[_detect_row_schema(row)] += 1

    suggested_schema = schema_votes.most_common(1)[0][0]

    # ── Bucketing by (schema, category) ──────────────────────
    bucket_counter: Counter = Counter()
    for row in scan_rows:
        schema = _detect_row_schema(row)
        cat = _infer_category(row)
        bucket_counter[(schema, cat)] += 1

    dataset_buckets = []
    for (schema, cat), count in sorted(
        bucket_counter.items(), key=lambda x: -x[1]
    ):
        dataset_buckets.append({
            "schema": schema,
            "category": cat,
            "count": count,
            "fraction": round(count / scanned, 4),
        })

    # ── Language detection ────────────────────────────────────
    en_count = 0
    non_en_count = 0
    for row in scan_rows:
        text = _text_from_row(row)
        if _is_mostly_ascii(text):
            en_count += 1
        else:
            non_en_count += 1

    # ── PII risk (counts only, never raw values) ─────────────
    email_hits = 0
    phone_hits = 0
    rows_with_pii = 0
    for row in scan_rows:
        text = _text_from_row(row)
        has_email = bool(_EMAIL_RE.search(text))
        has_phone = bool(_PHONE_RE.search(text))
        if has_email:
            email_hits += 1
        if has_phone:
            phone_hits += 1
        if has_email or has_phone:
            rows_with_pii += 1

    # ── Duplicate detection ──────────────────────────────────
    hashes: Counter = Counter()
    for row in scan_rows:
        hashes[_row_hash(row)] += 1
    dup_rows = sum(c - 1 for c in hashes.values() if c > 1)
    duplicate_rate = round(dup_rows / scanned, 4) if scanned > 0 else 0.0

    # ── Category distribution ────────────────────────────────
    cat_dist: Counter = Counter()
    for row in scan_rows:
        cat_dist[_infer_category(row)] += 1
    categories = {k: v for k, v in sorted(cat_dist.items(), key=lambda x: -x[1])}

    # ── Warnings ─────────────────────────────────────────────
    warnings: List[str] = []
    if suggested_schema == "unknown":
        warnings.append(
            "Could not determine dataset schema. Fields do not match any known pattern."
        )
    if duplicate_rate > 0.05:
        warnings.append(
            f"High duplicate rate: {duplicate_rate:.1%} of scanned rows are exact duplicates."
        )
    if rows_with_pii > 0:
        pii_pct = rows_with_pii / scanned
        warnings.append(
            f"PII detected in {rows_with_pii}/{scanned} scanned rows ({pii_pct:.0%}). "
            f"Email patterns: {email_hits}, phone patterns: {phone_hits}."
        )
    mixed_schemas = [s for s, c in schema_votes.items() if s != suggested_schema and c > 0]
    if mixed_schemas:
        warnings.append(
            f"Mixed schemas detected: primary={suggested_schema}, "
            f"also found: {', '.join(mixed_schemas)}."
        )
    if scanned < total:
        warnings.append(
            f"Only scanned {scanned}/{total} rows. Results are approximate."
        )

    return {
        "row_count": total,
        "rows_scanned": scanned,
        "suggested_schema": suggested_schema,
        "dataset_buckets": dataset_buckets,
        "tags_summary": {
            "language": {"en": en_count, "non_en": non_en_count},
            "pii_risk": {
                "emails": email_hits,
                "phones": phone_hits,
                "rows_with_pii": rows_with_pii,
            },
            "duplicate_rate": duplicate_rate,
            "categories": categories,
        },
        "warnings": warnings,
        "exit_code": 0,
    }


# ── Bucket export ─────────────────────────────────────────────────


@dataclass
class ExportSummary:
    """Summary returned by write_bucket_exports()."""

    bucket_paths: Dict[str, str] = field(default_factory=dict)
    rows_per_bucket: Dict[str, int] = field(default_factory=dict)
    schemas_per_bucket: Dict[str, str] = field(default_factory=dict)
    total_rows_written: int = 0
    suggested_next_steps_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_paths": self.bucket_paths,
            "rows_per_bucket": self.rows_per_bucket,
            "schemas_per_bucket": self.schemas_per_bucket,
            "total_rows_written": self.total_rows_written,
            "suggested_next_steps_path": self.suggested_next_steps_path,
        }


def _bucket_filename(schema: str, category: str) -> str:
    """Derive a safe filename for a bucket: <schema>__<category>.jsonl"""
    safe_cat = re.sub(r"[^a-z0-9_]", "_", category.lower()).strip("_") or "uncategorized"
    return f"{schema}__{safe_cat}.jsonl"


def _suggested_next_steps(
    classification: Dict[str, Any],
    export: "ExportSummary",
) -> Dict[str, Any]:
    """Build suggested_next_steps.json content from classification + export."""
    steps: List[Dict[str, Any]] = []

    schema = classification.get("suggested_schema", "unknown")
    pii = classification.get("tags_summary", {}).get("pii_risk", {})
    dup_rate = classification.get("tags_summary", {}).get("duplicate_rate", 0.0)

    # Step: schema-specific action
    if schema == "sft":
        steps.append({
            "action": "run_pipeline",
            "reason": "Dataset matches SFT schema — run full Verifily pipeline.",
            "command": "verifily pipeline --config verifily.yaml",
        })
    elif schema == "qa":
        steps.append({
            "action": "run_pipeline",
            "reason": "Dataset matches QA schema — run full Verifily pipeline with QA checks.",
            "command": "verifily pipeline --config verifily.yaml",
        })
    elif schema == "classification":
        steps.append({
            "action": "run_report",
            "reason": "Classification dataset detected — generate a quality report.",
            "command": "verifily report --dataset <path> --schema classification",
        })
    elif schema == "chat":
        steps.append({
            "action": "run_pipeline",
            "reason": "Chat dataset detected — run pipeline with chat schema.",
            "command": "verifily pipeline --config verifily.yaml",
        })
    else:
        steps.append({
            "action": "manual_review",
            "reason": "Schema could not be detected. Manual review recommended.",
        })

    # Step: dedup if high rate
    if dup_rate > 0.05:
        steps.append({
            "action": "deduplicate",
            "reason": f"Duplicate rate is {dup_rate:.1%} — consider deduplication before training.",
        })

    # Step: PII scrubbing
    if pii.get("rows_with_pii", 0) > 0:
        steps.append({
            "action": "scrub_pii",
            "reason": f"PII detected in {pii['rows_with_pii']} rows — scrub before training.",
        })

    # Step: per-bucket actions
    for bucket_name, path in export.bucket_paths.items():
        rows = export.rows_per_bucket.get(bucket_name, 0)
        steps.append({
            "action": "review_bucket",
            "bucket": bucket_name,
            "rows": rows,
            "path": path,
        })

    return {
        "suggested_schema": schema,
        "total_buckets": len(export.bucket_paths),
        "total_rows_exported": export.total_rows_written,
        "steps": steps,
    }


def write_bucket_exports(
    classification: Dict[str, Any],
    rows: List[Dict[str, Any]],
    out_dir: str,
    *,
    min_bucket_rows: int = 1,
) -> ExportSummary:
    """Write per-bucket JSONL files and suggested_next_steps.json.

    Args:
        classification: Result dict from classify_dataset().
        rows: Original dataset rows (full, not just scanned).
        out_dir: Root output directory. Files land in <out_dir>/classification/buckets/.
        min_bucket_rows: Skip buckets with fewer rows than this.

    Returns:
        ExportSummary with paths, counts, and schemas.
    """
    base = Path(out_dir) / "classification"
    buckets_dir = base / "buckets"
    buckets_dir.mkdir(parents=True, exist_ok=True)

    # Assign each row to its bucket
    bucket_rows: Dict[str, List[Dict[str, Any]]] = {}
    bucket_schemas: Dict[str, str] = {}
    for row in rows:
        schema = _detect_row_schema(row)
        cat = _infer_category(row)
        name = _bucket_filename(schema, cat)
        bucket_rows.setdefault(name, []).append(row)
        bucket_schemas[name] = schema

    summary = ExportSummary()

    for name, brows in sorted(bucket_rows.items()):
        if len(brows) < min_bucket_rows:
            continue
        path = buckets_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for row in brows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        summary.bucket_paths[name] = str(path)
        summary.rows_per_bucket[name] = len(brows)
        summary.schemas_per_bucket[name] = bucket_schemas[name]
        summary.total_rows_written += len(brows)

    # Write suggested_next_steps.json
    next_steps = _suggested_next_steps(classification, summary)
    next_steps_path = base / "suggested_next_steps.json"
    with open(next_steps_path, "w", encoding="utf-8") as f:
        json.dump(next_steps, f, indent=2, ensure_ascii=False)
    summary.suggested_next_steps_path = str(next_steps_path)

    return summary
