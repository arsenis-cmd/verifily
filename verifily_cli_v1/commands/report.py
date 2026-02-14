"""verifily report — dataset statistics and PII scan."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union

from rich.console import Console
from rich.table import Table

from verifily_cli_v1.core.io import read_jsonl, write_json

console = Console(stderr=True)

# ── PII patterns ────────────────────────────────────────────────

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}


# ── Public API ──────────────────────────────────────────────────

def dataset_report(
    dataset_path: Union[str, Path],
    schema: str = "sft",
) -> Dict[str, Any]:
    """Generate a report for a JSONL dataset.

    Returns:
        {
            "path": str,
            "row_count": int,
            "schema": str,
            "field_stats": {field: {"present": int, "empty": int, "avg_len": float}},
            "tag_distribution": {tag_key: {tag_value: count}},
            "pii_scan": {type: {"count": int, "rows": [int]}},
            "pii_clean": bool,
        }
    """
    rows = read_jsonl(dataset_path)
    row_count = len(rows)

    # Field stats
    field_counts: Dict[str, Dict[str, Any]] = {}
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())
    for field in sorted(all_fields):
        if field == "tags":
            continue
        present = sum(1 for r in rows if field in r)
        empty = sum(1 for r in rows if field in r and not str(r[field]).strip())
        lengths = [len(str(r[field])) for r in rows if field in r and str(r[field]).strip()]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        field_counts[field] = {"present": present, "empty": empty, "avg_len": round(avg_len, 1)}

    # Tag distribution
    tag_dist: Dict[str, Counter] = {}
    for row in rows:
        tags = row.get("tags", {})
        if isinstance(tags, dict):
            for k, v in tags.items():
                if k not in tag_dist:
                    tag_dist[k] = Counter()
                tag_dist[k][str(v)] += 1
    tag_distribution = {k: dict(v.most_common()) for k, v in tag_dist.items()}

    # PII scan
    pii_results: Dict[str, Dict[str, Any]] = {}
    for pii_type, pattern in PII_PATTERNS.items():
        hits: List[int] = []
        for i, row in enumerate(rows):
            text = " ".join(str(v) for k, v in row.items() if k != "tags")
            if pattern.search(text):
                hits.append(i)
        pii_results[pii_type] = {"count": len(hits), "rows": hits[:10]}

    pii_clean = all(r["count"] == 0 for r in pii_results.values())
    total_pii = sum(r["count"] for r in pii_results.values())

    return {
        "path": str(dataset_path),
        "row_count": row_count,
        "schema": schema,
        "field_stats": field_counts,
        "tag_distribution": tag_distribution,
        "pii_scan": pii_results,
        "pii_total_hits": total_pii,
        "pii_clean": pii_clean,
    }


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    dataset: str,
    schema: str = "sft",
    output: str | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate and display dataset report. Returns report dict."""
    report = dataset_report(dataset, schema=schema)

    console.print(f"\n[bold]Dataset Report:[/bold] {dataset}\n")

    # Summary
    console.print(f"  Rows: [cyan]{report['row_count']}[/cyan]")
    console.print(f"  Schema: [cyan]{report['schema']}[/cyan]")
    pii_style = "green" if report["pii_clean"] else "red bold"
    pii_label = "CLEAN" if report["pii_clean"] else f"{report['pii_total_hits']} HITS"
    console.print(f"  PII: [{pii_style}]{pii_label}[/{pii_style}]")

    # Field stats table
    if report["field_stats"]:
        tbl = Table(title="Field Statistics", show_lines=False)
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Present", justify="right")
        tbl.add_column("Empty", justify="right")
        tbl.add_column("Avg Length", justify="right")
        for field, stats in report["field_stats"].items():
            tbl.add_row(field, str(stats["present"]), str(stats["empty"]), str(stats["avg_len"]))
        console.print()
        console.print(tbl)

    # Tag distribution
    if report["tag_distribution"]:
        console.print("\n[bold]Tag Distribution:[/bold]")
        for tag_key, dist in report["tag_distribution"].items():
            parts = [f"{v}: {c}" for v, c in dist.items()]
            console.print(f"  {tag_key}: {', '.join(parts)}")

    # PII details
    if not report["pii_clean"]:
        console.print("\n[bold red]PII Detections:[/bold red]")
        for pii_type, data in report["pii_scan"].items():
            if data["count"] > 0:
                console.print(f"  {pii_type}: {data['count']} rows")

    console.print()

    # Write output if requested
    if output:
        write_json(output, report)
        console.print(f"Report written to: {output}\n")

    return report
