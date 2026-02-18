"""verifily report — dataset statistics and PII scan."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table

from verifily_cli_v1.core.io import read_jsonl, write_json
from verifily_cli_v1.core.pii import PII_PATTERNS, scan_dataset as pii_scan_dataset

console = Console(stderr=True)


# ── Public API ──────────────────────────────────────────────────

def dataset_report(
    dataset_path: Union[str, Path],
    schema: str = "sft",
    *,
    use_ner: bool = False,
    min_confidence: float = 0.0,
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

    # PII scan (using unified scanner)
    pii_result = pii_scan_dataset(rows, use_ner=use_ner, min_confidence=min_confidence)
    pii_results = pii_result["pii_scan"]
    pii_clean = pii_result["pii_clean"]
    total_pii = pii_result["pii_total_hits"]

    # Quality analysis
    from verifily_cli_v1.core.quality import analyze_quality
    quality = analyze_quality(rows, schema=schema)

    return {
        "path": str(dataset_path),
        "row_count": row_count,
        "schema": schema,
        "field_stats": field_counts,
        "tag_distribution": tag_distribution,
        "pii_scan": pii_results,
        "pii_total_hits": total_pii,
        "pii_clean": pii_clean,
        "quality": quality.to_dict(),
    }


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    dataset: str,
    schema: str = "sft",
    output: str | None = None,
    verbose: bool = False,
    use_ner: bool = False,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """Generate and display dataset report. Returns report dict."""
    report = dataset_report(dataset, schema=schema, use_ner=use_ner, min_confidence=min_confidence)

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

    # Quality analysis
    if "quality" in report:
        q = report["quality"]
        score = q.get("quality_score", 0)
        score_style = "green" if score >= 80 else "yellow" if score >= 50 else "red"
        console.print(f"\n[bold]Quality Score:[/bold] [{score_style}]{score}/100[/{score_style}]")
        for issue in q.get("issues", []):
            icon = "!" if issue["severity"] == "warning" else "X" if issue["severity"] == "error" else "*"
            console.print(f"  {icon} {issue['description']}")
        stats = q.get("stats", {})
        if stats.get("type_token_ratio"):
            console.print(f"  Vocabulary: {stats['unique_tokens']:,} unique / {stats['total_tokens']:,} total (TTR: {stats['type_token_ratio']:.3f})")

    console.print()

    # Write output if requested
    if output:
        write_json(output, report)
        console.print(f"Report written to: {output}\n")

    return report
