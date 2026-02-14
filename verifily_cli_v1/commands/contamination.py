"""verifily contamination — detect train/eval overlap."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

from rich.console import Console
from rich.table import Table

from verifily_cli_v1.core.hashing import sha256_string
from verifily_cli_v1.core.io import read_jsonl, write_json

console = Console(stderr=True)


# ── N-gram Jaccard ──────────────────────────────────────────────

def _ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _row_text(row: Dict[str, Any]) -> str:
    """Extract searchable text from a row (excludes tags)."""
    parts = []
    for key in ("instruction", "output", "input", "question", "answer", "context", "text"):
        if key in row and row[key]:
            parts.append(str(row[key]).strip())
    return " ".join(parts) if parts else json.dumps(row, sort_keys=True)


# ── Public API ──────────────────────────────────────────────────

def check_contamination(
    train_path: Union[str, Path],
    eval_path: Union[str, Path],
    *,
    exact_threshold: float = 0.0,
    near_threshold: float = 0.15,
    jaccard_cutoff: float = 0.70,
    ngram_size: int = 3,
) -> Dict[str, Any]:
    """Check for contamination between train and eval datasets.

    Args:
        train_path: path to training JSONL
        eval_path: path to eval JSONL
        exact_threshold: max allowed exact overlap fraction (0 = any exact match fails)
        near_threshold: max allowed near-duplicate fraction before WARN
        jaccard_cutoff: Jaccard similarity threshold for near-duplicate
        ngram_size: character n-gram size for Jaccard

    Returns:
        {
            "train_rows": int,
            "eval_rows": int,
            "exact_overlaps": int,
            "exact_overlap_fraction": float,
            "exact_overlap_indices": [(train_idx, eval_idx)],
            "near_duplicates": int,
            "near_duplicate_fraction": float,
            "near_duplicate_pairs": [(train_idx, eval_idx, similarity)],
            "status": "PASS" | "FAIL" | "WARN",
            "exit_code": 0 | 1 | 2,
            "reasons": [str],
        }
    """
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    # Build train hash index
    train_hashes: Dict[str, int] = {}
    train_texts: List[str] = []
    train_ngrams: List[Set[str]] = []

    for i, row in enumerate(train_rows):
        text = _row_text(row)
        h = sha256_string(text)
        train_hashes[h] = i
        train_texts.append(text)
        train_ngrams.append(_ngrams(text, ngram_size))

    # Check eval rows
    exact_overlaps: List[Tuple[int, int]] = []
    near_dupes: List[Tuple[int, int, float]] = []
    exact_eval_indices: Set[int] = set()

    for j, row in enumerate(eval_rows):
        text = _row_text(row)
        h = sha256_string(text)

        # Exact match
        if h in train_hashes:
            exact_overlaps.append((train_hashes[h], j))
            exact_eval_indices.add(j)
            continue

        # Near-duplicate (skip if already exact)
        eval_ng = _ngrams(text, ngram_size)
        best_sim = 0.0
        best_train_idx = -1
        for i, tng in enumerate(train_ngrams):
            sim = _jaccard(eval_ng, tng)
            if sim > best_sim:
                best_sim = sim
                best_train_idx = i
        if best_sim >= jaccard_cutoff:
            near_dupes.append((best_train_idx, j, round(best_sim, 4)))

    # Compute fractions
    eval_count = len(eval_rows)
    exact_frac = len(exact_overlaps) / eval_count if eval_count else 0.0
    near_frac = len(near_dupes) / eval_count if eval_count else 0.0

    # Determine status
    reasons: List[str] = []
    if exact_frac > exact_threshold:
        status = "FAIL"
        exit_code = 1
        reasons.append(
            f"Exact overlap: {len(exact_overlaps)}/{eval_count} eval rows "
            f"({exact_frac:.1%}) exceed threshold ({exact_threshold:.1%})"
        )
    elif near_frac > near_threshold:
        status = "WARN"
        exit_code = 2
        reasons.append(
            f"Near-duplicates: {len(near_dupes)}/{eval_count} eval rows "
            f"({near_frac:.1%}) exceed threshold ({near_threshold:.1%})"
        )
    else:
        status = "PASS"
        exit_code = 0
        reasons.append("No significant contamination detected")

    if exact_overlaps and exact_frac <= exact_threshold:
        reasons.append(
            f"Note: {len(exact_overlaps)} exact overlaps found but within threshold"
        )
    if near_dupes and near_frac <= near_threshold:
        reasons.append(
            f"Note: {len(near_dupes)} near-duplicates found but within threshold"
        )

    return {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_rows": len(train_rows),
        "eval_rows": eval_count,
        "exact_overlaps": len(exact_overlaps),
        "exact_overlap_fraction": round(exact_frac, 4),
        "exact_overlap_indices": exact_overlaps[:50],
        "near_duplicates": len(near_dupes),
        "near_duplicate_fraction": round(near_frac, 4),
        "near_duplicate_pairs": near_dupes[:50],
        "jaccard_cutoff": jaccard_cutoff,
        "status": status,
        "exit_code": exit_code,
        "reasons": reasons,
    }


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    train: str,
    eval_set: str,
    jaccard_cutoff: float = 0.70,
    output: str | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run contamination check. Returns result dict."""
    result = check_contamination(
        train, eval_set,
        jaccard_cutoff=jaccard_cutoff,
    )

    console.print(f"\n[bold]Contamination Check[/bold]")
    console.print(f"  Train: {train} ({result['train_rows']} rows)")
    console.print(f"  Eval:  {eval_set} ({result['eval_rows']} rows)")

    tbl = Table(show_lines=False)
    tbl.add_column("Check", style="cyan")
    tbl.add_column("Count", justify="right")
    tbl.add_column("Fraction", justify="right")
    tbl.add_row("Exact overlaps", str(result["exact_overlaps"]), f"{result['exact_overlap_fraction']:.1%}")
    tbl.add_row("Near duplicates", str(result["near_duplicates"]), f"{result['near_duplicate_fraction']:.1%}")
    console.print()
    console.print(tbl)

    if verbose and result["exact_overlap_indices"]:
        console.print("\n[dim]Exact overlap pairs (train_idx, eval_idx):[/dim]")
        for t, e in result["exact_overlap_indices"][:10]:
            console.print(f"  train[{t}] == eval[{e}]")

    if verbose and result["near_duplicate_pairs"]:
        console.print("\n[dim]Near-duplicate pairs (train_idx, eval_idx, similarity):[/dim]")
        for t, e, s in result["near_duplicate_pairs"][:10]:
            console.print(f"  train[{t}] ~ eval[{e}] (Jaccard={s:.3f})")

    status = result["status"]
    status_styles = {"PASS": "green bold", "FAIL": "red bold", "WARN": "yellow bold"}
    style = status_styles.get(status, "white")
    console.print(f"\n[{style}]CONTAMINATION: {status}[/{style}]")
    for reason in result["reasons"]:
        console.print(f"  {reason}")
    console.print()

    if output:
        write_json(output, result)
        console.print(f"Results written to: {output}\n")

    return result
