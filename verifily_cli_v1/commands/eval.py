"""verifily eval — evaluate a training run."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.io import read_json
from verifily_cli_v1.core.subprocess import check_module_available, run_module

console = Console()


def _display_eval_results(run_dir: Path, verbose: bool = False) -> bool:
    """Try to load and display cached eval results. Returns True if found."""
    # Check common locations for eval results
    candidates = [
        run_dir / "eval" / "eval_results.json",
        run_dir / "eval_results.json",
    ]

    results = None
    for path in candidates:
        if path.exists():
            results = read_json(path)
            break

    if results is None:
        return False

    console.print(
        Panel("[bold]Evaluation Results[/bold]", border_style="green")
    )

    # Display aggregate metrics — check "overall" first (verifily_train format),
    # then "aggregate", then fall back to top-level numeric keys.
    aggregate = results.get("overall", results.get("aggregate", None))
    if aggregate is None:
        # Fall back: collect top-level numeric values
        aggregate = {k: v for k, v in results.items() if isinstance(v, (int, float))}

    if isinstance(aggregate, dict) and aggregate:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan", min_width=20)
        table.add_column("Value", justify="right", min_width=12)

        for key, value in sorted(aggregate.items()):
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            elif isinstance(value, int):
                table.add_row(key, str(value))

        console.print(table)

    # Display per-slice if available
    slices = results.get("slices", results.get("per_slice", {}))
    if slices and verbose:
        console.print("\n[bold]Per-Slice Breakdown[/bold]")
        for slice_key, slice_val in slices.items():
            # Handle nested structure: slices.source.squad_human_test
            if isinstance(slice_val, dict):
                for sub_name, metrics in slice_val.items():
                    console.print(f"\n  [cyan]{slice_key}={sub_name}[/cyan]")
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            if isinstance(v, float):
                                console.print(f"    {k}: {v:.4f}")
                            else:
                                console.print(f"    {k}: {v}")

    return True


def run(
    run_dir: str,
    slice_by: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Evaluate a training run and display results."""
    rd = Path(run_dir)

    if not rd.exists():
        console.print(f"[red bold]Error:[/red bold] Run directory not found: {rd}")
        raise SystemExit(1)

    console.print(
        Panel(
            f"[bold]Verifily Eval[/bold]\nRun: {rd}",
            border_style="blue",
        )
    )

    # ── Try cached results first ──
    if _display_eval_results(rd, verbose):
        console.print(f"\n[dim]Next:[/dim] verifily compare --runs {rd}")
        return

    # ── No cached results: try running eval via verifily_train ──
    if not check_module_available("verifily_train"):
        console.print(
            "[red bold]Error:[/red bold] No eval_results.json found and verifily_train is not installed."
        )
        console.print(
            "\n  Expected file: [cyan]{}/eval/eval_results.json[/cyan]".format(rd)
        )
        console.print("  Or install verifily_train to run evaluation live.")
        raise SystemExit(1)

    console.print("  Running evaluation...\n")

    args = ["eval", "--run", str(rd.resolve())]
    if slice_by:
        args.extend(["--slice-by", slice_by])

    result = run_module("verifily_train", args, capture=False)

    if not result.ok:
        console.print(f"\n[red bold]Evaluation failed[/red bold] (exit code {result.returncode})")
        if result.stderr and verbose:
            console.print(f"[dim]{result.stderr}[/dim]")
        raise SystemExit(1)

    # Try to display results that were just generated
    _display_eval_results(rd, verbose)

    console.print(f"\n[dim]Next:[/dim] verifily compare --runs {rd}")
