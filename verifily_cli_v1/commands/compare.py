"""verifily compare — compare metrics across training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.io import read_json, read_yaml

console = Console()


def _load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load eval results from a run directory."""
    candidates = [
        run_dir / "eval" / "eval_results.json",
        run_dir / "eval_results.json",
    ]

    for path in candidates:
        if path.exists():
            return read_json(path)

    return {}


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    """Load config from a run directory."""
    candidates = [
        run_dir / "config.yaml",
        run_dir / "train_config.yaml",
        run_dir / "transform_config.yaml",
    ]

    for path in candidates:
        if path.exists():
            return read_yaml(path)

    return {}


def _extract_metric(results: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract a metric value from eval results, searching nested structures."""
    # Direct lookup
    if metric in results:
        val = results[metric]
        if isinstance(val, (int, float)):
            return float(val)

    # Check "overall" (verifily_train format) then "aggregate"
    for key in ("overall", "aggregate"):
        container = results.get(key, {})
        if isinstance(container, dict) and metric in container:
            val = container[metric]
            if isinstance(val, (int, float)):
                return float(val)

    # Check per_slice / slices — handle nested structure (slices.source.name.metric)
    slices = results.get("slices", results.get("per_slice", {}))
    if isinstance(slices, dict):
        vals: List[float] = []
        for slice_val in slices.values():
            if isinstance(slice_val, dict):
                # Direct: slices.slice_name.metric
                if metric in slice_val:
                    v = slice_val[metric]
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                else:
                    # Nested: slices.tag_key.tag_value.metric
                    for inner in slice_val.values():
                        if isinstance(inner, dict) and metric in inner:
                            v = inner[metric]
                            if isinstance(v, (int, float)):
                                vals.append(float(v))
        if vals:
            return sum(vals) / len(vals)

    return None


def run(
    runs: str,
    metric: str = "f1",
    verbose: bool = False,
) -> None:
    """Compare metrics across multiple training runs."""
    run_paths = [Path(r.strip()) for r in runs.split(",")]

    # Validate run directories exist
    for rd in run_paths:
        if not rd.exists():
            console.print(f"[red bold]Error:[/red bold] Run directory not found: {rd}")
            raise SystemExit(1)

    console.print(
        Panel(
            f"[bold]Verifily Compare[/bold]\n"
            f"Runs:   {len(run_paths)}\n"
            f"Metric: {metric}",
            border_style="blue",
        )
    )

    # ── Load all results ──
    run_data: List[Dict[str, Any]] = []
    all_metrics: set = set()

    for rd in run_paths:
        results = _load_run_metrics(rd)
        config = _load_run_config(rd)

        if not results:
            console.print(f"  [yellow]Warning:[/yellow] No eval results in {rd}")

        # Collect all available metric names
        aggregate = results.get("overall", results.get("aggregate", {}))
        if isinstance(aggregate, dict):
            for k, v in aggregate.items():
                if isinstance(v, (int, float)):
                    all_metrics.add(k)

        run_data.append({
            "path": rd,
            "name": rd.name,
            "results": results,
            "config": config,
        })

    # ── Build comparison table ──
    table = Table(
        title="Run Comparison",
        show_header=True,
        header_style="bold",
        title_style="bold",
    )
    table.add_column("Run", style="cyan", min_width=20)

    # Determine which metrics to show
    metrics_to_show = [metric]
    # Add common extras
    common_extras = ["exact_match", "f1", "accuracy", "perplexity"]
    for m in common_extras:
        if m in all_metrics and m not in metrics_to_show:
            metrics_to_show.append(m)

    for m in metrics_to_show:
        table.add_column(m, justify="right", min_width=12)

    # Find best value per metric for highlighting
    best_values: Dict[str, float] = {}
    for m in metrics_to_show:
        vals = []
        for rd_info in run_data:
            v = _extract_metric(rd_info["results"], m)
            if v is not None:
                # For perplexity, lower is better
                vals.append(v)
        if vals:
            if m == "perplexity":
                best_values[m] = min(vals)
            else:
                best_values[m] = max(vals)

    # Populate rows
    for rd_info in run_data:
        row = [rd_info["name"]]
        for m in metrics_to_show:
            val = _extract_metric(rd_info["results"], m)
            if val is None:
                row.append("[dim]—[/dim]")
            else:
                formatted = f"{val:.4f}"
                if m in best_values and val == best_values[m] and len(run_data) > 1:
                    formatted = f"[green bold]{val:.4f}[/green bold]"
                row.append(formatted)
        table.add_row(*row)

    console.print(table)

    # ── Config diff (if verbose) ──
    if verbose and len(run_data) > 1:
        console.print("\n[bold]Config Differences[/bold]")

        def _flatten(d: dict, prefix: str = "") -> Dict[str, Any]:
            items: Dict[str, Any] = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(_flatten(v, key))
                else:
                    items[key] = v
            return items

        flat_configs = [_flatten(rd_info["config"]) for rd_info in run_data]
        all_keys = set()
        for fc in flat_configs:
            all_keys.update(fc.keys())

        diff_table = Table(show_header=True, header_style="bold")
        diff_table.add_column("Setting", style="cyan")
        for rd_info in run_data:
            diff_table.add_column(rd_info["name"])

        for key in sorted(all_keys):
            values = [str(fc.get(key, "—")) for fc in flat_configs]
            if len(set(values)) > 1:  # Only show differing values
                diff_table.add_row(key, *values)

        if diff_table.row_count > 0:
            console.print(diff_table)
        else:
            console.print("  [dim]No config differences found.[/dim]")

    console.print(
        f"\n[dim]Next:[/dim] verifily reproduce --run {run_paths[0]}"
    )
