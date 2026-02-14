"""verifily history — regression tracking across runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.table import Table

from verifily_cli_v1.core.io import read_json, write_json

console = Console(stderr=True)


# ── Public API ──────────────────────────────────────────────────

def load_run_metric(run_dir: Union[str, Path], metric: str = "f1") -> Optional[float]:
    """Extract a metric value from a run's eval results."""
    d = Path(run_dir)
    for candidate in ["eval/eval_results.json", "eval_results.json"]:
        p = d / candidate
        if p.exists():
            data = read_json(p)
            # Try overall.metric, then aggregate.metric, then top-level
            for section in ["overall", "aggregate"]:
                if isinstance(data.get(section), dict):
                    val = data[section].get(metric)
                    if val is not None:
                        return float(val)
            val = data.get(metric)
            if val is not None:
                return float(val)
    return None


def analyze_history(
    runs_dir: Union[str, Path],
    metric: str = "f1",
    regression_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Analyze metric history across sorted run directories.

    Args:
        runs_dir: directory containing run subdirectories
        metric: metric name to track
        regression_threshold: minimum drop to flag as regression

    Returns:
        {
            "runs": [{"name": str, "metric": float|None, "delta": float|None, "status": str}],
            "baseline": float|None,
            "best": float|None,
            "worst": float|None,
            "regressions": [{"run": str, "from": float, "to": float, "delta": float}],
            "has_regression": bool,
            "exit_code": 0 | 2,
        }
    """
    d = Path(runs_dir)
    if not d.is_dir():
        return {"runs": [], "baseline": None, "best": None, "worst": None,
                "regressions": [], "has_regression": False, "exit_code": 0}

    # Discover run directories (sorted by name for deterministic ordering)
    run_dirs = sorted([p for p in d.iterdir() if p.is_dir()])

    runs: List[Dict[str, Any]] = []
    values: List[float] = []
    regressions: List[Dict[str, Any]] = []

    prev_value: Optional[float] = None

    for run_path in run_dirs:
        name = run_path.name
        value = load_run_metric(run_path, metric)

        entry: Dict[str, Any] = {"name": name, "metric": value, "delta": None, "status": "OK"}

        if value is not None:
            values.append(value)
            if prev_value is not None:
                delta = value - prev_value
                entry["delta"] = round(delta, 4)
                if delta < -regression_threshold:
                    entry["status"] = "REGRESSION"
                    regressions.append({
                        "run": name,
                        "from": prev_value,
                        "to": value,
                        "delta": round(delta, 4),
                    })
                elif delta > regression_threshold:
                    entry["status"] = "IMPROVED"
            prev_value = value
        else:
            entry["status"] = "NO_DATA"

        runs.append(entry)

    baseline = values[0] if values else None
    best = max(values) if values else None
    worst = min(values) if values else None
    has_regression = len(regressions) > 0

    return {
        "runs": runs,
        "metric": metric,
        "regression_threshold": regression_threshold,
        "baseline": baseline,
        "best": best,
        "worst": worst,
        "regressions": regressions,
        "has_regression": has_regression,
        "exit_code": 2 if has_regression else 0,
    }


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    runs_dir: str,
    metric: str = "f1",
    threshold: float = 0.02,
    output: str | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run history analysis. Returns result dict."""
    result = analyze_history(runs_dir, metric=metric, regression_threshold=threshold)

    console.print(f"\n[bold]Run History:[/bold] {runs_dir}")
    console.print(f"  Metric: {metric}  |  Regression threshold: {threshold}\n")

    if not result["runs"]:
        console.print("[yellow]No runs found.[/yellow]\n")
        return result

    tbl = Table(show_lines=False)
    tbl.add_column("Run", style="cyan")
    tbl.add_column("Metric", justify="right")
    tbl.add_column("Delta", justify="right")
    tbl.add_column("Status")

    for entry in result["runs"]:
        val = f"{entry['metric']:.4f}" if entry["metric"] is not None else "—"
        delta = f"{entry['delta']:+.4f}" if entry["delta"] is not None else "—"
        status = entry["status"]
        if status == "REGRESSION":
            status_str = "[red bold]REGRESSION[/red bold]"
            delta = f"[red]{delta}[/red]"
        elif status == "IMPROVED":
            status_str = "[green]IMPROVED[/green]"
            delta = f"[green]{delta}[/green]"
        elif status == "NO_DATA":
            status_str = "[yellow]NO_DATA[/yellow]"
        else:
            status_str = "[dim]OK[/dim]"
        tbl.add_row(entry["name"], val, delta, status_str)

    console.print(tbl)

    if result["regressions"]:
        console.print(f"\n[red bold]REGRESSIONS DETECTED: {len(result['regressions'])}[/red bold]")
        for reg in result["regressions"]:
            console.print(
                f"  {reg['run']}: {metric} dropped {reg['from']:.4f} → {reg['to']:.4f} "
                f"(Δ = {reg['delta']:+.4f})"
            )
    else:
        console.print(f"\n[green bold]No regressions detected.[/green bold]")

    console.print(f"\n  Baseline: {result['baseline']:.4f}" if result["baseline"] else "")
    console.print(f"  Best:     {result['best']:.4f}" if result["best"] else "")
    console.print(f"  Worst:    {result['worst']:.4f}" if result["worst"] else "")
    console.print()

    if output:
        write_json(output, result)
        console.print(f"History written to: {output}\n")

    return result
