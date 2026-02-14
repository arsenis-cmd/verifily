"""verifily train — launch training via verifily_train subprocess."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.io import read_yaml, write_json
from verifily_cli_v1.core.subprocess import check_module_available, run_module

console = Console()


def _auto_run_name() -> str:
    """Generate an automatic run directory name."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}"


def run(
    config: str,
    run_dir: Optional[str] = None,
    plan: bool = False,
    verbose: bool = False,
) -> None:
    """Launch a training run by delegating to verifily_train."""
    config_path = Path(config)

    if not config_path.exists():
        console.print(f"[red bold]Error:[/red bold] Config not found: {config_path}")
        console.print("[dim]Hint:[/dim] Run 'verifily init' to generate a config.")
        raise SystemExit(1)

    # Parse config for display
    try:
        cfg = read_yaml(config_path)
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] Invalid config: {e}")
        raise SystemExit(1)

    # Determine run directory
    if run_dir:
        rd = Path(run_dir)
    else:
        base = cfg.get("output", {}).get("dir", "runs")
        rd = Path(base) / _auto_run_name()

    # Extract config values — support both wizard schema and verifily_train schema
    model_name = (
        cfg.get("base_model")
        or cfg.get("model", {}).get("name")
        or "N/A"
    )
    task = (
        cfg.get("task")
        or cfg.get("model", {}).get("task")
        or "N/A"
    )
    dataset = (
        cfg.get("data_paths", {}).get("train")
        or cfg.get("data", {}).get("train")
        or "N/A"
    )
    training = cfg.get("training", {})
    epochs = training.get("num_epochs") or training.get("epochs") or "N/A"
    lora_enabled = cfg.get("lora", {}).get("enabled", training.get("use_lora", False))

    console.print(
        Panel(
            "[bold]Verifily Train[/bold]\n"
            f"Config:  {config_path}\n"
            f"Run dir: {rd}\n"
            f"Model:   {model_name}\n"
            f"Task:    {task}",
            border_style="blue",
        )
    )

    # ── Plan mode ──
    if plan:
        console.print("\n[yellow bold]--plan mode:[/yellow bold] dry run, no training launched\n")

        plan_table = Table(title="Training Plan", show_header=True, header_style="bold")
        plan_table.add_column("Setting", style="cyan")
        plan_table.add_column("Value")

        plan_table.add_row("Model", str(model_name))
        plan_table.add_row("Dataset", str(dataset))
        plan_table.add_row("Epochs", str(epochs))
        plan_table.add_row("Batch size", str(training.get("batch_size", "N/A")))
        plan_table.add_row("Learning rate", str(training.get("learning_rate", "N/A")))
        plan_table.add_row("LoRA", str(lora_enabled))
        plan_table.add_row("Run dir", str(rd))
        console.print(plan_table)

        console.print(
            f"\n[dim]Next:[/dim] verifily train --config {config_path} --run-dir {rd}"
        )
        return

    # ── Check verifily_train is available ──
    if not check_module_available("verifily_train"):
        console.print(
            "[red bold]Error:[/red bold] verifily_train is not installed.\n"
        )
        console.print("  Install it with:")
        console.print("  [cyan]pip install -e path/to/verifily_train[/cyan]")
        console.print(
            "\n  Or install from the demo directory:"
        )
        console.print(
            "  [cyan]cd verifily_train && pip install -e .[/cyan]"
        )
        raise SystemExit(1)

    # ── Launch training ──
    console.print("  Launching training...\n")

    args = ["train", "--config", str(config_path.resolve())]
    if run_dir:
        args.extend(["--run-dir", str(rd.resolve())])

    result = run_module("verifily_train", args, capture=False)

    if not result.ok:
        console.print(f"\n[red bold]Training failed[/red bold] (exit code {result.returncode})")
        if result.stderr and verbose:
            console.print(f"[dim]{result.stderr}[/dim]")
        raise SystemExit(1)

    console.print(f"\n[green bold]Training complete.[/green bold]")
    console.print(f"  Run artifacts: {rd}")
    console.print(f"\n[dim]Next:[/dim] verifily eval --run {rd}")
