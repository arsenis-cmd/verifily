"""verifily init — interactive wizard to generate config YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from verifily_cli_v1.core.io import write_yaml

console = Console()


def _build_transform_config() -> dict:
    """Interactively build a transform config."""
    console.print("\n[bold cyan]Transform Configuration[/bold cyan]\n")

    input_path = Prompt.ask(
        "  Input data path", default="data/raw_input.jsonl"
    )
    output_dir = Prompt.ask("  Output directory", default="datasets/")
    dataset_name = Prompt.ask("  Dataset name", default="my_dataset_v1")
    task = Prompt.ask("  Task type", choices=["sft", "classification"], default="sft")
    schema = "sft" if task == "sft" else "classification"

    privacy = Prompt.ask(
        "  PII handling", choices=["redact", "hash", "none"], default="redact"
    )
    dedupe_exact = Confirm.ask("  Enable exact deduplication?", default=True)
    dedupe_fuzzy = Confirm.ask("  Enable fuzzy deduplication?", default=False)

    return {
        "input": {"path": input_path, "format": "jsonl"},
        "output": {"dir": output_dir, "name": dataset_name},
        "labeling": {"task": task, "strategy": "heuristic"},
        "synthetic": {"enabled": False},
        "dedupe": {"exact": dedupe_exact, "fuzzy": dedupe_fuzzy},
        "privacy": {"pii_removal": privacy != "none", "audit_log": False},
        "seed": 42,
    }


def _build_train_config() -> dict:
    """Interactively build a train config."""
    console.print("\n[bold cyan]Training Configuration[/bold cyan]\n")

    model_name = Prompt.ask(
        "  Base model", default="google/flan-t5-base"
    )
    dataset_path = Prompt.ask(
        "  Training dataset path", default="datasets/my_dataset_v1/dataset.jsonl"
    )
    task = Prompt.ask("  Task type", choices=["sft", "classification"], default="sft")
    epochs = Prompt.ask("  Number of epochs", default="3")
    batch_size = Prompt.ask("  Batch size", default="8")
    lr = Prompt.ask("  Learning rate", default="2e-4")
    use_lora = Confirm.ask("  Use LoRA (parameter-efficient)?", default=True)
    output_dir = Prompt.ask("  Run output directory", default="runs/")

    config: dict = {
        "model": {"name": model_name, "task": task},
        "data": {"train": dataset_path},
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "use_lora": use_lora,
        },
        "output": {"dir": output_dir},
        "seed": 42,
    }

    if use_lora:
        config["lora"] = {"r": 16, "alpha": 32, "dropout": 0.05}

    return config


def run(
    out: Optional[str] = None,
) -> None:
    """Interactive wizard to generate a Verifily config YAML."""
    console.print(
        Panel(
            "[bold]Verifily Config Wizard[/bold]\n"
            "Generate a transform.yaml or train.yaml in seconds.",
            border_style="blue",
        )
    )

    config_type = Prompt.ask(
        "\n  What config do you need?",
        choices=["transform", "train"],
        default="transform",
    )

    if config_type == "transform":
        config = _build_transform_config()
        default_name = "transform.yaml"
    else:
        config = _build_train_config()
        default_name = "train.yaml"

    if out:
        out_path = Path(out)
    else:
        out_path = Path.cwd() / default_name

    write_yaml(out_path, config)

    console.print(f"\n[green bold]Created:[/green bold] {out_path}")
    if config_type == "transform":
        console.print(
            f"[dim]Next:[/dim] verifily transform --config {out_path}"
        )
    else:
        console.print(
            f"[dim]Next:[/dim] verifily train --config {out_path}"
        )
