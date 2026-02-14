"""Verifily Train CLI — the main entry point."""

import logging
import sys
from typing import Optional

import click

from verifily_train import __version__


def _count_gpus() -> int:
    """Count available CUDA GPUs."""
    try:
        import torch
        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
        stream=sys.stderr,
    )


# ======================================================================
# Root group
# ======================================================================

@click.group()
@click.version_option(version=__version__, prog_name="verifily")
def cli():
    """Verifily Train — dataset-aware fine-tuning in one command."""
    pass


# ======================================================================
# verifily train
# ======================================================================

@cli.command()
@click.option("--config", "config_path", default="train.yaml", help="Path to training YAML config.")
@click.option("--dataset", default=None, help="Verifily dataset version ref (e.g. ds_abc@v3).")
@click.option("--base-model", default=None, help="HuggingFace model ID (overrides config).")
@click.option("--task", default=None, type=click.Choice(["sft", "classification"]), help="Task type.")
@click.option("--output-dir", default=None, help="Run artifact output directory.")
@click.option("--device", default=None, help="auto | cuda | mps | cpu.")
@click.option("--mode", default=None, type=click.Choice(["local", "managed"]), help="Compute mode.")
@click.option("--epochs", default=None, type=int, help="Override num_epochs.")
@click.option("--batch-size", default=None, type=int, help="Override batch_size.")
@click.option("--lr", default=None, type=float, help="Override learning_rate.")
@click.option("--lora-r", default=None, type=int, help="Override LoRA rank.")
@click.option("--quantization", default=None, type=click.Choice(["none", "4bit", "8bit"]))
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--name", default=None, help="Human-readable run name.")
@click.option("--distributed", is_flag=True, help="Enable multi-GPU distributed training via Accelerate.")
@click.option("--num-gpus", default=None, type=int, help="Number of GPUs for distributed training.")
@click.option("--dry-run", is_flag=True, help="Validate config, create run dir, but do not train.")
@click.option("--verbose", is_flag=True, help="Enable DEBUG logging.")
def train(config_path, dataset, base_model, task, output_dir, device, mode,
          epochs, batch_size, lr, lora_r, quantization, seed, name,
          distributed, num_gpus, dry_run, verbose):
    """Run a training job."""
    _setup_logging(verbose)

    if mode == "managed":
        click.echo("Managed mode is not yet available. Coming soon.")
        sys.exit(10)

    from verifily_train.config import TrainConfig
    from verifily_train.trainer import train as run_train

    cfg = TrainConfig.from_yaml(config_path)
    cfg = cfg.merge_cli_overrides(
        dataset=dataset, base_model=base_model, task=task,
        output_dir=output_dir, device=device, epochs=epochs,
        batch_size=batch_size, lr=lr, lora_r=lora_r,
        quantization=quantization, seed=seed, name=name,
    )

    if distributed:
        cfg.compute.distributed = True
        if num_gpus:
            cfg.compute.num_gpus = num_gpus

    # If distributed and not already inside an accelerate launcher, relaunch
    import os
    if cfg.compute.distributed and "ACCELERATE_LAUNCHED" not in os.environ:
        import subprocess
        ngpus = cfg.compute.num_gpus or "all"
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(ngpus) if ngpus != "all" else str(_count_gpus()),
            "--mixed_precision", "fp16" if cfg.compute.fp16 else ("bf16" if cfg.compute.bf16 else "no"),
            "-m", "verifily_train.cli", "train",
            "--config", config_path,
        ]
        # Forward CLI overrides so the relaunched process uses the same settings
        if dataset:
            cmd.extend(["--dataset", dataset])
        if base_model:
            cmd.extend(["--base-model", base_model])
        if task:
            cmd.extend(["--task", task])
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        if device:
            cmd.extend(["--device", device])
        if epochs is not None:
            cmd.extend(["--epochs", str(epochs)])
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])
        if lr is not None:
            cmd.extend(["--lr", str(lr)])
        if lora_r is not None:
            cmd.extend(["--lora-r", str(lora_r)])
        if quantization:
            cmd.extend(["--quantization", quantization])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if name:
            cmd.extend(["--name", name])
        if dry_run:
            cmd.append("--dry-run")
        if verbose:
            cmd.append("--verbose")
        os.environ["ACCELERATE_LAUNCHED"] = "1"
        click.echo(f"Launching distributed training: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    run = run_train(cfg, dry_run=dry_run)

    click.echo("")
    if dry_run:
        click.echo(f"Dry run complete. Artifacts: {run.artifact_path}")
    else:
        click.echo(f"Run completed: {run.run_id}")
        click.echo(f"  Duration:    {run.duration_seconds}s")
        click.echo(f"  Train loss:  {run.metrics.get('train_loss', 'N/A')}")
        click.echo(f"  Artifacts:   {run.artifact_path}")
        click.echo(f"  Repro hash:  {run.reproducibility_hash}")
        click.echo("")
        click.echo(f"Run `verifily eval --run {run.artifact_path}` for full evaluation.")


# ======================================================================
# verifily eval
# ======================================================================

@cli.command(name="eval")
@click.option("--run", "run_path", required=True, help="Path to run artifact directory.")
@click.option("--test-data", default=None, help="Path to test JSONL (overrides run config).")
@click.option("--metrics", default=None, help="Comma-separated metric names.")
@click.option("--slice-by", default=None, help="Comma-separated tag keys to slice by.")
@click.option("--hard-examples", default=50, type=int, help="Number of worst examples to output.")
@click.option("--device", default="auto", help="Device for inference.")
@click.option("--batch-size", default=16, type=int, help="Inference batch size.")
@click.option("--verbose", is_flag=True)
def eval_cmd(run_path, test_data, metrics, slice_by, hard_examples, device, batch_size, verbose):
    """Evaluate a trained model on a test set."""
    _setup_logging(verbose)

    import json
    from pathlib import Path
    from verifily_train.evaluator import EvalResult

    # Check for cached eval results
    cached_path = Path(run_path) / "eval" / "eval_results.json"
    if cached_path.exists() and not test_data:
        with open(cached_path) as f:
            data = json.load(f)
        result = EvalResult(**{k: v for k, v in data.items() if k in EvalResult.__dataclass_fields__})
    else:
        from verifily_train.evaluator import evaluate

        metric_list = metrics.split(",") if metrics else None
        slice_list = slice_by.split(",") if slice_by else None

        result = evaluate(
            run_path=run_path,
            test_data=test_data,
            metrics=metric_list,
            slice_by=slice_list,
            hard_examples_n=hard_examples,
            device=device,
            batch_size=batch_size,
        )

    click.echo("")
    click.echo(f"Evaluation: {result.run_id}")
    click.echo(f"  Test set: {result.num_examples} examples")
    click.echo("")
    click.echo("  Overall:")
    for k, v in result.overall.items():
        if isinstance(v, float):
            click.echo(f"    {k}: {v:.4f}")
        else:
            click.echo(f"    {k}: {v}")

    if result.slices:
        for tag_key, tag_vals in result.slices.items():
            click.echo(f"\n  Sliced by [{tag_key}]:")
            for tv, tv_metrics in tag_vals.items():
                if isinstance(tv_metrics, dict):
                    n = tv_metrics.get("n", "?")
                    detail = ", ".join(f"{k}={v:.4f}" for k, v in tv_metrics.items() if isinstance(v, float) and k != "n")
                    click.echo(f"    {tag_key}={tv} (n={n}): {detail}")

    hard_path = Path(run_path) / "eval" / "hard_examples.jsonl"
    if hard_path.exists():
        click.echo(f"\n  Hard examples: {hard_path}")

    click.echo(f"\n  Full results: {run_path}/eval/eval_results.json")
    click.echo("  Note: Slice metrics show correlation, not causation.")


# ======================================================================
# verifily compare
# ======================================================================

@cli.command(name="compare")
@click.option("--runs", required=True, help="Comma-separated run paths.")
@click.option("--metric", default="f1", help="Metric to compare (default: f1).")
@click.option("--slice-by", default=None, help="Tag key for slice breakdown.")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "csv"]))
@click.option("--output", default=None, help="Write comparison to file (default: stdout).")
def compare_cmd(runs, metric, slice_by, fmt, output):
    """Compare metrics across multiple runs."""
    from verifily_train.compare import compare, format_comparison

    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    result = compare(run_list, metric=metric, slice_by=slice_by)

    if fmt == "csv":
        import csv, io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["run", metric, "delta"])
        for run_id in result.runs:
            writer.writerow([run_id, result.overall[run_id], result.deltas[run_id]])
        text = buf.getvalue()
    else:
        text = format_comparison(result, fmt=fmt)

    if output:
        with open(output, "w") as f:
            f.write(text)
        click.echo(f"Comparison written to {output}")
    else:
        click.echo(text)


# ======================================================================
# verifily reproduce
# ======================================================================

@cli.command()
@click.option("--run", "run_path", required=True, help="Path to run artifact directory.")
@click.option("--verify-only", is_flag=True, help="Check hashes without retraining.")
@click.option("--verbose", is_flag=True)
def reproduce(run_path, verify_only, verbose):
    """Verify or reproduce a training run."""
    _setup_logging(verbose)

    if not verify_only:
        click.echo("Full reproduction (retraining) is not yet implemented.")
        click.echo("Use --verify-only to check reproducibility hashes.")
        sys.exit(1)

    from verifily_train.reproduce import verify, format_verify_result

    result = verify(run_path)
    click.echo(format_verify_result(result))


# ======================================================================
# verifily dashboard
# ======================================================================

@cli.command()
@click.option("--runs-dir", default="runs/", help="Directory containing run artifacts.")
@click.option("--port", default=7860, type=int, help="Port for the web UI.")
@click.option("--share", is_flag=True, help="Create a public Gradio link.")
def dashboard(runs_dir, port, share):
    """Launch the Verifily Train web dashboard."""
    from verifily_train.web import launch
    launch(runs_dir=runs_dir, port=port, share=share)


# ======================================================================
# verifily tune
# ======================================================================

@cli.command()
@click.option("--config", "config_path", required=True, help="Base training YAML config.")
@click.option("--param", multiple=True, help="Param search range: 'training.learning_rate=1e-5,2e-4,5e-4'")
@click.option("--strategy", default="grid", type=click.Choice(["grid", "random"]), help="Search strategy.")
@click.option("--max-trials", default=None, type=int, help="Max trials for random search.")
@click.option("--metric", default="eval_loss", help="Metric to optimise (lower is better).")
@click.option("--verbose", is_flag=True)
def tune(config_path, param, strategy, max_trials, metric, verbose):
    """Auto-tune hyperparameters via grid or random search."""
    _setup_logging(verbose)

    from verifily_train.tuner import run_tuning

    # Parse param specs: "training.learning_rate=1e-5,2e-4" -> dict
    param_grid = {}
    for p in param:
        key, vals_str = p.split("=", 1)
        values = []
        for v in vals_str.split(","):
            v = v.strip()
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)
        param_grid[key.strip()] = values

    if not param_grid:
        click.echo("Error: provide at least one --param flag.")
        sys.exit(1)

    best = run_tuning(
        config_path=config_path,
        param_grid=param_grid,
        strategy=strategy,
        max_trials=max_trials,
        metric=metric,
    )

    click.echo(f"\nBest trial: {best['run_id']}")
    click.echo(f"  {metric}: {best['score']}")
    click.echo(f"  Params: {best['params']}")
    click.echo(f"  Artifacts: {best['artifact_path']}")


# ======================================================================
# Entry point
# ======================================================================

def main():
    cli()
