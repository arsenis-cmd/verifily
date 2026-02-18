"""Verifily CLI — Typer app with all subcommands."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from verifily_cli_v1 import __version__

console = Console(stderr=True)


# ── License gating ────────────────────────────────────────────
def _require_license(command_name: str) -> None:
    """Check license for a paid command.  Raises SystemExit on failure."""
    from verifily_cli_v1.core.licensing import check_license, LicenseError, Tier
    try:
        info = check_license(command_name)
        if info.is_trial:
            console.print(
                f"[dim]Trial active — {info.days_remaining} days remaining.  "
                f"Upgrade at https://verifily.io/pricing[/dim]"
            )
    except LicenseError as exc:
        console.print(f"\n[red bold]License required[/red bold]\n\n{exc}")
        raise SystemExit(1)

app = typer.Typer(
    name="verifily",
    help=(
        "Verifily — ML data quality gate.\n\n"
        "Ingest, validate, and ship datasets with confidence.\n"
        "Exit codes: 0=SHIP, 1=DONT_SHIP, 2=INVESTIGATE, 3=CONTRACT_FAIL, 4=TOOL_ERROR."
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=(
        "Quick start:\n"
        "  verifily quickstart my_project        Scaffold a working project\n"
        "  verifily pipeline --config verifily.yaml --ci\n"
        "  verifily version                      Show version info\n\n"
        "Common commands:\n"
        "  verifily ingest --in data.csv --out datasets/v1 --schema sft\n"
        "  verifily report --dataset data/train.jsonl\n"
        "  verifily contamination --train train.jsonl --eval eval.jsonl\n"
        "  verifily fingerprint --dataset data/artifact\n"
        "  verifily diff-datasets ds_a.jsonl ds_b.jsonl\n"
        "  verifily ci-init --github\n"
        "  verifily serve --port 8000\n\n"
        f"Verifily v{__version__} — Deterministic ML decision infrastructure.\n"
        "Docs: SECURITY.md, docs/guarantees.md, docs/reliability.md"
    ),
)


# ── Sub-app groups (keep top-level --help clean) ─────────────
admin_app = typer.Typer(help="Admin commands: API keys, projects, teams, RBAC.")
billing_app = typer.Typer(help="Billing: events, invoices, plans, usage, checkout.")
monitor_app = typer.Typer(help="Continuous monitoring: start, stop, status, history.")
ws_app = typer.Typer(help="Workspaces: orgs, projects, keys.")
app.add_typer(admin_app, name="admin")
app.add_typer(billing_app, name="billing")
app.add_typer(monitor_app, name="monitor")
app.add_typer(ws_app, name="ws")


def _version_callback(value: bool) -> None:
    if value:
        from rich.panel import Panel
        c = Console()
        c.print(Panel(f"[bold]Verifily CLI[/bold] v{__version__}", border_style="blue"))
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit.",
        callback=_version_callback, is_eager=True,
    ),
) -> None:
    """Verifily — ML data quality gate."""
    cmd = ctx.invoked_subcommand
    if cmd is not None:
        _require_license(cmd)


# ── init ─────────────────────────────────────────────────────────

@app.command()
def init(
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="Output path for the generated config file."
    ),
) -> None:
    """Interactive wizard to generate a transform.yaml or train.yaml config.

    Example:
      verifily init
      verifily init --out my_config.yaml
    """
    _run_safe(lambda: _init_impl(out))


def _init_impl(out: Optional[str]) -> None:
    from verifily_cli_v1.commands.init import run
    run(out=out)


# ── doctor ───────────────────────────────────────────────────────

@app.command()
def doctor(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Optional config file to validate."
    ),
    deploy: bool = typer.Option(
        False, "--deploy", help="Run deployment readiness checks."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Check Python, platform, packages, write permissions, and optional tools.

    Exit codes: 0=healthy, 2=warnings only, 3=fatal issues.

    Example:
      verifily doctor
      verifily doctor --config train.yaml
      verifily doctor --deploy
    """
    _run_safe(lambda: _doctor_impl(config, deploy, verbose), verbose=verbose)


def _doctor_impl(config: Optional[str], deploy: bool, verbose: bool) -> None:
    if deploy:
        exit_code = _doctor_deploy_impl(verbose)
    else:
        from verifily_cli_v1.commands.doctor import run
        exit_code = run(config=config, verbose=verbose)
    if exit_code != 0:
        raise SystemExit(exit_code)


def _doctor_deploy_impl(verbose: bool) -> int:
    """Run deployment readiness checks.
    
    Returns:
        Exit code: 0=OK, 1=FAIL, 2=WARN
    """
    from verifily_cli_v1.core.deploy_config import load_deploy_config, validate_deploy_config
    from verifily_cli_v1.core.runtime_paths import get_runtime_paths
    
    exit_code = 0
    
    console.print("[bold]Verifily Deployment Doctor[/bold]\n")
    
    # 1. Config loading
    console.print("[bold]Configuration:[/bold]")
    try:
        deploy_config = load_deploy_config()
        config_valid, config_errors = validate_deploy_config(deploy_config)
        
        if config_valid:
            console.print("  [green]✓[/green] Config loaded and valid")
        else:
            console.print("  [red]✗[/red] Config validation failed:")
            for err in config_errors:
                console.print(f"    - {err}")
            exit_code = max(exit_code, 1)
    except Exception as e:
        console.print(f"  [red]✗[/red] Config error: {e}")
        exit_code = max(exit_code, 1)
    
    # 2. Runtime paths
    console.print("\n[bold]Runtime Paths:[/bold]")
    runtime_paths = get_runtime_paths()
    path_errors = runtime_paths.validate_writable()
    
    if path_errors:
        console.print("  [red]✗[/red] Path errors:")
        for err in path_errors:
            console.print(f"    - {err}")
        exit_code = max(exit_code, 1)
    else:
        console.print("  [green]✓[/green] All paths writable")
        for name, path in runtime_paths.describe_paths().items():
            console.print(f"    {name}: {path}")
    
    # 3. Production warnings
    console.print("\n[bold]Deployment Posture:[/bold]")
    warnings = []
    
    if deploy_config.auth.enabled:
        if deploy_config.auth.api_key and not deploy_config.auth.workspaces_enabled:
            warnings.append("Using legacy shared-key mode (consider workspaces)")
    else:
        warnings.append("Auth disabled - server is open")
    
    if runtime_paths.is_default_home() and deploy_config.is_production_like():
        warnings.append("VERIFILY_HOME is /tmp (may be cleared on reboot)")
    
    if not deploy_config.persistence.any_enabled():
        warnings.append("No persistence enabled - data will be lost on restart")
    
    if warnings:
        for warning in warnings:
            console.print(f"  [yellow]⚠[/yellow] {warning}")
        exit_code = max(exit_code, 2)
    else:
        console.print("  [green]✓[/green] No warnings")
    
    # Summary
    console.print()
    if exit_code == 0:
        console.print("[bold green]All checks passed[/bold green]")
    elif exit_code == 2:
        console.print("[bold yellow]Checks passed with warnings[/bold yellow]")
    else:
        console.print("[bold red]Checks failed[/bold red]")
    
    return exit_code


# ── transform ────────────────────────────────────────────────────

@app.command(hidden=True)
def transform(
    input_path: str = typer.Option(
        ..., "--in", help="Raw input dataset file or directory."
    ),
    output_path: str = typer.Option(
        ..., "--out", help="Output dataset directory."
    ),
    format: str = typer.Option(
        "jsonl", "--format", "-f", help="Output format: jsonl or parquet."
    ),
    schema: str = typer.Option(
        "sft", "--schema", "-s", help="Target schema: sft, qa, or classification."
    ),
    privacy: str = typer.Option(
        "redact", "--privacy", "-p", help="PII handling: redact, hash, or none."
    ),
    expand: Optional[int] = typer.Option(
        None, "--expand", "-e", help="Expansion factor (recorded in manifest, not generated)."
    ),
    plan: bool = typer.Option(False, "--plan", help="Dry run: show plan without writing files."),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Validate, normalize, and package raw data into a training-ready dataset.

    Example:
      verifily transform --in data/raw.jsonl --out datasets/v1
      verifily transform --in data/ --out datasets/v1 --schema classification --plan
    """
    _run_safe(
        lambda: _transform_impl(input_path, output_path, format, schema, privacy, expand, plan, verbose),
        verbose=verbose,
    )


def _transform_impl(
    input_path: str, output_path: str, format: str, schema: str,
    privacy: str, expand: Optional[int], plan: bool, verbose: bool,
) -> None:
    from verifily_cli_v1.commands.transform import run
    run(
        input_path=input_path,
        output_path=output_path,
        format=format,
        schema=schema,
        privacy=privacy,
        expand=expand,
        plan=plan,
        verbose=verbose,
    )


# ── train ────────────────────────────────────────────────────────

@app.command(hidden=True)
def train(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to train.yaml config file."
    ),
    run_dir: Optional[str] = typer.Option(
        None, "--run-dir", "-r", help="Output run directory (auto-generated if omitted)."
    ),
    plan: bool = typer.Option(False, "--plan", help="Dry run: show plan without training."),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Launch a training run using verifily_train.

    Example:
      verifily train --config train.yaml
      verifily train --config train.yaml --run-dir runs/exp_01 --plan
    """
    _run_safe(
        lambda: _train_impl(config, run_dir, plan, verbose),
        verbose=verbose,
    )


def _train_impl(config: str, run_dir: Optional[str], plan: bool, verbose: bool) -> None:
    from verifily_cli_v1.commands.train import run
    run(config=config, run_dir=run_dir, plan=plan, verbose=verbose)


# ── eval ─────────────────────────────────────────────────────────

@app.command(name="eval")
def eval_cmd(
    run_dir: str = typer.Option(
        None, "--run", help="Path to the run directory to evaluate."
    ),
    predictions: Optional[str] = typer.Option(
        None, "--predictions", help="JSONL file with predictions (for --compute mode)."
    ),
    references: Optional[str] = typer.Option(
        None, "--references", help="JSONL file with references (for --compute mode)."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write eval_results.json to this path."
    ),
    slice_by: Optional[str] = typer.Option(
        None, "--slice-by", help="Optional tag key for sliced metrics."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Evaluate a training run and display metrics.

    Two modes:
      1. Display pre-computed results: verifily eval --run runs/my_run
      2. Compute metrics from files: verifily eval --predictions preds.jsonl --references refs.jsonl

    Example:
      verifily eval --run runs/my_run
      verifily eval --predictions preds.jsonl --references refs.jsonl -o eval_results.json
    """
    _run_safe(
        lambda: _eval_impl(run_dir, predictions, references, output, slice_by, verbose),
        verbose=verbose,
    )


def _eval_impl(
    run_dir: Optional[str],
    predictions: Optional[str],
    references: Optional[str],
    output: Optional[str],
    slice_by: Optional[str],
    verbose: bool,
) -> None:
    if predictions and references:
        # Compute mode: calculate real metrics
        from verifily_cli_v1.core.io import read_jsonl
        from verifily_cli_v1.core.eval_metrics import compute_metrics
        from rich.table import Table
        from rich.panel import Panel

        pred_rows = read_jsonl(predictions)
        ref_rows = read_jsonl(references)

        pred_texts = [str(r.get("output", r.get("text", ""))).strip() for r in pred_rows]
        ref_texts = [str(r.get("output", r.get("text", ""))).strip() for r in ref_rows]

        result = compute_metrics(pred_texts, ref_texts, include_per_example=verbose)

        tbl = Table(title="Evaluation Metrics", show_header=True)
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Value", justify="right")
        tbl.add_row("Exact Match", f"{result.exact_match:.4f}")
        tbl.add_row("Token F1", f"{result.f1:.4f}")
        tbl.add_row("BLEU", f"{result.bleu:.4f}")
        tbl.add_row("ROUGE-L", f"{result.rouge_l:.4f}")
        tbl.add_row("Length Ratio", f"{result.length_ratio:.4f}")
        tbl.add_row("Examples", str(result.num_examples))
        console.print(tbl)

        if output:
            from verifily_cli_v1.core.io import write_json
            write_json(output, result.to_dict())
            console.print(f"\nResults written to: {output}")
    elif run_dir:
        from verifily_cli_v1.commands.eval import run
        run(run_dir=run_dir, slice_by=slice_by, verbose=verbose)
    else:
        console.print("[red]Provide --run or both --predictions and --references[/red]")
        raise SystemExit(1)


# ── compare ──────────────────────────────────────────────────────

@app.command(hidden=True)
def compare(
    runs: str = typer.Option(
        ..., "--runs", help="Comma-separated list of run directories."
    ),
    metric: str = typer.Option(
        "f1", "--metric", "-m", help="Primary metric to compare: f1, exact_match, accuracy."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show config diff between runs."),
) -> None:
    """Compare metrics across multiple training runs.

    Example:
      verifily compare --runs runs/a,runs/b,runs/c --metric f1
    """
    _run_safe(
        lambda: _compare_impl(runs, metric, verbose),
        verbose=verbose,
    )


def _compare_impl(runs: str, metric: str, verbose: bool) -> None:
    from verifily_cli_v1.commands.compare import run
    run(runs=runs, metric=metric, verbose=verbose)


# ── reproduce ────────────────────────────────────────────────────

@app.command(hidden=True)
def reproduce(
    run_dir: str = typer.Option(
        ..., "--run", help="Path to the run directory to verify."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show hash details."),
) -> None:
    """Verify the reproducibility of a training or transform run via hash chains.

    Example:
      verifily reproduce --run runs/my_run
    """
    _run_safe(
        lambda: _reproduce_impl(run_dir, verbose),
        verbose=verbose,
    )


def _reproduce_impl(run_dir: str, verbose: bool) -> None:
    from verifily_cli_v1.commands.reproduce import run
    run(run_dir=run_dir, verbose=verbose)


# ── report ──────────────────────────────────────────────────────

@app.command()
def report(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Path to JSONL dataset file."
    ),
    schema: str = typer.Option(
        "sft", "--schema", "-s", help="Expected schema: sft, classification, chat, qa, etc."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write JSON report to file."
    ),
    use_ner: bool = typer.Option(
        False, "--ner/--no-ner", help="Use spaCy NER for PII detection (requires pip install verifily[pii])."
    ),
    pii_confidence: float = typer.Option(
        0.0, "--pii-confidence", help="Minimum confidence threshold for PII matches."
    ),
    server: Optional[str] = typer.Option(
        None, "--server", help="Run via remote API server."
    ),
    server_api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for remote server auth."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Generate a dataset report with field statistics and PII scan.

    Use --server to run via a remote API instead of locally.

    Example:
      verifily report --dataset data/train.jsonl
      verifily report --dataset data/train.jsonl --ner
      verifily report --dataset data/train.jsonl --server https://api.verifily.io
    """
    def _impl():
        from verifily_cli_v1.core.remote import get_api_url
        remote_url = get_api_url(server)
        if remote_url:
            from verifily_cli_v1.core.remote import remote_report
            console.print(f"[dim]Running via remote server: {remote_url}[/dim]")
            result = remote_report(
                server=remote_url,
                api_key=server_api_key or os.environ.get("VERIFILY_API_KEY", ""),
                dataset_path=dataset,
                schema=schema,
            )
            import json as _json
            console.print(_json.dumps(result, indent=2))
            return
        _report_impl(dataset, schema, output, verbose, use_ner, pii_confidence)

    _run_safe(_impl, verbose=verbose)


def _report_impl(
    dataset: str, schema: str, output: Optional[str], verbose: bool,
    use_ner: bool = False, pii_confidence: float = 0.0,
) -> None:
    ds_path = Path(dataset)
    if not ds_path.exists():
        console.print(f"[red bold]Error:[/red bold] Dataset not found: {ds_path}")
        console.print("[dim]Hint:[/dim] Use [bold]verifily ingest --in <raw> --out <dir>[/bold] to create a dataset,")
        console.print("      or check the path and try again.")
        raise SystemExit(1)
    from verifily_cli_v1.commands.report import run
    run(dataset=dataset, schema=schema, output=output, verbose=verbose,
        use_ner=use_ner, min_confidence=pii_confidence)


# ── contamination ───────────────────────────────────────────────

@app.command()
def contamination(
    train: str = typer.Option(
        ..., "--train", help="Path to training JSONL dataset."
    ),
    eval_set: str = typer.Option(
        ..., "--eval", help="Path to eval JSONL dataset."
    ),
    jaccard_cutoff: float = typer.Option(
        0.70, "--jaccard", help="Jaccard similarity cutoff for near-duplicates."
    ),
    num_perm: int = typer.Option(
        128, "--num-perm", help="Number of MinHash permutations for LSH."
    ),
    no_lsh: bool = typer.Option(
        False, "--no-lsh", help="Disable LSH, use brute-force O(N^2) comparison."
    ),
    sample_train: Optional[int] = typer.Option(
        None, "--sample-train", help="Deterministically sample N train rows."
    ),
    sample_eval: Optional[int] = typer.Option(
        None, "--sample-eval", help="Deterministically sample N eval rows."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write JSON results to file."
    ),
    server: Optional[str] = typer.Option(
        None, "--server", help="Run via remote API server (e.g. https://api.verifily.io)."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for remote server auth."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Detect train/eval contamination via exact and near-duplicate matching.

    Uses MinHash LSH by default for scalable O(N) near-duplicate detection.
    Use --no-lsh to fall back to brute-force O(N^2) comparison.

    Use --server to run via a remote API instead of locally.

    Exit codes: 0=PASS, 1=FAIL (exact leaks), 2=WARN (near-duplicates).

    Example:
      verifily contamination --train data/train.jsonl --eval data/eval.jsonl
      verifily contamination --train train.jsonl --eval eval.jsonl --num-perm 128
      verifily contamination --train train.jsonl --eval eval.jsonl --server https://api.verifily.io
    """
    def _impl():
        from verifily_cli_v1.core.remote import get_api_url
        remote_url = get_api_url(server)

        if remote_url:
            from verifily_cli_v1.core.remote import remote_contamination
            console.print(f"[dim]Running via remote server: {remote_url}[/dim]")
            result = remote_contamination(
                server=remote_url,
                api_key=api_key or os.environ.get("VERIFILY_API_KEY", ""),
                train_path=train,
                eval_path=eval_set,
                jaccard_cutoff=jaccard_cutoff,
            )
            if output:
                Path(output).write_text(json.dumps(result, indent=2))
            import json as _json
            console.print(_json.dumps(result, indent=2))
            raise SystemExit(result.get("exit_code", 0))

        from verifily_cli_v1.commands.contamination import run
        result = run(train=train, eval_set=eval_set, jaccard_cutoff=jaccard_cutoff,
                     output=output, verbose=verbose, num_perm=num_perm,
                     use_lsh=not no_lsh, sample_train=sample_train,
                     sample_eval=sample_eval)
        raise SystemExit(result["exit_code"])

    _run_safe(_impl, verbose=verbose)


# ── pipeline ────────────────────────────────────────────────────

@app.command()
def pipeline(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to verifily.yaml pipeline config."
    ),
    ci: bool = typer.Option(
        False, "--ci", help="CI mode: JSON output to stdout, clean exit codes."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Directory to write pipeline artifacts."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
    wandb: Optional[bool] = typer.Option(
        None, "--wandb/--no-wandb", help="Log results to Weights & Biases (overrides config)."
    ),
    wandb_project: Optional[str] = typer.Option(
        None, "--wandb-project", help="W&B project name (default: verifily)."
    ),
    mlflow: Optional[bool] = typer.Option(
        None, "--mlflow/--no-mlflow", help="Log results to MLflow (overrides config)."
    ),
    mlflow_experiment: Optional[str] = typer.Option(
        None, "--mlflow-experiment", help="MLflow experiment name (default: verifily)."
    ),
    server: Optional[str] = typer.Option(
        None, "--server", help="Run via remote API server (e.g. https://api.verifily.io)."
    ),
    server_api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for remote server auth."
    ),
) -> None:
    """Run end-to-end pipeline: contract → report → contamination → decision.

    Exit codes: 0=SHIP, 1=DONT_SHIP, 2=INVESTIGATE, 3=CONTRACT_FAIL, 4=TOOL_ERROR.

    Use --server to run via a remote API instead of locally.

    Example:
      verifily pipeline --config verifily.yaml
      verifily pipeline --config verifily.yaml --ci
      verifily pipeline --config verifily.yaml --wandb --wandb-project my-project
      verifily pipeline --config verifily.yaml --server https://api.verifily.io
    """
    def _impl():
        from verifily_cli_v1.core.remote import get_api_url
        remote_url = get_api_url(server)

        if remote_url:
            from verifily_cli_v1.core.remote import remote_pipeline
            console.print(f"[dim]Running via remote server: {remote_url}[/dim]")
            result = remote_pipeline(
                server=remote_url,
                api_key=server_api_key or os.environ.get("VERIFILY_API_KEY", ""),
                config_path=config,
                ci=ci,
            )
            import json as _json
            if ci:
                print(_json.dumps(result, indent=2))
            else:
                console.print(_json.dumps(result, indent=2))
            exit_code = result.get("decision", {}).get("exit_code", EXIT_TOOL_ERROR)
            raise SystemExit(exit_code)

        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red bold]Error:[/red bold] Config not found: {config_path}")
            console.print("[dim]Hint:[/dim] Run [bold]verifily quickstart <path>[/bold] to scaffold a project,")
            console.print("      or specify a valid config with [bold]--config <path>[/bold].")
            raise SystemExit(EXIT_TOOL_ERROR)

        # Apply CLI integration overrides to config before running
        _integration_overrides = {}
        if wandb is not None:
            _integration_overrides.setdefault("wandb", {})["enabled"] = wandb
        if wandb_project is not None:
            _integration_overrides.setdefault("wandb", {})["project"] = wandb_project
        if mlflow is not None:
            _integration_overrides.setdefault("mlflow", {})["enabled"] = mlflow
        if mlflow_experiment is not None:
            _integration_overrides.setdefault("mlflow", {})["experiment_name"] = mlflow_experiment

        try:
            from verifily_cli_v1.commands.pipeline import run
            result = run(
                config=config, ci=ci, output=output, verbose=verbose,
                integration_overrides=_integration_overrides if _integration_overrides else None,
            )
            exit_code = result.get("decision", {}).get("exit_code", EXIT_TOOL_ERROR)
            raise SystemExit(exit_code)
        except SystemExit:
            raise
        except Exception as e:
            if verbose:
                import traceback as tb
                console.print(f"\n[red bold]Error:[/red bold] {e}")
                console.print(tb.format_exc())
            else:
                console.print(f"\n[red bold]Error:[/red bold] {e}")
            raise SystemExit(EXIT_TOOL_ERROR)

    _run_safe(_impl, verbose=verbose)


# ── history ─────────────────────────────────────────────────────

@app.command(hidden=True)
def history(
    runs_dir: str = typer.Option(
        ..., "--runs", help="Directory containing run subdirectories."
    ),
    metric: str = typer.Option(
        "f1", "--metric", "-m", help="Metric to track: f1, exact_match, accuracy."
    ),
    threshold: float = typer.Option(
        0.02, "--threshold", "-t", help="Regression threshold (minimum drop to flag)."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Write JSON results to file."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Track metric history and detect regressions across runs.

    Exit codes: 0=no regression, 2=regression detected.

    Example:
      verifily history --runs runs/ --metric f1
    """
    def _impl():
        from verifily_cli_v1.commands.history import run
        result = run(runs_dir=runs_dir, metric=metric, threshold=threshold,
                     output=output, verbose=verbose)
        raise SystemExit(result["exit_code"])

    _run_safe(_impl, verbose=verbose)


# ── contract-check ──────────────────────────────────────────────

@app.command(name="contract-check")
def contract_check(
    run_dir: Optional[str] = typer.Option(
        None, "--run", help="Run directory to validate."
    ),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="JSONL dataset to validate."
    ),
    schema: str = typer.Option(
        "sft", "--schema", "-s", help="Expected schema: sft or classification."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Validate run contract and/or dataset schema.

    Example:
      verifily contract-check --run runs/my_run
      verifily contract-check --dataset data/train.jsonl --schema sft
    """
    def _impl():
        from verifily_cli_v1.commands.contract_check import run
        ok = run(run_dir=run_dir, dataset=dataset, schema=schema, verbose=verbose)
        if not ok:
            raise SystemExit(1)

    _run_safe(_impl, verbose=verbose)


# ── ingest ──────────────────────────────────────────────────────

@app.command()
def ingest(
    input_path: str = typer.Option(
        ..., "--in", help="Input CSV, JSONL, Parquet file, or HuggingFace URI (hf://dataset-name)."
    ),
    output_path: str = typer.Option(
        ..., "--out", help="Output dataset artifact directory."
    ),
    schema: str = typer.Option(
        "auto", "--schema", "-s",
        help="Target schema: auto, sft, classification, chat, qa, summarization, translation, rm_pairwise.",
    ),
    map_args: Optional[list[str]] = typer.Option(
        None, "--map",
        help="Column mapping canonical:source (repeatable). E.g. --map question:prompt",
    ),
    tag_args: Optional[list[str]] = typer.Option(
        None, "--tag",
        help="Tag to attach to every row key:value (repeatable). E.g. --tag source:customer",
    ),
    id_col: Optional[str] = typer.Option(
        None, "--id-col", help="Source column to use as row ID (otherwise auto-generated).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Ingest only the first N rows.",
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail on any invalid row (default: drop and warn).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and preview without writing files.",
    ),
    expand_chat: bool = typer.Option(
        True, "--expand-chat/--no-expand-chat",
        help="Expand chat messages into SFT pairs (default: True).",
    ),
    flatten_sep: str = typer.Option(
        ".", "--flatten-sep", help="Separator for flattened nested field names.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Convert CSV, JSONL, or Parquet into a canonical Verifily dataset artifact.

    Produces: dataset.jsonl, manifest.json, hashes.json, report.json.

    Examples:
      verifily ingest --in data.csv --out datasets/v1 --schema sft
      verifily ingest --in data.parquet --out datasets/v1 --schema auto
      verifily ingest --in data.csv --out datasets/v1 --map question:prompt --map answer:completion
      verifily ingest --in data.jsonl --out datasets/v1 --schema auto --tag source:customer
      verifily ingest --in chat.jsonl --out datasets/v1 --schema chat
    """
    _run_safe(
        lambda: _ingest_impl(
            input_path, output_path, schema, map_args, tag_args,
            id_col, limit, strict, dry_run, verbose, expand_chat, flatten_sep,
        ),
        verbose=verbose,
    )


def _ingest_impl(
    input_path: str, output_path: str, schema: str,
    map_args: Optional[list[str]], tag_args: Optional[list[str]],
    id_col: Optional[str], limit: Optional[int],
    strict: bool, dry_run: bool, verbose: bool,
    expand_chat: bool = True, flatten_sep: str = ".",
) -> None:
    from verifily_cli_v1.commands.ingest import run
    run(
        input_path=input_path,
        output_path=output_path,
        schema=schema,
        map_args=map_args,
        tag_args=tag_args,
        id_col=id_col,
        limit=limit,
        strict=strict,
        dry_run=dry_run,
        verbose=verbose,
        expand_chat=expand_chat,
        flatten_sep=flatten_sep,
    )


# ── serve ──────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host to bind to (default: 127.0.0.1)."
    ),
    port: int = typer.Option(
        8000, "--port", "-p", help="Port to listen on."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", help="Project root directory (auto-discover verifily.yaml)."
    ),
    allow_nonlocal: bool = typer.Option(
        False, "--allow-nonlocal",
        help="Allow binding to non-localhost addresses (use with caution).",
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development.",
    ),
    prod: bool = typer.Option(
        False, "--prod", help="Run in production mode (strict workspace security, json logs).",
    ),
    workspace_root: Optional[str] = typer.Option(
        None, "--workspace-root",
        help="Workspace root directory for file operations (default: ./workspace).",
    ),
    enable_docs: bool = typer.Option(
        False, "--enable-docs", help="Force-enable /docs even in prod mode.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Start a local HTTP API server wrapping Verifily pipeline, contamination, and report.

    Binds to 127.0.0.1 by default. Use --allow-nonlocal to override.

    Example:
      verifily serve
      verifily serve --port 8099 --prod --allow-nonlocal
      verifily serve --project /path/to/project
      verifily serve --prod --workspace-root /data/workspace
    """
    _run_safe(
        lambda: _serve_impl(host, port, project, allow_nonlocal, reload, prod, workspace_root, enable_docs),
        verbose=verbose,
    )


def _serve_impl(
    host: str, port: int, project: Optional[str],
    allow_nonlocal: bool, reload: bool,
    prod: bool, workspace_root: Optional[str],
    enable_docs_flag: bool,
) -> None:
    import os
    from pathlib import Path
    
    # Set environment variables from CLI flags
    if project:
        os.environ["VERIFILY_PROJECT"] = str(project)
    if prod:
        os.environ["VERIFILY_PROD"] = "1"
        os.environ["VERIFILY_LOG_FORMAT"] = "json"
    if workspace_root:
        os.environ["VERIFILY_WORKSPACE_ROOT"] = str(workspace_root)
    if enable_docs_flag:
        os.environ["VERIFILY_ENABLE_DOCS"] = "1"
    elif prod:
        os.environ["VERIFILY_ENABLE_DOCS"] = "0"

    # Load production settings
    try:
        from verifily_cli_v1.core.api.settings import load_production_settings
        settings = load_production_settings()
        settings_valid = True
    except ValueError as e:
        console.print(f"[yellow]Settings warning:[/yellow] {e}")
        from verifily_cli_v1.core.api.settings import ProductionSettings
        settings = ProductionSettings()
        settings_valid = False

    # Print startup info
    if settings_valid:
        settings.print_startup_info()
    else:
        console.print("[bold]Verifily API Server[/bold]")
        console.print(f"  Mode:        {settings.prod_mode and 'PRODUCTION' or 'DEVELOPMENT'}")
        console.print(f"  Bind:        {settings.bind_host}:{settings.port}")
        console.print()

    # Validate workspace
    if prod or settings.prod_mode:
        if not settings.workspace_root.exists():
            console.print(f"[red]Error:[/red] Workspace root does not exist: {settings.workspace_root}")
            console.print(f"  Create it: mkdir -p {settings.workspace_root}")
            raise SystemExit(1)

    from verifily_cli_v1.core.api.server import start_server
    from verifily_cli_v1.core.api.settings import load_settings as load_base_settings
    
    base_settings = load_base_settings(
        bind=host,
        port=port,
        allow_nonlocal=allow_nonlocal,
        enable_docs=enable_docs_flag if enable_docs_flag else not prod,
    )

    start_server(
        host=host, port=port,
        allow_nonlocal=allow_nonlocal,
        reload=reload,
        settings=base_settings,
    )


# ── usage ──────────────────────────────────────────────────────

@app.command(hidden=True)
def usage(
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s",
        help="Verifily API server URL.",
    ),
    window: int = typer.Option(
        0, "--window", "-w",
        help="Time window in minutes (0 = all time).",
    ),
    group_by: str = typer.Option(
        "key_project", "--group-by", "-g",
        help="Grouping mode: key_project, key, project, or total.",
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Output raw JSON instead of a table.",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k",
        help="API key (or set VERIFILY_API_KEY env var).",
        envvar="VERIFILY_API_KEY",
    ),
) -> None:
    """Query usage accounting data from a running Verifily API server.

    Example:
      verifily usage
      verifily usage --server http://localhost:8080 --window 60 --group-by total
      verifily usage --json
    """
    _run_safe(
        lambda: _usage_impl(server, window, group_by, json_output, api_key),
    )


def _usage_impl(
    server: str, window: int, group_by: str,
    json_output: bool, api_key: Optional[str],
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.usage(window_minutes=window, group_by=group_by)

    if json_output:
        out_console = Console()
        out_console.print_json(json_mod.dumps(resp.model_dump(exclude_none=True), indent=2))
        return

    from rich.table import Table

    if resp.total is not None:
        table = Table(title="Usage — Total")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for k, v in resp.total.items():
            table.add_row(k, str(v))
        console.print(table)
    elif resp.buckets is not None:
        table = Table(title=f"Usage — group_by={group_by}")
        if group_by in ("key_project", "key"):
            table.add_column("api_key_id")
        if group_by in ("key_project", "project"):
            table.add_column("project_id")
        table.add_column("requests", justify="right")
        table.add_column("ship", justify="right")
        table.add_column("dont_ship", justify="right")
        table.add_column("investigate", justify="right")
        table.add_column("rows_in", justify="right")
        table.add_column("rows_out", justify="right")
        table.add_column("elapsed_ms", justify="right")
        for b in resp.buckets:
            row = []
            if group_by in ("key_project", "key"):
                row.append(b.api_key_id or "")
            if group_by in ("key_project", "project"):
                row.append(b.project_id or "")
            row.extend([
                str(b.requests),
                str(b.decisions_ship),
                str(b.decisions_dont_ship),
                str(b.decisions_investigate),
                str(b.rows_in),
                str(b.rows_out),
                str(b.elapsed_ms_sum),
            ])
            table.add_row(*row)
        console.print(table)
    else:
        console.print("[dim]No usage data.[/dim]")


# ── retrain ──────────────────────────────────────────────────────

@app.command(hidden=True)
def retrain(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Path to dataset artifact directory."
    ),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", "-b", help="Baseline run directory."
    ),
    contaminated: Optional[str] = typer.Option(
        None, "--contaminated", help="Contaminated run directory (for degradation test)."
    ),
    mode: str = typer.Option(
        "mock", "--mode", "-m", help="Training mode: mock (default) or real."
    ),
    output: Optional[str] = typer.Option(
        None, "--out", "-o", help="Output directory for run artifacts."
    ),
    seed_val: int = typer.Option(
        42, "--seed", help="Random seed for deterministic mock metrics."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Run mock (or real) retrain and produce Verifily artifacts.

    Mock mode is safe and fast — produces deterministic eval metrics and decisions.
    Real mode requires VERIFILY_ENABLE_REAL_TRAIN=1 env var.

    Example:
      verifily retrain --dataset datasets/v1
      verifily retrain --dataset datasets/v1 --baseline runs/run_clean --out /tmp/retrain
    """
    _run_safe(
        lambda: _retrain_impl(dataset, baseline, contaminated, mode, output, seed_val, verbose),
        verbose=verbose,
    )


def _retrain_impl(
    dataset: str, baseline: Optional[str], contaminated: Optional[str],
    mode: str, output: Optional[str], seed: int, verbose: bool,
) -> None:
    from verifily_cli_v1.core.api.retrain import run_retrain_api

    payload = {
        "dataset_dir": dataset,
        "base_run_dir": baseline,
        "contaminated_run_dir": contaminated,
        "mode": mode,
        "output_dir": output,
        "seed": seed,
        "project_id": "cli",
    }
    result = run_retrain_api(payload)

    out = Console()
    decision = result["decision"]
    out.print(f"\n[bold]Retrain Complete[/bold]")
    out.print(f"  Run dir: {result['run_dir']}")
    out.print(f"  Decision: [{'green' if decision['recommendation'] == 'SHIP' else 'red'}]{decision['recommendation']}[/]")
    out.print(f"  Exit code: {decision['exit_code']}")
    out.print(f"  Metrics: {result['eval_summary']}")
    if verbose:
        out.print(f"  Artifacts:")
        for k, v in result["artifacts"].items():
            out.print(f"    {k}: {v}")


# ── monitor-start ────────────────────────────────────────────────

@monitor_app.command(name="start")
def monitor_start(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to verifily.yaml pipeline config."
    ),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Seconds between ticks."
    ),
    max_ticks: int = typer.Option(
        0, "--max-ticks", help="Max ticks (0=unlimited)."
    ),
    rolling_window: int = typer.Option(
        20, "--rolling-window", help="Max ticks kept in history."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
) -> None:
    """Start a continuous gating monitor on a running Verifily server.

    Example:
      verifily monitor-start --config verifily.yaml --interval 60 --max-ticks 10
    """
    _run_safe(lambda: _monitor_start_impl(config, interval, max_ticks, rolling_window, server, api_key))


def _monitor_start_impl(
    config: str, interval: int, max_ticks: int, rolling_window: int,
    server: str, api_key: Optional[str],
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.start_monitor(
        config_path=config,
        interval_seconds=interval,
        max_ticks=max_ticks,
        rolling_window=rolling_window,
    )
    out = Console()
    out.print(f"[bold green]Monitor started[/bold green]")
    out.print(f"  monitor_id: {resp.monitor_id}")
    out.print(f"  status: {resp.status}")


# ── monitor-stop ─────────────────────────────────────────────────

@monitor_app.command(name="stop")
def monitor_stop(
    monitor_id: str = typer.Option(
        ..., "--monitor-id", "-m", help="Monitor ID to stop."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
) -> None:
    """Stop a running monitor.

    Example:
      verifily monitor-stop --monitor-id abc123
    """
    _run_safe(lambda: _monitor_stop_impl(monitor_id, server, api_key))


def _monitor_stop_impl(monitor_id: str, server: str, api_key: Optional[str]) -> None:
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.stop_monitor(monitor_id)
    out = Console()
    out.print(f"[bold yellow]Monitor stopped[/bold yellow]")
    out.print(f"  monitor_id: {resp.get('monitor_id', monitor_id)}")


# ── monitor-status ───────────────────────────────────────────────

@monitor_app.command(name="status")
def monitor_status(
    monitor_id: str = typer.Option(
        ..., "--monitor-id", "-m", help="Monitor ID."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Check the status of a monitor.

    Example:
      verifily monitor-status --monitor-id abc123
    """
    _run_safe(lambda: _monitor_status_impl(monitor_id, server, api_key, json_output))


def _monitor_status_impl(
    monitor_id: str, server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.monitor_status(monitor_id)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(exclude_none=True), indent=2))
    else:
        out.print(f"[bold]Monitor Status[/bold]")
        out.print(f"  monitor_id: {resp.monitor_id}")
        out.print(f"  status: {resp.status}")
        out.print(f"  tick_count: {resp.tick_count}")
        if resp.last_tick:
            out.print(f"  last_decision: {resp.last_tick.get('decision', 'N/A')}")
        if resp.error:
            out.print(f"  [red]error: {resp.error}[/red]")


# ── monitor-history ──────────────────────────────────────────────

@monitor_app.command(name="history")
def monitor_history(
    monitor_id: str = typer.Option(
        ..., "--monitor-id", "-m", help="Monitor ID."
    ),
    last_n: int = typer.Option(
        0, "--last-n", "-n", help="Return last N ticks (0=all)."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """View tick history for a monitor.

    Example:
      verifily monitor-history --monitor-id abc123 --last-n 5
    """
    _run_safe(lambda: _monitor_history_impl(monitor_id, last_n, server, api_key, json_output))


def _monitor_history_impl(
    monitor_id: str, last_n: int, server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.monitor_history(monitor_id, last_n=last_n)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(exclude_none=True), indent=2))
    else:
        from rich.table import Table
        table = Table(title=f"Monitor History — {monitor_id}")
        table.add_column("Tick", justify="right")
        table.add_column("Decision")
        table.add_column("F1", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Regression")
        table.add_column("Contamination")
        table.add_column("Contract")
        for t in resp.ticks:
            delta_str = f"{t.get('delta', 0):.4f}" if t.get("delta") is not None else "-"
            f1_str = f"{t.get('metric_value', 0):.4f}" if t.get("metric_value") is not None else "-"
            table.add_row(
                str(t.get("tick_number", "")),
                t.get("decision", ""),
                f1_str,
                delta_str,
                "YES" if t.get("regression_detected") else "no",
                "PASS" if t.get("contamination_pass") else "FAIL",
                "PASS" if t.get("contract_pass") else "FAIL",
            )
        out.print(table)
        out.print(f"  Total ticks: {resp.total_ticks}")


# ── quickstart ────────────────────────────────────────────────────

@app.command()
def quickstart(
    path: str = typer.Argument(
        "./verifily_quickstart", help="Project directory to create."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing directory."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON instead of Rich panel."),
) -> None:
    """Scaffold a working Verifily project in 3 minutes.

    Creates sample data, a baseline run, and a demo script that runs ingest + pipeline.

    Example:
      verifily quickstart
      verifily quickstart ./my_project --force
      verifily quickstart --json
    """
    _run_safe(lambda: _quickstart_impl(path, force, json_output))


def _quickstart_impl(path: str, force: bool, json_output: bool) -> None:
    import json as json_mod
    from verifily_cli_v1.commands.quickstart import scaffold

    result = scaffold(path, force=force)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(result, indent=2))
        return

    from rich.panel import Panel
    lines = [
        f"[bold green]Project created at:[/bold green] {result['project_dir']}",
        "",
        "[bold]Next steps:[/bold]",
    ]
    for i, step in enumerate(result["next_steps"], 1):
        lines.append(f"  {i}. [cyan]{step}[/cyan]")
    console.print(Panel("\n".join(lines), title="Verifily Quickstart", border_style="blue"))


# ── check ─────────────────────────────────────────────────────────

@app.command()
def check(
    file: str = typer.Argument(..., help="Path to CSV, JSONL, or Parquet file."),
    schema: Optional[str] = typer.Option(
        None, "--schema", "-s", help="Override auto-detected schema."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Quick quality check on any dataset file.

    Auto-detects schema, scans for PII, checks for duplicates and empty fields.

    Example:
      verifily check data.csv
      verifily check train.jsonl
      verifily check data.parquet --schema sft
    """
    _run_safe(lambda: _check_impl(file, schema, json_output))


def _check_impl(file: str, schema_override: Optional[str], json_output: bool) -> None:
    import csv
    import hashlib
    from rich.table import Table
    from rich.panel import Panel

    p = Path(file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file}")

    # Read rows
    suffix = p.suffix.lower()
    rows = []
    if suffix == ".jsonl":
        from verifily_cli_v1.core.io import read_jsonl
        rows = read_jsonl(p)
    elif suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(p))
            rows = table.to_pydict()
            keys = list(rows.keys())
            rows = [dict(zip(keys, vals)) for vals in zip(*rows.values())]
        except ImportError:
            raise RuntimeError("pip install pyarrow to read Parquet files.")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not rows:
        console.print("[red]File is empty.[/red]")
        raise SystemExit(1)

    # Schema detection
    fields = set(rows[0].keys())
    detected = schema_override
    if not detected:
        try:
            from verifily_cli_v1.core.schemas import detect_schema_from_fields
            detected = detect_schema_from_fields(fields)
        except ValueError:
            detected = "unknown"

    # Empty fields
    total_cells = len(rows) * len(fields)
    empty_count = sum(
        1 for row in rows for v in row.values()
        if v is None or (isinstance(v, str) and v.strip() == "")
    )

    # Duplicates (hash text content)
    seen = set()
    dup_count = 0
    for row in rows:
        h = hashlib.md5(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
        if h in seen:
            dup_count += 1
        seen.add(h)

    # PII scan
    from verifily_cli_v1.core.pii import scan_dataset
    pii_result = scan_dataset(rows)
    pii_hits = pii_result.get("pii_total_hits", 0)

    # Quality analysis
    from verifily_cli_v1.core.quality import analyze_quality
    quality = analyze_quality(rows)
    near_dup_count = quality.stats.get("near_duplicate_count", 0)

    # Status
    issues = []
    if pii_hits > 0:
        issues.append(f"{pii_hits} PII hits")
    if dup_count > 0:
        issues.append(f"{dup_count} exact duplicates")
    if near_dup_count > 0:
        issues.append(f"{near_dup_count} near-duplicates")
    if empty_count > total_cells * 0.1:
        issues.append(f"{empty_count} empty fields ({empty_count*100//total_cells}%)")
    if quality.quality_score < 50:
        issues.append(f"quality score {quality.quality_score}/100")

    status = "PASS" if not issues else "WARN"
    status_style = "green bold" if status == "PASS" else "yellow bold"

    if json_output:
        out = {
            "file": str(p),
            "rows": len(rows),
            "fields": sorted(fields),
            "schema": detected,
            "quality_score": quality.quality_score,
            "empty_fields": empty_count,
            "duplicates": dup_count,
            "near_duplicates": near_dup_count,
            "pii_hits": pii_hits,
            "status": status,
            "quality": quality.to_dict(),
        }
        Console().print_json(json.dumps(out))
        return

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold")
    tbl.add_column()
    tbl.add_row("File", p.name)
    tbl.add_row("Rows", f"{len(rows):,}")
    tbl.add_row("Fields", ", ".join(sorted(fields)))
    tbl.add_row("Schema", detected or "unknown")
    score_style = "green" if quality.quality_score >= 80 else "yellow" if quality.quality_score >= 50 else "red"
    tbl.add_row("Quality score", f"[{score_style}]{quality.quality_score}/100[/{score_style}]")
    tbl.add_row("Empty fields", f"{empty_count:,}" + (f" ({empty_count*100//total_cells}%)" if total_cells else ""))
    tbl.add_row("Duplicates", f"{dup_count:,}" + (f" ({dup_count*100//len(rows):.1f}%)" if rows else ""))
    if near_dup_count > 0:
        tbl.add_row("Near-duplicates", f"{near_dup_count:,}" + f" ({near_dup_count*100//len(rows):.0f}%)")
    tbl.add_row("PII detected", str(pii_hits))
    tbl.add_row("Status", f"[{status_style}]{status}[/{status_style}]")

    c = Console()
    c.print(Panel(tbl, title="Verifily Check", border_style="blue"))

    # Print quality issues
    for qi in quality.issues:
        icon = "!" if qi.severity == "warning" else "X" if qi.severity == "error" else "*"
        c.print(f"  {icon} {qi.description}")


# ── ci-init ──────────────────────────────────────────────────────

@app.command(name="ci-init")
def ci_init(
    github: bool = typer.Option(False, "--github", help="Generate GitHub Actions workflow."),
    gitlab: bool = typer.Option(False, "--gitlab", help="Generate GitLab CI config."),
    path: str = typer.Option(".", "--path", "-p", help="Project root directory."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing CI files."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Generate CI config files for Verifily pipeline gating.

    Must specify exactly one of --github or --gitlab.

    Example:
      verifily ci-init --github
      verifily ci-init --gitlab --path ./my_project
      verifily ci-init --github --force --json
    """
    _run_safe(lambda: _ci_init_impl(github, gitlab, path, force, json_output))


def _ci_init_impl(
    github: bool, gitlab: bool, path: str, force: bool, json_output: bool,
) -> None:
    import json as json_mod
    from verifily_cli_v1.commands.ci_init import generate_ci

    if github == gitlab:
        raise ValueError("Specify exactly one of --github or --gitlab.")

    provider = "github" if github else "gitlab"
    result = generate_ci(provider, path, force=force)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(result, indent=2))
        return

    console.print(f"\n[bold green]CI config created:[/bold green] {result['file_path']}")
    console.print(f"  Provider: {result['provider']}")
    console.print(f"  Path: {result['relative_path']}")
    console.print()


# ── fingerprint ──────────────────────────────────────────────────

@app.command()
def fingerprint(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Dataset path (JSONL file or artifact directory)."
    ),
    schema: str = typer.Option(
        "auto", "--schema", "-s", help="Schema type: sft, classification, or auto."
    ),
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="Output directory for fingerprint.json."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for MinHash."),
    json_output: bool = typer.Option(False, "--json", help="Print fingerprint JSON to stdout."),
) -> None:
    """Compute a privacy-safe fingerprint for a dataset.

    The fingerprint contains no raw text — only statistical summaries
    and hash-based signatures for comparison.

    Example:
      verifily fingerprint --dataset data/artifact
      verifily fingerprint --dataset data.jsonl --out /tmp/fp --json
    """
    _run_safe(lambda: _fingerprint_impl(dataset, schema, out, seed, json_output))


def _fingerprint_impl(
    dataset: str, schema: str, out: Optional[str], seed: int, json_output: bool,
) -> None:
    import json as json_mod
    from verifily_cli_v1.commands.fingerprint_cmd import run_fingerprint

    result = run_fingerprint(dataset, schema=schema, out=out, seed=seed)

    if json_output:
        out_console = Console()
        out_console.print_json(json_mod.dumps(result, indent=2))
        return

    from rich.table import Table
    table = Table(title="Dataset Fingerprint")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Schema", result.get("schema", "unknown"))
    table.add_row("Rows", str(result.get("rows", 0)))
    ls = result.get("length_stats", {})
    table.add_row("Length (min/med/p95/max)",
                  f"{ls.get('min',0):.0f} / {ls.get('median',0):.0f} / {ls.get('p95',0):.0f} / {ls.get('max',0):.0f}")
    table.add_row("Exact dup rate", f"{result.get('exact_dup_rate', 0):.4f}")
    table.add_row("MinHash perms", str(len(result.get("minhash_signature", []))))
    table.add_row("Written to", result.get("fingerprint_path", "N/A"))
    console.print(table)


# ── diff-datasets ────────────────────────────────────────────────

@app.command(name="diff-datasets")
def diff_datasets(
    dataset_a: str = typer.Argument(..., help="First dataset path or artifact dir."),
    dataset_b: str = typer.Argument(..., help="Second dataset path or artifact dir."),
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="Output directory for diff.json and diff.txt."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Compare two datasets using privacy-safe fingerprints.

    Loads existing fingerprints or computes them on the fly.

    Example:
      verifily diff-datasets data/artifact_a data/artifact_b
      verifily diff-datasets ds1.jsonl ds2.jsonl --out /tmp/diff --json
    """
    _run_safe(lambda: _diff_datasets_impl(dataset_a, dataset_b, out, json_output))


def _diff_datasets_impl(
    dataset_a: str, dataset_b: str, out: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_cli_v1.commands.fingerprint_cmd import run_diff

    result = run_diff(dataset_a, dataset_b, out=out)

    if json_output:
        out_console = Console()
        out_console.print_json(json_mod.dumps(result, indent=2))
        return

    from rich.table import Table
    table = Table(title="Dataset Diff")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Similarity (MinHash Jaccard)", f"{result['similarity_score']:.4f}")
    table.add_row("Exact hash overlap estimate", f"{result['exact_dup_overlap_estimate']:.4f}")
    table.add_row("Row delta", f"{result['row_delta']:+d}")
    table.add_row("Dup rate delta", f"{result['dup_rate_delta']:+.4f}")
    ld = result.get("length_deltas", {})
    table.add_row("Length deltas (min/med/p95/max)",
                  f"{ld.get('min',0):+.0f} / {ld.get('median',0):+.0f} / {ld.get('p95',0):+.0f} / {ld.get('max',0):+.0f}")
    console.print(table)


# ── org-create ──────────────────────────────────────────────────

@app.command(name="org-create", hidden=True)
def org_create(
    name: str = typer.Option(
        ..., "--name", "-n", help="Organization name."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a new organization.

    Example:
      verifily org-create --name "My Team"
      verifily org-create --name "My Team" --json
    """
    _run_safe(lambda: _org_create_impl(name, server, api_key, json_output))


def _org_create_impl(name: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.create_org(name=name)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Organization created[/bold green]")
        out.print(f"  id: {resp.id}")
        out.print(f"  name: {resp.name}")


# ── org-list ────────────────────────────────────────────────────

@app.command(name="org-list", hidden=True)
def org_list(
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List organizations you belong to.

    Example:
      verifily org-list
      verifily org-list --json
    """
    _run_safe(lambda: _org_list_impl(server, api_key, json_output))


def _org_list_impl(server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.list_orgs()
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        from rich.table import Table
        table = Table(title="Organizations")
        table.add_column("ID", style="bold")
        table.add_column("Name")
        table.add_column("Created By")
        for org in resp.orgs:
            table.add_row(org.id, org.name, org.created_by)
        out.print(table)


# ── project-create ──────────────────────────────────────────────

@app.command(name="project-create", hidden=True)
def project_create(
    org_id: str = typer.Option(
        ..., "--org", help="Organization ID."
    ),
    name: str = typer.Option(
        ..., "--name", "-n", help="Project name."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a new project within an organization.

    Example:
      verifily project-create --org abc123 --name "LLM Training v2"
    """
    _run_safe(lambda: _project_create_impl(org_id, name, server, api_key, json_output))


def _project_create_impl(
    org_id: str, name: str, server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.create_project(org_id=org_id, name=name)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Project created[/bold green]")
        out.print(f"  id: {resp.id}")
        out.print(f"  org_id: {resp.org_id}")
        out.print(f"  name: {resp.name}")


# ── project-list ────────────────────────────────────────────────

@app.command(name="project-list", hidden=True)
def project_list(
    org_id: Optional[str] = typer.Option(
        None, "--org", help="Filter by organization ID."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List projects.

    Example:
      verifily project-list
      verifily project-list --org abc123
    """
    _run_safe(lambda: _project_list_impl(org_id, server, api_key, json_output))


def _project_list_impl(
    org_id: Optional[str], server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.list_projects(org_id=org_id)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        from rich.table import Table
        table = Table(title="Projects")
        table.add_column("ID", style="bold")
        table.add_column("Org ID")
        table.add_column("Name")
        table.add_column("Created By")
        for proj in resp.projects:
            table.add_row(proj.id, proj.org_id, proj.name, proj.created_by)
        out.print(table)


# ── security-check ────────────────────────────────────────────────


@app.command(name="security-check", hidden=True)
def security_check(
    target: str = typer.Argument(".", help="Directory to scan for leaked secrets."),
    verbose: bool = typer.Option(False, "--verbose", help="Show all scanned files."),
) -> None:
    """Scan files for leaked secrets (API keys, tokens, credentials).

    Exits 0 if clean, 1 if suspected leaks found.
    """
    _run_safe(lambda: _security_check_impl(target, verbose))


def _security_check_impl(target: str, verbose: bool) -> None:
    from rich.console import Console
    from verifily_cli_v1.core.secrets import scan_directory_for_secrets

    out = Console(stderr=True)
    findings = scan_directory_for_secrets(target, verbose=verbose)

    if not findings:
        out.print("[green]No suspected secret leaks found.[/green]")
        return

    out.print(f"[red]Found {len(findings)} suspected secret leak(s):[/red]")
    for f in findings:
        out.print(f"  {f['file']}:{f['line']} ({f['pattern']})")
    raise SystemExit(1)


# ── badge ────────────────────────────────────────────────────────

@app.command(hidden=True)
def badge(
    decision: Optional[str] = typer.Option(
        None, "--decision", "-d",
        help="Decision value: SHIP, DONT_SHIP, INVESTIGATE, CONTRACT_FAIL, TOOL_ERROR.",
    ),
    from_run: Optional[str] = typer.Option(
        None, "--from-run",
        help="Load decision from a run or artifact directory.",
    ),
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="Write badge JSON to this file path.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print badge JSON to stdout."),
) -> None:
    """Generate a Shields.io-compatible badge JSON for a pipeline decision.

    Specify either --decision or --from-run.

    Example:
      verifily badge --decision SHIP --json
      verifily badge --from-run verifily_artifacts/ --out badge.json
    """
    _run_safe(lambda: _badge_impl(decision, from_run, out, json_output))


def _badge_impl(
    decision: Optional[str], from_run: Optional[str],
    out: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_cli_v1.commands.badge import generate_badge, badge_from_run, write_badge

    if decision and from_run:
        raise ValueError("Specify either --decision or --from-run, not both.")
    if not decision and not from_run:
        raise ValueError("Specify either --decision or --from-run.")

    if from_run:
        badge_data = badge_from_run(from_run)
    else:
        badge_data = generate_badge(decision)

    if out:
        write_badge(badge_data, out)
        console.print(f"Badge written to: {out}")

    if json_output or not out:
        out_console = Console()
        out_console.print_json(json_mod.dumps(badge_data, indent=2))


# ── bundle ───────────────────────────────────────────────────────

@app.command(hidden=True)
def bundle(
    run_dir: str = typer.Option(
        ..., "--run", "-r", help="Run or artifact directory to bundle.",
    ),
    out: str = typer.Option(
        ..., "--out", "-o", help="Output zip file path.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print manifest JSON to stdout."),
) -> None:
    """Create a shareable zip bundle of run artifacts (no raw data).

    Includes decision, eval results, audit logs, hashes, and config.
    Excludes dataset.jsonl, model weights, and raw data files.

    Example:
      verifily bundle --run verifily_artifacts/ --out report.zip
      verifily bundle --run runs/my_run --out report.zip --json
    """
    _run_safe(lambda: _bundle_impl(run_dir, out, json_output))


def _bundle_impl(run_dir: str, out: str, json_output: bool) -> None:
    import json as json_mod
    from verifily_cli_v1.commands.bundle import create_bundle

    result = create_bundle(run_dir, out)

    if json_output:
        out_console = Console()
        out_console.print_json(json_mod.dumps(result["manifest"], indent=2))
    else:
        console.print(f"\n[bold]Bundle created:[/bold] {result['bundle_path']}")
        console.print(f"  Files included: {len(result['files_included'])}")
        if result["files_excluded"]:
            console.print(f"  Files excluded: {len(result['files_excluded'])} (raw data)")
        for f in result["files_included"]:
            console.print(f"    + {f}")
        console.print()


# ── version ──────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Show Verifily version, Python version, and platform."""
    import platform

    from rich.table import Table

    table = Table(show_header=False, border_style="blue", title="Verifily", title_style="bold")
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Version", __version__)
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", f"{platform.system()} {platform.machine()}")

    # Detect installed extras
    extras = []
    try:
        import fastapi  # noqa: F401
        extras.append("api")
    except ImportError:
        pass
    try:
        import httpx  # noqa: F401
        extras.append("sdk")
    except ImportError:
        pass
    table.add_row("Extras", ", ".join(extras) if extras else "none")

    c = Console()
    c.print(table)


# ── billing-events ────────────────────────────────────────────────

# ── license management ────────────────────────────────────────

@app.command()
def login(
    key: str = typer.Option(
        ..., "--key", "-k", help="Your Verifily license key (VFY-...)."
    ),
) -> None:
    """Activate a Verifily license key.

    Example:
      verifily login --key VFY-PRO-a1b2c3d4-20270216-<sig>
    """
    def _impl():
        from verifily_cli_v1.core.licensing import save_license_key
        info = save_license_key(key)
        console.print(f"\n[green bold]License activated![/green bold]")
        console.print(f"  Tier:           {info.tier.value}")
        console.print(f"  Organization:   {info.org_hash}")
        console.print(f"  Days remaining: {info.days_remaining}")
        console.print(f"\n  Saved to ~/.verifily/license.json")
    _run_safe(_impl)


@app.command()
def account() -> None:
    """Show current license status."""
    def _impl():
        from verifily_cli_v1.core.licensing import license_status
        from rich.table import Table
        status = license_status()
        table = Table(
            show_header=False, border_style="blue",
            title="Verifily License", title_style="bold",
        )
        table.add_column("Key", style="dim")
        table.add_column("Value")
        for k, v in status.items():
            table.add_row(k, str(v))
        console.print(table)
    _run_safe(_impl)


# ── billing ──────────────────────────────────────────────────

@billing_app.command(name="events")
def billing_events(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project ID."
    ),
    window: int = typer.Option(
        0, "--window", "-w", help="Time window in minutes (0=all)."
    ),
    limit: int = typer.Option(
        100, "--limit", "-n", help="Max events to return."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Query billing events from a running Verifily server.

    Example:
      verifily billing-events
      verifily billing-events --project my-proj --window 60 --json
    """
    _run_safe(lambda: _billing_events_impl(project, window, limit, server, api_key, json_output))


def _billing_events_impl(
    project: Optional[str], window: int, limit: int,
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.billing_events(project_id=project, window_minutes=window, limit=limit)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    from rich.table import Table
    table = Table(title="Billing Events")
    table.add_column("Timestamp")
    table.add_column("Endpoint")
    table.add_column("Project")
    table.add_column("Units")
    table.add_column("Status", justify="right")
    import datetime as _dt
    for evt in resp.events:
        ts_str = _dt.datetime.fromtimestamp(evt.ts).strftime("%H:%M:%S")
        units_str = ", ".join(f"{k}={v}" for k, v in evt.units.items() if v)
        table.add_row(ts_str, evt.endpoint, evt.project_id, units_str, str(evt.status_code))
    console.print(table)
    console.print(f"  Total: {resp.total}")


# ── billing-preview ──────────────────────────────────────────────

@billing_app.command(name="preview")
def billing_preview(
    project: str = typer.Option(
        "default", "--project", "-p", help="Project ID."
    ),
    plan: str = typer.Option(
        "FREE", "--plan", help="Plan ID: FREE, STARTER, PRO, ENTERPRISE."
    ),
    window: int = typer.Option(
        43200, "--window", "-w", help="Window in minutes (default 30 days)."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Preview an invoice for a project and plan.

    Example:
      verifily billing-preview --project my-proj --plan STARTER
      verifily billing-preview --json
    """
    _run_safe(lambda: _billing_preview_impl(project, plan, window, server, api_key, json_output))


def _billing_preview_impl(
    project: str, plan: str, window: int,
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.invoice_preview(project_id=project, plan_id=plan, window_minutes=window)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    from rich.table import Table
    table = Table(title=f"Invoice Preview — {resp.plan_id} plan for '{resp.project_id}'")
    table.add_column("Line Item")
    table.add_column("Quantity", justify="right")
    table.add_column("Included", justify="right")
    table.add_column("Overage", justify="right")
    table.add_column("Amount ($)", justify="right")
    for line in resp.lines:
        table.add_row(
            line.label,
            str(line.quantity),
            str(line.included),
            str(line.overage),
            f"${line.amount_cents / 100:.2f}",
        )
    console.print(table)
    console.print(f"  Monthly base:  ${resp.monthly_base_cents / 100:.2f}")
    console.print(f"  Subtotal:      ${resp.subtotal_cents / 100:.2f}")
    console.print(f"  Tax:           ${resp.tax_cents / 100:.2f}")
    console.print(f"  [bold]Total:         ${resp.total_cents / 100:.2f}[/bold]")


# ── plans ──────────────────────────────────────────────────────────

@billing_app.command(name="plans")
def billing_plans_cmd(
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List available billing plans.

    Example:
      verifily plans
      verifily plans --json
    """
    _run_safe(lambda: _plans_impl(server, api_key, json_output))


def _plans_impl(server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.billing_plans()

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    from rich.table import Table
    table = Table(title="Billing Plans")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Base $/mo", justify="right")
    table.add_column("Incl. Requests", justify="right")
    table.add_column("Incl. Rows", justify="right")
    table.add_column("Incl. Bytes", justify="right")
    for p in resp.plans:
        table.add_row(
            p.id, p.name,
            f"${p.monthly_base_cents / 100:.2f}",
            f"{p.included_requests:,}",
            f"{p.included_rows:,}",
            f"{p.included_bytes:,}",
        )
    console.print(table)


# ── estimate ──────────────────────────────────────────────────────

@billing_app.command(name="estimate")
def billing_estimate_cmd(
    plan: str = typer.Option("FREE", "--plan", help="Plan ID: FREE, STARTER, PRO, ENTERPRISE."),
    window: int = typer.Option(43200, "--window", "-w", help="Window in minutes (default 30 days)."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project."),
    group_by: str = typer.Option("total", "--group-by", help="Grouping: total, project, api_key."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Estimate billing for current usage.

    Example:
      verifily estimate --plan PRO --window 60
      verifily estimate --plan STARTER --json
    """
    _run_safe(lambda: _estimate_impl(plan, window, project, group_by, server, api_key, json_output))


def _estimate_impl(
    plan: str, window: int, project: Optional[str], group_by: str,
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.billing_estimate(plan=plan, window_minutes=window, project_id=project)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    from rich.table import Table
    table = Table(title=f"Billing Estimate — {resp.plan_id} plan")
    table.add_column("Line Item")
    table.add_column("Quantity", justify="right")
    table.add_column("Included", justify="right")
    table.add_column("Overage", justify="right")
    table.add_column("Amount ($)", justify="right")
    for line in resp.lines:
        table.add_row(
            line.label,
            str(line.quantity),
            str(line.included),
            str(line.overage),
            f"${line.amount_cents / 100:.2f}",
        )
    console.print(table)
    console.print(f"  Monthly base:  ${resp.monthly_base_cents / 100:.2f}")
    console.print(f"  Subtotal:      ${resp.subtotal_cents / 100:.2f}")
    console.print(f"  Tax:           ${resp.tax_cents / 100:.2f}")
    console.print(f"  [bold]Total:         ${resp.total_cents / 100:.2f}[/bold]")


# ── invoice ───────────────────────────────────────────────────────

@billing_app.command(name="invoice")
def billing_invoice_cmd(
    plan: str = typer.Option(..., "--plan", help="Plan ID."),
    period_days: int = typer.Option(30, "--period-days", help="Period length in days."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project ID."),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output directory."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Generate a billing invoice.

    Example:
      verifily invoice --plan PRO --period-days 30
      verifily invoice --plan STARTER --out ./invoices --json
    """
    _run_safe(lambda: _invoice_impl(plan, period_days, project, out, server, api_key, json_output))


def _invoice_impl(
    plan: str, period_days: int, project: Optional[str], out_dir: Optional[str],
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from datetime import datetime as _dt, timezone as _tz
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    now = _dt.now(_tz.utc)
    period_end = now.isoformat()
    period_start = (now - __import__("datetime").timedelta(days=period_days)).isoformat()

    resp = client.billing_invoice(
        plan=plan,
        period_start=period_start,
        period_end=period_end,
        project_id=project,
    )

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    if out_dir:
        from pathlib import Path
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        invoice_file = out_path / f"{resp.invoice_id}.json"
        invoice_file.write_text(json_mod.dumps(resp.model_dump(), indent=2))
        console.print(f"  Invoice saved to: {invoice_file}")

    from rich.table import Table
    table = Table(title=f"Invoice {resp.invoice_id} — {resp.plan_id}")
    table.add_column("Line Item")
    table.add_column("Quantity", justify="right")
    table.add_column("Included", justify="right")
    table.add_column("Overage", justify="right")
    table.add_column("Amount ($)", justify="right")
    for line in resp.lines:
        table.add_row(
            line.label,
            str(line.quantity),
            str(line.included),
            str(line.overage),
            f"${line.amount_cents / 100:.2f}",
        )
    console.print(table)
    console.print(f"  Monthly base:  ${resp.monthly_base_cents / 100:.2f}")
    console.print(f"  [bold]Total:         ${resp.total_cents / 100:.2f}[/bold]")
    console.print(f"  Period: {resp.period_start} → {resp.period_end}")


# ── usage-export ──────────────────────────────────────────────────

@billing_app.command(name="usage-export")
def billing_usage_export_cmd(
    format: str = typer.Option("csv", "--format", "-f", help="Export format: csv or jsonl."),
    period_days: int = typer.Option(30, "--period-days", help="Period in days."),
    group_by: str = typer.Option("day", "--group-by", help="Grouping: day, day_project, day_api_key."),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output directory."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
) -> None:
    """Export usage data as CSV or JSONL.

    Example:
      verifily usage-export --format csv --period-days 30
      verifily usage-export --format jsonl --out ./exports
    """
    _run_safe(lambda: _usage_export_impl(format, period_days, group_by, out, server, api_key))


def _usage_export_impl(
    fmt: str, period_days: int, group_by: str, out_dir: Optional[str],
    server: str, api_key: Optional[str],
) -> None:
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    content = client.billing_usage_export(format=fmt, period_days=period_days, group_by=group_by)

    if out_dir:
        from pathlib import Path
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ext = "csv" if fmt == "csv" else "jsonl"
        export_file = out_path / f"usage_export.{ext}"
        export_file.write_text(content)
        console.print(f"  Export saved to: {export_file}")
    else:
        console.print(content)


# ── billing-usage ──────────────────────────────────────────────────

@billing_app.command(name="usage")
def billing_usage_cmd(
    period: Optional[str] = typer.Option(None, "--period", help="YYYY-MM billing period (default: current)."),
    group_by: str = typer.Option("total", "--group-by", help="Grouping: total, api_key, project."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project ID."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", envvar="VERIFILY_API_KEY", help="API key."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Query processed rows usage for a billing period."""
    _run_safe(lambda: _billing_usage_impl(period, group_by, project, server, api_key, json_output))


def _billing_usage_impl(
    period: Optional[str],
    group_by: str,
    project: Optional[str],
    server: str,
    api_key: Optional[str],
    json_output: bool,
) -> None:
    from verifily_sdk import VerifilyClient
    client = VerifilyClient(base_url=server, api_key=api_key)
    data = client.billing_usage(period=period, group_by=group_by, project_id=project)
    if json_output:
        import json as _json
        console.print(_json.dumps(data, indent=2))
        return
    usage = data.get("usage", {})
    table = Table(title=f"Billing Usage — {data.get('period', '?')}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Processed Rows", f"{usage.get('processed_rows', 0):,}")
    table.add_row("Bytes Processed", f"{usage.get('bytes_processed', 0):,}")
    table.add_row("Decisions", f"{usage.get('decisions', 0):,}")
    table.add_row("Requests", f"{usage.get('requests', 0):,}")
    if data.get("project_id"):
        table.add_row("Project", data["project_id"])
    console.print(table)


# ── checkout ──────────────────────────────────────────────────────

@billing_app.command(name="checkout")
def checkout_cmd(
    plan: str = typer.Option("pro", "--plan", help="Plan to subscribe to."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project ID."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Start a Stripe Checkout session for Pro subscription.

    Example:
      verifily checkout --plan pro --project my-proj
    """
    _run_safe(lambda: _checkout_impl(plan, project, server, api_key, json_output))


def _checkout_impl(
    plan: str, project: Optional[str],
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.checkout(plan=plan, project_id=project)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    console.print(f"  Checkout URL: [bold blue]{resp.checkout_url}[/bold blue]")
    console.print(f"  Customer ID:  {resp.stripe_customer_id}")
    console.print(f"  Plan:         {resp.plan}")


# ── subscription ──────────────────────────────────────────────────

@billing_app.command(name="subscription")
def subscription_cmd(
    project: str = typer.Option("default", "--project", "-p", help="Project ID."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Check subscription status for a project.

    Example:
      verifily subscription --project my-proj
    """
    _run_safe(lambda: _subscription_impl(project, server, api_key, json_output))


def _subscription_impl(
    project: str, server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.subscription(project_id=project)

    if json_output:
        out = Console()
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
        return

    from rich.table import Table
    table = Table(title=f"Subscription — {resp.project_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Org", resp.org_id)
    table.add_row("Project", resp.project_id)
    table.add_row("Plan", resp.plan)
    table.add_row("Status", resp.status)
    table.add_row("Customer ID", resp.stripe_customer_id or "-")
    table.add_row("Subscription ID", resp.stripe_subscription_id or "-")
    console.print(table)


# ── admin-project-create ─────────────────────────────────────────

@admin_app.command(name="project-create")
def admin_project_create(
    id: str = typer.Option(..., "--id", help="Project ID."),
    name: str = typer.Option(..., "--name", "-n", help="Project name."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a project in the auth registry (admin).

    Example:
      verifily admin-project-create --id proj-1 --name "Demo Project"
    """
    _run_safe(lambda: _admin_project_create_impl(id, name, server, api_key, json_output))


def _admin_project_create_impl(
    id: str, name: str, server: str, api_key: Optional[str], json_output: bool
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_create_project(id=id, name=name)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print("[bold green]Project created[/bold green]")
        out.print(f"  id: {resp.id}")
        out.print(f"  name: {resp.name}")


# ── admin-key-create ─────────────────────────────────────────────

@admin_app.command(name="key-create")
def admin_key_create(
    id: str = typer.Option(..., "--id", help="Key ID."),
    name: str = typer.Option(..., "--name", "-n", help="Key name."),
    raw_key: str = typer.Option(..., "--raw-key", help="Raw API key (min 8 chars)."),
    scopes: str = typer.Option(..., "--scopes", help="Comma-separated scopes."),
    projects: str = typer.Option("*", "--projects", help="Comma-separated project IDs or '*'."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a scoped API key (admin).

    Example:
      verifily admin-key-create --id key-1 --name "CI Key" --raw-key sk-secret123 --scopes pipeline:run,report:read --projects demo
    """
    _run_safe(lambda: _admin_key_create_impl(id, name, raw_key, scopes, projects, server, api_key, json_output))


def _admin_key_create_impl(
    id: str, name: str, raw_key: str, scopes: str, projects: str,
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    scope_list = [s.strip() for s in scopes.split(",") if s.strip()]
    project_list = [p.strip() for p in projects.split(",") if p.strip()]
    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_create_key(
        id=id, name=name, raw_key=raw_key, scopes=scope_list, projects_allowed=project_list,
    )
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print("[bold green]Key created[/bold green]")
        out.print(f"  id: {resp.id}")
        out.print(f"  name: {resp.name}")
        out.print(f"  scopes: {', '.join(resp.scopes)}")
        out.print(f"  projects: {', '.join(resp.projects_allowed)}")


# ── admin-key-list ───────────────────────────────────────────────

@admin_app.command(name="key-list")
def admin_key_list(
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List all API keys (admin, hashes redacted).

    Example:
      verifily admin-key-list
    """
    _run_safe(lambda: _admin_key_list_impl(server, api_key, json_output))


def _admin_key_list_impl(server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_list_keys()
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        from rich.table import Table
        table = Table(title="API Keys")
        table.add_column("ID", style="bold")
        table.add_column("Name")
        table.add_column("Scopes")
        table.add_column("Projects")
        table.add_column("Disabled")
        for k in resp.keys:
            table.add_row(k.id, k.name, ", ".join(k.scopes), ", ".join(k.projects_allowed), str(k.disabled))
        out.print(table)


# ── admin-key-disable ────────────────────────────────────────────

@admin_app.command(name="key-disable")
def admin_key_disable(
    key_id: str = typer.Option(..., "--key-id", help="Key ID to disable."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Disable an API key (admin).

    Example:
      verifily admin-key-disable --key-id key-1
    """
    _run_safe(lambda: _admin_key_disable_impl(key_id, server, api_key, json_output))


def _admin_key_disable_impl(key_id: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_disable_key(key_id)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold yellow]Key disabled[/bold yellow]: {resp.id}")


# ── admin-key-rotate ─────────────────────────────────────────────

@admin_app.command(name="key-rotate")
def admin_key_rotate(
    key_id: str = typer.Option(..., "--key-id", help="Key ID to rotate."),
    raw_key: str = typer.Option(..., "--raw-key", help="New raw API key (min 8 chars)."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Rotate an API key (admin).

    Example:
      verifily admin-key-rotate --key-id key-1 --raw-key sk-newsecret456
    """
    _run_safe(lambda: _admin_key_rotate_impl(key_id, raw_key, server, api_key, json_output))


def _admin_key_rotate_impl(key_id: str, raw_key: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_rotate_key(key_id, raw_key=raw_key)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Key rotated[/bold green]: {resp.id}")


# ── Teams: whoami ────────────────────────────────────────────────

@admin_app.command(name="whoami")
def whoami_cmd(
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show identity of the current API key (teams mode)."""
    _run_safe(lambda: _whoami_impl(server, api_key, json_output))


def _whoami_impl(server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.whoami()
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold]API Key ID:[/bold] {resp.api_key_id}")
        out.print(f"[bold]Org ID:[/bold]    {resp.org_id}")
        out.print(f"[bold]Scopes:[/bold]    {', '.join(resp.scopes)}")
        out.print(f"[bold]Projects:[/bold]  {', '.join(resp.project_ids)}")


# ── Teams: admin-org-create ─────────────────────────────────────

@admin_app.command(name="org-create")
def admin_org_create_cmd(
    id: str = typer.Option(..., "--id", help="Organization ID."),
    name: str = typer.Option(..., "--name", help="Organization name."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create an organization (teams admin)."""
    _run_safe(lambda: _admin_org_create_impl(id, name, server, api_key, json_output))


def _admin_org_create_impl(id: str, name: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_create_org(id=id, name=name)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Org created[/bold green]: {resp.id} ({resp.name})")


# ── Teams: admin-user-create ────────────────────────────────────

@admin_app.command(name="user-create")
def admin_user_create_cmd(
    id: str = typer.Option(..., "--id", help="User ID."),
    email: str = typer.Option(..., "--email", help="User email."),
    name: str = typer.Option(..., "--name", help="User display name."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a user (teams admin)."""
    _run_safe(lambda: _admin_user_create_impl(id, email, name, server, api_key, json_output))


def _admin_user_create_impl(id: str, email: str, name: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_create_user(id=id, email=email, name=name)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]User created[/bold green]: {resp.id} ({resp.name})")


# ── Teams: admin-member-add ─────────────────────────────────────

@admin_app.command(name="member-add")
def admin_member_add_cmd(
    user_id: str = typer.Option(..., "--user-id", help="User ID."),
    org_id: str = typer.Option(..., "--org-id", help="Organization ID."),
    role: str = typer.Option("member", "--role", help="Role: owner, admin, member, readonly."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Add a membership (teams admin)."""
    _run_safe(lambda: _admin_member_add_impl(user_id, org_id, role, server, api_key, json_output))


def _admin_member_add_impl(user_id: str, org_id: str, role: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_add_membership(user_id=user_id, org_id=org_id, role=role)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Membership added[/bold green]: {resp.user_id} -> {resp.org_id} ({resp.role})")


# ── Teams: admin-team-project-create ────────────────────────────

@admin_app.command(name="team-project-create")
def admin_team_project_create_cmd(
    id: str = typer.Option(..., "--id", help="Project ID."),
    org_id: str = typer.Option(..., "--org-id", help="Organization ID."),
    name: str = typer.Option(..., "--name", help="Project name."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a project in an organization (teams admin)."""
    _run_safe(lambda: _admin_team_project_create_impl(id, org_id, name, server, api_key, json_output))


def _admin_team_project_create_impl(id: str, org_id: str, name: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_create_team_project(id=id, org_id=org_id, name=name)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Project created[/bold green]: {resp.id} ({resp.name}) in org {resp.org_id}")


# ── Teams: admin-key-issue ──────────────────────────────────────

@admin_app.command(name="key-issue")
def admin_key_issue_cmd(
    id: str = typer.Option(..., "--id", help="Key ID."),
    org_id: str = typer.Option(..., "--org-id", help="Organization ID."),
    name: str = typer.Option(..., "--name", help="Key display name."),
    raw_key: str = typer.Option(..., "--raw-key", help="Raw API key (min 8 chars)."),
    scopes: str = typer.Option(..., "--scopes", help="Comma-separated scopes (run:write,run:read,usage:read,admin:write)."),
    project_ids: str = typer.Option(..., "--project-ids", help="Comma-separated project IDs (or * for all)."),
    created_by: str = typer.Option(..., "--created-by", help="User ID who creates this key."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Issue a scoped API key (teams admin)."""
    scopes_list = [s.strip() for s in scopes.split(",") if s.strip()]
    project_ids_list = [p.strip() for p in project_ids.split(",") if p.strip()]
    _run_safe(lambda: _admin_key_issue_impl(
        id, org_id, name, raw_key, scopes_list, project_ids_list, created_by,
        server, api_key, json_output,
    ))


def _admin_key_issue_impl(
    id: str, org_id: str, name: str, raw_key: str,
    scopes: list, project_ids: list, created_by: str,
    server: str, api_key: Optional[str], json_output: bool,
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.admin_issue_api_key(
        id=id, org_id=org_id, name=name, raw_key=raw_key,
        scopes=scopes, project_ids=project_ids, created_by=created_by,
    )
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp.model_dump(), indent=2))
    else:
        out.print(f"[bold green]Key issued[/bold green]: {resp.id}")
        out.print(f"  Org: {resp.org_id}")
        out.print(f"  Scopes: {', '.join(resp.scopes)}")
        out.print(f"  Projects: {', '.join(resp.project_ids)}")


# ── lineage ─────────────────────────────────────────────────────

@app.command(hidden=True)
def lineage(
    run_dir: str = typer.Option(
        ..., "--run", "-r", help="Path to run directory (e.g., runs/model_v1)."
    ),
    output: Optional[str] = typer.Option(
        None, "--out", "-o", help="Output directory for lineage artifacts."
    ),
    mermaid: bool = typer.Option(
        False, "--mermaid", help="Output Mermaid diagram to stdout."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output JSON graph to stdout."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show detailed output."
    ),
) -> None:
    """Visualize dataset lineage for a run.

    Shows the complete chain: raw data → transforms → dataset →
    contamination check → train → evaluation → decision.

    Example:
      verifily lineage --run runs/model_v1
      verifily lineage --run runs/model_v1 --mermaid > graph.mmd
      verifily lineage --run runs/model_v1 --json > graph.json
      verifily lineage --run runs/model_v1 --out ./lineage/
    """
    _run_safe(lambda: _lineage_impl(run_dir, output, mermaid, json_output, verbose))


def _lineage_impl(
    run_dir: str, output: Optional[str], mermaid: bool, json_output: bool, verbose: bool,
) -> None:
    from verifily_cli_v1.commands.lineage import run
    exit_code = run(
        run_dir=run_dir,
        output=output,
        mermaid=mermaid,
        json_output=json_output,
        verbose=verbose,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


# ── score ────────────────────────────────────────────────────────

@app.command(hidden=True)
def score(
    run_dir: str = typer.Option(
        ..., "--run", "-r", help="Path to run directory (e.g., runs/model_v1)."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output JSON to stdout."
    ),
    no_write: bool = typer.Option(
        False, "--no-write", help="Compute only, don't write artifacts."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show detailed output."
    ),
) -> None:
    """Compute Risk Score and Health Index for a run.

    Calculates quantitative assessments:
    - Dataset Risk Score (0-100, higher = riskier): Is dataset safe to train on?
    - Model Health Index (0-100, higher = healthier): Is model safe to ship?

    Example:
      verifily score --run runs/model_v1
      verifily score --run runs/model_v1 --json
      verifily score --run runs/model_v1 --no-write
    """
    _run_safe(lambda: _score_impl(run_dir, json_output, no_write, verbose))


def _score_impl(
    run_dir: str, json_output: bool, no_write: bool, verbose: bool,
) -> None:
    from verifily_cli_v1.commands.score import run
    exit_code = run(
        run_dir=run_dir,
        json_output=json_output,
        no_write=no_write,
        verbose=verbose,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


# ── registry ─────────────────────────────────────────────────────

@app.command(name="registry", hidden=True)
def registry_cmd(
    action: str = typer.Argument(
        ..., help="Action: register, promote, list, history"
    ),
    run_dir: Optional[str] = typer.Option(
        None, "--run", "-r", help="Run directory (for register action)."
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model ID (for promote/history actions)."
    ),
    version: Optional[str] = typer.Option(
        None, "--version", "-v", help="Version (for promote action)."
    ),
    stage: Optional[str] = typer.Option(
        None, "--stage", "-s", help="Stage filter (for list) or target stage (for promote)."
    ),
    reason: str = typer.Option(
        "", "--reason", help="Promotion reason."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output JSON to stdout."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show detailed output."
    ),
) -> None:
    """Model Registry - manage model versions and promotions.

    Register models from runs, promote through stages (none → staging → production),
    and track model history.

    Examples:
      verifily registry register --run runs/model_v1
      verifily registry promote fraud_detector v3 production
      verifily registry list --stage production
      verifily registry history fraud_detector
    """
    _run_safe(lambda: _registry_impl(
        action, run_dir, model_id, version, stage, reason, json_output, verbose
    ))


def _registry_impl(
    action: str,
    run_dir: Optional[str],
    model_id: Optional[str],
    version: Optional[str],
    stage: Optional[str],
    reason: str,
    json_output: bool,
    verbose: bool,
) -> None:
    from verifily_cli_v1.commands import registry_cmd as reg
    
    action = action.lower()
    
    if action == "register":
        if not run_dir:
            console.print("[red]Error:[/red] --run required for register action")
            raise SystemExit(1)
        exit_code = reg.cmd_register(
            run_dir=run_dir,
            model_id=model_id,
            version=version,
            json_output=json_output,
            verbose=verbose,
        )
    elif action == "promote":
        if not model_id or not version or not stage:
            console.print("[red]Error:[/red] --model, --version, and --stage required for promote")
            raise SystemExit(1)
        exit_code = reg.cmd_promote(
            model_id=model_id,
            version=version,
            target_stage=stage,
            reason=reason,
            json_output=json_output,
            verbose=verbose,
        )
    elif action == "list":
        exit_code = reg.cmd_list(
            stage=stage,
            model_id=model_id,
            json_output=json_output,
        )
    elif action == "history":
        if not model_id:
            console.print("[red]Error:[/red] --model required for history")
            raise SystemExit(1)
        exit_code = reg.cmd_history(
            model_id=model_id,
            json_output=json_output,
        )
    else:
        console.print(f"[red]Error:[/red] Unknown action: {action}")
        console.print("Valid actions: register, promote, list, history")
        raise SystemExit(1)
    
    if exit_code != 0:
        raise SystemExit(exit_code)


# ── budget ───────────────────────────────────────────────────────

@app.command(hidden=True)
def budget(
    action: str = typer.Argument(
        ..., help="Action: status, set-policy"
    ),
    project_id: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project ID (default: default)"
    ),
    daily_limit: Optional[int] = typer.Option(
        None, "--daily", help="Daily limit in units (for set-policy)"
    ),
    monthly_limit: Optional[int] = typer.Option(
        None, "--monthly", help="Monthly limit in units (for set-policy)"
    ),
    hard_block: bool = typer.Option(
        True, "--hard-block/--soft-block", help="Hard vs soft enforcement (for set-policy)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON (CI mode)."
    ),
) -> None:
    """Budget management: check status and configure limits.

    Examples:
      verifily budget status
      verifily budget status --project my-project
      verifily budget set-policy --project my-project --daily 1000 --monthly 10000
    """
    _run_safe(lambda: _budget_impl(
        action=action,
        project_id=project_id or "default",
        daily_limit=daily_limit,
        monthly_limit=monthly_limit,
        hard_block=hard_block,
        json_output=json_output,
    ))


def _budget_impl(
    action: str,
    project_id: str,
    daily_limit: Optional[int],
    monthly_limit: Optional[int],
    hard_block: bool,
    json_output: bool,
) -> None:
    from verifily_cli_v1.core.budget import BudgetPolicy, budget_store
    
    action = action.lower()
    
    if action == "status":
        status = budget_store.get_status(project_id)
        
        if json_output:
            import json
            print(json.dumps(status, indent=2))
        else:
            from rich.table import Table
            from rich.panel import Panel
            
            policy = status["policy"]
            usage = status["usage"]
            
            # Mode color
            mode = usage["mode"]
            mode_style = {
                "pass": "green",
                "warn": "yellow",
                "block": "red",
            }.get(mode, "white")
            
            table = Table(title=f"Budget Status: {project_id}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Usage", style="white")
            table.add_column("Status", style="white")
            
            daily_pct = usage["daily_used"] / policy["daily_limit_units"] * 100 if policy["daily_limit_units"] else 0
            monthly_pct = usage["monthly_used"] / policy["monthly_limit_units"] * 100 if policy["monthly_limit_units"] else 0
            
            table.add_row(
                "Daily Limit",
                str(policy["daily_limit_units"]),
                f"{usage['daily_used']}/{policy['daily_limit_units']} ({daily_pct:.0f}%)",
                "✓" if mode != "block" else "✗"
            )
            table.add_row(
                "Monthly Limit",
                str(policy["monthly_limit_units"]),
                f"{usage['monthly_used']}/{policy['monthly_limit_units']} ({monthly_pct:.0f}%)",
                "✓" if mode != "block" else "✗"
            )
            table.add_row(
                "Remaining",
                f"Daily: {usage['daily_remaining']}, Monthly: {usage['monthly_remaining']}",
                "",
                ""
            )
            table.add_row(
                "Policy",
                f"Hard Block: {policy['hard_block']}",
                "",
                ""
            )
            table.add_row(
                "Next Reset",
                status["next_reset"],
                f"({status['seconds_until_reset'] // 3600}h {(status['seconds_until_reset'] % 3600) // 60}m)",
                ""
            )
            
            console.print()
            console.print(Panel(table, border_style=mode_style, title=f"[bold {mode_style}]Mode: {mode.upper()}[/bold {mode_style}]"))
            console.print()
            
            if mode == "block":
                console.print("[red]⚠ Budget exceeded - requests will be blocked[/red]")
            elif mode == "warn":
                console.print("[yellow]⚠ Budget warning - approaching limits[/yellow]")
            else:
                console.print("[green]✓ Within budget limits[/green]")
    
    elif action == "set-policy":
        if daily_limit is None or monthly_limit is None:
            console.print("[red]Error:[/red] --daily and --monthly required for set-policy")
            raise SystemExit(1)
        
        policy = BudgetPolicy(
            project_id=project_id,
            daily_limit_units=daily_limit,
            monthly_limit_units=monthly_limit,
            hard_block=hard_block,
        )
        budget_store.set_policy(policy)
        
        if json_output:
            import json
            print(json.dumps({
                "success": True,
                "policy": {
                    "project_id": project_id,
                    "daily_limit_units": daily_limit,
                    "monthly_limit_units": monthly_limit,
                    "hard_block": hard_block,
                }
            }, indent=2))
        else:
            console.print(f"[green]✓ Budget policy set for {project_id}[/green]")
            console.print(f"  Daily limit: {daily_limit} units")
            console.print(f"  Monthly limit: {monthly_limit} units")
            console.print(f"  Hard block: {hard_block}")
    
    else:
        console.print(f"[red]Error:[/red] Unknown action: {action}")
        console.print("Valid actions: status, set-policy")
        raise SystemExit(1)


# ── org ──────────────────────────────────────────────────────────

org_app = typer.Typer(help="Multi-tenant org/project management")
app.add_typer(org_app, name="org")


@org_app.command("create")
def org_create(
    name: str = typer.Option(..., "--name", "-n", help="Organization name"),
    server: str = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for auth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a new organization (ADMIN only)."""
    _run_safe(lambda: _org_create_impl(name, server, api_key, json_output))


def _org_create_impl(name: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    from verifily_sdk import VerifilyClient
    
    client = VerifilyClient(base_url=server, api_key=api_key)
    try:
        result = client.create_org(name=name)
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓ Created organization[/green]")
            console.print(f"  org_id: {result['org_id']}")
            console.print(f"  name: {result['name']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    finally:
        client.close()


# ── project ──────────────────────────────────────────────────────

project_app = typer.Typer(help="Project management")
app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    org_id: str = typer.Option(..., "--org", "-o", help="Organization ID"),
    name: str = typer.Option(..., "--name", "-n", help="Project name"),
    server: str = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for auth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a new project (ADMIN only)."""
    _run_safe(lambda: _project_create_impl(org_id, name, server, api_key, json_output))


def _project_create_impl(org_id: str, name: str, server: str, api_key: Optional[str], json_output: bool) -> None:
    from verifily_sdk import VerifilyClient
    
    client = VerifilyClient(base_url=server, api_key=api_key)
    try:
        result = client.create_project(org_id=org_id, name=name)
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓ Created project[/green]")
            console.print(f"  project_id: {result['project_id']}")
            console.print(f"  org_id: {result['org_id']}")
            console.print(f"  name: {result['name']}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    finally:
        client.close()


@project_app.command("list")
def project_list(
    org_id: Optional[str] = typer.Option(None, "--org", "-o", help="Filter by org"),
    server: str = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for auth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List projects (ADMIN, DEV, VIEWER)."""
    _run_safe(lambda: _project_list_impl(org_id, server, api_key, json_output))


def _project_list_impl(org_id: Optional[str], server: str, api_key: Optional[str], json_output: bool) -> None:
    from verifily_sdk import VerifilyClient
    
    client = VerifilyClient(base_url=server, api_key=api_key)
    try:
        result = client.list_projects(org_id=org_id)
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            from rich.table import Table
            table = Table(title="Projects")
            table.add_column("project_id", style="cyan")
            table.add_column("org_id", style="magenta")
            table.add_column("name", style="green")
            
            for p in result.get("projects", []):
                table.add_row(p["project_id"], p["org_id"], p["name"])
            
            console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    finally:
        client.close()


# ── key ──────────────────────────────────────────────────────────

key_app = typer.Typer(help="API key management")
app.add_typer(key_app, name="key")


@key_app.command("create")
def key_create(
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID"),
    role: str = typer.Option(..., "--role", "-r", help="Role: admin, dev, viewer"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Key label"),
    server: str = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Admin API key for auth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a new API key (ADMIN only).
    
    WARNING: The secret is displayed ONCE. Save it immediately!
    """
    _run_safe(lambda: _key_create_impl(project_id, role, label, server, api_key, json_output))


def _key_create_impl(
    project_id: str,
    role: str,
    label: Optional[str],
    server: str,
    api_key: Optional[str],
    json_output: bool,
) -> None:
    from verifily_sdk import VerifilyClient
    
    client = VerifilyClient(base_url=server, api_key=api_key)
    try:
        result = client.create_key(project_id=project_id, role=role, label=label)
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓ Created API key[/green]")
            console.print(f"  key_id: {result['key_id']}")
            console.print(f"  role: {result['role']}")
            console.print(f"  project_id: {result['project_id']}")
            console.print(f"  [bold red]secret: {result['secret']}[/bold red]")
            console.print()
            console.print("[yellow]⚠ Save this secret now - it will not be shown again![/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    finally:
        client.close()


@key_app.command("revoke")
def key_revoke(
    key_id: str = typer.Argument(..., help="Key ID to revoke"),
    server: str = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Admin API key for auth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Revoke an API key (ADMIN only)."""
    _run_safe(lambda: _key_revoke_impl(key_id, server, api_key, json_output))


def _key_revoke_impl(
    key_id: str,
    server: str,
    api_key: Optional[str],
    json_output: bool,
) -> None:
    from verifily_sdk import VerifilyClient
    
    client = VerifilyClient(base_url=server, api_key=api_key)
    try:
        result = client.revoke_key(key_id=key_id)
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            if result.get("was_active"):
                console.print(f"[green]✓ Revoked key {key_id}[/green]")
            else:
                console.print(f"[yellow]Key {key_id} was already revoked[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    finally:
        client.close()


@key_app.command("list")
def key_list(
    project_id: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
    server: str = typer.Option("http://127.0.0.1:8000", "--server", "-s", help="Server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Admin API key for auth"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List API keys (ADMIN only)."""
    _run_safe(lambda: _key_list_impl(project_id, server, api_key, json_output))


def _key_list_impl(
    project_id: Optional[str],
    server: str,
    api_key: Optional[str],
    json_output: bool,
) -> None:
    from verifily_sdk import VerifilyClient
    
    client = VerifilyClient(base_url=server, api_key=api_key)
    try:
        result = client.list_keys(project_id=project_id)
        if json_output:
            import json
            print(json.dumps(result, indent=2))
        else:
            from rich.table import Table
            table = Table(title="API Keys")
            table.add_column("key_id", style="cyan")
            table.add_column("role", style="magenta")
            table.add_column("project_id", style="green")
            table.add_column("is_active", style="yellow")
            table.add_column("label", style="dim")
            
            for k in result.get("keys", []):
                table.add_row(
                    k["key_id"][:16] + "...",
                    k["role"],
                    k["project_id"],
                    "✓" if k["is_active"] else "✗",
                    k.get("label", "-") or "-",
                )
            
            console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    finally:
        client.close()


# ── Workspaces ───────────────────────────────────────────────────


@ws_app.command(name="org-create")
def ws_org_create_cmd(
    name: str = typer.Option(..., "--name", help="Organization name."),
    bootstrap_token: Optional[str] = typer.Option(
        None, "--bootstrap-token", help="Bootstrap token for first org."
    ),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create an organization (workspaces mode)."""
    _run_safe(lambda: _ws_org_create_impl(name, bootstrap_token, server, api_key, json_output))


def _ws_org_create_impl(
    name: str, bootstrap_token: Optional[str], server: str, api_key: Optional[str], json_output: bool
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.ws_create_org(name=name, bootstrap_token=bootstrap_token)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp, indent=2))
    else:
        out.print(f"[bold green]Org created[/bold green]: {resp['org_id']} ({resp['name']})")


@ws_app.command(name="project-create")
def ws_project_create_cmd(
    org: str = typer.Option(..., "--org", help="Organization ID."),
    name: str = typer.Option(..., "--name", help="Project name."),
    plan: str = typer.Option("free", "--plan", help="Billing plan."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a project within an org (workspaces mode)."""
    _run_safe(lambda: _ws_project_create_impl(org, name, plan, server, api_key, json_output))


def _ws_project_create_impl(
    org_id: str, name: str, plan: str, server: str, api_key: Optional[str], json_output: bool
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.ws_create_project(org_id=org_id, name=name, billing_plan=plan)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp, indent=2))
    else:
        out.print(f"[bold green]Project created[/bold green]: {resp['project_id']} ({resp['name']})")


@ws_app.command(name="key-create")
def ws_key_create_cmd(
    project: str = typer.Option(..., "--project", help="Project ID."),
    role: str = typer.Option(..., "--role", help="Role: admin, editor, or viewer."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create an API key for a project (workspaces mode)."""
    _run_safe(lambda: _ws_key_create_impl(project, role, server, api_key, json_output))


def _ws_key_create_impl(
    project_id: str, role: str, server: str, api_key: Optional[str], json_output: bool
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.ws_create_key(project_id=project_id, role=role)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp, indent=2))
    else:
        out.print(f"[bold green]Key created[/bold green]: {resp['api_key_id']} (role={resp['role']})")
        out.print(f"[bold yellow]API Key (save now, shown once):[/bold yellow] {resp['api_key']}")


@ws_app.command(name="key-revoke")
def ws_key_revoke_cmd(
    project: str = typer.Option(..., "--project", help="Project ID."),
    key_id: str = typer.Option(..., "--key-id", help="API key ID to revoke."),
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Revoke an API key (workspaces mode)."""
    _run_safe(lambda: _ws_key_revoke_impl(project, key_id, server, api_key, json_output))


def _ws_key_revoke_impl(
    project_id: str, key_id: str, server: str, api_key: Optional[str], json_output: bool
) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.ws_revoke_key(project_id=project_id, api_key_id=key_id)
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp, indent=2))
    else:
        out.print(f"[bold green]Key revoked[/bold green]: ok={resp['ok']}")


@ws_app.command(name="me")
def ws_me_cmd(
    server: str = typer.Option(
        "http://127.0.0.1:8000", "--server", "-s", help="Verifily API server URL."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key.", envvar="VERIFILY_API_KEY",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show identity of current key (workspaces mode)."""
    _run_safe(lambda: _ws_me_impl(server, api_key, json_output))


def _ws_me_impl(server: str, api_key: Optional[str], json_output: bool) -> None:
    import json as json_mod
    from verifily_sdk import VerifilyClient

    client = VerifilyClient(base_url=server, api_key=api_key)
    resp = client.ws_me()
    out = Console()
    if json_output:
        out.print_json(json_mod.dumps(resp, indent=2))
    else:
        out.print(f"[bold]Org:[/bold]     {resp['org_id']}")
        out.print(f"[bold]Project:[/bold] {resp['project_id']}")
        out.print(f"[bold]Role:[/bold]    {resp['role']}")
        out.print(f"[bold]Key ID:[/bold]  {resp['api_key_id']}")


# ── backup ───────────────────────────────────────────────────────

@app.command(hidden=True)
def backup(
    output: str = typer.Option(..., "--out", "-o", help="Output path for backup archive (.tar.gz)"),
    include_logs: bool = typer.Option(False, "--include-logs", help="Include log files in backup"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create a backup of operational metadata.
    
    Backs up usage events, jobs events, monitor events, and workspaces store.
    Does NOT include raw datasets or run artifacts.
    
    Example:
      verifily backup --out /backups/verifily_$(date +%Y%m%d).tar.gz
    """
    _run_safe(lambda: _backup_impl(output, include_logs, json_output))


def _backup_impl(output: str, include_logs: bool, json_output: bool) -> None:
    from pathlib import Path
    from verifily_cli_v1.core.backup_restore import create_backup
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = create_backup(output_path, include_logs=include_logs)
    
    if json_output:
        import json
        print(json.dumps(result, indent=2))
    else:
        console.print(f"[green]✓ Backup created[/green]")
        console.print(f"  Path: {result['output_path']}")
        console.print(f"  Files: {result['files_backed_up']}")
        console.print(f"  Size: {result['total_bytes']:,} bytes (compressed: {result['backup_bytes']:,})")


# ── restore ──────────────────────────────────────────────────────

@app.command(hidden=True)
def restore(
    file: str = typer.Option(..., "--file", "-f", help="Backup archive path (.tar.gz)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Restore from a backup archive.
    
    Requires --force if files already exist.
    
    Example:
      verifily restore --file /backups/verifily_20240115.tar.gz --force
    """
    _run_safe(lambda: _restore_impl(file, force, json_output))


def _restore_impl(file: str, force: bool, json_output: bool) -> None:
    from pathlib import Path
    from verifily_cli_v1.core.backup_restore import restore_backup
    
    backup_path = Path(file)
    if not backup_path.exists():
        console.print(f"[red]Error:[/red] Backup file not found: {backup_path}")
        raise SystemExit(1)
    
    result = restore_backup(backup_path, force=force)
    
    if json_output:
        import json
        print(json.dumps(result, indent=2))
    else:
        console.print(f"[green]✓ Restore completed[/green]")
        console.print(f"  Files restored: {result['files_restored']}")
        console.print(f"  Report: {result['restore_report']}")


# ── verify-run ───────────────────────────────────────────────────

@app.command(hidden=True)
def verify_run(
    run: str = typer.Option(..., "--run", "-r", help="Path to run directory"),
    integrity: bool = typer.Option(True, "--integrity/--no-integrity", help="Verify hash chain"),
    signature: bool = typer.Option(True, "--signature/--no-signature", help="Verify signature"),
    audit: bool = typer.Option(True, "--audit/--no-audit", help="Run redaction audit"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Verify a run's integrity, signature, and redaction compliance.
    
    Performs comprehensive verification:
    - Hash chain verification (detect tampering)
    - Signature verification (if signed)
    - PII/secrets audit (redaction compliance)
    
    Example:
      verifily verify-run --run ./runs/run_20250120_120000
      verifily verify-run --run ./runs/run_001 --json
    """
    _run_safe(lambda: _verify_run_impl(run, integrity, signature, audit, json_output))


def _verify_run_impl(
    run: str,
    check_integrity: bool,
    check_signature: bool,
    check_audit: bool,
    json_output: bool,
) -> None:
    from pathlib import Path
    from verifily_cli_v1.core.integrity import verify_hash_chain, detect_artifact_tampering
    from verifily_cli_v1.core.signed_decisions import verify_decision_signature, has_signature
    from verifily_cli_v1.core.redaction_audit import audit_run_directory
    
    run_path = Path(run)
    if not run_path.exists():
        console.print(f"[red]Error:[/red] Run directory not found: {run_path}")
        raise SystemExit(1)
    
    results = {
        "run_id": run_path.name,
        "path": str(run_path),
        "checks": {},
        "overall_status": "PASS",
    }
    
    # Integrity check
    if check_integrity:
        try:
            chain_result = verify_hash_chain(run_path)
            tampering = detect_artifact_tampering(run_path)
            
            results["checks"]["integrity"] = {
                "status": chain_result["status"],
                "tampering_detected": tampering["tampering_detected"],
                "confidence": tampering.get("confidence", "UNKNOWN"),
                "errors": chain_result.get("errors", []),
            }
            
            if chain_result["status"] == "TAMPERED" or tampering["tampering_detected"]:
                results["overall_status"] = "FAIL"
            elif chain_result["status"] != "VERIFIED":
                results["overall_status"] = "WARN"
                
        except Exception as e:
            results["checks"]["integrity"] = {"status": "ERROR", "error": str(e)}
            results["overall_status"] = "FAIL"
    
    # Signature check
    if check_signature:
        if has_signature(run_path):
            try:
                sig_result = verify_decision_signature(run_path / "decision.json")
                results["checks"]["signature"] = {
                    "present": True,
                    "valid": sig_result["valid"],
                    "algorithm": sig_result.get("algorithm", "unknown"),
                    "key_id": sig_result.get("key_id"),
                }
                if not sig_result["valid"]:
                    results["overall_status"] = "FAIL"
            except Exception as e:
                results["checks"]["signature"] = {"present": True, "valid": False, "error": str(e)}
                results["overall_status"] = "FAIL"
        else:
            results["checks"]["signature"] = {"present": False, "valid": None}
    
    # Redaction audit
    if check_audit:
        try:
            audit_report = audit_run_directory(run_path)
            results["checks"]["redaction"] = {
                "status": audit_report.status,
                "findings_count": len(audit_report.findings),
                "findings_by_type": audit_report.summary.get("findings_by_type", {}),
            }
            if audit_report.status == "FAIL":
                results["overall_status"] = "FAIL"
            elif audit_report.status == "WARN" and results["overall_status"] == "PASS":
                results["overall_status"] = "WARN"
        except Exception as e:
            results["checks"]["redaction"] = {"status": "ERROR", "error": str(e)}
    
    # Output
    if json_output:
        import json
        print(json.dumps(results, indent=2))
    else:
        status_color = {
            "PASS": "green",
            "WARN": "yellow",
            "FAIL": "red",
        }.get(results["overall_status"], "white")
        
        console.print(f"\n[bold]Verification Results for {run_path.name}[/bold]")
        console.print(f"Overall Status: [{status_color}]{results['overall_status']}[/{status_color}]")
        console.print()
        
        # Integrity
        if "integrity" in results["checks"]:
            ic = results["checks"]["integrity"]
            ic_status = ic.get("status", "UNKNOWN")
            ic_color = "green" if ic_status == "VERIFIED" else "red" if ic_status in ("TAMPERED", "ERROR") else "yellow"
            console.print(f"[bold]Integrity:[/bold] [{ic_color}]{ic_status}[/{ic_color}]")
            if ic.get("tampering_detected"):
                console.print(f"  [red]⚠ Tampering detected (confidence: {ic.get('confidence', 'UNKNOWN')})[/red]")
            if ic.get("errors"):
                for error in ic["errors"]:
                    console.print(f"  [red]  • {error}[/red]")
        
        # Signature
        if "signature" in results["checks"]:
            sc = results["checks"]["signature"]
            if sc["present"]:
                sig_status = "VALID" if sc["valid"] else "INVALID"
                sig_color = "green" if sc["valid"] else "red"
                console.print(f"[bold]Signature:[/bold] [{sig_color}]{sig_status}[/{sig_color}] ({sc.get('algorithm', 'unknown')})")
            else:
                console.print("[bold]Signature:[/bold] [dim]Not present[/dim]")
        
        # Redaction
        if "redaction" in results["checks"]:
            rc = results["checks"]["redaction"]
            rc_status = rc.get("status", "UNKNOWN")
            rc_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(rc_status, "white")
            console.print(f"[bold]Redaction:[/bold] [{rc_color}]{rc_status}[/{rc_color}] ({rc['findings_count']} findings)")
        
        console.print()
    
    # Exit code
    if results["overall_status"] == "FAIL":
        raise SystemExit(1)
    elif results["overall_status"] == "WARN":
        raise SystemExit(2)


# ── drift ───────────────────────────────────────────────────────

@app.command(hidden=True)
def drift(
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline dataset or artifact directory"),
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate dataset or artifact directory"),
    warn_sim: float = typer.Option(0.65, "--warn-sim", help="Similarity threshold for WARN"),
    fail_sim: float = typer.Option(0.45, "--fail-sim", help="Similarity threshold for FAIL"),
    warn_tag: float = typer.Option(0.15, "--warn-tag", help="Tag shift threshold for WARN"),
    fail_tag: float = typer.Option(0.30, "--fail-tag", help="Tag shift threshold for FAIL"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Detect dataset drift between baseline and candidate.
    
    Compares dataset distributions using MinHash similarity and tag analysis.
    Supports both direct JSONL paths and artifact directories.
    
    Drift detection helps identify when your dataset has changed meaningfully
    since the last production deployment, enabling proactive retraining decisions.
    
    Exit codes:
      0 = PASS (no significant drift)
      2 = WARN (moderate drift detected)
      1 = FAIL (significant drift detected)
    
    Examples:
      verifily drift --baseline datasets/production/ --candidate datasets/candidate/
      verifily drift --baseline data/train_v1.jsonl --candidate data/train_v2.jsonl --json
      verifily drift --baseline runs/run_001/ --candidate runs/run_002/ --fail-sim 0.40
    """
    _run_safe(lambda: _drift_impl(baseline, candidate, warn_sim, fail_sim, warn_tag, fail_tag, json_output))


def _drift_impl(
    baseline: str,
    candidate: str,
    warn_sim: float,
    fail_sim: float,
    warn_tag: float,
    fail_tag: float,
    json_output: bool,
) -> None:
    from pathlib import Path
    from verifily_cli_v1.core.drift import detect_drift, format_drift_report, DriftStatus
    
    baseline_path = Path(baseline)
    candidate_path = Path(candidate)
    
    if not baseline_path.exists():
        console.print(f"[red]Error:[/red] Baseline not found: {baseline_path}")
        raise SystemExit(1)
    
    if not candidate_path.exists():
        console.print(f"[red]Error:[/red] Candidate not found: {candidate_path}")
        raise SystemExit(1)
    
    try:
        with console.status("[bold green]Detecting drift..."):
            result = detect_drift(
                baseline_dataset=baseline_path,
                candidate_dataset=candidate_path,
                min_similarity_warn=warn_sim,
                min_similarity_fail=fail_sim,
                max_tag_shift_warn=warn_tag,
                max_tag_shift_fail=fail_tag,
            )
    except Exception as e:
        console.print(f"[red]Error:[/red] Drift detection failed: {e}")
        raise SystemExit(1)
    
    # Output
    if json_output:
        print(result.to_json())
    else:
        console.print()
        console.print(format_drift_report(result))
        console.print()
        
        # Status-specific message
        if result.status == DriftStatus.PASS:
            console.print("[green]✓ No significant drift detected[/green]")
        elif result.status == DriftStatus.WARN:
            console.print("[yellow]⚠ Moderate drift detected - review recommended[/yellow]")
        else:
            console.print("[red]✗ Significant drift detected - retraining may be needed[/red]")
        
        console.print()
    
    # Exit code
    if result.status == DriftStatus.FAIL:
        raise SystemExit(1)
    elif result.status == DriftStatus.WARN:
        raise SystemExit(2)


# ── ready ───────────────────────────────────────────────────────

@app.command(hidden=True)
def ready(
    run: str = typer.Option(..., "--run", "-r", help="Path to run directory"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check if a run is ready for production deployment.
    
    Validates production readiness:
    - Contract integrity (required files present)
    - Hash chain validity (no tampering)
    - Privacy safety (redaction audit)
    - Contamination risk (train/eval overlap)
    - Regression risk (metric degradation)
    - Config sanity (reasonable thresholds)
    - Reproducibility (environment captured)
    
    Exit codes:
      0 = READY (all checks pass)
      1 = NOT READY (failures or blocking warnings)
    
    Example:
      verifily ready --run ./runs/run_001
      verifily ready --run ./runs/run_001 --json
    """
    _run_safe(lambda: _ready_impl(run, json_output))


def _ready_impl(run: str, json_output: bool) -> None:
    from pathlib import Path
    from verifily_cli_v1.core.readiness import validate_readiness, format_readiness_report, ReadinessStatus
    
    run_path = Path(run)
    if not run_path.exists():
        console.print(f"[red]Error:[/red] Run directory not found: {run_path}")
        raise SystemExit(1)
    
    try:
        report = validate_readiness(run_path)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to validate readiness: {e}")
        raise SystemExit(1)
    
    # Output
    if json_output:
        print(report.to_json())
    else:
        console.print()
        console.print(format_readiness_report(report))
        console.print()
        
        # Overall status message
        if report.overall_status == ReadinessStatus.PASS:
            console.print("[green bold]✓ This run is READY for production deployment[/green bold]")
        elif report.overall_status == ReadinessStatus.WARN:
            console.print("[yellow bold]⚠ This run has warnings - review before deployment[/yellow bold]")
        else:
            console.print("[red bold]✗ This run is NOT READY for production[/red bold]")
        
        console.print()
    
    # Exit code
    if report.overall_status == ReadinessStatus.FAIL:
        raise SystemExit(1)
    elif report.overall_status == ReadinessStatus.WARN:
        # Warnings don't block by default, but could with --strict flag
        pass


# ── benchmark ────────────────────────────────────────────────────

@app.command(hidden=True)
def benchmark(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset path for ingest benchmark"),
    train: Optional[str] = typer.Option(None, "--train", "-t", help="Training dataset for contamination benchmark"),
    eval: Optional[str] = typer.Option(None, "--eval", "-e", help="Eval dataset for contamination benchmark"),
    pipeline_run: Optional[str] = typer.Option(None, "--pipeline", "-p", help="Run directory for pipeline benchmark"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of runs for averaging"),
) -> None:
    """Benchmark Verifily performance.
    
    Measures performance for ingest, contamination detection, and pipeline operations.
    Outputs rows/sec, elapsed time, and memory usage.
    
    Examples:
      verifily benchmark --dataset data/train.jsonl
      verifily benchmark --dataset data/train.jsonl --runs 5
      verifily benchmark --train data/train.jsonl --eval data/eval.jsonl
      verifily benchmark --pipeline ./runs/run_001
    """
    _run_safe(lambda: _benchmark_impl(dataset, train, eval, pipeline_run, json_output, runs))


def _benchmark_impl(
    dataset: Optional[str],
    train: Optional[str],
    eval: Optional[str],
    pipeline_run: Optional[str],
    json_output: bool,
    runs: int,
) -> None:
    from pathlib import Path
    from verifily_cli_v1.core.benchmark import (
        benchmark_ingest,
        benchmark_contamination,
        benchmark_pipeline,
        format_benchmark_table,
        BenchmarkReport,
    )
    
    report = BenchmarkReport()
    has_benchmark = False
    
    # Ingest benchmark
    if dataset:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            console.print(f"[red]Error:[/red] Dataset not found: {dataset_path}")
            raise SystemExit(1)
        
        try:
            with console.status("[bold green]Running ingest benchmark..."):
                result = benchmark_ingest(dataset_path, runs=runs)
            report.add_result(result)
            has_benchmark = True
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Ingest benchmark failed: {e}")
    
    # Contamination benchmark
    if train and eval:
        train_path = Path(train)
        eval_path = Path(eval)
        
        if not train_path.exists():
            console.print(f"[red]Error:[/red] Train dataset not found: {train_path}")
            raise SystemExit(1)
        if not eval_path.exists():
            console.print(f"[red]Error:[/red] Eval dataset not found: {eval_path}")
            raise SystemExit(1)
        
        try:
            with console.status("[bold green]Running contamination benchmark..."):
                result = benchmark_contamination(train_path, eval_path, runs=1)
            report.add_result(result)
            has_benchmark = True
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Contamination benchmark failed: {e}")
    
    # Pipeline benchmark
    if pipeline_run:
        run_path = Path(pipeline_run)
        if not run_path.exists():
            console.print(f"[red]Error:[/red] Run directory not found: {run_path}")
            raise SystemExit(1)
        
        try:
            with console.status("[bold green]Running pipeline benchmark..."):
                result = benchmark_pipeline(run_path, plan=True, runs=1)
            report.add_result(result)
            has_benchmark = True
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Pipeline benchmark failed: {e}")
    
    if not has_benchmark:
        console.print("[yellow]Warning:[/yellow] No benchmarks specified. Use --dataset, --train/--eval, or --pipeline")
        raise SystemExit(1)
    
    # Calculate summary
    if report.results:
        total_rows = sum(r.rows for r in report.results)
        total_time = sum(r.elapsed_ms for r in report.results)
        avg_rows_per_sec = sum(r.rows_per_sec for r in report.results) / len(report.results)
        
        report.summary = {
            "total_benchmarks": len(report.results),
            "total_rows_processed": total_rows,
            "total_time_ms": round(total_time, 2),
            "avg_rows_per_sec": round(avg_rows_per_sec, 2),
        }
    
    # Output
    if json_output:
        import json
        print(report.to_json())
    else:
        console.print()
        console.print(format_benchmark_table(report))
        
        # Performance assessment
        avg_rps = report.summary.get("avg_rows_per_sec", 0)
        if avg_rps > 10000:
            console.print("\n[green]✓ Excellent performance[/green] (>10K rows/sec)")
        elif avg_rps > 5000:
            console.print("\n[green]✓ Good performance[/green] (>5K rows/sec)")
        elif avg_rps > 1000:
            console.print("\n[yellow]⚠ Moderate performance[/yellow] (>1K rows/sec)")
        else:
            console.print("\n[red]⚠ Low performance[/red] (<1K rows/sec)")
        
        console.print("\n[dim]Note: Verifily prioritizes determinism and accuracy over raw speed.[/dim]")
        console.print()


# ── nl2sql ────────────────────────────────────────────────────────

nl2sql_app = typer.Typer(help="NL2SQL dataset tools: validate, fingerprint, split, gate.")
app.add_typer(nl2sql_app, name="nl2sql")


@nl2sql_app.command("validate")
def nl2sql_validate(
    input_path: str = typer.Option(..., "--in", help="Path to NL2SQL JSONL dataset."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write JSON results to file."),
    strict: bool = typer.Option(False, "--strict", help="Fail on first invalid row."),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Validate NL2SQL dataset rows for required fields.

    Checks that each row has: question, sql, and schema/schema_ref.

    Example:
      verifily nl2sql validate --in data/nl2sql.jsonl
    """
    def _impl():
        from verifily_cli_v1.commands.nl2sql import run_validate
        result = run_validate(input_path=input_path, output=output, strict=strict, verbose=verbose)
        raise SystemExit(0 if result["status"] == "PASS" else 1)
    _run_safe(_impl, verbose=verbose)


@nl2sql_app.command("fingerprint")
def nl2sql_fingerprint(
    input_path: str = typer.Option(..., "--in", help="Path to NL2SQL JSONL dataset."),
    out_dir: str = typer.Option(..., "--out", help="Output directory for enriched dataset."),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Compute SQL fingerprints and templates for each row.

    Enriches rows with sql_fingerprint, sql_template, template_fingerprint.

    Example:
      verifily nl2sql fingerprint --in data/nl2sql.jsonl --out data/nl2sql_fp
    """
    def _impl():
        from verifily_cli_v1.commands.nl2sql import run_fingerprint
        run_fingerprint(input_path=input_path, out_dir=out_dir, verbose=verbose)
    _run_safe(_impl, verbose=verbose)


@nl2sql_app.command("split")
def nl2sql_split(
    input_path: str = typer.Option(..., "--in", help="Path to NL2SQL JSONL dataset."),
    out_dir: str = typer.Option(..., "--out-dir", help="Output directory for train/eval split."),
    eval_ratio: float = typer.Option(0.1, "--eval-ratio", help="Fraction of groups for eval."),
    group: str = typer.Option("template", "--group", help="Grouping key: template, db_id, or schema_ref."),
    seed: int = typer.Option(42, "--seed", help="Random seed for deterministic split."),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """Leakage-resistant train/eval split grouped by SQL template.

    Groups rows by template_fingerprint (or db_id/schema_ref) so no
    structural duplicates leak across the split boundary.

    Example:
      verifily nl2sql split --in dataset.jsonl --out-dir splits/ --eval-ratio 0.1
      verifily nl2sql split --in dataset.jsonl --out-dir splits/ --group db_id
    """
    def _impl():
        from verifily_cli_v1.commands.nl2sql import run_split
        run_split(input_path=input_path, out_dir=out_dir, eval_ratio=eval_ratio,
                  group=group, seed=seed, verbose=verbose)
    _run_safe(_impl, verbose=verbose)


@nl2sql_app.command("gate")
def nl2sql_gate(
    train_path: str = typer.Option(..., "--train", help="Path to training JSONL dataset."),
    eval_path: str = typer.Option(..., "--eval", help="Path to eval JSONL dataset."),
    jaccard_cutoff: float = typer.Option(0.70, "--jaccard", help="Jaccard cutoff for near-dup questions."),
    num_perm: int = typer.Option(128, "--num-perm", help="MinHash permutations for LSH."),
    no_lsh: bool = typer.Option(False, "--no-lsh", help="Disable LSH, use brute-force."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write JSON results to file."),
    verbose: bool = typer.Option(False, "--verbose", help="Show detailed output."),
) -> None:
    """NL2SQL-specific contamination gate.

    Three-tier leakage check:
      1. Exact SQL overlap (by sql_fingerprint)
      2. Template overlap (by template_fingerprint)
      3. Near-duplicate question overlap (n-gram Jaccard)

    Exit codes: 0=PASS, 1=FAIL (exact leaks), 2=WARN (template/near-dup).

    Example:
      verifily nl2sql gate --train train.jsonl --eval eval.jsonl
    """
    def _impl():
        from verifily_cli_v1.commands.nl2sql import run_gate
        result = run_gate(train=train_path, eval_set=eval_path, jaccard_cutoff=jaccard_cutoff,
                         num_perm=num_perm, use_lsh=not no_lsh, output=output, verbose=verbose)
        raise SystemExit(result["exit_code"])
    _run_safe(_impl, verbose=verbose)


# ── Error handling ───────────────────────────────────────────────

EXIT_TOOL_ERROR = 4

def _run_safe(fn, verbose: bool = False) -> None:
    """Run a function with clean error handling."""
    try:
        fn()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise SystemExit(130)
    except Exception as e:
        if verbose:
            console.print(f"\n[red bold]Error:[/red bold] {e}")
            console.print(traceback.format_exc())
        else:
            console.print(f"\n[red bold]Error:[/red bold] {e}")
            console.print("[dim]Run with --verbose for full traceback.[/dim]")
        raise SystemExit(1)
