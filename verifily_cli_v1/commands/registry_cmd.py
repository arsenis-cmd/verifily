"""CLI commands for the Model Registry.

Usage:
    verifily registry register --run runs/model_x
    verifily registry promote fraud_detector v3 production
    verifily registry list --stage production
    verifily registry history fraud_detector
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.registry import (
    ModelStage,
    PromotionError,
    RegistrationError,
    registry_store,
    configure_registry_persistence,
)

console = Console(stderr=True)


def cmd_register(
    run_dir: str,
    *,
    model_id: Optional[str] = None,
    version: Optional[str] = None,
    json_output: bool = False,
    verbose: bool = False,
) -> int:
    """Register a model from a run directory.
    
    Args:
        run_dir: Path to run directory
        model_id: Optional model ID (auto-detected if not provided)
        version: Optional version (auto-generated if not provided)
        json_output: If True, output JSON to stdout
        verbose: If True, show detailed output
        
    Returns:
        Exit code (0 = success, 1 = error)
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        console.print(f"[red]Error:[/red] Run directory not found: {run_path}")
        return 1
    
    # Configure persistence if enabled
    if os.environ.get("VERIFILY_REGISTRY_PERSIST") == "1":
        data_dir = os.environ.get("VERIFILY_DATA_DIR", ".verifily")
        persist_path = Path(data_dir) / "registry.jsonl"
        configure_registry_persistence(persist_path)
    
    try:
        from verifily_cli_v1.core.registry import RegistrationRequest
        
        request = RegistrationRequest(
            run_dir=str(run_path),
            model_id=model_id,
            version=version,
            registered_by="cli",
        )
        
        record = registry_store.register(request)
        
        if json_output:
            import json
            sys.stdout.write(json.dumps(record.to_dict(), indent=2))
            sys.stdout.write("\n")
            return 0
        
        # Rich output
        console.print()
        console.print(Panel.fit(
            f"[bold green]Model Registered[/bold green]\n\n"
            f"Model ID: [cyan]{record.model_id}[/cyan]\n"
            f"Version: [cyan]{record.version}[/cyan]\n"
            f"Stage: [yellow]{record.stage}[/yellow]\n"
            f"Decision: [green]{record.decision}[/green]\n"
            f"Health: {record.health_index:.1f} | Risk: {record.risk_score:.1f}",
            title="Registry",
            border_style="green",
        ))
        
        if record.metrics:
            console.print()
            console.print("[bold]Metrics:[/bold]")
            for k, v in record.metrics.items():
                console.print(f"  {k}: {v:.3f}")
        
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print(f"  verifily registry promote {record.model_id} {record.version} staging")
        console.print()
        
        return 0
        
    except RegistrationError as e:
        console.print(f"[red]Registration failed:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def cmd_promote(
    model_id: str,
    version: str,
    target_stage: str,
    *,
    reason: str = "",
    json_output: bool = False,
    verbose: bool = False,
) -> int:
    """Promote a model to a new stage.
    
    Args:
        model_id: Model identifier
        version: Model version
        target_stage: Target stage (staging/production/archived)
        reason: Promotion reason
        json_output: If True, output JSON to stdout
        verbose: If True, show detailed output
        
    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Configure persistence if enabled
    if os.environ.get("VERIFILY_REGISTRY_PERSIST") == "1":
        data_dir = os.environ.get("VERIFILY_DATA_DIR", ".verifily")
        persist_path = Path(data_dir) / "registry.jsonl"
        configure_registry_persistence(persist_path)
    
    # Validate stage
    valid_stages = [s.value for s in ModelStage]
    if target_stage not in valid_stages:
        console.print(f"[red]Error:[/red] Invalid stage '{target_stage}'")
        console.print(f"Valid stages: {', '.join(valid_stages)}")
        return 1
    
    try:
        from verifily_cli_v1.core.registry import PromotionRequest
        
        request = PromotionRequest(
            model_id=model_id,
            version=version,
            target_stage=target_stage,
            promoted_by="cli",
            reason=reason,
        )
        
        record = registry_store.promote(request)
        
        if json_output:
            import json
            sys.stdout.write(json.dumps(record.to_dict(), indent=2))
            sys.stdout.write("\n")
            return 0
        
        # Rich output
        console.print()
        
        # Color based on stage
        stage_color = {
            ModelStage.PRODUCTION.value: "green",
            ModelStage.STAGING.value: "yellow",
            ModelStage.ARCHIVED.value: "dim",
        }.get(target_stage, "white")
        
        console.print(Panel.fit(
            f"[bold]Model Promoted[/bold]\n\n"
            f"Model ID: [cyan]{record.model_id}[/cyan]\n"
            f"Version: [cyan]{record.version}[/cyan]\n"
            f"Stage: [{stage_color}]{record.stage}[/{stage_color}]\n"
            f"Health: {record.health_index:.1f} | Risk: {record.risk_score:.1f}",
            title="Registry",
            border_style="blue",
        ))
        
        # Show promotion history
        if record.promotion_history:
            console.print()
            console.print("[bold]Promotion History:[/bold]")
            for h in record.promotion_history[-3:]:  # Show last 3
                console.print(
                    f"  {h['from']} → {h['to']} "
                    f"([dim]{h['at'][:10]}[/dim])"
                )
        
        console.print()
        return 0
        
    except PromotionError as e:
        console.print(f"[red]Promotion blocked:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1


def cmd_list(
    *,
    stage: Optional[str] = None,
    model_id: Optional[str] = None,
    json_output: bool = False,
) -> int:
    """List registered models.
    
    Args:
        stage: Filter by stage
        model_id: Filter by model ID
        json_output: If True, output JSON to stdout
        
    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Configure persistence if enabled
    if os.environ.get("VERIFILY_REGISTRY_PERSIST") == "1":
        data_dir = os.environ.get("VERIFILY_DATA_DIR", ".verifily")
        persist_path = Path(data_dir) / "registry.jsonl"
        configure_registry_persistence(persist_path)
    
    try:
        records = registry_store.list(stage=stage, model_id=model_id)
        
        if json_output:
            import json
            sys.stdout.write(json.dumps([r.to_dict() for r in records], indent=2))
            sys.stdout.write("\n")
            return 0
        
        if not records:
            console.print()
            console.print("[dim]No models found.[/dim]")
            console.print()
            return 0
        
        # Rich table output
        table = Table(title="Model Registry")
        table.add_column("Model ID", style="cyan")
        table.add_column("Version")
        table.add_column("Stage", style="bold")
        table.add_column("Health", justify="right")
        table.add_column("Risk", justify="right")
        table.add_column("F1/Metric", justify="right")
        table.add_column("Decision")
        
        for r in records:
            # Stage color
            stage_style = {
                ModelStage.PRODUCTION.value: "green",
                ModelStage.STAGING.value: "yellow",
                ModelStage.NONE.value: "dim",
                ModelStage.ARCHIVED.value: "dim",
            }.get(r.stage, "white")
            
            # Health color
            health_val = r.health_index
            health_style = "green" if health_val >= 60 else "yellow" if health_val >= 40 else "red"
            
            # Risk color
            risk_val = r.risk_score
            risk_style = "green" if risk_val <= 25 else "yellow" if risk_val <= 50 else "red"
            
            # Metric display
            metric_val = r.metrics.get("f1", r.metrics.get("accuracy", 0))
            
            table.add_row(
                r.model_id,
                r.version,
                f"[{stage_style}]{r.stage}[/{stage_style}]",
                f"[{health_style}]{health_val:.1f}[/{health_style}]",
                f"[{risk_style}]{risk_val:.1f}[/{risk_style}]",
                f"{metric_val:.3f}" if metric_val else "-",
                r.decision,
            )
        
        console.print()
        console.print(table)
        console.print()
        console.print(f"[dim]Total: {len(records)} models[/dim]")
        console.print()
        return 0
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


def cmd_history(
    model_id: str,
    *,
    json_output: bool = False,
) -> int:
    """Show history of a model (all versions).
    
    Args:
        model_id: Model identifier
        json_output: If True, output JSON to stdout
        
    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Configure persistence if enabled
    if os.environ.get("VERIFILY_REGISTRY_PERSIST") == "1":
        data_dir = os.environ.get("VERIFILY_DATA_DIR", ".verifily")
        persist_path = Path(data_dir) / "registry.jsonl"
        configure_registry_persistence(persist_path)
    
    try:
        records = registry_store.history(model_id)
        
        if not records:
            console.print(f"[red]No history found for model:[/red] {model_id}")
            return 1
        
        if json_output:
            import json
            sys.stdout.write(json.dumps([r.to_dict() for r in records], indent=2))
            sys.stdout.write("\n")
            return 0
        
        # Rich output
        console.print()
        console.print(f"[bold]Model History:[/bold] [cyan]{model_id}[/cyan]")
        console.print()
        
        for r in records:
            stage_color = {
                ModelStage.PRODUCTION.value: "green",
                ModelStage.STAGING.value: "yellow",
                ModelStage.ARCHIVED.value: "dim",
            }.get(r.stage, "white")
            
            console.print(Panel(
                f"Version: [cyan]{r.version}[/cyan] | "
                f"Stage: [{stage_color}]{r.stage}[/{stage_color}]\n"
                f"Created: [dim]{r.created_at[:19]}[/dim] | "
                f"By: {r.registered_by}\n"
                f"Health: {r.health_index:.1f} | Risk: {r.risk_score:.1f} | "
                f"Decision: {r.decision}",
                border_style="dim",
            ))
            
            if r.promotion_history:
                console.print("  [dim]Promotions:[/dim]")
                for h in r.promotion_history:
                    console.print(f"    {h['from']} → {h['to']} ({h['at'][:10]})")
            console.print()
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
