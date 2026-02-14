"""CLI command for computing Risk Score and Health Index.

Usage:
    verifily score --run runs/model_v1
    verifily score --run runs/model_v1 --json
    verifily score --run runs/model_v1 --no-write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.io import read_json
from verifily_cli_v1.core.scoring import (
    compute_dataset_risk,
    compute_model_health,
    compute_verdict,
    write_score_artifacts,
)

console = Console(stderr=True)


def run(
    run_dir: str,
    *,
    json_output: bool = False,
    no_write: bool = False,
    verbose: bool = False,
) -> int:
    """Execute the score command.
    
    Args:
        run_dir: Path to run directory
        json_output: If True, output JSON to stdout
        no_write: If True, compute only (don't write artifacts)
        verbose: If True, show detailed output
        
    Returns:
        Exit code (0 = success, 1 = error)
    """
    run_path = Path(run_dir)
    
    # Validate run directory
    if not run_path.exists():
        console.print(f"[red]Error:[/red] Run directory not found: {run_path}")
        return 1
    
    if not run_path.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {run_path}")
        return 1
    
    # Load artifacts
    try:
        report = _load_json(run_path / "report.json")
        contamination = _load_json(run_path / "contamination_results.json")
        contract = _load_json(run_path / "contract.json")
        decision = _load_json(run_path / "decision.json")
        eval_results = _load_json(run_path / "eval" / "eval_results.json")
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to load artifacts: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1
    
    # Compute scores
    try:
        risk_score = compute_dataset_risk(
            report_result=report,
            contamination_result=contamination,
            contract_result=contract,
        )
        
        health_index = compute_model_health(
            decision_result=decision,
            eval_results=eval_results,
            reproducibility_ok=contract.get("valid", False) if contract else False,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to compute scores: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1
    
    # JSON output mode
    if json_output:
        from verifily_cli_v1.core.scoring import RiskHealthSummary
        verdict, recommendations = compute_verdict(risk_score, health_index)
        summary = RiskHealthSummary(
            risk_score=risk_score,
            health_index=health_index,
            verdict=verdict,
            recommendations=recommendations,
        )
        sys.stdout.write(json.dumps(summary.to_dict(), indent=2))
        sys.stdout.write("\n")
        return 0
    
    # Rich output mode
    _display_scores(risk_score, health_index, run_path.name)
    
    # Write artifacts unless --no-write
    if not no_write:
        try:
            artifacts = write_score_artifacts(risk_score, health_index, run_path)
            console.print()
            console.print("[green]Artifacts written:[/green]")
            for name, path in artifacts.items():
                console.print(f"  • {name}: {path}")
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to write artifacts: {e}")
            return 1
    
    console.print()
    return 0


def _load_json(path: Path) -> Optional[dict]:
    """Load JSON file if it exists, else return None."""
    if path.exists():
        try:
            return read_json(path)
        except Exception:
            pass
    return None


def _display_scores(risk: "RiskScore", health: "HealthIndex", run_name: str) -> None:
    """Display scores using Rich formatting."""
    from verifily_cli_v1.core.scoring import compute_verdict
    
    verdict, recommendations = compute_verdict(risk, health)
    
    # Color mapping
    risk_color = {
        "LOW": "green",
        "MEDIUM": "yellow",
        "HIGH": "orange3",
        "CRITICAL": "red",
    }.get(risk.level, "white")
    
    health_color = {
        "EXCELLENT": "green",
        "GOOD": "blue",
        "FAIR": "yellow",
        "POOR": "red",
    }.get(health.level, "white")
    
    verdict_color = {
        "APPROVED": "green",
        "REVIEW_REQUIRED": "yellow",
        "BLOCKED": "red",
    }.get(verdict, "white")
    
    # Main panel
    console.print()
    console.print(Panel.fit(
        f"[bold]{run_name}[/bold]\n\n"
        f"Dataset Risk Score: [{risk_color}]{risk.total:.1f}/100[/] ({risk.level})\n"
        f"Model Health Index: [{health_color}]{health.total:.1f}/100[/] ({health.level})\n\n"
        f"Verdict: [{verdict_color}]{verdict}[/]",
        title="[bold]Risk & Health Assessment[/bold]",
        border_style="blue",
    ))
    
    # Risk breakdown table
    risk_table = Table(title=f"Risk Components (Total: {risk.total:.1f})")
    risk_table.add_column("Component", style="cyan")
    risk_table.add_column("Contribution", justify="right")
    risk_table.add_column("Detail")
    
    for c in sorted(risk.components, key=lambda x: x.contribution, reverse=True)[:5]:
        contrib_color = "red" if c.contribution > 15 else "yellow" if c.contribution > 5 else "green"
        risk_table.add_row(
            c.name,
            f"[{contrib_color}]{c.contribution:.1f}[/]",
            c.detail[:50] + "..." if len(c.detail) > 50 else c.detail,
        )
    
    console.print()
    console.print(risk_table)
    
    # Health breakdown table
    health_table = Table(title=f"Health Components (Total: {health.total:.1f})")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Contribution", justify="right")
    health_table.add_column("Detail")
    
    for c in sorted(health.components, key=lambda x: x.contribution, reverse=True)[:5]:
        contrib_color = "green" if c.contribution > 20 else "yellow" if c.contribution > 10 else "red"
        health_table.add_row(
            c.name,
            f"[{contrib_color}]{c.contribution:.1f}[/]",
            c.detail[:50] + "..." if len(c.detail) > 50 else c.detail,
        )
    
    console.print()
    console.print(health_table)
    
    # Recommendations
    if recommendations:
        console.print()
        console.print("[bold]Recommendations:[/bold]")
        for rec in recommendations:
            console.print(f"  • {rec}")
