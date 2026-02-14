"""verifily doctor — environment and dependency health checks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.env import run_all_checks
from verifily_cli_v1.core.io import read_yaml

console = Console()

_STATUS_STYLE = {
    "PASS": "[green]PASS[/green]",
    "FAIL": "[red bold]FAIL[/red bold]",
    "WARN": "[yellow]WARN[/yellow]",
}

# Exit codes
EXIT_HEALTHY = 0
EXIT_WARNINGS = 2
EXIT_FATAL = 3


def run(config: Optional[str] = None, verbose: bool = False) -> int:
    """Run all environment checks.

    Returns:
        0 = healthy, 2 = warnings only, 3 = fatal issues.
    """
    console.print(
        Panel("[bold]Verifily Doctor[/bold]", border_style="blue")
    )

    checks = run_all_checks()

    # Optional config validation
    if config:
        cfg_path = Path(config)
        if cfg_path.exists():
            try:
                data = read_yaml(cfg_path)
                if data:
                    checks.append(
                        type("EnvCheck", (), {
                            "name": f"Config: {cfg_path.name}",
                            "status": "PASS",
                            "detail": f"{len(data)} top-level keys",
                        })()
                    )
                else:
                    checks.append(
                        type("EnvCheck", (), {
                            "name": f"Config: {cfg_path.name}",
                            "status": "FAIL",
                            "detail": "Empty or invalid YAML",
                        })()
                    )
            except Exception as e:
                checks.append(
                    type("EnvCheck", (), {
                        "name": f"Config: {cfg_path.name}",
                        "status": "FAIL",
                        "detail": str(e) if verbose else "Parse error",
                    })()
                )
        else:
            checks.append(
                type("EnvCheck", (), {
                    "name": f"Config: {cfg_path.name}",
                    "status": "FAIL",
                    "detail": "File not found",
                })()
            )

    # Build table
    table = Table(show_header=True, header_style="bold", pad_edge=True)
    table.add_column("Check", style="cyan", min_width=20)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Detail", min_width=30)

    has_failure = False
    has_warning = False
    for c in checks:
        styled = _STATUS_STYLE.get(c.status, c.status)
        table.add_row(c.name, styled, c.detail)
        if c.status == "FAIL":
            has_failure = True
        elif c.status == "WARN":
            has_warning = True

    console.print(table)

    if has_failure:
        console.print("\n[red bold]Fatal issues detected.[/red bold] Fix the FAIL items above before proceeding.")
        return EXIT_FATAL
    elif has_warning:
        console.print("\n[yellow bold]Warnings detected.[/yellow bold] Verifily will work, but some features may be limited.")
        return EXIT_WARNINGS
    else:
        console.print("\n[green bold]All checks passed.[/green bold]")
        return EXIT_HEALTHY
