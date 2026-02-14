"""CLI command for visualizing dataset lineage.

Usage:
    verifily lineage --run runs/model_v1
    verifily lineage --run runs/model_v1 --mermaid
    verifily lineage --run runs/model_v1 --json
    verifily lineage --run runs/model_v1 --output ./lineage
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from verifily_cli_v1.core.lineage_graph import (
    LineageGraph,
    build_lineage_graph,
    write_lineage_artifacts,
)

console = Console(stderr=True)


def run(
    run_dir: str,
    *,
    output: Optional[str] = None,
    mermaid: bool = False,
    json_output: bool = False,
    verbose: bool = False,
) -> int:
    """Execute the lineage command.

    Args:
        run_dir: Path to the run directory
        output: Optional directory to write artifacts
        mermaid: If True, output Mermaid diagram to stdout
        json_output: If True, output JSON to stdout
        verbose: If True, show detailed output

    Returns:
        Exit code (0 = success, 1 = error)
    """
    run_path = Path(run_dir)

    # Validate run directory exists
    if not run_path.exists():
        console.print(f"[red]Error:[/red] Run directory not found: {run_path}")
        return 1

    if not run_path.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {run_path}")
        return 1

    # Build the lineage graph
    try:
        graph = build_lineage_graph(run_path)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to build lineage graph: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        return 1

    # Output based on format flags
    if json_output:
        # JSON output to stdout
        sys.stdout.write(graph.to_json())
        sys.stdout.write("\n")
        return 0

    if mermaid:
        # Mermaid output to stdout
        sys.stdout.write(graph.to_mermaid())
        sys.stdout.write("\n")
        return 0

    # Default: Rich tree visualization
    tree_text = graph.to_rich_tree()

    # Print to console
    console.print()
    console.print(Panel.fit(
        tree_text,
        title=f"[bold]Lineage: {run_path.name}[/bold]",
        border_style="blue",
    ))

    # Node counts
    node_counts = {}
    for node in graph.nodes:
        node_counts[node.type.value] = node_counts.get(node.type.value, 0) + 1

    console.print()
    console.print("[dim]Summary:[/dim]")
    for node_type, count in sorted(node_counts.items()):
        console.print(f"  [dim]{node_type}:[/dim] {count}")
    console.print(f"  [dim]edges:[/dim] {len(graph.edges)}")

    # Write artifacts if output directory specified
    if output:
        try:
            artifacts = write_lineage_artifacts(graph, output)
            console.print()
            console.print("[green]Artifacts written:[/green]")
            for artifact_type, path in artifacts.items():
                console.print(f"  • {artifact_type}: {path}")
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to write artifacts: {e}")
            return 1

    console.print()
    return 0
