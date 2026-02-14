"""verifily reproduce — verify run reproducibility via hash chains."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from verifily_cli_v1.core.hashing import sha256_file, verify_hashes
from verifily_cli_v1.core.io import read_json

console = Console()


def run(
    run_dir: str,
    verbose: bool = False,
) -> None:
    """Verify the reproducibility status of a training or transform run."""
    rd = Path(run_dir)

    if not rd.exists():
        console.print(f"[red bold]Error:[/red bold] Run directory not found: {rd}")
        raise SystemExit(1)

    console.print(
        Panel(
            f"[bold]Verifily Reproduce[/bold]\nRun: {rd}",
            border_style="blue",
        )
    )

    # ── Look for hashes.json ──
    hashes_path = rd / "hashes.json"
    if not hashes_path.exists():
        console.print(
            "[red bold]Error:[/red bold] hashes.json not found in run directory.\n"
            "  This run was not created with Verifily or is missing integrity data."
        )
        raise SystemExit(1)

    hashes_data = read_json(hashes_path)
    file_hashes = hashes_data.get("files", {})
    expected_chain = hashes_data.get("chain_hash", "")

    # Handle Transform v1 format: {dataset_hash: "sha256:...", config_hash: ...}
    if not file_hashes and any(k.endswith("_hash") for k in hashes_data if k not in ("chain_hash", "manifest_hash", "hash_algorithm")):
        _hash_to_file = {
            "dataset_hash": "dataset.jsonl",
            "config_hash": "transform_config.yaml",
        }
        for hash_key, filename in _hash_to_file.items():
            if hash_key in hashes_data:
                raw = hashes_data[hash_key]
                # Strip "sha256:" prefix if present
                file_hashes[filename] = raw.replace("sha256:", "") if isinstance(raw, str) else raw

    if not file_hashes:
        console.print("[yellow]Warning:[/yellow] hashes.json contains no file hashes.")
        # Try manifest.json as fallback
        manifest_path = rd / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            file_hashes = manifest.get("file_hashes", {})
            expected_chain = manifest.get("chain_hash", "")

    if not file_hashes:
        console.print("[red bold]Error:[/red bold] No file hashes found to verify.")
        raise SystemExit(1)

    # ── Verify each file ──
    results = verify_hashes(rd, file_hashes)

    table = Table(
        title="File Integrity",
        show_header=True,
        header_style="bold",
    )
    table.add_column("File", style="cyan", min_width=25)
    table.add_column("Status", justify="center", min_width=10)
    if verbose:
        table.add_column("Hash (first 16)", min_width=20)

    all_pass = True
    for check in results:
        if check["status"] == "PASS":
            status = "[green]PASS[/green]"
        elif check["status"] == "MISSING":
            status = "[red bold]MISSING[/red bold]"
            all_pass = False
        else:
            status = "[red bold]FAIL[/red bold]"
            all_pass = False

        row = [check["file"], status]
        if verbose:
            hash_preview = check["actual"][:16] + "..." if check["actual"] else "—"
            row.append(hash_preview)
        table.add_row(*row)

    console.print(table)

    # ── Chain hash ──
    if expected_chain:
        from verifily_cli_v1.core.hashing import compute_chain_hash

        actual_hashes = {c["file"]: c["actual"] for c in results if c["actual"]}
        actual_chain = compute_chain_hash(actual_hashes)
        chain_match = actual_chain == expected_chain

        if chain_match:
            console.print(f"\n  Chain hash: [green]VERIFIED[/green]")
        else:
            console.print(f"\n  Chain hash: [red bold]MISMATCH[/red bold]")
            all_pass = False

        if verbose:
            console.print(f"    Expected: {expected_chain[:32]}...")
            console.print(f"    Actual:   {actual_chain[:32]}...")

    # ── Manifest hash ──
    manifest_hash = hashes_data.get("manifest_hash", "")
    manifest_path = rd / "manifest.json"
    if manifest_hash and manifest_path.exists():
        actual_manifest_hash = sha256_file(manifest_path)
        if actual_manifest_hash == manifest_hash:
            console.print(f"  Manifest:   [green]VERIFIED[/green]")
        else:
            console.print(f"  Manifest:   [red bold]MISMATCH[/red bold]")
            all_pass = False

    # ── Run metadata ──
    run_meta_path = rd / "run_meta.json"
    if run_meta_path.exists() and verbose:
        meta = read_json(run_meta_path)
        console.print(f"\n  Run ID:     {meta.get('run_id', 'N/A')}")
        console.print(f"  Created:    {meta.get('created_at', 'N/A')}")
        console.print(f"  Seed:       {meta.get('seed', 'N/A')}")

    # ── Summary ──
    if all_pass:
        console.print(
            "\n[green bold]Reproducibility: VERIFIED[/green bold]"
            "  All files match their recorded hashes."
        )
    else:
        console.print(
            "\n[red bold]Reproducibility: BROKEN[/red bold]"
            "  One or more files have been modified since the run was created."
        )
        raise SystemExit(1)

    console.print(f"\n[dim]Next:[/dim] verifily eval --run {rd}")
