"""Verifily Transform CLI — data transformation pipeline."""

import logging
import sys
import time

import click

from verifily_transform import __version__

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
        stream=sys.stderr,
    )


@click.group()
@click.version_option(version=__version__, prog_name="verifily-transform")
def cli():
    """Verifily Transform — raw data to training-ready datasets."""
    pass


@cli.command()
@click.option("--config", "config_path", required=True, help="Path to transform YAML config.")
@click.option("--seed", default=None, type=int, help="Override random seed.")
@click.option("--dry-run", is_flag=True, help="Validate config and show pipeline plan, don't execute.")
@click.option("--verbose", is_flag=True, help="Enable DEBUG logging.")
def run(config_path, seed, dry_run, verbose):
    """Execute a data transformation pipeline."""
    _setup_logging(verbose)
    import random
    import numpy as np

    from verifily_transform.config import TransformConfig
    from verifily_transform.ingest import ingest
    from verifily_transform.normalize import normalize
    from verifily_transform.label import label
    from verifily_transform.synthesize import synthesize
    from verifily_transform.dedupe import deduplicate
    from verifily_transform.filters import apply_filters
    from verifily_transform.artifacts import package_artifact
    from verifily_transform.utils import generate_transform_id, redact_pii, utcnow_iso

    cfg = TransformConfig.from_yaml(config_path)
    if seed is not None:
        cfg.seed = seed

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    transform_id = generate_transform_id()
    stats = {"transform_id": transform_id}

    click.echo(f"Verifily Transform v{__version__}")
    click.echo(f"  Run ID:  {transform_id}")
    click.echo(f"  Input:   {cfg.input.path} ({cfg.input.format})")
    click.echo(f"  Output:  {cfg.output.dir.rstrip('/')}/{cfg.output.name}")
    click.echo(f"  Task:    {cfg.labeling.task}")
    click.echo(f"  Synth:   {'ON (x' + str(cfg.synthetic.expansion_factor) + ')' if cfg.synthetic.enabled else 'OFF'}")
    click.echo(f"  Dedupe:  exact={cfg.dedupe.exact}, fuzzy={cfg.dedupe.fuzzy}")
    click.echo(f"  Privacy: pii={cfg.privacy.pii_removal}")
    click.echo("")

    if dry_run:
        click.echo("DRY RUN — config validated, pipeline plan shown above.")
        return

    start = time.time()

    # === STEP 1: Ingest ===
    click.echo("Step 1/6: Ingesting raw data...")
    raw_rows = ingest(cfg.input)
    stats["raw_rows"] = len(raw_rows)
    click.echo(f"  {len(raw_rows)} raw rows ingested")

    # === STEP 2: Normalize ===
    click.echo("Step 2/6: Normalizing schema...")
    normalized = normalize(
        raw_rows, cfg.labeling.task,
        instruction_field=cfg.labeling.instruction_field,
        output_field=cfg.labeling.output_field,
    )
    stats["normalized_rows"] = len(normalized)
    click.echo(f"  {len(normalized)} rows normalized")

    # === STEP 3: Label ===
    click.echo("Step 3/6: Labeling...")
    labeled = label(normalized, cfg.labeling)
    stats["labeled_rows"] = len(labeled)
    click.echo(f"  {len(labeled)} rows labeled")

    # === STEP 4: PII Redaction ===
    if cfg.privacy.pii_removal:
        click.echo("Step 3.5/6: Redacting PII...")
        pii_count = 0
        for row in labeled:
            for key in ["instruction", "output", "input", "text"]:
                if key in row and row[key]:
                    original = row[key]
                    row[key] = redact_pii(row[key])
                    if row[key] != original:
                        pii_count += 1
        stats["pii_redactions"] = pii_count
        click.echo(f"  {pii_count} fields redacted")

    # === STEP 5: Synthesize ===
    synthetic_rows = []
    if cfg.synthetic.enabled:
        click.echo("Step 4/6: Generating synthetic data...")
        synthetic_rows = synthesize(labeled, cfg.synthetic, cfg.labeling.task)
        stats["synthetic_rows"] = len(synthetic_rows)
        click.echo(f"  {len(synthetic_rows)} synthetic rows generated")
    else:
        click.echo("Step 4/6: Synthesis disabled, skipping")

    # Merge original + synthetic
    all_rows = labeled + synthetic_rows

    # === STEP 6: Deduplicate ===
    click.echo("Step 5/6: Deduplicating...")
    deduped = deduplicate(all_rows, cfg.dedupe, cfg.labeling.task)
    stats["deduped_rows"] = len(deduped)
    click.echo(f"  {len(deduped)} rows after dedup")

    # === STEP 7: Filter ===
    click.echo("Step 6/6: Filtering...")
    filtered, rejection_counts = apply_filters(
        deduped,
        cfg.labeling.task,
        seed_rows=labeled if cfg.synthetic.enabled else None,
        min_length=cfg.synthetic.filters.min_length,
        max_length=cfg.synthetic.filters.max_length,
        leakage_check=cfg.synthetic.filters.leakage_check,
        pii_removal=cfg.privacy.pii_removal,
    )
    stats["filtered_rows"] = len(filtered)
    stats["rejections"] = rejection_counts
    click.echo(f"  {len(filtered)} rows passed filters")

    # === Package artifact ===
    click.echo("\nPackaging artifact...")
    elapsed = time.time() - start
    stats["duration_seconds"] = round(elapsed, 2)

    output_dir = package_artifact(filtered, cfg, transform_id, stats)

    click.echo("")
    click.echo("=" * 50)
    click.echo("Transform complete!")
    click.echo(f"  Run ID:     {transform_id}")
    click.echo(f"  Rows:       {stats['raw_rows']} raw -> {len(filtered)} final")
    click.echo(f"  Duration:   {elapsed:.1f}s")
    click.echo(f"  Artifact:   {output_dir}/")
    click.echo(f"  Dataset:    {output_dir}/dataset.jsonl")
    click.echo(f"  Manifest:   {output_dir}/manifest.json")
    click.echo("=" * 50)
    click.echo("")
    click.echo(f"Ready for training:")
    click.echo(f"  verifily train --config train.yaml --data {output_dir}/dataset.jsonl")

    # Audit log
    if cfg.privacy.audit_log:
        import json
        audit_path = f"{output_dir}/audit_log.json"
        with open(audit_path, "w") as f:
            json.dump({
                "transform_id": transform_id,
                "pii_removal": cfg.privacy.pii_removal,
                "pii_redactions": stats.get("pii_redactions", 0),
                "rows_filtered": stats.get("rejections", {}),
                "timestamp": utcnow_iso(),
            }, f, indent=2)
        click.echo(f"  Audit log:  {audit_path}")


@cli.command()
@click.option("--dir", "manifest_dir", required=True, help="Path to dataset artifact directory.")
def verify(manifest_dir):
    """Verify dataset integrity against its manifest."""
    from verifily_transform.manifest import verify_manifest, format_verification

    results = verify_manifest(manifest_dir)
    click.echo(format_verification(results))


def main():
    cli()
