"""Thin wrappers that call existing Verifily core/CLI functions.

Each runner returns plain dicts — the server layer converts to Pydantic models.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from verifily_cli_v1.core.io import read_jsonl, read_yaml, write_json, write_yaml


def _timestamp_id() -> str:
    """Generate a short timestamp + random ID for output dirs."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


# ── Pipeline ─────────────────────────────────────────────────────

def run_pipeline_api(
    *,
    config_path: Optional[str] = None,
    project_path: Optional[str] = None,
    plan: bool = False,
    ci: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full pipeline and return a result dict.

    Either config_path or project_path must be provided.
    """
    from verifily_cli_v1.commands.pipeline import run_pipeline

    t0 = time.monotonic()

    # Resolve config
    resolved_config = _resolve_pipeline_config(config_path, project_path, overrides)

    # Determine output directory
    output_dir: Optional[str] = None
    if not plan:
        output_dir = str(
            Path(tempfile.gettempdir())
            / "verifily_api_runs"
            / _timestamp_id()
        )

    result = run_pipeline(
        resolved_config, ci=ci, output_dir=output_dir,
        request_id=request_id, mode="api", project_id=project_id,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Build response
    decision = result.get("decision", {})
    contamination = result.get("contamination")
    contract = result.get("contract")
    report_data = result.get("report")

    response: Dict[str, Any] = {
        "exit_code": decision.get("exit_code", 4),
        "decision": decision,
        "config_path": str(resolved_config),
        "elapsed_ms": elapsed_ms,
    }

    if contamination:
        response["contamination"] = {
            "status": contamination.get("status", "UNKNOWN"),
            "exact_overlaps": contamination.get("exact_overlaps", 0),
            "near_duplicates": contamination.get("near_duplicates", 0),
            "exact_overlap_fraction": contamination.get("exact_overlap_fraction", 0.0),
            "near_duplicate_fraction": contamination.get("near_duplicate_fraction", 0.0),
        }

    if contract:
        response["contract"] = {
            "valid": contract.get("valid", False),
            "checks": contract.get("checks", []),
            "has_eval": contract.get("has_eval", False),
        }

    if report_data:
        response["report_summary"] = {
            "row_count": report_data.get("row_count", 0),
            "pii_clean": report_data.get("pii_clean", True),
            "pii_total_hits": report_data.get("pii_total_hits", 0),
        }

    if output_dir and not plan:
        response["output_dir"] = output_dir
    else:
        response["output_dir"] = None

    # Include usage data from pipeline
    usage = result.get("usage")
    if usage:
        response["usage"] = usage

    return response


def _resolve_pipeline_config(
    config_path: Optional[str],
    project_path: Optional[str],
    overrides: Optional[Dict[str, Any]],
) -> str:
    """Resolve to a concrete config file path."""
    if config_path:
        resolved = Path(config_path)
    elif project_path:
        resolved = Path(project_path) / "verifily.yaml"
        if not resolved.exists():
            raise FileNotFoundError(
                f"No verifily.yaml found in project: {project_path}"
            )
    else:
        raise ValueError("Either config_path or project_path must be provided.")

    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    # If overrides, merge into a temp config
    if overrides:
        cfg = read_yaml(resolved)
        cfg.update(overrides)
        tmp = Path(tempfile.gettempdir()) / f"verifily_api_{uuid.uuid4().hex[:8]}.yaml"
        write_yaml(tmp, cfg)
        return str(tmp)

    return str(resolved)


# ── Contamination ────────────────────────────────────────────────

def run_contamination_api(
    *,
    train_path: str,
    eval_path: str,
    jaccard_cutoff: float = 0.70,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run contamination check and return result dict."""
    from verifily_cli_v1.commands.contamination import check_contamination

    t0 = time.monotonic()
    result = check_contamination(
        train_path, eval_path, jaccard_cutoff=jaccard_cutoff,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if out_path:
        write_json(out_path, result)

    return {
        "status": result["status"],
        "exit_code": result["exit_code"],
        "train_rows": result["train_rows"],
        "eval_rows": result["eval_rows"],
        "exact_overlaps": result["exact_overlaps"],
        "exact_overlap_fraction": result["exact_overlap_fraction"],
        "near_duplicates": result["near_duplicates"],
        "near_duplicate_fraction": result["near_duplicate_fraction"],
        "reasons": result["reasons"],
        "elapsed_ms": elapsed_ms,
    }


# ── Report ───────────────────────────────────────────────────────

def run_report_api(
    *,
    dataset_path: str,
    schema: str = "sft",
    sample: int = 0,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run dataset report and return result dict."""
    from verifily_cli_v1.commands.report import dataset_report

    t0 = time.monotonic()
    report = dataset_report(dataset_path, schema=schema)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # PII summary: counts only, never raw PII
    pii_summary = {
        pii_type: data["count"]
        for pii_type, data in report.get("pii_scan", {}).items()
    }

    response: Dict[str, Any] = {
        "row_count": report["row_count"],
        "schema": report["schema"],
        "field_stats": report["field_stats"],
        "tag_distribution": report["tag_distribution"],
        "pii_summary": pii_summary,
        "pii_clean": report["pii_clean"],
        "exit_code": 0,
        "elapsed_ms": elapsed_ms,
    }

    # Optionally include sample rows (strip PII-containing fields of raw values)
    if sample > 0:
        rows = read_jsonl(dataset_path)
        response["sample_rows"] = rows[:sample]

    if out_dir:
        from verifily_cli_v1.core.io import ensure_dir
        out = ensure_dir(out_dir)
        write_json(out / "report.json", report)

    return response


# ── Classify ────────────────────────────────────────────────────

def run_classify_api(
    *,
    dataset_path: Optional[str] = None,
    inline_rows: Optional[list] = None,
    output_dir: Optional[str] = None,
    max_rows_scan: int = 500,
    export_buckets: bool = False,
    min_bucket_rows: int = 1,
) -> Dict[str, Any]:
    """Run dataset classification and return result dict."""
    from verifily_cli_v1.core.classify import classify_dataset, write_bucket_exports

    t0 = time.monotonic()

    # Load rows from file or inline
    if inline_rows is not None:
        rows = inline_rows
    elif dataset_path:
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if p.suffix == ".csv":
            import csv
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
        else:
            rows = read_jsonl(dataset_path)
    else:
        raise ValueError("Either dataset_path or inline_rows must be provided.")

    classification = classify_dataset(rows, max_rows_scan=max_rows_scan)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Write artifact if output_dir set
    artifacts: Dict[str, str] = {}
    export_summary: Optional[Dict[str, Any]] = None
    if output_dir:
        from verifily_cli_v1.core.io import ensure_dir
        out = ensure_dir(Path(output_dir) / "classification")
        artifact_path = out / "classification.json"
        write_json(artifact_path, classification)
        artifacts["classification_json"] = str(artifact_path)

        # Bucket exports
        if export_buckets:
            summary = write_bucket_exports(
                classification, rows, output_dir,
                min_bucket_rows=min_bucket_rows,
            )
            artifacts["suggested_next_steps"] = summary.suggested_next_steps_path or ""
            for bucket_name, bucket_path in summary.bucket_paths.items():
                artifacts[f"bucket_{bucket_name}"] = bucket_path
            export_summary = summary.to_dict()

    result: Dict[str, Any] = {
        "classification": classification,
        "artifacts": artifacts,
        "elapsed_ms": elapsed_ms,
    }
    if export_summary is not None:
        result["export_summary"] = export_summary
    return result
