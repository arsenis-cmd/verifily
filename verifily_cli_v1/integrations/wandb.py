"""Weights & Biases integration — log pipeline results to W&B.

Opt-in: set ``wandb.enabled: true`` in verifily.yaml or pass ``--wandb`` on CLI.
Requires ``wandb``: ``pip install verifily[wandb]``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WandbConfig:
    """Configuration for W&B logging."""
    enabled: bool = False
    project: str = "verifily"
    entity: Optional[str] = None
    tags: Optional[List[str]] = None
    run_name: Optional[str] = None


def _import_wandb():
    """Lazy-import wandb."""
    try:
        import wandb
        return wandb
    except ImportError:
        raise ImportError(
            "W&B support requires wandb. "
            "Install with: pip install verifily[wandb]"
        )


def wandb_config_from_dict(d: Dict[str, Any]) -> WandbConfig:
    """Build WandbConfig from a config dict (e.g. from verifily.yaml)."""
    return WandbConfig(
        enabled=bool(d.get("enabled", False)),
        project=str(d.get("project", "verifily")),
        entity=d.get("entity"),
        tags=d.get("tags"),
        run_name=d.get("run_name"),
    )


def log_pipeline_run(
    results: Dict[str, Any],
    config: WandbConfig,
) -> Optional[str]:
    """Log a pipeline run to W&B. Returns the W&B run URL or None.

    Logs:
    - decision.recommendation, exit_code, confidence as summary
    - decision.metrics (f1, accuracy, etc.)
    - decision.deltas as comparison metrics
    - risk_score, health_index as gauges
    - contamination stats
    - PII counts from report
    - Full results dict as a W&B artifact (JSON)
    """
    wandb = _import_wandb()

    decision = results.get("decision", {})
    run_id = results.get("run_id", "unknown")

    run = wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.run_name or f"verifily-{run_id}",
        tags=config.tags or [],
        config={"verifily_config": results.get("config", "")},
    )

    # Scalar metrics
    log_data: Dict[str, Any] = {
        "decision/exit_code": decision.get("exit_code", -1),
        "decision/confidence": decision.get("confidence", 0.0),
    }

    # Decision metrics (f1, accuracy, etc.)
    for k, v in decision.get("metrics", {}).items():
        if isinstance(v, (int, float)):
            log_data[f"metrics/{k}"] = v

    # Deltas (comparison to baseline)
    for k, v in (decision.get("deltas") or {}).items():
        if isinstance(v, (int, float)):
            log_data[f"deltas/{k}"] = v

    # Risk score and health index
    if "risk_score" in results:
        log_data["risk_score"] = results["risk_score"].get("total", 0)
    if "health_index" in results:
        log_data["health_index"] = results["health_index"].get("total", 0)

    # Contamination
    contam = results.get("contamination", {})
    if contam:
        log_data["contamination/exact_overlaps"] = contam.get("exact_overlaps", 0)
        log_data["contamination/near_duplicates"] = contam.get("near_duplicates", 0)

    # PII
    report = results.get("report", {})
    if report:
        log_data["pii/total_hits"] = report.get("pii_total_hits", 0)
        log_data["pii/clean"] = 1 if report.get("pii_clean", True) else 0

    wandb.log(log_data)

    # Save full results as artifact
    artifact = wandb.Artifact(
        f"pipeline-results-{run_id}", type="results"
    )
    with artifact.new_file("results.json") as f:
        json.dump(results, f, indent=2, default=str)
    wandb.log_artifact(artifact)

    # Log decision as summary for table views
    wandb.run.summary["recommendation"] = decision.get("recommendation", "UNKNOWN")

    url = run.url
    wandb.finish()
    return url
