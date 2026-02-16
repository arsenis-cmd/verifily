"""MLflow integration — log pipeline metrics and sync model registry.

Opt-in: set ``mlflow.enabled: true`` in verifily.yaml or pass ``--mlflow`` on CLI.
Requires ``mlflow``: ``pip install verifily[mlflow]``
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MLflowConfig:
    """Configuration for MLflow logging."""
    enabled: bool = False
    tracking_uri: Optional[str] = None  # defaults to MLFLOW_TRACKING_URI env var
    experiment_name: str = "verifily"
    register_model: bool = True  # auto-register on SHIP


def _import_mlflow():
    """Lazy-import mlflow."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        raise ImportError(
            "MLflow support requires mlflow. "
            "Install with: pip install verifily[mlflow]"
        )


def mlflow_config_from_dict(d: Dict[str, Any]) -> MLflowConfig:
    """Build MLflowConfig from a config dict (e.g. from verifily.yaml)."""
    return MLflowConfig(
        enabled=bool(d.get("enabled", False)),
        tracking_uri=d.get("tracking_uri"),
        experiment_name=str(d.get("experiment_name", "verifily")),
        register_model=bool(d.get("register_model", True)),
    )


def log_pipeline_run(
    results: Dict[str, Any],
    config: MLflowConfig,
) -> Optional[str]:
    """Log pipeline results to MLflow. Returns the MLflow run ID.

    Creates an MLflow run with:
    - Params: config path, recommendation
    - Metrics: f1, accuracy, risk_score, health_index, contamination counts
    - Tags: decision, verifily version
    - Artifacts: full results JSON
    - Model registration on SHIP decision (if enabled)
    """
    mlflow = _import_mlflow()

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    decision = results.get("decision", {})
    run_id = results.get("run_id", "run")

    with mlflow.start_run(run_name=f"verifily-{run_id}") as run:
        # Params
        mlflow.log_param("config_path", results.get("config", ""))
        mlflow.log_param("recommendation", decision.get("recommendation", "UNKNOWN"))

        # Metrics from decision
        for k, v in decision.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        mlflow.log_metric("exit_code", decision.get("exit_code", -1))
        mlflow.log_metric("confidence", decision.get("confidence", 0.0))

        # Risk/health scores
        if "risk_score" in results:
            mlflow.log_metric("risk_score", results["risk_score"].get("total", 0))
        if "health_index" in results:
            mlflow.log_metric("health_index", results["health_index"].get("total", 0))

        # Contamination
        contam = results.get("contamination", {})
        if contam:
            mlflow.log_metric("contamination_exact_overlaps", contam.get("exact_overlaps", 0))
            mlflow.log_metric("contamination_near_duplicates", contam.get("near_duplicates", 0))

        # Tags
        mlflow.set_tag("verifily.decision", decision.get("recommendation", ""))
        usage = results.get("usage", {})
        if usage:
            mlflow.set_tag("verifily.version", usage.get("verifily_version", ""))

        # Artifact: full results JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="verifily_results_"
        ) as f:
            json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(f.name, "verifily")

        # Register model on SHIP decision
        if config.register_model and decision.get("recommendation") == "SHIP":
            registry_info = results.get("model_registry", {})
            if registry_info.get("registered"):
                model_name = registry_info.get("model_id", "verifily-model")
                try:
                    mlflow.register_model(
                        f"runs:/{run.info.run_id}/verifily", model_name
                    )
                except Exception:
                    pass  # Model registration is best-effort

        return run.info.run_id
