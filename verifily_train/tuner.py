"""Verifily Train auto-hyperparameter tuning: grid and random search."""

import itertools
import logging
import random
from typing import Any, Dict, List, Optional

from verifily_train.config import TrainConfig
from verifily_train.trainer import train

logger = logging.getLogger(__name__)


def _set_nested(cfg: TrainConfig, dotted_key: str, value: Any) -> None:
    """Set a value on config using a dotted key like 'training.learning_rate'."""
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _generate_grid_trials(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations for grid search."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    trials = []
    for combo in itertools.product(*values):
        trials.append(dict(zip(keys, combo)))
    return trials


def _generate_random_trials(
    param_grid: Dict[str, List[Any]], max_trials: int,
) -> List[Dict[str, Any]]:
    """Sample random combinations from the grid."""
    trials = []
    for _ in range(max_trials):
        trial = {}
        for key, values in param_grid.items():
            trial[key] = random.choice(values)
        trials.append(trial)
    return trials


def run_tuning(
    config_path: str,
    param_grid: Dict[str, List[Any]],
    strategy: str = "grid",
    max_trials: Optional[int] = None,
    metric: str = "eval_loss",
) -> Dict[str, Any]:
    """Run hyperparameter search.

    Args:
        config_path: path to base training YAML.
        param_grid: {dotted_param_key: [value1, value2, ...]}.
        strategy: "grid" or "random".
        max_trials: cap on number of trials (required for random).
        metric: metric to minimise from run.metrics (e.g. "eval_loss", "train_loss").

    Returns:
        Dict with best trial info: run_id, score, params, artifact_path.
    """
    if strategy == "grid":
        trials = _generate_grid_trials(param_grid)
    else:
        if max_trials is None:
            max_trials = min(20, len(_generate_grid_trials(param_grid)))
        trials = _generate_random_trials(param_grid, max_trials)

    logger.info("Tuning: %d trials (%s search)", len(trials), strategy)
    logger.info("Param grid: %s", param_grid)
    logger.info("Optimising: %s (lower is better)", metric)

    results = []
    for i, trial_params in enumerate(trials, 1):
        logger.info("--- Trial %d/%d: %s ---", i, len(trials), trial_params)

        cfg = TrainConfig.from_yaml(config_path)
        cfg.name = f"tune_trial_{i}"
        for key, val in trial_params.items():
            _set_nested(cfg, key, val)

        try:
            run = train(cfg)
            score = run.metrics.get(metric)
            if score is None:
                logger.warning("Metric '%s' not found in run metrics, using train_loss", metric)
                score = run.metrics.get("train_loss", float("inf"))
            results.append({
                "trial": i,
                "params": trial_params,
                "score": score,
                "run_id": run.run_id,
                "artifact_path": run.artifact_path,
                "metrics": run.metrics,
            })
            logger.info("Trial %d: %s = %.6f", i, metric, score)
        except Exception as e:
            logger.error("Trial %d failed: %s", i, e)
            results.append({
                "trial": i,
                "params": trial_params,
                "score": float("inf"),
                "run_id": "FAILED",
                "artifact_path": "",
                "metrics": {},
            })

    # Find best (lowest score)
    best = min(results, key=lambda r: r["score"])

    logger.info("=" * 60)
    logger.info("Tuning complete. Best trial: %d", best["trial"])
    logger.info("  %s = %.6f", metric, best["score"])
    logger.info("  Params: %s", best["params"])
    logger.info("  Run: %s", best["run_id"])
    logger.info("=" * 60)

    return best
