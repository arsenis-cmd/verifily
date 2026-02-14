"""Verifily Train compare: diff metrics across runs."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from verifily_train.errors import EvalError

logger = logging.getLogger(__name__)


@dataclass
class CompareResult:
    runs: List[str] = field(default_factory=list)
    metric: str = "f1"
    overall: Dict[str, float] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    slices: Optional[Dict[str, Dict[str, float]]] = None
    config_diff: Optional[Dict[str, Dict[str, Any]]] = None


def _load_eval_results(run_path: str) -> Dict[str, Any]:
    """Load eval_results.json from a run directory."""
    p = Path(run_path) / "eval" / "eval_results.json"
    if not p.exists():
        raise EvalError(
            f"No eval results in {run_path}. Run `verifily eval --run {run_path}` first."
        )
    with open(p) as f:
        return json.load(f)


def _load_config(run_path: str) -> Optional[Dict[str, Any]]:
    """Load config.yaml from a run directory, if it exists."""
    cfg_path = Path(run_path) / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _diff_configs(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Find keys where configs differ across runs.

    Returns: {key: {run_label: value, ...}} for every key that differs.
    """
    if not configs:
        return {}

    all_keys: set = set()
    for cfg in configs.values():
        all_keys.update(_flatten_dict(cfg).keys())

    diffs: Dict[str, Dict[str, Any]] = {}
    run_labels = list(configs.keys())
    flat = {lbl: _flatten_dict(cfg) for lbl, cfg in configs.items()}
    for key in sorted(all_keys):
        vals = [flat[lbl].get(key, "<absent>") for lbl in run_labels]
        if len(set(str(v) for v in vals)) > 1:
            diffs[key] = {lbl: flat[lbl].get(key, "<absent>") for lbl in run_labels}
    return diffs


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict with dot-separated keys."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items


def compare(
    run_paths: List[str],
    metric: str = "f1",
    slice_by: Optional[str] = None,
) -> CompareResult:
    """Compare a single metric across multiple runs.

    Args:
        run_paths: list of run artifact directory paths.
        metric: metric name to compare (must exist in eval results).
        slice_by: optional tag key to break down comparison.

    Returns:
        CompareResult with per-run values and deltas from the first run.
    """
    if len(run_paths) < 2:
        raise EvalError("compare() requires at least 2 runs")

    evals = []
    for rp in run_paths:
        evals.append((rp, _load_eval_results(rp)))

    # Overall
    overall: Dict[str, float] = {}
    for rp, ev in evals:
        val = ev.get("overall", {}).get(metric)
        if val is None:
            raise EvalError(f"Metric '{metric}' not found in eval results for {rp}")
        label = ev.get("run_id", Path(rp).name)
        overall[label] = float(val)

    # Deltas vs first run
    baseline = list(overall.values())[0]
    deltas = {k: round(v - baseline, 6) for k, v in overall.items()}

    # Slices
    sliced = None
    if slice_by:
        sliced = {}
        for rp, ev in evals:
            label = ev.get("run_id", Path(rp).name)
            tag_data = ev.get("slices", {}).get(slice_by, {})
            for tag_val, tag_metrics in tag_data.items():
                sliced.setdefault(tag_val, {})[label] = tag_metrics.get(metric, None)

    # Config diff
    config_diff = None
    configs: Dict[str, Dict[str, Any]] = {}
    for rp, ev in evals:
        label = ev.get("run_id", Path(rp).name)
        cfg = _load_config(rp)
        if cfg is not None:
            configs[label] = cfg
    if len(configs) >= 2:
        config_diff = _diff_configs(configs) or None

    result = CompareResult(
        runs=[ev.get("run_id", Path(rp).name) for rp, ev in evals],
        metric=metric,
        overall=overall,
        deltas=deltas,
        slices=sliced,
        config_diff=config_diff,
    )

    logger.info("Comparison (%s): %s", metric, overall)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_comparison(result: CompareResult, fmt: str = "table") -> str:
    """Format a CompareResult as a human-readable string."""
    if fmt == "json":
        import json as _json
        data = {
            "runs": result.runs,
            "metric": result.metric,
            "overall": result.overall,
            "deltas": result.deltas,
            "slices": result.slices,
        }
        if result.config_diff:
            data["config_diff"] = result.config_diff
        return _json.dumps(data, indent=2, default=str)

    lines = [
        f"Comparison: {result.metric}",
        "",
    ]

    # Header
    header = f"  {'Run':<35} {'Overall':>10}"
    sep = f"  {'-'*35} {'-'*10}"

    if result.slices:
        tag_vals = sorted(result.slices.keys())
        for tv in tag_vals:
            header += f"  {tv:>15}"
            sep += f"  {'-'*15}"

    lines.append(header)
    lines.append(sep)

    # Rows
    for run_id in result.runs:
        val = result.overall.get(run_id, 0)
        row = f"  {run_id:<35} {val:>10.4f}"
        if result.slices:
            for tv in tag_vals:
                sv = (result.slices.get(tv) or {}).get(run_id)
                row += f"  {sv:>15.4f}" if sv is not None else f"  {'N/A':>15}"
        lines.append(row)

    # Deltas
    lines.append("")
    delta_line = f"  {'Delta (vs first)':<35}"
    for run_id in result.runs:
        d = result.deltas.get(run_id, 0)
        delta_line += f" {d:>+10.4f}"
    lines.append(delta_line)

    # Config diff
    if result.config_diff:
        lines.append("")
        lines.append("  Config differences:")
        for key, vals in result.config_diff.items():
            lines.append(f"    {key}:")
            for run_id, val in vals.items():
                lines.append(f"      {run_id}: {val}")

    lines.append("")
    lines.append("  Note: Slice metrics show correlation, not causation.")

    return "\n".join(lines)
