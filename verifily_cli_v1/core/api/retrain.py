"""RETRAIN runner — mock training with real Verifily artifact generation.

Mock mode (default): produces a complete run directory with deterministic
eval metrics, passes through the real decision engine, writes all contract-
required artifacts.

Real mode: gated behind VERIFILY_ENABLE_REAL_TRAIN=1 env var.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from verifily_cli_v1.commands.pipeline import make_decision
from verifily_cli_v1.core.hashing import compute_chain_hash, sha256_file, sha256_string
from verifily_cli_v1.core.io import ensure_dir, read_json, write_json, write_yaml
from verifily_cli_v1.core.usage import UsageMeter


def run_retrain_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a RETRAIN job.  Returns result dict."""
    t0 = time.monotonic()

    # Parse payload
    project_id = payload.get("project_id", "default")
    dataset_dir = payload.get("dataset_dir")
    base_run_dir = payload.get("base_run_dir")
    contaminated_run_dir = payload.get("contaminated_run_dir")
    metric = payload.get("metric", "f1")
    mode = payload.get("mode", "mock")
    output_dir = payload.get("output_dir")
    seed = payload.get("seed", 42)
    notes = payload.get("notes")
    request_id = payload.get("request_id")

    # Validate dataset_dir
    ds = Path(dataset_dir)
    if not ds.is_dir():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    dataset_jsonl = ds / "dataset.jsonl"
    dataset_hashes = ds / "hashes.json"
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {dataset_dir}")
    if not dataset_hashes.exists():
        raise FileNotFoundError(f"hashes.json not found in {dataset_dir}")

    # Real mode gate
    if mode == "real":
        if os.environ.get("VERIFILY_ENABLE_REAL_TRAIN") != "1":
            raise ValueError(
                "Real training is disabled. Set VERIFILY_ENABLE_REAL_TRAIN=1 "
                "to enable. Use mode='mock' for safe mock training."
            )
        raise NotImplementedError(
            "Real training mode is not yet implemented. Use mode='mock'."
        )

    # Generate run directory
    run_dir = _make_run_dir(output_dir, project_id)

    # Initialize usage meter
    run_id = Path(run_dir).name
    meter = UsageMeter(
        run_id=run_id, request_id=request_id, mode="api",
        ci=True, project_id=project_id,
    )

    # ── Mock mode: generate artifacts ──────────────────────────

    # Read dataset hash for deterministic metrics
    ds_hashes = read_json(dataset_hashes)
    dataset_hash = ds_hashes.get("chain_hash", "")
    if not dataset_hash:
        # Fallback: hash of first file entry
        files = ds_hashes.get("files", {})
        dataset_hash = next(iter(files.values()), "default_hash") if files else "default_hash"

    # 1. config.yaml
    config_data = {
        "task": "sft",
        "mode": "mock",
        "seed": seed,
        "metric": metric,
        "dataset_dir": str(dataset_dir),
        "base_run_dir": base_run_dir,
        "contaminated_run_dir": contaminated_run_dir,
        "notes": notes,
    }
    write_yaml(Path(run_dir) / "config.yaml", config_data)

    # 2. environment.json
    env_data = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "arch": platform.machine(),
        "mode": "mock",
    }
    write_json(Path(run_dir) / "environment.json", env_data)

    # 3. eval/eval_results.json (deterministic)
    mock_metrics = _mock_eval(dataset_hash, seed)
    eval_data = {
        "run_id": run_id,
        "num_examples": 10,
        "overall": mock_metrics,
        "slices": {},
        "hard_examples": [],
        "eval_duration_seconds": 0.01,
    }
    ensure_dir(Path(run_dir) / "eval")
    write_json(Path(run_dir) / "eval" / "eval_results.json", eval_data)

    # 4. run_meta.json
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_meta = {
        "run_id": run_id,
        "status": "completed",
        "task": "sft",
        "mode": "mock",
        "seed": seed,
        "started_at": now_iso,
        "completed_at": now_iso,
        "duration_seconds": 0.01,
        "metrics": {"train_loss": 0.0},
        "artifact_path": run_dir,
    }
    write_json(Path(run_dir) / "run_meta.json", run_meta)

    # 5. hashes.json (hash all written files)
    files_to_hash = [
        "config.yaml",
        "environment.json",
        "eval/eval_results.json",
        "run_meta.json",
    ]
    file_hashes = {}
    for fname in files_to_hash:
        fpath = Path(run_dir) / fname
        if fpath.exists():
            file_hashes[fname] = sha256_file(fpath)
    chain_hash = compute_chain_hash(file_hashes)
    hashes_data = {"files": file_hashes, "chain_hash": chain_hash}
    write_json(Path(run_dir) / "hashes.json", hashes_data)

    # ── Decision ──────────────────────────────────────────────

    # Contract result (mock run always passes — we just wrote all files)
    contract_result = {"valid": True, "checks": [], "has_eval": True}
    meter.record_contract(valid=True, elapsed_ms=0)

    # Baseline
    baseline_results = None
    if base_run_dir:
        baseline_results = _load_eval_results(base_run_dir)

    # Contamination (optional)
    contamination_result = None
    if contaminated_run_dir:
        contamination_result = {
            "status": "FAIL",
            "exact_overlaps": 1,
            "near_duplicates": 0,
            "exact_overlap_fraction": 0.1,
            "near_duplicate_fraction": 0.0,
        }
        # Degrade metrics for contaminated scenario
        mock_metrics["f1"] = max(0.0, mock_metrics["f1"] - 0.15)
        mock_metrics["exact_match"] = max(0.0, mock_metrics["exact_match"] - 0.15)
        eval_data["overall"] = mock_metrics
        write_json(Path(run_dir) / "eval" / "eval_results.json", eval_data)
        # Re-hash
        file_hashes["eval/eval_results.json"] = sha256_file(Path(run_dir) / "eval" / "eval_results.json")
        chain_hash = compute_chain_hash(file_hashes)
        hashes_data = {"files": file_hashes, "chain_hash": chain_hash}
        write_json(Path(run_dir) / "hashes.json", hashes_data)

    meter.record_contamination(
        status=contamination_result["status"] if contamination_result else "PASS",
        checked_rows=0,
        elapsed_ms=0,
    )

    # Ship criteria (reasonable defaults)
    ship_criteria = {
        "min_f1": 0.50,
        "min_exact_match": 0.40,
        "max_f1_regression": 0.05,
        "max_pii_hits": 0,
    }

    decision = make_decision(
        contract_result=contract_result,
        contamination_result=contamination_result,
        eval_results=eval_data,
        baseline_results=baseline_results,
        ship_criteria=ship_criteria,
    )

    meter.record_decision(
        decision=decision["recommendation"],
        exit_code=decision["exit_code"],
        elapsed_ms=0,
    )

    # 6. Decision artifacts
    ensure_dir(Path(run_dir) / "decision")
    write_json(Path(run_dir) / "decision" / "decision.json", decision)
    decision_txt = (
        f"Recommendation: {decision['recommendation']}\n"
        f"Exit code: {decision['exit_code']}\n"
        f"Confidence: {decision['confidence']}\n"
        f"Reasons:\n"
    )
    for r in decision.get("reasons", []):
        decision_txt += f"  - {r}\n"
    (Path(run_dir) / "decision" / "decision.txt").write_text(decision_txt)

    # 7. Usage + audit
    total_ms = int((time.monotonic() - t0) * 1000)
    meter.finalize(total_elapsed_ms=total_ms)
    usage_dict = meter.to_dict()
    write_json(Path(run_dir) / "usage.json", usage_dict)

    # Audit log
    audit_entry = {
        "ts": now_iso,
        "step": "retrain",
        "status": decision["recommendation"],
        "run_id": run_id,
        "project_id": project_id,
        "mode": mode,
        "exit_code": decision["exit_code"],
    }
    with open(Path(run_dir) / "audit_log.jsonl", "a") as f:
        f.write(json.dumps(audit_entry, separators=(",", ":")) + "\n")

    # Build artifact paths
    artifacts = {
        "config": str(Path(run_dir) / "config.yaml"),
        "environment": str(Path(run_dir) / "environment.json"),
        "hashes": str(Path(run_dir) / "hashes.json"),
        "run_meta": str(Path(run_dir) / "run_meta.json"),
        "eval_results": str(Path(run_dir) / "eval" / "eval_results.json"),
        "decision_json": str(Path(run_dir) / "decision" / "decision.json"),
        "decision_txt": str(Path(run_dir) / "decision" / "decision.txt"),
        "usage": str(Path(run_dir) / "usage.json"),
        "audit_log": str(Path(run_dir) / "audit_log.jsonl"),
    }

    return {
        "job_type": "RETRAIN",
        "run_dir": run_dir,
        "decision": decision,
        "exit_code": decision["exit_code"],
        "eval_summary": mock_metrics,
        "usage": usage_dict,
        "artifacts": artifacts,
        "elapsed_ms": total_ms,
    }


# ── Helpers ──────────────────────────────────────────────────────

def _make_run_dir(output_dir: Optional[str], project_id: str) -> str:
    """Create and return a unique run directory path."""
    if output_dir:
        base = Path(output_dir) / "runs"
    else:
        base = Path("/tmp/verifily_retrain") / project_id / "runs"
    ensure_dir(base)

    # Deterministic counter based on existing dirs
    existing = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("retrain_"))
    counter = len(existing) + 1
    run_name = f"retrain_{counter:04d}"
    run_dir = base / run_name
    ensure_dir(run_dir)
    return str(run_dir)


def _mock_eval(dataset_hash: str, seed: int) -> Dict[str, float]:
    """Generate deterministic eval metrics from dataset hash + seed."""
    h = int(hashlib.sha256(f"{dataset_hash}:{seed}".encode()).hexdigest(), 16)
    f1 = 0.60 + (h % 2000) / 10000.0  # range [0.60, 0.80]
    exact_match = f1 - 0.10 - (h % 500) / 10000.0
    return {
        "f1": round(f1, 4),
        "exact_match": round(max(0.0, exact_match), 4),
    }


def _load_eval_results(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load eval_results.json from a run directory."""
    for candidate in ("eval/eval_results.json", "eval_results.json"):
        p = Path(run_dir) / candidate
        if p.exists():
            return read_json(p)
    return None
