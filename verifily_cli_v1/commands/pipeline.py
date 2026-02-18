"""verifily pipeline — end-to-end CI gate: contract → report → contamination → decision."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.panel import Panel

from verifily_cli_v1.core.audit import AuditLogger
from verifily_cli_v1.core.io import read_json, read_yaml, write_json
from verifily_cli_v1.core.lineage_graph import LineageGraphBuilder, write_lineage_artifacts
from verifily_cli_v1.core.registry import (
    RegistrationError,
    registry_store,
    configure_registry_persistence,
)
from verifily_cli_v1.core.scoring import (
    compute_dataset_risk,
    compute_model_health,
    write_score_artifacts,
)
from verifily_cli_v1.core.usage import UsageMeter

console = Console(stderr=True)

# ── Exit codes ──────────────────────────────────────────────────

EXIT_SHIP = 0
EXIT_DONT_SHIP = 1
EXIT_INVESTIGATE = 2
EXIT_CONTRACT_FAIL = 3
EXIT_TOOL_ERROR = 4

EXIT_CODE_TABLE = {
    EXIT_SHIP: "SHIP",
    EXIT_DONT_SHIP: "DONT_SHIP",
    EXIT_INVESTIGATE: "INVESTIGATE",
    EXIT_CONTRACT_FAIL: "CONTRACT_FAIL",
    EXIT_TOOL_ERROR: "TOOL_ERROR",
}


# ── Threshold validation ────────────────────────────────────────

def validate_thresholds(criteria: Dict[str, Any]) -> None:
    """Validate ship_if thresholds at load time. Raises ValueError on invalid."""
    checks = [
        ("min_f1", 0.0, 1.0),
        ("min_exact_match", 0.0, 1.0),
        ("max_f1_regression", 0.0, None),
        ("max_pii_hits", 0, None),
    ]
    for key, lo, hi in checks:
        val = criteria.get(key)
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            raise ValueError(f"ship_if.{key} must be a number, got {type(val).__name__}")
        if lo is not None and val < lo:
            raise ValueError(f"ship_if.{key}={val} invalid (must be >= {lo})")
        if hi is not None and val > hi:
            raise ValueError(f"ship_if.{key}={val} invalid (must be <= {hi})")


# ── Decision logic ──────────────────────────────────────────────

# Blocker precedence (highest to lowest):
# 1. Contract invalid
# 2. PII violation
# 3. Contamination FAIL
# 4. Eval missing (when thresholds require metrics)
# 5. F1 below threshold
# 6. Exact match below threshold
# 7. F1 regression exceeded

def make_decision(
    *,
    contract_result: Dict[str, Any],
    contamination_result: Dict[str, Any] | None,
    report_result: Dict[str, Any] | None = None,
    eval_results: Dict[str, Any] | None,
    baseline_results: Dict[str, Any] | None,
    ship_criteria: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a SHIP / DONT_SHIP / INVESTIGATE decision.

    Blocker precedence: contract > PII > contamination > eval missing > metric thresholds.
    Any blocker → DONT_SHIP. Risk flags only → INVESTIGATE. No flags → SHIP.

    Returns:
        {
            "recommendation": str,
            "exit_code": int,
            "confidence": float,
            "reasons": [str],
            "risk_flags": [str],
            "blocker_precedence": [str],
            "metrics": dict,
            "baseline_metrics": dict | None,
            "deltas": dict | None,
        }
    """
    reasons: list[str] = []
    risk_flags: list[str] = []
    blockers: list[str] = []

    # 1. Contract (immediate return — highest precedence)
    if not contract_result.get("valid"):
        missing = [c["file"] for c in contract_result.get("checks", []) if c.get("status") == "FAIL"]
        return {
            "recommendation": "DONT_SHIP",
            "exit_code": EXIT_CONTRACT_FAIL,
            "confidence": 1.0,
            "reasons": ["Run contract failed — required artifacts missing"],
            "risk_flags": ["contract_fail"],
            "blocker_precedence": ["CONTRACT_INVALID"],
            "metrics": {},
            "baseline_metrics": None,
            "deltas": None,
            "missing_files": missing,
        }

    # 2. PII violation
    max_pii_hits = ship_criteria.get("max_pii_hits")
    if max_pii_hits is not None and report_result:
        pii_hits = report_result.get("pii_total_hits", 0)
        if pii_hits > max_pii_hits:
            blockers.append(f"PII hits ({pii_hits}) exceed max allowed ({max_pii_hits})")
            risk_flags.append("pii_violation")

    # 3. Contamination
    if contamination_result:
        cont_status = contamination_result.get("status", "PASS")
        if cont_status == "FAIL":
            blockers.append(
                f"Contamination FAIL: {contamination_result.get('exact_overlaps', 0)} exact overlaps, "
                f"{contamination_result.get('near_duplicates', 0)} near-duplicates"
            )
            risk_flags.append("contamination_fail")
        elif cont_status == "WARN":
            risk_flags.append("contamination_warn")
            reasons.append(
                f"Contamination WARN: {contamination_result.get('near_duplicates', 0)} near-duplicates"
            )

    # 4. Metrics extraction
    metrics: Dict[str, float] = {}
    baseline_metrics: Dict[str, float] | None = None
    deltas: Dict[str, float] | None = None

    if eval_results:
        overall = eval_results.get("overall", eval_results.get("aggregate", {}))
        if isinstance(overall, dict):
            metrics = {k: v for k, v in overall.items() if isinstance(v, (int, float))}

    if baseline_results:
        b_overall = baseline_results.get("overall", baseline_results.get("aggregate", {}))
        if isinstance(b_overall, dict):
            baseline_metrics = {k: v for k, v in b_overall.items() if isinstance(v, (int, float))}

    if metrics and baseline_metrics:
        deltas = {}
        for k in metrics:
            if k in baseline_metrics:
                deltas[k] = round(metrics[k] - baseline_metrics[k], 6)

    # Quality gate (optional)
    min_quality = ship_criteria.get("min_quality_score")
    if min_quality is not None and report_result:
        q_score = report_result.get("quality", {}).get("quality_score")
        if q_score is not None and q_score < min_quality:
            blockers.append(f"Quality score ({q_score}) below threshold ({min_quality})")
            risk_flags.append("quality_below_threshold")

    # 5. Eval missing (when thresholds require metrics)
    min_f1 = ship_criteria.get("min_f1")
    min_exact_match = ship_criteria.get("min_exact_match")
    if not metrics and (min_f1 is not None or min_exact_match is not None):
        blockers.append("Eval results missing or invalid — cannot verify metric thresholds")
        risk_flags.append("eval_missing")

    # 6. F1 threshold
    if min_f1 is not None and "f1" in metrics:
        if metrics["f1"] < min_f1:
            blockers.append(f"F1 ({metrics['f1']:.4f}) below threshold ({min_f1})")
            risk_flags.append("f1_below_threshold")
        else:
            reasons.append(f"F1 ({metrics['f1']:.4f}) meets threshold ({min_f1})")

    # 7. Exact match threshold
    if min_exact_match is not None and "exact_match" in metrics:
        if metrics["exact_match"] < min_exact_match:
            blockers.append(f"Exact match ({metrics['exact_match']:.4f}) below threshold ({min_exact_match})")
            risk_flags.append("exact_match_below_threshold")
        else:
            reasons.append(f"Exact match ({metrics['exact_match']:.4f}) meets threshold ({min_exact_match})")

    # 8. F1 regression
    max_regression = ship_criteria.get("max_f1_regression")
    if max_regression is not None and deltas and "f1" in deltas:
        if deltas["f1"] < -max_regression:
            blockers.append(
                f"F1 regression ({deltas['f1']:+.4f}) exceeds max allowed (-{max_regression})"
            )
            risk_flags.append("regression_exceeded")

    # ── Final decision ──────────────────────────────────────────
    if blockers:
        recommendation = "DONT_SHIP"
        exit_code = EXIT_DONT_SHIP
        # More blockers = higher confidence in DONT_SHIP (0.1–0.3)
        confidence = min(0.3, 0.1 * len(blockers))
        reasons = blockers + reasons
    elif risk_flags:
        recommendation = "INVESTIGATE"
        exit_code = EXIT_INVESTIGATE
        # Fewer risk flags = higher confidence (0.5–0.8)
        confidence = max(0.5, 0.8 - 0.1 * len(risk_flags))
        reasons.append("Risk flags present but no hard blockers")
    else:
        recommendation = "SHIP"
        exit_code = EXIT_SHIP
        # More passing criteria = higher confidence (0.9–1.0)
        passing = len([r for r in reasons if "meets" in r.lower() or "passed" in r.lower()])
        confidence = min(1.0, 0.9 + 0.02 * passing)
        if not reasons:
            reasons.append("All criteria passed")

    return {
        "recommendation": recommendation,
        "exit_code": exit_code,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "risk_flags": risk_flags,
        "blocker_precedence": list(blockers),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "deltas": deltas,
    }


# ── Integration logging ────────────────────────────────────────

def _log_integrations(
    results: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    ci: bool = False,
    verbose: bool = False,
) -> None:
    """Log pipeline results to enabled integrations (W&B, MLflow).

    Reads ``wandb:`` and ``mlflow:`` sections from the pipeline config.
    Failures never block the pipeline.
    """
    # W&B
    wandb_cfg = cfg.get("wandb", {})
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled"):
        try:
            from verifily_cli_v1.integrations.wandb import (
                log_pipeline_run,
                wandb_config_from_dict,
            )
            wc = wandb_config_from_dict(wandb_cfg)
            url = log_pipeline_run(results, wc)
            if url and not ci:
                console.print(f"  W&B: {url}")
        except Exception as e:
            if verbose:
                console.print(f"[dim]W&B logging skipped: {e}[/dim]")

    # MLflow
    mlflow_cfg = cfg.get("mlflow", {})
    if isinstance(mlflow_cfg, dict) and mlflow_cfg.get("enabled"):
        try:
            from verifily_cli_v1.integrations.mlflow import (
                log_pipeline_run,
                mlflow_config_from_dict,
            )
            mc = mlflow_config_from_dict(mlflow_cfg)
            run_id = log_pipeline_run(results, mc)
            if run_id and not ci:
                console.print(f"  MLflow run: {run_id}")
        except Exception as e:
            if verbose:
                console.print(f"[dim]MLflow logging skipped: {e}[/dim]")


# ── Pipeline orchestrator ───────────────────────────────────────

def run_pipeline(
    config_path: Union[str, Path],
    *,
    ci: bool = False,
    output_dir: str | None = None,
    verbose: bool = False,
    request_id: str | None = None,
    mode: str = "cli",
    project_id: str | None = None,
    integration_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full Verifily pipeline from a config file.

    Config YAML schema:
        run_dir: path
        train_data: path
        eval_data: path
        baseline_run: path (optional)
        ship_if:
            min_f1: float
            min_exact_match: float (optional)
            max_f1_regression: float (optional)
            max_pii_hits: int (optional)
    """
    from verifily_cli_v1.commands.contract_check import validate_run_contract
    from verifily_cli_v1.commands.contamination import check_contamination
    from verifily_cli_v1.commands.report import dataset_report

    pipeline_t0 = time.monotonic()
    run_id = uuid.uuid4().hex[:12]

    audit = AuditLogger(run_id=run_id, request_id=request_id, project=project_id)
    meter = UsageMeter(run_id=run_id, request_id=request_id, mode=mode, ci=ci, project_id=project_id)

    cfg = read_yaml(config_path)

    # Merge CLI integration overrides (--wandb, --mlflow flags)
    if integration_overrides:
        for section, overrides in integration_overrides.items():
            cfg.setdefault(section, {}).update(overrides)

    config_dir = Path(config_path).parent

    # Resolve paths relative to config file
    def _resolve(p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (config_dir / pp).resolve()

    run_dir = _resolve(cfg["run_dir"])
    train_data = _resolve(cfg["train_data"])
    eval_data = _resolve(cfg["eval_data"])
    baseline_run = _resolve(cfg["baseline_run"]) if cfg.get("baseline_run") else None
    ship_criteria = cfg.get("ship_if", {})

    # Validate thresholds fail-fast
    try:
        validate_thresholds(ship_criteria)
    except ValueError as e:
        decision = {
            "recommendation": "DONT_SHIP",
            "exit_code": EXIT_TOOL_ERROR,
            "confidence": 1.0,
            "reasons": [f"Invalid threshold: {e}"],
            "risk_flags": ["invalid_thresholds"],
            "blocker_precedence": [f"INVALID_THRESHOLD: {e}"],
            "metrics": {},
            "baseline_metrics": None,
            "deltas": None,
        }
        results_err: Dict[str, Any] = {"config": str(config_path), "run_id": run_id, "decision": decision}
        _print_decision(decision, ci=ci)
        if output_dir:
            _write_artifacts(results_err, output_dir, ci=ci)
        return results_err

    results: Dict[str, Any] = {"config": str(config_path), "run_id": run_id}

    # Step 1: Contract check
    if not ci:
        console.print(Panel("[bold]Step 1/4: Contract Check[/bold]", border_style="blue"))
    audit.start("CONTRACT", inputs={"run_dir": str(run_dir)})
    t0 = time.monotonic()
    contract = validate_run_contract(run_dir)
    contract_ms = int((time.monotonic() - t0) * 1000)
    results["contract"] = contract
    meter.record_contract(valid=contract["valid"], elapsed_ms=contract_ms)
    if contract["valid"]:
        audit.ok("CONTRACT", elapsed_ms=contract_ms, summary={"valid": True})
    else:
        audit.fail("CONTRACT", elapsed_ms=contract_ms, summary={"valid": False})
    if not ci:
        status_style = "green" if contract["valid"] else "red"
        console.print(f"  [{status_style}]{'PASS' if contract['valid'] else 'FAIL'}[/{status_style}]")

    if not contract["valid"]:
        decision = make_decision(
            contract_result=contract,
            contamination_result=None,
            report_result=None,
            eval_results=None,
            baseline_results=None,
            ship_criteria=ship_criteria,
        )
        results["decision"] = decision
        meter.record_decision(
            decision=decision["recommendation"],
            exit_code=decision["exit_code"],
            elapsed_ms=0,
        )
        total_ms = int((time.monotonic() - pipeline_t0) * 1000)
        meter.finalize(total_elapsed_ms=total_ms)
        results["usage"] = meter.to_dict()
        _print_decision(decision, ci=ci)
        if output_dir:
            _write_artifacts(results, output_dir, ci=ci)
            audit.write(output_dir)
            write_json(Path(output_dir) / "usage.json", meter.to_dict())
        return results

    # Step 2: Report
    if not ci:
        console.print(Panel("[bold]Step 2/4: Dataset Report[/bold]", border_style="blue"))
    audit.start("REPORT", inputs={"train_data": str(train_data)})
    t0 = time.monotonic()
    report = dataset_report(train_data)
    report_ms = int((time.monotonic() - t0) * 1000)
    results["report"] = report
    train_bytes = train_data.stat().st_size if train_data.exists() else 0
    meter.record_report(
        rows_in=report.get("row_count", 0),
        bytes_in=train_bytes,
        elapsed_ms=report_ms,
    )
    if report.get("pii_clean", True):
        audit.ok("REPORT", elapsed_ms=report_ms, summary={
            "row_count": report.get("row_count", 0),
            "pii_clean": True,
        })
    else:
        audit.warn("REPORT", elapsed_ms=report_ms, summary={
            "row_count": report.get("row_count", 0),
            "pii_clean": False,
            "pii_total_hits": report.get("pii_total_hits", 0),
        })
    if not ci:
        pii_label = "CLEAN" if report["pii_clean"] else f"{report['pii_total_hits']} PII HITS"
        pii_style = "green" if report["pii_clean"] else "red"
        console.print(f"  {report['row_count']} rows | PII: [{pii_style}]{pii_label}[/{pii_style}]")

    # Step 3: Contamination
    if not ci:
        console.print(Panel("[bold]Step 3/4: Contamination Gate[/bold]", border_style="blue"))
    audit.start("CONTAMINATION", inputs={
        "train_data": str(train_data),
        "eval_data": str(eval_data),
    })
    t0 = time.monotonic()
    contamination = check_contamination(train_data, eval_data)
    contam_ms = int((time.monotonic() - t0) * 1000)
    results["contamination"] = contamination
    meter.record_contamination(
        status=contamination["status"],
        checked_rows=contamination.get("train_rows", 0) + contamination.get("eval_rows", 0),
        elapsed_ms=contam_ms,
    )
    contam_summary = {
        "status": contamination["status"],
        "exact_overlaps": contamination.get("exact_overlaps", 0),
        "near_duplicates": contamination.get("near_duplicates", 0),
    }
    if contamination["status"] == "FAIL":
        audit.fail("CONTAMINATION", elapsed_ms=contam_ms, summary=contam_summary)
    elif contamination["status"] == "WARN":
        audit.warn("CONTAMINATION", elapsed_ms=contam_ms, summary=contam_summary)
    else:
        audit.ok("CONTAMINATION", elapsed_ms=contam_ms, summary=contam_summary)
    if not ci:
        c_style = {"PASS": "green", "FAIL": "red", "WARN": "yellow"}.get(contamination["status"], "white")
        console.print(
            f"  [{c_style}]{contamination['status']}[/{c_style}] — "
            f"{contamination['exact_overlaps']} exact, {contamination['near_duplicates']} near"
        )

    # Step 4: Decision
    if not ci:
        console.print(Panel("[bold]Step 4/4: Decision[/bold]", border_style="blue"))
    audit.start("DECISION")
    t0 = time.monotonic()

    eval_results = None
    for candidate in ["eval/eval_results.json", "eval_results.json"]:
        p = run_dir / candidate
        if p.exists():
            eval_results = read_json(p)
            break

    baseline_results = None
    if baseline_run:
        for candidate in ["eval/eval_results.json", "eval_results.json"]:
            p = baseline_run / candidate
            if p.exists():
                baseline_results = read_json(p)
                break

    decision = make_decision(
        contract_result=contract,
        contamination_result=contamination,
        report_result=report,
        eval_results=eval_results,
        baseline_results=baseline_results,
        ship_criteria=ship_criteria,
    )
    decision_ms = int((time.monotonic() - t0) * 1000)
    results["decision"] = decision
    meter.record_decision(
        decision=decision["recommendation"],
        exit_code=decision["exit_code"],
        elapsed_ms=decision_ms,
    )
    audit.ok("DECISION", elapsed_ms=decision_ms, summary={
        "recommendation": decision["recommendation"],
        "exit_code": decision["exit_code"],
        "confidence": decision["confidence"],
    })

    # Finalize usage
    total_ms = int((time.monotonic() - pipeline_t0) * 1000)
    meter.finalize(total_elapsed_ms=total_ms)
    results["usage"] = meter.to_dict()

    # ── Integration logging (opt-in) ─────────────────────────
    _log_integrations(results, cfg, ci=ci, verbose=verbose)

    _print_decision(decision, ci=ci)

    if output_dir:
        _write_artifacts(results, output_dir, ci=ci)
        audit.write(output_dir)
        write_json(Path(output_dir) / "usage.json", meter.to_dict())

        # Generate lineage graph for the run
        try:
            builder = LineageGraphBuilder()
            graph = builder.build_from_run(Path(output_dir))
            lineage_files = write_lineage_artifacts(graph, output_dir)
            if not ci:
                console.print(f"  Lineage: {lineage_files.get('json', 'N/A')}")
        except Exception as e:
            if verbose:
                console.print(f"[dim]Lineage generation skipped: {e}[/dim]")

        # Compute and write Risk Score and Health Index
        try:
            risk_score = compute_dataset_risk(
                report_result=results.get("report"),
                contamination_result=results.get("contamination"),
                contract_result=results.get("contract"),
            )
            health_index = compute_model_health(
                decision_result=results.get("decision"),
                eval_results=eval_results,
                reproducibility_ok=results.get("contract", {}).get("valid", False),
            )
            score_files = write_score_artifacts(risk_score, health_index, output_dir)
            if not ci:
                console.print(f"  Risk Score: {risk_score.total:.1f} ({risk_score.level})")
                console.print(f"  Health Index: {health_index.total:.1f} ({health_index.level})")
            
            # Add to results for API response
            results["risk_score"] = risk_score.to_dict()
            results["health_index"] = health_index.to_dict()
        except Exception as e:
            if verbose:
                console.print(f"[dim]Score computation skipped: {e}[/dim]")

        # Auto-register model if decision is SHIP
        if decision.get("recommendation") == "SHIP":
            try:
                # Configure persistence if enabled
                import os
                if os.environ.get("VERIFILY_REGISTRY_PERSIST") == "1":
                    data_dir = os.environ.get("VERIFILY_DATA_DIR", ".verifily")
                    persist_path = Path(data_dir) / "registry.jsonl"
                    configure_registry_persistence(persist_path)
                
                from verifily_cli_v1.core.registry import RegistrationRequest
                
                reg_request = RegistrationRequest(
                    run_dir=str(output_dir),
                    registered_by="pipeline",
                )
                record = registry_store.register(reg_request)
                
                if not ci:
                    console.print(f"  Registry: {record.model_id}:{record.version} (stage: {record.stage})")
                
                results["model_registry"] = {
                    "registered": True,
                    "model_id": record.model_id,
                    "version": record.version,
                    "stage": record.stage,
                }
            except RegistrationError as e:
                if not ci:
                    console.print(f"  [yellow]Registry: {e}[/yellow]")
                results["model_registry"] = {
                    "registered": False,
                    "error": str(e),
                }
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Auto-registration skipped: {e}[/dim]")

    return results


def _print_decision(decision: Dict[str, Any], ci: bool = False) -> None:
    """Print decision summary.

    CI mode: raw JSON to stdout, deterministic final line to stderr.
    Interactive mode: Rich-formatted output to stderr.
    """
    rec = decision["recommendation"]
    code = decision["exit_code"]
    conf = decision["confidence"]
    style_map = {
        "SHIP": "green bold",
        "DONT_SHIP": "red bold",
        "INVESTIGATE": "yellow bold",
    }
    style = style_map.get(rec, "white")

    if ci:
        # CI mode: raw JSON to stdout (no Rich markup/ANSI)
        sys.stdout.write(json.dumps(decision, indent=2) + "\n")
        sys.stdout.flush()
    else:
        console.print(f"\n  [{style}]DECISION: {rec}[/{style}]  (confidence: {conf:.0%}, exit: {code})")
        for reason in decision["reasons"]:
            console.print(f"    • {reason}")
        if decision["risk_flags"]:
            console.print(f"    Risk flags: {', '.join(decision['risk_flags'])}")
        console.print()

    # Always print deterministic final line to stderr
    console.print(f"Decision: {rec} (exit {code})")


def _write_artifacts(results: Dict[str, Any], output_dir: str, ci: bool = False) -> None:
    """Write pipeline artifacts to output directory."""
    from verifily_cli_v1.core.io import ensure_dir
    out = ensure_dir(output_dir)
    write_json(out / "pipeline_result.json", results)
    if "decision" in results:
        write_json(out / "decision_summary.json", results["decision"])
    if not ci:
        console.print(f"Artifacts written deterministically to: {out}\n")


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    config: str,
    ci: bool = False,
    output: str | None = None,
    verbose: bool = False,
    request_id: str | None = None,
    mode: str = "cli",
    project_id: str | None = None,
    integration_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the pipeline and return results."""
    result = run_pipeline(
        config, ci=ci, output_dir=output, verbose=verbose,
        request_id=request_id, mode=mode, project_id=project_id,
        integration_overrides=integration_overrides,
    )
    return result
