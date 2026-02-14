"""Risk Score and Health Index computation for Verifily.

Provides quantitative assessments of:
- Dataset Risk Score (0-100, higher = riskier): Is this dataset safe to train on?
- Model Health Index (0-100, higher = healthier): Is this model safe to ship?

All scoring is deterministic and based only on metadata (no raw data).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from verifily_cli_v1.core.io import write_json


@dataclass
class ScoreComponent:
    """A single component contributing to a score.
    
    Attributes:
        name: Component name (e.g., "PII Hits", "Contamination")
        value: Raw value (0-100 scale, interpreted based on context)
        weight: Weight multiplier (0.0-1.0) for contribution to total
        detail: Human-readable explanation
        contribution: Computed contribution to final score (value * weight)
    """
    name: str
    value: float
    weight: float
    detail: str
    contribution: float = field(default=0.0)

    def __post_init__(self):
        if self.contribution == 0.0:
            self.contribution = self.value * self.weight


@dataclass
class RiskScore:
    """Dataset Risk Score (0-100, higher = riskier).
    
    Interprets risk levels as:
    - 0-25: LOW (safe to use)
    - 26-50: MEDIUM (review recommended)
    - 51-75: HIGH (significant concerns)
    - 76-100: CRITICAL (do not use)
    """
    total: float
    components: List[ScoreComponent]
    summary: str
    level: str = field(default="")
    
    def __post_init__(self):
        # Clamp to valid range
        self.total = max(0.0, min(100.0, self.total))
        # Determine level
        if self.total <= 25:
            self.level = "LOW"
        elif self.total <= 50:
            self.level = "MEDIUM"
        elif self.total <= 75:
            self.level = "HIGH"
        else:
            self.level = "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": round(self.total, 2),
            "level": self.level,
            "summary": self.summary,
            "components": [
                {
                    "name": c.name,
                    "value": round(c.value, 2),
                    "weight": round(c.weight, 2),
                    "detail": c.detail,
                    "contribution": round(c.contribution, 2),
                }
                for c in self.components
            ],
        }


@dataclass
class HealthIndex:
    """Model Health Index (0-100, higher = healthier).
    
    Interprets health levels as:
    - 0-25: POOR (not shippable)
    - 26-50: FAIR (major concerns)
    - 51-75: GOOD (minor issues)
    - 76-100: EXCELLENT (ready to ship)
    """
    total: float
    components: List[ScoreComponent]
    summary: str
    level: str = field(default="")
    
    def __post_init__(self):
        # Clamp to valid range
        self.total = max(0.0, min(100.0, self.total))
        # Determine level
        if self.total >= 76:
            self.level = "EXCELLENT"
        elif self.total >= 51:
            self.level = "GOOD"
        elif self.total >= 26:
            self.level = "FAIR"
        else:
            self.level = "POOR"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": round(self.total, 2),
            "level": self.level,
            "summary": self.summary,
            "components": [
                {
                    "name": c.name,
                    "value": round(c.value, 2),
                    "weight": round(c.weight, 2),
                    "detail": c.detail,
                    "contribution": round(c.contribution, 2),
                }
                for c in self.components
            ],
        }


@dataclass
class RiskHealthSummary:
    """Combined summary of both risk and health scores."""
    risk_score: RiskScore
    health_index: HealthIndex
    verdict: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score.to_dict(),
            "health_index": self.health_index.to_dict(),
            "verdict": self.verdict,
            "recommendations": self.recommendations,
        }


def _clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def compute_dataset_risk(
    report_result: Optional[Dict[str, Any]],
    contamination_result: Optional[Dict[str, Any]],
    contract_result: Optional[Dict[str, Any]],
    privacy_ctx: Optional[Dict[str, Any]] = None,
    min_rows_threshold: int = 100,
) -> RiskScore:
    """Compute Dataset Risk Score (0-100, higher = riskier).
    
    Components:
    - PII hits from report (weight: 0.30)
    - Duplicate rate (weight: 0.15)
    - Schema/contract issues (weight: 0.25)
    - Contamination status (weight: 0.20)
    - Privacy mode penalty/bonus (weight: 0.10)
    - Dataset size risk (weight: 0.10)
    
    Args:
        report_result: Dataset report output
        contamination_result: Contamination check results
        contract_result: Contract validation results
        privacy_ctx: Privacy context (mode, etc.)
        min_rows_threshold: Minimum recommended rows
        
    Returns:
        RiskScore with total 0-100 and component breakdown
    """
    components: List[ScoreComponent] = []
    
    # 1. PII Risk (0-30 contribution)
    pii_value = 0.0
    pii_detail = "No PII detected"
    if report_result:
        pii_hits = report_result.get("pii_total_hits", 0)
        pii_clean = report_result.get("pii_clean", True)
        if not pii_clean and pii_hits > 0:
            # Scale: 1-10 hits = 10-30 risk, 10+ hits = 30 risk
            pii_value = min(30, 10 + (pii_hits * 2))
            pii_detail = f"{pii_hits} PII hits detected"
        elif not pii_clean:
            pii_value = 15
            pii_detail = "PII detected (count unknown)"
    else:
        pii_value = 20  # Unknown = moderate risk
        pii_detail = "No report available - PII status unknown"
    
    components.append(ScoreComponent(
        name="PII Risk",
        value=pii_value,
        weight=1.0,  # Value already scaled to contribution
        detail=pii_detail,
    ))
    
    # 2. Duplicate Rate (0-15 contribution)
    dup_value = 0.0
    dup_detail = "No duplicates detected"
    if report_result:
        # Look for duplicate info in report
        row_count = report_result.get("row_count", 0)
        # Estimate dup rate from any available metrics
        dup_rate = report_result.get("duplicate_rate", 0)
        if dup_rate > 0:
            dup_value = min(15, dup_rate * 100 * 0.5)  # 30% dups = 15 risk
            dup_detail = f"{dup_rate:.1%} estimated duplicate rate"
    components.append(ScoreComponent(
        name="Duplicate Rate",
        value=dup_value,
        weight=1.0,
        detail=dup_detail,
    ))
    
    # 3. Contract/Schema Issues (0-25 contribution)
    contract_value = 0.0
    contract_detail = "Contract valid"
    if contract_result:
        valid = contract_result.get("valid", False)
        if not valid:
            missing = contract_result.get("missing_files", [])
            failed = [c for c in contract_result.get("checks", []) if c.get("status") == "FAIL"]
            contract_value = min(25, 10 + len(missing) * 5 + len(failed) * 5)
            contract_detail = f"Contract invalid: {len(missing)} missing, {len(failed)} failed checks"
    else:
        contract_value = 15
        contract_detail = "No contract validation performed"
    components.append(ScoreComponent(
        name="Contract Issues",
        value=contract_value,
        weight=1.0,
        detail=contract_detail,
    ))
    
    # 4. Contamination Status (0-20 contribution)
    contam_value = 0.0
    contam_detail = "No contamination check performed"
    if contamination_result:
        status = contamination_result.get("status", "UNKNOWN")
        if status == "FAIL":
            exact = contamination_result.get("exact_overlaps", 0)
            near = contamination_result.get("near_duplicates", 0)
            contam_value = 50  # High penalty, but cap at contribution
            contam_detail = f"Contamination FAIL: {exact} exact, {near} near duplicates"
        elif status == "WARN":
            contam_value = 20
            contam_detail = "Contamination WARN: review recommended"
        elif status == "PASS":
            contam_value = 0
            contam_detail = "Contamination PASS"
    components.append(ScoreComponent(
        name="Contamination",
        value=contam_value,
        weight=1.0,
        detail=contam_detail,
    ))
    
    # 5. Privacy Mode Context (±10 bonus/penalty)
    privacy_value = 0.0
    privacy_detail = "Local mode (neutral)"
    if privacy_ctx:
        mode = privacy_ctx.get("privacy_mode", "local").lower()
        if mode == "local":
            privacy_value = -5  # Bonus (reduces risk)
            privacy_detail = "LOCAL mode: privacy-preserving"
        elif mode == "remote":
            privacy_value = 10  # Penalty
            privacy_detail = "REMOTE mode: data leaves local environment"
    components.append(ScoreComponent(
        name="Privacy Context",
        value=max(0, privacy_value),  # Store positive, apply as adjustment
        weight=1.0 if privacy_value > 0 else 0,  # Only apply if penalty
        detail=privacy_detail,
    ))
    # Apply bonus separately
    bonus = abs(min(0, privacy_value))
    
    # 6. Dataset Size Risk (0-10 contribution)
    size_value = 0.0
    size_detail = "Dataset size adequate"
    if report_result:
        row_count = report_result.get("row_count", 0)
        if row_count < min_rows_threshold:
            size_value = min(10, (min_rows_threshold - row_count) / min_rows_threshold * 10)
            size_detail = f"Small dataset: {row_count} rows (threshold: {min_rows_threshold})"
    components.append(ScoreComponent(
        name="Dataset Size",
        value=size_value,
        weight=1.0,
        detail=size_detail,
    ))
    
    # Calculate total
    total = sum(c.contribution for c in components) - bonus
    total = _clamp(total)
    
    # Generate summary
    top_risks = sorted(components, key=lambda c: c.contribution, reverse=True)[:2]
    if total <= 25:
        summary = f"LOW RISK: Dataset appears safe. Top factors: {', '.join(r.name for r in top_risks)}"
    elif total <= 50:
        summary = f"MEDIUM RISK: Review recommended. Main concerns: {', '.join(r.name for r in top_risks)}"
    elif total <= 75:
        summary = f"HIGH RISK: Significant concerns. Critical issues: {', '.join(r.name for r in top_risks)}"
    else:
        summary = f"CRITICAL RISK: Do not use this dataset. Critical: {', '.join(r.name for r in top_risks)}"
    
    return RiskScore(total=total, components=components, summary=summary)


def compute_model_health(
    decision_result: Optional[Dict[str, Any]],
    eval_results: Optional[Dict[str, Any]],
    regression_result: Optional[Dict[str, Any]] = None,
    reproducibility_ok: bool = False,
) -> HealthIndex:
    """Compute Model Health Index (0-100, higher = healthier).
    
    Components:
    - Decision outcome (weight: 0.40)
    - Evaluation metric score (weight: 0.30)
    - Regression status (weight: 0.15)
    - Reproducibility verified (weight: 0.15)
    
    Args:
        decision_result: Decision gate output
        eval_results: Evaluation results with metrics
        regression_result: Regression detection results
        reproducibility_ok: Whether reproducibility hash verified
        
    Returns:
        HealthIndex with total 0-100 and component breakdown
    """
    components: List[ScoreComponent] = []
    
    # 1. Decision Outcome (0-40 contribution)
    decision_value = 0.0
    decision_detail = "No decision available"
    if decision_result:
        recommendation = decision_result.get("recommendation", "UNKNOWN")
        confidence = decision_result.get("confidence", 0.5)
        
        if recommendation == "SHIP":
            decision_value = 40 * confidence
            decision_detail = f"SHIP decision (confidence: {confidence:.0%})"
        elif recommendation == "INVESTIGATE":
            decision_value = 20 * confidence
            decision_detail = f"INVESTIGATE decision (confidence: {confidence:.0%})"
        elif recommendation == "DONT_SHIP":
            decision_value = 5  # Small base score even for blocked
            decision_detail = "DONT_SHIP decision"
    else:
        decision_value = 10
        decision_detail = "No decision recorded"
    components.append(ScoreComponent(
        name="Decision Outcome",
        value=decision_value,
        weight=1.0,
        detail=decision_detail,
    ))
    
    # 2. Evaluation Metrics (0-30 contribution)
    metric_value = 0.0
    metric_detail = "No evaluation metrics"
    if eval_results:
        overall = eval_results.get("overall", eval_results.get("aggregate", {}))
        if isinstance(overall, dict):
            # Use F1 if available, otherwise accuracy, otherwise best available
            f1 = overall.get("f1")
            acc = overall.get("accuracy")
            em = overall.get("exact_match")
            
            score = None
            if f1 is not None:
                score = f1
                metric_detail = f"F1 score: {f1:.3f}"
            elif acc is not None:
                score = acc
                metric_detail = f"Accuracy: {acc:.3f}"
            elif em is not None:
                score = em
                metric_detail = f"Exact match: {em:.3f}"
            
            if score is not None:
                # Scale 0-1 to 0-30
                metric_value = score * 30
            else:
                metric_value = 10
                metric_detail = "No recognized metrics in eval results"
    else:
        metric_value = 5
        metric_detail = "No evaluation performed"
    components.append(ScoreComponent(
        name="Evaluation Metrics",
        value=metric_value,
        weight=1.0,
        detail=metric_detail,
    ))
    
    # 3. Regression Status (0-15 contribution, penalty)
    regression_value = 15.0  # Start full, subtract if clean
    regression_detail = "No regression check performed"
    if regression_result:
        detected = regression_result.get("regression_detected", False)
        if detected:
            delta = regression_result.get("delta", 0)
            regression_value = 0  # Full penalty
            regression_detail = f"Regression detected: {delta:+.3f} drop"
        else:
            regression_value = 15
            regression_detail = "No regression detected"
    else:
        regression_value = 10  # Neutral if unknown
        regression_detail = "No baseline comparison available"
    components.append(ScoreComponent(
        name="Regression Status",
        value=regression_value,
        weight=1.0,
        detail=regression_detail,
    ))
    
    # 4. Reproducibility (0-15 contribution)
    repro_value = 0.0
    repro_detail = "Reproducibility not verified"
    if reproducibility_ok:
        repro_value = 15
        repro_detail = "Reproducibility hash verified"
    else:
        repro_value = 5
        repro_detail = "Reproducibility check failed or not performed"
    components.append(ScoreComponent(
        name="Reproducibility",
        value=repro_value,
        weight=1.0,
        detail=repro_detail,
    ))
    
    # Calculate total
    total = sum(c.contribution for c in components)
    total = _clamp(total)
    
    # Generate summary
    top_factors = sorted(components, key=lambda c: c.contribution, reverse=True)[:2]
    if total >= 76:
        summary = f"EXCELLENT: Model ready to ship. Strengths: {', '.join(f.name for f in top_factors)}"
    elif total >= 51:
        summary = f"GOOD: Minor issues. Best factors: {', '.join(f.name for f in top_factors)}"
    elif total >= 26:
        summary = f"FAIR: Major concerns. Best factors: {', '.join(f.name for f in top_factors)}"
    else:
        summary = f"POOR: Not shippable. Only positives: {', '.join(f.name for f in top_factors)}"
    
    return HealthIndex(total=total, components=components, summary=summary)


def compute_verdict(
    risk_score: RiskScore,
    health_index: HealthIndex,
) -> Tuple[str, List[str]]:
    """Compute overall verdict and recommendations.
    
    Returns:
        Tuple of (verdict string, list of recommendations)
    """
    recommendations: List[str] = []
    
    # Build verdict based on risk/health matrix
    if risk_score.level == "CRITICAL" or health_index.level == "POOR":
        verdict = "BLOCKED"
    elif risk_score.level == "HIGH" and health_index.level in ("FAIR", "POOR"):
        verdict = "BLOCKED"
    elif risk_score.level in ("LOW", "MEDIUM") and health_index.level in ("GOOD", "EXCELLENT"):
        verdict = "APPROVED"
    else:
        verdict = "REVIEW_REQUIRED"
    
    # Generate recommendations
    if risk_score.total > 50:
        recommendations.append("Address dataset quality issues before training")
    if health_index.total < 50:
        recommendations.append("Investigate model performance concerns")
    if any(c.name == "Contamination" and c.contribution > 10 for c in risk_score.components):
        recommendations.append("Review and fix data contamination")
    if any(c.name == "PII Risk" and c.contribution > 10 for c in risk_score.components):
        recommendations.append("Remove or mask PII from dataset")
    if not recommendations:
        if verdict == "APPROVED":
            recommendations.append("Dataset and model meet quality thresholds")
        else:
            recommendations.append("Review all score components before proceeding")
    
    return verdict, recommendations


def write_score_artifacts(
    risk_score: RiskScore,
    health_index: HealthIndex,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Write all scoring artifacts to output directory.
    
    Writes:
    - risk_score.json: Full risk score data
    - risk_score.txt: Human-readable summary
    - health_index.json: Full health index data
    - health_index.txt: Human-readable summary
    - risk_health_summary.json: Combined summary
    
    Args:
        risk_score: Computed risk score
        health_index: Computed health index
        output_dir: Directory to write artifacts
        
    Returns:
        Dict mapping artifact name to file path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    artifacts: Dict[str, Path] = {}
    
    # Compute verdict
    verdict, recommendations = compute_verdict(risk_score, health_index)
    summary = RiskHealthSummary(
        risk_score=risk_score,
        health_index=health_index,
        verdict=verdict,
        recommendations=recommendations,
    )
    
    # Write JSON artifacts
    risk_path = out / "risk_score.json"
    write_json(risk_path, risk_score.to_dict())
    artifacts["risk_score_json"] = risk_path
    
    health_path = out / "health_index.json"
    write_json(health_path, health_index.to_dict())
    artifacts["health_index_json"] = health_path
    
    summary_path = out / "risk_health_summary.json"
    write_json(summary_path, summary.to_dict())
    artifacts["summary_json"] = summary_path
    
    # Write TXT artifacts (human-readable)
    risk_txt = out / "risk_score.txt"
    risk_txt.write_text(_format_risk_text(risk_score))
    artifacts["risk_score_txt"] = risk_txt
    
    health_txt = out / "health_index.txt"
    health_txt.write_text(_format_health_text(health_index))
    artifacts["health_index_txt"] = health_txt
    
    return artifacts


def _format_risk_text(risk: RiskScore) -> str:
    """Format risk score as human-readable text."""
    lines = [
        "=" * 50,
        "DATASET RISK SCORE",
        "=" * 50,
        f"",
        f"Total Risk: {risk.total:.1f}/100 ({risk.level})",
        f"Summary: {risk.summary}",
        f"",
        "Component Breakdown:",
        "-" * 50,
    ]
    for c in sorted(risk.components, key=lambda x: x.contribution, reverse=True):
        lines.append(f"  {c.name:20s}: {c.contribution:5.1f}  ({c.detail})")
    lines.extend([
        "-" * 50,
        "Interpretation: 0-25 LOW, 26-50 MEDIUM, 51-75 HIGH, 76-100 CRITICAL",
        "=" * 50,
    ])
    return "\n".join(lines)


def _format_health_text(health: HealthIndex) -> str:
    """Format health index as human-readable text."""
    lines = [
        "=" * 50,
        "MODEL HEALTH INDEX",
        "=" * 50,
        f"",
        f"Total Health: {health.total:.1f}/100 ({health.level})",
        f"Summary: {health.summary}",
        f"",
        "Component Breakdown:",
        "-" * 50,
    ]
    for c in sorted(health.components, key=lambda x: x.contribution, reverse=True):
        lines.append(f"  {c.name:20s}: {c.contribution:5.1f}  ({c.detail})")
    lines.extend([
        "-" * 50,
        "Interpretation: 76-100 EXCELLENT, 51-75 GOOD, 26-50 FAIR, 0-25 POOR",
        "=" * 50,
    ])
    return "\n".join(lines)


def compute_scores_from_pipeline_result(
    pipeline_result: Dict[str, Any],
) -> Tuple[RiskScore, HealthIndex]:
    """Convenience function to compute both scores from pipeline output.
    
    Args:
        pipeline_result: Full pipeline result dict
        
    Returns:
        Tuple of (RiskScore, HealthIndex)
    """
    # Extract inputs from pipeline result
    report = pipeline_result.get("report")
    contamination = pipeline_result.get("contamination")
    contract = pipeline_result.get("contract")
    decision = pipeline_result.get("decision")
    
    # Get eval results from run_dir if available
    eval_results = None
    output_dir = pipeline_result.get("output_dir")
    if output_dir:
        eval_path = Path(output_dir) / "eval" / "eval_results.json"
        if eval_path.exists():
            try:
                eval_results = json.loads(eval_path.read_text())
            except Exception:
                pass
    
    # Compute scores
    risk = compute_dataset_risk(
        report_result=report,
        contamination_result=contamination,
        contract_result=contract,
    )
    
    health = compute_model_health(
        decision_result=decision,
        eval_results=eval_results,
        reproducibility_ok=contract.get("valid", False) if contract else False,
    )
    
    return risk, health
