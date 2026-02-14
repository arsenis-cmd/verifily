"""Enterprise policy engine — configurable gate conditions.

Evaluates pipeline results against enterprise policies.  Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PolicyConfig:
    """Enterprise policy configuration.  Each field is opt-in."""

    require_contamination_pass: bool = False
    require_reproducibility: bool = False
    block_if_pii_hits: Optional[int] = None
    min_f1_threshold: Optional[float] = None


@dataclass
class PolicyResult:
    """Result of policy evaluation."""

    allowed: bool
    violations: List[str] = field(default_factory=list)


def evaluate_policies(
    policy_cfg: PolicyConfig,
    pipeline_result: Dict[str, Any],
) -> PolicyResult:
    """Evaluate enterprise policies against a pipeline result dict.

    Returns PolicyResult with ``allowed=False`` if any violations found.
    """
    violations: List[str] = []

    # 1. Contamination pass
    if policy_cfg.require_contamination_pass:
        contam = pipeline_result.get("contamination", {})
        status = contam.get("status", "UNKNOWN")
        if status != "PASS":
            violations.append(
                f"Policy violation: contamination status is {status}, "
                f"but require_contamination_pass=True"
            )

    # 2. Reproducibility (contract valid)
    if policy_cfg.require_reproducibility:
        contract = pipeline_result.get("contract", {})
        if not contract.get("valid", False):
            violations.append(
                "Policy violation: contract is invalid, "
                "but require_reproducibility=True"
            )

    # 3. PII hits threshold
    if policy_cfg.block_if_pii_hits is not None:
        report = pipeline_result.get("report_summary", {})
        pii_hits = report.get("pii_total_hits", 0)
        if pii_hits > policy_cfg.block_if_pii_hits:
            violations.append(
                f"Policy violation: PII hits ({pii_hits}) exceed "
                f"threshold ({policy_cfg.block_if_pii_hits})"
            )

    # 4. Minimum F1 threshold
    if policy_cfg.min_f1_threshold is not None:
        decision = pipeline_result.get("decision", {})
        metrics = decision.get("metrics", {})
        f1 = metrics.get("f1")
        if f1 is not None and f1 < policy_cfg.min_f1_threshold:
            violations.append(
                f"Policy violation: F1 ({f1:.4f}) below "
                f"threshold ({policy_cfg.min_f1_threshold})"
            )
        elif f1 is None:
            violations.append(
                "Policy violation: F1 metric unavailable, "
                "but min_f1_threshold is configured"
            )

    return PolicyResult(allowed=len(violations) == 0, violations=violations)
