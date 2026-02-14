"""Production readiness validator for Verifily runs.

Checks if a run is ready for production deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ReadinessStatus(str, Enum):
    """Readiness check status."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class ReadinessCheck:
    """A single readiness check result."""
    
    name: str
    status: ReadinessStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ReadinessReport:
    """Complete readiness report."""
    
    run_id: str
    checks: List[ReadinessCheck] = field(default_factory=list)
    overall_status: ReadinessStatus = ReadinessStatus.PASS
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_check(self, check: ReadinessCheck) -> None:
        self.checks.append(check)
        # Update overall status
        if check.status == ReadinessStatus.FAIL:
            self.overall_status = ReadinessStatus.FAIL
        elif check.status == ReadinessStatus.WARN and self.overall_status == ReadinessStatus.PASS:
            self.overall_status = ReadinessStatus.WARN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overall_status": self.overall_status.value,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ReadinessError(Exception):
    """Readiness check failed."""
    pass


def check_contract_integrity(run_dir: Path) -> ReadinessCheck:
    """Check that all required contract files are present."""
    required_files = ["manifest.json", "decision.json"]
    missing = []
    
    for filename in required_files:
        if not (run_dir / filename).exists():
            missing.append(filename)
    
    if missing:
        return ReadinessCheck(
            name="contract_integrity",
            status=ReadinessStatus.FAIL,
            message=f"Missing required files: {', '.join(missing)}",
            details={"missing_files": missing},
        )
    
    # Check manifest validity
    manifest_path = run_dir / "manifest.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        if "run_id" not in manifest:
            return ReadinessCheck(
                name="contract_integrity",
                status=ReadinessStatus.FAIL,
                message="Invalid manifest: missing run_id",
                details={},
            )
        
        return ReadinessCheck(
            name="contract_integrity",
            status=ReadinessStatus.PASS,
            message="All required files present and valid",
            details={"files": required_files},
        )
    except (json.JSONDecodeError, IOError) as e:
        return ReadinessCheck(
            name="contract_integrity",
            status=ReadinessStatus.FAIL,
            message=f"Cannot read manifest: {e}",
            details={},
        )


def check_hash_chain(run_dir: Path) -> ReadinessCheck:
    """Check that hash chain is valid."""
    from verifily_cli_v1.core.integrity import verify_hash_chain
    
    try:
        result = verify_hash_chain(run_dir)
        status = result.get("status", "UNKNOWN")
        
        if status == "VERIFIED":
            return ReadinessCheck(
                name="hash_chain",
                status=ReadinessStatus.PASS,
                message="Hash chain verified successfully",
                details={"checks": len(result.get("checks", []))},
            )
        elif status == "TAMPERED":
            errors = result.get("errors", [])
            return ReadinessCheck(
                name="hash_chain",
                status=ReadinessStatus.FAIL,
                message=f"Tampering detected: {errors[0] if errors else 'Unknown'}" if errors else "Tampering detected",
                details={"errors": errors},
            )
        else:
            return ReadinessCheck(
                name="hash_chain",
                status=ReadinessStatus.WARN,
                message=f"Hash chain status: {status}",
                details={"status": status},
            )
    except Exception as e:
        return ReadinessCheck(
            name="hash_chain",
            status=ReadinessStatus.WARN,
            message=f"Could not verify hash chain: {e}",
            details={},
        )


def check_privacy_safety(run_dir: Path) -> ReadinessCheck:
    """Check privacy safety settings."""
    # Check for redaction audit
    audit_path = run_dir / "redaction_audit.json"
    
    if audit_path.exists():
        try:
            with open(audit_path) as f:
                audit = json.load(f)
            
            status = audit.get("status", "UNKNOWN")
            
            if status == "PASS":
                return ReadinessCheck(
                    name="privacy_safety",
                    status=ReadinessStatus.PASS,
                    message="Redaction audit passed",
                    details={"findings": audit.get("findings_count", 0)},
                )
            elif status == "FAIL":
                return ReadinessCheck(
                    name="privacy_safety",
                    status=ReadinessStatus.FAIL,
                    message=f"Redaction audit failed: {audit.get('findings_count', 0)} findings",
                    details={"findings_by_type": audit.get("summary", {}).get("findings_by_type", {})},
                )
            else:
                return ReadinessCheck(
                    name="privacy_safety",
                    status=ReadinessStatus.WARN,
                    message=f"Redaction audit: {status}",
                    details={},
                )
        except (json.JSONDecodeError, IOError) as e:
            return ReadinessCheck(
                name="privacy_safety",
                status=ReadinessStatus.WARN,
                message=f"Cannot read redaction audit: {e}",
                details={},
            )
    
    # No audit performed
    return ReadinessCheck(
        name="privacy_safety",
        status=ReadinessStatus.WARN,
        message="No redaction audit performed",
        details={"suggestion": "Run with --audit flag"},
    )


def check_contamination_risk(run_dir: Path) -> ReadinessCheck:
    """Check contamination detection results."""
    # Look for contamination results in decision or separate file
    decision_path = run_dir / "decision.json"
    
    if decision_path.exists():
        try:
            with open(decision_path) as f:
                decision = json.load(f)
            
            # Check if contamination check was part of pipeline
            checks = decision.get("checks", {})
            
            # Look for contamination failure
            if isinstance(checks, dict):
                for check_name, check_result in checks.items():
                    if "contamination" in check_name.lower():
                        if check_result in ("FAIL", "FAILED", False):
                            return ReadinessCheck(
                                name="contamination_risk",
                                status=ReadinessStatus.FAIL,
                                message="Contamination check failed",
                                details={"check": check_name},
                            )
            
            # Check decision status
            if decision.get("status") == "FAIL":
                return ReadinessCheck(
                    name="contamination_risk",
                    status=ReadinessStatus.WARN,
                    message="Decision gate failed - verify contamination status",
                    details={},
                )
            
            return ReadinessCheck(
                name="contamination_risk",
                status=ReadinessStatus.PASS,
                message="No contamination detected",
                details={},
            )
        except (json.JSONDecodeError, IOError) as e:
            return ReadinessCheck(
                name="contamination_risk",
                status=ReadinessStatus.WARN,
                message=f"Cannot read decision: {e}",
                details={},
            )
    
    return ReadinessCheck(
        name="contamination_risk",
        status=ReadinessStatus.WARN,
        message="No decision file found",
        details={},
    )


def check_regression_risk(run_dir: Path) -> ReadinessCheck:
    """Check for recent regressions."""
    # Look for regression indicators in decision
    decision_path = run_dir / "decision.json"
    
    if decision_path.exists():
        try:
            with open(decision_path) as f:
                decision = json.load(f)
            
            # Check for regression warnings
            summary = decision.get("summary", {})
            warnings = summary.get("warnings", [])
            
            regression_warnings = [w for w in warnings if "regression" in w.lower()]
            
            if regression_warnings:
                return ReadinessCheck(
                    name="regression_risk",
                    status=ReadinessStatus.WARN,
                    message=f"Regression warning: {regression_warnings[0]}",
                    details={"warnings": regression_warnings},
                )
            
            # Check for metric degradation indicators
            checks = decision.get("checks", {})
            for check_name, check_result in checks.items():
                if isinstance(check_result, dict):
                    if check_result.get("regression_detected"):
                        return ReadinessCheck(
                            name="regression_risk",
                            status=ReadinessStatus.WARN,
                            message="Regression detected in metrics",
                            details={"check": check_name},
                        )
            
            return ReadinessCheck(
                name="regression_risk",
                status=ReadinessStatus.PASS,
                message="No regression detected",
                details={},
            )
        except (json.JSONDecodeError, IOError) as e:
            return ReadinessCheck(
                name="regression_risk",
                status=ReadinessStatus.WARN,
                message=f"Cannot read decision: {e}",
                details={},
            )
    
    return ReadinessCheck(
        name="regression_risk",
        status=ReadinessStatus.WARN,
        message="No decision file found",
        details={},
    )


def check_config_sanity(run_dir: Path) -> ReadinessCheck:
    """Check configuration for absurd values."""
    manifest_path = run_dir / "manifest.json"
    
    if not manifest_path.exists():
        return ReadinessCheck(
            name="config_sanity",
            status=ReadinessStatus.WARN,
            message="No manifest to check",
            details={},
        )
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        warnings = []
        
        # Check for suspicious thresholds
        contracts = manifest.get("contracts", [])
        for contract in contracts:
            if isinstance(contract, dict):
                # Check for very low thresholds
                if contract.get("min_row_count", 0) < 10:
                    warnings.append(f"Very low min_row_count in {contract.get('name', 'unknown')}")
                
                # Check for very high null ratios
                if contract.get("max_null_ratio", 0) > 0.5:
                    warnings.append(f"Very high max_null_ratio in {contract.get('name', 'unknown')}")
        
        if warnings:
            return ReadinessCheck(
                name="config_sanity",
                status=ReadinessStatus.WARN,
                message=f"Configuration warnings: {warnings[0]}",
                details={"warnings": warnings},
            )
        
        return ReadinessCheck(
            name="config_sanity",
            status=ReadinessStatus.PASS,
            message="Configuration looks reasonable",
            details={},
        )
    except (json.JSONDecodeError, IOError) as e:
        return ReadinessCheck(
            name="config_sanity",
            status=ReadinessStatus.WARN,
            message=f"Cannot read manifest: {e}",
            details={},
        )


def check_reproducibility(run_dir: Path) -> ReadinessCheck:
    """Check if reproducibility artifacts are present."""
    # Check for environment.json or similar
    env_path = run_dir / "environment.json"
    
    if env_path.exists():
        try:
            with open(env_path) as f:
                env = json.load(f)
            
            has_seed = "seed" in env or "random_seed" in env
            has_versions = "versions" in env or "dependencies" in env
            
            if has_seed and has_versions:
                return ReadinessCheck(
                    name="reproducibility",
                    status=ReadinessStatus.PASS,
                    message="Reproducibility artifacts present",
                    details={"has_seed": has_seed, "has_versions": has_versions},
                )
            else:
                return ReadinessCheck(
                    name="reproducibility",
                    status=ReadinessStatus.WARN,
                    message="Partial reproducibility artifacts",
                    details={"has_seed": has_seed, "has_versions": has_versions},
                )
        except (json.JSONDecodeError, IOError) as e:
            return ReadinessCheck(
                name="reproducibility",
                status=ReadinessStatus.WARN,
                message=f"Cannot read environment file: {e}",
                details={},
            )
    
    # Check for version in manifest
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            if "version" in manifest:
                return ReadinessCheck(
                    name="reproducibility",
                    status=ReadinessStatus.WARN,
                    message="Version tracked but no full environment capture",
                    details={"version": manifest.get("version")},
                )
        except (json.JSONDecodeError, IOError):
            pass
    
    return ReadinessCheck(
        name="reproducibility",
        status=ReadinessStatus.WARN,
        message="No reproducibility artifacts found",
        details={"suggestion": "Add environment.json with seed and versions"},
    )


def validate_readiness(
    run_dir: Union[str, Path],
    run_id: Optional[str] = None,
) -> ReadinessReport:
    """Validate production readiness of a run.
    
    Args:
        run_dir: Path to run directory
        run_id: Optional run ID (extracted from directory if not provided)
        
    Returns:
        ReadinessReport with all checks
        
    Example:
        >>> report = validate_readiness("./runs/run_001")
        >>> print(report.overall_status)  # PASS, WARN, or FAIL
        >>> if report.overall_status == ReadinessStatus.PASS:
        ...     print("Ready for production!")
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        raise ReadinessError(f"Run directory not found: {run_path}")
    
    # Extract run_id from directory name if not provided
    if run_id is None:
        run_id = run_path.name
    
    report = ReadinessReport(run_id=run_id)
    
    # Run all checks
    checks = [
        check_contract_integrity(run_path),
        check_hash_chain(run_path),
        check_privacy_safety(run_path),
        check_contamination_risk(run_path),
        check_regression_risk(run_path),
        check_config_sanity(run_path),
        check_reproducibility(run_path),
    ]
    
    for check in checks:
        report.add_check(check)
    
    # Calculate summary
    pass_count = sum(1 for c in report.checks if c.status == ReadinessStatus.PASS)
    warn_count = sum(1 for c in report.checks if c.status == ReadinessStatus.WARN)
    fail_count = sum(1 for c in report.checks if c.status == ReadinessStatus.FAIL)
    
    report.summary = {
        "total_checks": len(report.checks),
        "passed": pass_count,
        "warnings": warn_count,
        "failed": fail_count,
    }
    
    return report


def format_readiness_report(report: ReadinessReport) -> str:
    """Format readiness report for display."""
    lines = []
    
    # Header
    status_emoji = {
        ReadinessStatus.PASS: "✅",
        ReadinessStatus.WARN: "⚠️",
        ReadinessStatus.FAIL: "❌",
    }
    
    lines.append(f"Readiness Report for {report.run_id}")
    lines.append("=" * 50)
    lines.append(f"Overall Status: {status_emoji[report.overall_status]} {report.overall_status.value}")
    lines.append("")
    
    # Individual checks
    for check in report.checks:
        emoji = status_emoji[check.status]
        lines.append(f"{emoji} {check.name.replace('_', ' ').title()}")
        lines.append(f"   Status: {check.status.value}")
        lines.append(f"   {check.message}")
        lines.append("")
    
    # Summary
    lines.append("-" * 50)
    lines.append(f"Summary: {report.summary.get('passed', 0)} passed, "
                f"{report.summary.get('warnings', 0)} warnings, "
                f"{report.summary.get('failed', 0)} failed")
    
    return "\n".join(lines)
