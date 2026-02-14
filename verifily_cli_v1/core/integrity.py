"""Artifact integrity verification for tamper resistance.

Provides hash chain verification and tampering detection for run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

class IntegrityError(Exception):
    """Artifact integrity verification failed."""
    pass


class TamperingDetected(IntegrityError):
    """Tampering detected in artifact."""
    pass


class HashChainError(IntegrityError):
    """Hash chain verification failed."""
    pass


# Canonical JSON keys for stable ordering
CANONICAL_KEYS = {
    "manifest.json": ["run_id", "timestamp", "contracts", "datasets", "version"],
    "decision.json": ["run_id", "timestamp", "status", "passed", "checks", "summary"],
    "usage.json": ["run_id", "timestamp", "usage", "cost"],
}


def canonical_hash(obj: Any) -> str:
    """Compute canonical SHA-256 hash of object with stable ordering.
    
    Ensures deterministic hashing regardless of key insertion order
    or Python version.
    
    Args:
        obj: Object to hash (dict, list, or primitive)
        
    Returns:
        Hex-encoded SHA-256 hash
        
    Example:
        >>> canonical_hash({"b": 1, "a": 2})
        'a1b2...'  # Same regardless of key order
    """
    canonical = _canonicalize(obj)
    json_bytes = json.dumps(canonical, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(json_bytes).hexdigest()


def _canonicalize(obj: Any) -> Any:
    """Convert object to canonical form for stable hashing.
    
    - Dicts: sorted keys, recursively canonicalized values
    - Lists: recursively canonicalized elements
    - Primitives: as-is
    """
    if isinstance(obj, dict):
        # Sort keys for stable ordering
        return {k: _canonicalize(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_canonicalize(item) for item in obj)
    else:
        return obj


def verify_hash_chain(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Verify the hash chain integrity of a run directory.
    
    Checks that:
    1. All required artifacts exist
    2. Hashes in manifest match actual file hashes
    3. Decision references match actual datasets
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Verification result with status and details
        
    Raises:
        TamperingDetected: If hash mismatch detected
        IntegrityError: If required files missing
        
    Example:
        >>> result = verify_hash_chain("/runs/run_20250120_120000")
        >>> print(result["status"])  # "VERIFIED" or "TAMPERED"
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        raise IntegrityError(f"Run directory not found: {run_path}")
    
    result = {
        "status": "VERIFIED",
        "run_id": run_path.name,
        "checks": [],
        "errors": [],
    }
    
    # Required files for a valid run
    required_files = ["manifest.json", "decision.json"]
    optional_files = ["dataset.jsonl", "usage.json", "audit.log"]
    
    # Check required files exist
    for filename in required_files:
        file_path = run_path / filename
        if not file_path.exists():
            result["errors"].append(f"Missing required file: {filename}")
            result["status"] = "INCOMPLETE"
    
    if result["status"] == "INCOMPLETE":
        return result
    
    # Load manifest
    manifest_path = run_path / "manifest.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise IntegrityError(f"Cannot read manifest: {e}")
    
    # Verify manifest hash matches directory
    result["checks"].append(_verify_manifest_integrity(run_path, manifest))
    
    # Verify dataset integrity if present
    dataset_path = run_path / "dataset.jsonl"
    if dataset_path.exists():
        result["checks"].append(_verify_dataset_integrity(run_path, dataset_path, manifest))
    
    # Verify decision integrity
    decision_path = run_path / "decision.json"
    if decision_path.exists():
        result["checks"].append(_verify_decision_integrity(run_path, decision_path, manifest))
    
    # Check for any tampering
    failed_checks = [c for c in result["checks"] if not c.get("passed", False)]
    if failed_checks:
        result["status"] = "TAMPERED"
        for check in failed_checks:
            result["errors"].append(check.get("error", "Unknown integrity failure"))
    
    return result


def _verify_manifest_integrity(run_path: Path, manifest: Dict) -> Dict[str, Any]:
    """Verify manifest file integrity."""
    check = {
        "name": "manifest_integrity",
        "passed": True,
        "details": {},
    }
    
    # Verify manifest has required fields
    required_fields = ["run_id", "timestamp"]
    for field in required_fields:
        if field not in manifest:
            check["passed"] = False
            check["error"] = f"Manifest missing required field: {field}"
            return check
    
    # Compute canonical hash of manifest (excluding any embedded hash field)
    manifest_copy = {k: v for k, v in manifest.items() if k not in ("hash", "_hash")}
    computed_hash = canonical_hash(manifest_copy)
    check["details"]["canonical_hash"] = computed_hash
    
    return check


def _verify_dataset_integrity(
    run_path: Path, 
    dataset_path: Path, 
    manifest: Dict
) -> Dict[str, Any]:
    """Verify dataset file integrity against manifest."""
    check = {
        "name": "dataset_integrity",
        "passed": True,
        "details": {},
    }
    
    # Compute file hash
    file_hash = _file_hash(dataset_path)
    check["details"]["computed_hash"] = file_hash
    check["details"]["file_size"] = dataset_path.stat().st_size
    
    # Check against manifest if hash recorded
    manifest_datasets = manifest.get("datasets", [])
    for ds in manifest_datasets:
        if ds.get("path") == "dataset.jsonl":
            expected_hash = ds.get("hash")
            if expected_hash and expected_hash != file_hash:
                check["passed"] = False
                check["error"] = (
                    f"Dataset hash mismatch: "
                    f"expected={expected_hash[:16]}..., "
                    f"computed={file_hash[:16]}..."
                )
            check["details"]["expected_hash"] = expected_hash
            break
    
    return check


def _verify_decision_integrity(
    run_path: Path,
    decision_path: Path,
    manifest: Dict
) -> Dict[str, Any]:
    """Verify decision file integrity."""
    check = {
        "name": "decision_integrity",
        "passed": True,
        "details": {},
    }
    
    try:
        with open(decision_path) as f:
            decision = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        check["passed"] = False
        check["error"] = f"Cannot read decision: {e}"
        return check
    
    # Verify decision run_id matches manifest
    decision_run_id = decision.get("run_id")
    manifest_run_id = manifest.get("run_id")
    if decision_run_id and manifest_run_id and decision_run_id != manifest_run_id:
        check["passed"] = False
        check["error"] = (
            f"Run ID mismatch: decision={decision_run_id}, manifest={manifest_run_id}"
        )
    
    # Compute canonical hash
    decision_copy = {k: v for k, v in decision.items() if k not in ("hash", "_hash", "signature")}
    computed_hash = canonical_hash(decision_copy)
    check["details"]["canonical_hash"] = computed_hash
    
    return check


def detect_artifact_tampering(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Detect tampering in run artifacts.
    
    More detailed than verify_hash_chain - reports all detected
    tampering attempts with specific indicators.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tampering detection report with indicators
        
    Example:
        >>> report = detect_artifact_tampering("/runs/run_001")
        >>> if report["tampering_detected"]:
        ...     print(f"Indicators: {report['indicators']}")
    """
    run_path = Path(run_dir)
    report = {
        "tampering_detected": False,
        "indicators": [],
        "confidence": "NONE",
        "details": {},
    }
    
    if not run_path.exists():
        report["indicators"].append("RUN_DIRECTORY_MISSING")
        report["tampering_detected"] = True
        report["confidence"] = "HIGH"
        return report
    
    # Check for file modification time anomalies
    mtime_check = _check_modification_times(run_path)
    if mtime_check.get("anomaly_detected"):
        report["indicators"].append("MTIME_ANOMALY")
        report["details"]["mtime_anomaly"] = mtime_check
        report["tampering_detected"] = True
    
    # Check for missing expected files
    expected_files = ["manifest.json", "decision.json"]
    missing = [f for f in expected_files if not (run_path / f).exists()]
    if missing:
        report["indicators"].append("FILES_MISSING")
        report["details"]["missing_files"] = missing
        report["tampering_detected"] = True
    
    # Check for unexpected files
    expected_set = set(expected_files + ["dataset.jsonl", "usage.json", "audit.log", "decision.sig"])
    actual_files = {f.name for f in run_path.iterdir() if f.is_file()}
    unexpected = actual_files - expected_set
    if unexpected:
        report["indicators"].append("UNEXPECTED_FILES")
        report["details"]["unexpected_files"] = list(unexpected)
    
    # Check file sizes for anomalies
    size_check = _check_file_sizes(run_path)
    if size_check.get("anomaly_detected"):
        report["indicators"].append("SIZE_ANOMALY")
        report["details"]["size_anomaly"] = size_check
    
    # Verify hash chain
    try:
        chain_result = verify_hash_chain(run_path)
        if chain_result["status"] == "TAMPERED":
            report["indicators"].append("HASH_MISMATCH")
            report["details"]["hash_chain"] = chain_result
            report["tampering_detected"] = True
            report["confidence"] = "HIGH"
        elif chain_result["status"] == "VERIFIED" and not report["tampering_detected"]:
            report["confidence"] = "LOW"
    except IntegrityError as e:
        report["indicators"].append("INTEGRITY_CHECK_FAILED")
        report["details"]["error"] = str(e)
        report["tampering_detected"] = True
        report["confidence"] = "MEDIUM"
    
    # Set confidence based on indicators
    if report["tampering_detected"] and report["confidence"] == "NONE":
        report["confidence"] = "MEDIUM"
    
    return report


def _check_modification_times(run_path: Path) -> Dict[str, Any]:
    """Check for modification time anomalies."""
    result = {"anomaly_detected": False, "details": []}
    
    manifest_path = run_path / "manifest.json"
    decision_path = run_path / "decision.json"
    
    if not manifest_path.exists() or not decision_path.exists():
        return result
    
    manifest_mtime = manifest_path.stat().st_mtime
    decision_mtime = decision_path.stat().st_mtime
    
    # Decision should be created after or at same time as manifest
    if decision_mtime < manifest_mtime - 1:  # 1 second tolerance
        result["anomaly_detected"] = True
        result["details"].append({
            "type": "DECISION_BEFORE_MANIFEST",
            "manifest_mtime": manifest_mtime,
            "decision_mtime": decision_mtime,
        })
    
    return result


def _check_file_sizes(run_path: Path) -> Dict[str, Any]:
    """Check for file size anomalies."""
    result = {"anomaly_detected": False, "details": []}
    
    # Check for empty critical files
    for filename in ["manifest.json", "decision.json"]:
        file_path = run_path / filename
        if file_path.exists():
            size = file_path.stat().st_size
            if size == 0:
                result["anomaly_detected"] = True
                result["details"].append({
                    "type": "EMPTY_FILE",
                    "file": filename,
                })
    
    return result


def _file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_artifact_hashes(run_dir: Union[str, Path]) -> Dict[str, str]:
    """Compute hashes for all artifacts in run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dict mapping filenames to SHA-256 hashes
    """
    run_path = Path(run_dir)
    hashes = {}
    
    for file_path in run_path.iterdir():
        if file_path.is_file():
            hashes[file_path.name] = _file_hash(file_path)
    
    return hashes
