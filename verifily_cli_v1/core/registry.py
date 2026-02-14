"""Lightweight local Model Registry for Verifily.

Manages model versions, promotion workflows (staging → production),
and release safety checks. All data is metadata-only (no raw datasets).

Storage is append-only JSONL for auditability.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from verifily_cli_v1.core.io import read_json, write_json
from verifily_cli_v1.core.io import read_jsonl, write_jsonl

logger = logging.getLogger("verifily.registry")


class ModelStage(str, Enum):
    """Model lifecycle stages."""
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class RegistryError(Exception):
    """Base exception for registry operations."""
    pass


class RegistrationError(RegistryError):
    """Raised when model registration fails safety checks."""
    pass


class PromotionError(RegistryError):
    """Raised when model promotion fails safety checks."""
    pass


@dataclass
class ModelRecord:
    """A registered model version.
    
    Attributes:
        model_id: Unique model identifier (e.g., "fraud_detector")
        version: Semantic version string (e.g., "v1.2.3")
        created_at: ISO timestamp of registration
        source_run: Path to the run directory that produced this model
        decision: Final decision (SHIP/INVESTIGATE/DONT_SHIP)
        risk_score: Dataset risk score (0-100)
        health_index: Model health index (0-100)
        metrics: Key metrics dict (f1, accuracy, etc.)
        lineage_hash: Hash of lineage graph for reproducibility
        stage: Current lifecycle stage
        registered_by: Identifier of who/what registered the model
        promotion_history: List of stage transitions
    """
    model_id: str
    version: str
    created_at: str
    source_run: str
    decision: str
    risk_score: float
    health_index: float
    metrics: Dict[str, float] = field(default_factory=dict)
    lineage_hash: str = ""
    stage: str = ModelStage.NONE.value
    registered_by: str = "unknown"
    promotion_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at,
            "source_run": self.source_run,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "health_index": self.health_index,
            "metrics": self.metrics,
            "lineage_hash": self.lineage_hash,
            "stage": self.stage,
            "registered_by": self.registered_by,
            "promotion_history": self.promotion_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRecord":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            created_at=data["created_at"],
            source_run=data["source_run"],
            decision=data["decision"],
            risk_score=data["risk_score"],
            health_index=data["health_index"],
            metrics=data.get("metrics", {}),
            lineage_hash=data.get("lineage_hash", ""),
            stage=data.get("stage", ModelStage.NONE.value),
            registered_by=data.get("registered_by", "unknown"),
            promotion_history=data.get("promotion_history", []),
        )


@dataclass
class RegistrationRequest:
    """Request to register a model from a run."""
    run_dir: str
    model_id: Optional[str] = None  # Auto-detect if not provided
    version: Optional[str] = None   # Auto-detect if not provided
    registered_by: str = "cli"


@dataclass
class PromotionRequest:
    """Request to promote a model to a new stage."""
    model_id: str
    version: str
    target_stage: str
    promoted_by: str = "cli"
    reason: str = ""


class RegistryStore:
    """Thread-safe singleton model registry store.
    
    Uses append-only JSONL for persistence. In-memory index for fast lookups.
    """
    
    # Production thresholds
    PRODUCTION_MIN_HEALTH = 60.0
    PRODUCTION_MAX_RISK = 50.0
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: Dict[str, ModelRecord] = {}  # key: "model_id:version"
        self._persist_path: Optional[Path] = None
        self._by_model: Dict[str, List[str]] = {}  # model_id -> list of version keys
    
    def configure_persistence(self, path: Optional[Union[str, Path]]) -> None:
        """Enable file-backed persistence and replay existing events.
        
        Args:
            path: Path to registry.jsonl file, or None to disable.
        """
        with self._lock:
            self._persist_path = Path(path) if path else None
        
        if path:
            self._replay(path)
    
    def _replay(self, path: Union[str, Path]) -> None:
        """Replay events from JSONL file to rebuild state."""
        p = Path(path)
        if not p.exists():
            return
        
        try:
            events = read_jsonl(p)
            for event in events:
                action = event.get("action")
                if action == "registered":
                    record = ModelRecord.from_dict(event["record"])
                    self._index_record(record)
                elif action == "promoted":
                    key = f"{event['model_id']}:{event['version']}"
                    record = self._models.get(key)
                    if record:
                        record.stage = event["new_stage"]
                        record.promotion_history.append({
                            "from": event["old_stage"],
                            "to": event["new_stage"],
                            "at": event["timestamp"],
                            "by": event.get("promoted_by", "unknown"),
                            "reason": event.get("reason", ""),
                        })
        except Exception as e:
            logger.warning("Registry replay failed for %s: %s", path, e)
    
    def _persist_event(self, event: Dict[str, Any]) -> None:
        """Append event to JSONL file (caller does NOT hold lock)."""
        if not self._persist_path:
            return
        
        try:
            # Ensure directory exists
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to file
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")
                f.flush()
        except Exception as e:
            logger.warning("Registry persist failed: %s", e)
    
    def _index_record(self, record: ModelRecord) -> str:
        """Add record to in-memory indexes (internal use)."""
        key = f"{record.model_id}:{record.version}"
        self._models[key] = record
        
        # Update by_model index
        if record.model_id not in self._by_model:
            self._by_model[record.model_id] = []
        if key not in self._by_model[record.model_id]:
            self._by_model[record.model_id].append(key)
        
        return key
    
    def _make_version(self, model_id: str) -> str:
        """Auto-generate next version for a model."""
        existing = self._by_model.get(model_id, [])
        if not existing:
            return "v1"
        
        # Find highest version number
        max_num = 0
        for key in existing:
            version = self._models[key].version
            if version.startswith("v"):
                try:
                    num = int(version[1:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        
        return f"v{max_num + 1}"
    
    def register(
        self,
        request: RegistrationRequest,
        *,
        skip_safety_checks: bool = False,
    ) -> ModelRecord:
        """Register a model from a run directory.
        
        Performs safety checks unless skip_safety_checks=True.
        
        Args:
            request: Registration request with run_dir and optional model_id/version
            skip_safety_checks: If True, bypass safety gates (use with caution)
            
        Returns:
            ModelRecord of the registered model
            
        Raises:
            RegistrationError: If safety checks fail
            FileNotFoundError: If run_dir doesn't exist
        """
        run_path = Path(request.run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        
        # Load required artifacts
        decision = self._load_json(run_path / "decision.json")
        contract = self._load_json(run_path / "contract.json")
        contamination = self._load_json(run_path / "contamination_results.json")
        risk_score = self._load_json(run_path / "risk_score.json")
        health_index = self._load_json(run_path / "health_index.json")
        eval_results = self._load_json(run_path / "eval" / "eval_results.json")
        
        # Safety checks
        if not skip_safety_checks:
            errors = []
            
            # Check 1: Decision must be SHIP
            if not decision or decision.get("recommendation") != "SHIP":
                rec = decision.get("recommendation", "UNKNOWN") if decision else "NONE"
                errors.append(f"Decision={rec} (required: SHIP)")
            
            # Check 2: Contract must be valid
            if not contract or not contract.get("valid", False):
                errors.append("Contract validation failed")
            
            # Check 3: Contamination must not be FAIL
            if contamination and contamination.get("status") == "FAIL":
                errors.append("Contamination check failed")
            
            if errors:
                raise RegistrationError(
                    f"Cannot register model: {'; '.join(errors)}"
                )
        
        # Extract model info
        model_id = request.model_id or self._detect_model_id(run_path)
        version = request.version or self._make_version(model_id)
        
        # Check for duplicate
        key = f"{model_id}:{version}"
        with self._lock:
            if key in self._models:
                raise RegistrationError(
                    f"Model {model_id} version {version} already exists"
                )
        
        # Build metrics dict
        metrics = {}
        if eval_results:
            overall = eval_results.get("overall", eval_results.get("aggregate", {}))
            if isinstance(overall, dict):
                for k in ["f1", "accuracy", "exact_match", "precision", "recall"]:
                    if k in overall:
                        metrics[k] = overall[k]
        
        # Get lineage hash
        lineage = self._load_json(run_path / "lineage_graph.json")
        lineage_hash = ""
        if lineage:
            lineage_hash = lineage.get("root_id", "")
        
        # Create record
        record = ModelRecord(
            model_id=model_id,
            version=version,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            source_run=str(run_path),
            decision=decision.get("recommendation", "UNKNOWN") if decision else "UNKNOWN",
            risk_score=risk_score.get("total", 50.0) if risk_score else 50.0,
            health_index=health_index.get("total", 50.0) if health_index else 50.0,
            metrics=metrics,
            lineage_hash=lineage_hash,
            stage=ModelStage.NONE.value,
            registered_by=request.registered_by,
        )
        
        # Store
        with self._lock:
            self._index_record(record)
        
        # Persist
        self._persist_event({
            "action": "registered",
            "timestamp": record.created_at,
            "record": record.to_dict(),
        })
        
        logger.info("Registered model %s version %s", model_id, version)
        return record
    
    def promote(
        self,
        request: PromotionRequest,
        *,
        skip_safety_checks: bool = False,
    ) -> ModelRecord:
        """Promote a model to a new stage.
        
        Args:
            request: Promotion request
            skip_safety_checks: If True, bypass production promotion checks
            
        Returns:
            Updated ModelRecord
            
        Raises:
            PromotionError: If model not found or safety checks fail
        """
        key = f"{request.model_id}:{request.version}"
        
        with self._lock:
            record = self._models.get(key)
        
        if not record:
            raise PromotionError(
                f"Model {request.model_id} version {request.version} not found"
            )
        
        old_stage = record.stage
        new_stage = request.target_stage
        
        # Validate stage transition
        valid_transitions = {
            ModelStage.NONE.value: [ModelStage.STAGING.value],
            ModelStage.STAGING.value: [ModelStage.PRODUCTION.value, ModelStage.ARCHIVED.value],
            ModelStage.PRODUCTION.value: [ModelStage.ARCHIVED.value],
            ModelStage.ARCHIVED.value: [],
        }
        
        if new_stage not in valid_transitions.get(old_stage, []):
            raise PromotionError(
                f"Invalid promotion: {old_stage} -> {new_stage}"
            )
        
        # Production safety checks
        if new_stage == ModelStage.PRODUCTION.value and not skip_safety_checks:
            errors = []
            
            if record.health_index < self.PRODUCTION_MIN_HEALTH:
                errors.append(
                    f"Health Index {record.health_index:.1f} below threshold "
                    f"{self.PRODUCTION_MIN_HEALTH}"
                )
            
            if record.risk_score > self.PRODUCTION_MAX_RISK:
                errors.append(
                    f"Risk Score {record.risk_score:.1f} above threshold "
                    f"{self.PRODUCTION_MAX_RISK}"
                )
            
            if record.decision != "SHIP":
                errors.append(f"Decision was {record.decision} (required: SHIP)")
            
            if errors:
                raise PromotionError(
                    f"Promotion blocked: {'; '.join(errors)}"
                )
        
        # Update record
        record.stage = new_stage
        record.promotion_history.append({
            "from": old_stage,
            "to": new_stage,
            "at": datetime.datetime.utcnow().isoformat() + "Z",
            "by": request.promoted_by,
            "reason": request.reason,
        })
        
        # Persist
        self._persist_event({
            "action": "promoted",
            "timestamp": record.promotion_history[-1]["at"],
            "model_id": record.model_id,
            "version": record.version,
            "old_stage": old_stage,
            "new_stage": new_stage,
            "promoted_by": request.promoted_by,
            "reason": request.reason,
        })
        
        logger.info(
            "Promoted %s:%s from %s to %s",
            request.model_id, request.version, old_stage, new_stage
        )
        return record
    
    def get(self, model_id: str, version: str) -> Optional[ModelRecord]:
        """Get a specific model version."""
        key = f"{model_id}:{version}"
        with self._lock:
            return self._models.get(key)
    
    def list(
        self,
        *,
        stage: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> List[ModelRecord]:
        """List models, optionally filtered by stage or model_id."""
        with self._lock:
            if model_id:
                keys = self._by_model.get(model_id, [])
                records = [self._models[k] for k in keys]
            else:
                records = list(self._models.values())
        
        if stage:
            records = [r for r in records if r.stage == stage]
        
        # Sort by created_at desc
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records
    
    def history(self, model_id: str) -> List[ModelRecord]:
        """Get full history of a model (all versions)."""
        return self.list(model_id=model_id)
    
    def get_latest(self, model_id: str, stage: Optional[str] = None) -> Optional[ModelRecord]:
        """Get the latest version of a model."""
        records = self.list(model_id=model_id, stage=stage)
        return records[0] if records else None
    
    def reset(self) -> None:
        """Clear all state (for test isolation)."""
        with self._lock:
            self._models.clear()
            self._by_model.clear()
            self._persist_path = None
    
    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Safely load JSON file."""
        if not path.exists():
            return None
        try:
            return read_json(path)
        except Exception:
            return None
    
    def _detect_model_id(self, run_path: Path) -> str:
        """Auto-detect model ID from run directory."""
        # Try run_meta.json
        run_meta = self._load_json(run_path / "run_meta.json")
        if run_meta and "run_id" in run_meta:
            run_id = run_meta["run_id"]
            # Clean up run_id to make a valid model_id
            return run_id.replace("run_", "").replace("-", "_").lower()
        
        # Fallback to directory name
        return run_path.name.replace("run_", "").replace("-", "_").lower()


# Singleton instance
registry_store = RegistryStore()


def register_model(
    run_dir: Union[str, Path],
    *,
    model_id: Optional[str] = None,
    version: Optional[str] = None,
    registered_by: str = "cli",
) -> ModelRecord:
    """Convenience function to register a model.
    
    Args:
        run_dir: Path to run directory
        model_id: Optional model ID (auto-detected if not provided)
        version: Optional version (auto-generated if not provided)
        registered_by: Identifier of registrant
        
    Returns:
        ModelRecord of registered model
    """
    request = RegistrationRequest(
        run_dir=str(run_dir),
        model_id=model_id,
        version=version,
        registered_by=registered_by,
    )
    return registry_store.register(request)


def promote_model(
    model_id: str,
    version: str,
    target_stage: str,
    *,
    promoted_by: str = "cli",
    reason: str = "",
) -> ModelRecord:
    """Convenience function to promote a model.
    
    Args:
        model_id: Model identifier
        version: Model version
        target_stage: Target stage (staging/production/archived)
        promoted_by: Identifier of promoter
        reason: Promotion reason
        
    Returns:
        Updated ModelRecord
    """
    request = PromotionRequest(
        model_id=model_id,
        version=version,
        target_stage=target_stage,
        promoted_by=promoted_by,
        reason=reason,
    )
    return registry_store.promote(request)


def configure_registry_persistence(path: Optional[Union[str, Path]]) -> None:
    """Configure registry persistence.
    
    Args:
        path: Path to registry.jsonl file, or None to disable persistence.
    """
    registry_store.configure_persistence(path)
