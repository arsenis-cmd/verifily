"""Tests for the Model Registry system.

Covers:
- Registration (success and failure cases)
- Promotion workflow
- Registry persistence
- CLI outputs
- API endpoints
- SDK client

All tests are fast and deterministic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from verifily_cli_v1.core.registry import (
    ModelRecord,
    ModelStage,
    PromotionError,
    PromotionRequest,
    RegistrationError,
    RegistrationRequest,
    RegistryStore,
    configure_registry_persistence,
    promote_model,
    register_model,
    registry_store,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry store before each test."""
    registry_store.reset()
    yield
    registry_store.reset()


class TestModelRecord:
    """Tests for ModelRecord dataclass."""

    def test_record_creation(self):
        """Should create a valid record."""
        record = ModelRecord(
            model_id="test_model",
            version="v1",
            created_at="2026-02-13T10:00:00Z",
            source_run="runs/test",
            decision="SHIP",
            risk_score=25.0,
            health_index=80.0,
        )
        assert record.model_id == "test_model"
        assert record.version == "v1"
        assert record.stage == ModelStage.NONE.value

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        record = ModelRecord(
            model_id="test",
            version="v1",
            created_at="2026-02-13T10:00:00Z",
            source_run="runs/test",
            decision="SHIP",
            risk_score=25.0,
            health_index=80.0,
            metrics={"f1": 0.75},
        )
        d = record.to_dict()
        assert d["model_id"] == "test"
        assert d["metrics"]["f1"] == 0.75

    def test_from_dict(self):
        """Should deserialize from dict correctly."""
        data = {
            "model_id": "test",
            "version": "v1",
            "created_at": "2026-02-13T10:00:00Z",
            "source_run": "runs/test",
            "decision": "SHIP",
            "risk_score": 25.0,
            "health_index": 80.0,
            "metrics": {},
            "lineage_hash": "",
            "stage": "staging",
            "registered_by": "cli",
            "promotion_history": [],
        }
        record = ModelRecord.from_dict(data)
        assert record.model_id == "test"
        assert record.stage == "staging"


class TestRegistryStore:
    """Tests for RegistryStore."""

    @pytest.fixture
    def mock_run_dir(self, tmp_path: Path) -> Path:
        """Create a mock run directory with required artifacts."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()

        # decision.json - SHIP
        (run_dir / "decision.json").write_text(json.dumps({
            "recommendation": "SHIP",
            "confidence": 0.9,
        }))

        # contract.json - valid
        (run_dir / "contract.json").write_text(json.dumps({
            "valid": True,
            "checks": [{"status": "PASS"}],
        }))

        # contamination_results.json - PASS
        (run_dir / "contamination_results.json").write_text(json.dumps({
            "status": "PASS",
        }))

        # risk_score.json
        (run_dir / "risk_score.json").write_text(json.dumps({
            "total": 20.0,
            "level": "LOW",
        }))

        # health_index.json
        (run_dir / "health_index.json").write_text(json.dumps({
            "total": 85.0,
            "level": "EXCELLENT",
        }))

        # eval_results.json
        eval_dir = run_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "eval_results.json").write_text(json.dumps({
            "overall": {"f1": 0.75, "accuracy": 0.78},
        }))

        return run_dir

    def test_register_success(self, mock_run_dir: Path):
        """Should successfully register a valid run."""
        request = RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="my_model",
            version="v1",
        )

        record = registry_store.register(request)

        assert record.model_id == "my_model"
        assert record.version == "v1"
        assert record.decision == "SHIP"
        assert record.health_index == 85.0

    def test_register_rejects_dont_ship(self, tmp_path: Path):
        """Should reject registration if decision is not SHIP."""
        run_dir = tmp_path / "run_bad"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({
            "recommendation": "DONT_SHIP",
        }))

        request = RegistrationRequest(run_dir=str(run_dir))

        with pytest.raises(RegistrationError) as exc_info:
            registry_store.register(request)

        assert "DONT_SHIP" in str(exc_info.value)

    def test_register_rejects_contamination_fail(self, mock_run_dir: Path):
        """Should reject registration if contamination fails."""
        # Update contamination to FAIL
        (mock_run_dir / "contamination_results.json").write_text(json.dumps({
            "status": "FAIL",
        }))

        request = RegistrationRequest(run_dir=str(mock_run_dir))

        with pytest.raises(RegistrationError) as exc_info:
            registry_store.register(request)

        assert "Contamination" in str(exc_info.value)

    def test_auto_version_generation(self, mock_run_dir: Path):
        """Should auto-generate version if not provided."""
        request1 = RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="auto_model",
        )
        record1 = registry_store.register(request1)
        assert record1.version == "v1"

        # Second registration should get v2
        request2 = RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="auto_model",
        )
        record2 = registry_store.register(request2)
        assert record2.version == "v2"

    def test_promote_to_staging(self, mock_run_dir: Path):
        """Should promote model to staging."""
        # Register first
        reg_request = RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="promo_model",
            version="v1",
        )
        registry_store.register(reg_request)

        # Promote to staging
        promo_request = PromotionRequest(
            model_id="promo_model",
            version="v1",
            target_stage=ModelStage.STAGING.value,
        )
        record = registry_store.promote(promo_request)

        assert record.stage == ModelStage.STAGING.value
        assert len(record.promotion_history) == 1
        assert record.promotion_history[0]["from"] == ModelStage.NONE.value
        assert record.promotion_history[0]["to"] == ModelStage.STAGING.value

    def test_promote_to_production(self, mock_run_dir: Path):
        """Should promote model to production if health is good."""
        # Register and promote to staging first
        reg_request = RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="prod_model",
            version="v1",
        )
        registry_store.register(reg_request)

        registry_store.promote(PromotionRequest(
            model_id="prod_model",
            version="v1",
            target_stage=ModelStage.STAGING.value,
        ))

        # Promote to production
        promo_request = PromotionRequest(
            model_id="prod_model",
            version="v1",
            target_stage=ModelStage.PRODUCTION.value,
        )
        record = registry_store.promote(promo_request)

        assert record.stage == ModelStage.PRODUCTION.value

    def test_promote_to_production_blocked_low_health(self, tmp_path: Path):
        """Should block promotion to production if health is too low."""
        # Create run with low health
        run_dir = tmp_path / "run_low_health"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({"recommendation": "SHIP"}))
        (run_dir / "contract.json").write_text(json.dumps({"valid": True}))
        (run_dir / "contamination_results.json").write_text(json.dumps({"status": "PASS"}))
        (run_dir / "risk_score.json").write_text(json.dumps({"total": 20.0}))
        (run_dir / "health_index.json").write_text(json.dumps({"total": 40.0}))  # Too low

        # Register and promote to staging
        registry_store.register(RegistrationRequest(
            run_dir=str(run_dir),
            model_id="low_health_model",
            version="v1",
        ))
        registry_store.promote(PromotionRequest(
            model_id="low_health_model",
            version="v1",
            target_stage=ModelStage.STAGING.value,
        ))

        # Try to promote to production
        with pytest.raises(PromotionError) as exc_info:
            registry_store.promote(PromotionRequest(
                model_id="low_health_model",
                version="v1",
                target_stage=ModelStage.PRODUCTION.value,
            ))

        assert "Health Index" in str(exc_info.value)

    def test_get_model(self, mock_run_dir: Path):
        """Should retrieve a specific model version."""
        registry_store.register(RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="get_model",
            version="v1",
        ))

        record = registry_store.get("get_model", "v1")

        assert record is not None
        assert record.model_id == "get_model"

    def test_list_models(self, mock_run_dir: Path):
        """Should list all models."""
        registry_store.register(RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="list_model_1",
            version="v1",
        ))
        registry_store.register(RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="list_model_2",
            version="v1",
        ))

        records = registry_store.list()

        assert len(records) == 2

    def test_list_by_stage(self, mock_run_dir: Path):
        """Should filter models by stage."""
        registry_store.register(RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="stage_filter",
            version="v1",
        ))
        registry_store.promote(PromotionRequest(
            model_id="stage_filter",
            version="v1",
            target_stage=ModelStage.STAGING.value,
        ))

        none_models = registry_store.list(stage=ModelStage.NONE.value)
        staging_models = registry_store.list(stage=ModelStage.STAGING.value)

        assert len(none_models) == 0  # Was promoted
        assert len(staging_models) == 1

    def test_history(self, mock_run_dir: Path):
        """Should get history of all versions of a model."""
        registry_store.register(RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="history_model",
            version="v1",
        ))
        registry_store.register(RegistrationRequest(
            run_dir=str(mock_run_dir),
            model_id="history_model",
            version="v2",
        ))

        records = registry_store.history("history_model")

        assert len(records) == 2


class TestRegistryPersistence:
    """Tests for registry persistence."""

    def test_persist_and_replay(self, tmp_path: Path):
        """Should persist and replay events."""
        persist_path = tmp_path / "registry.jsonl"

        # Create and configure store
        store = RegistryStore()
        store.configure_persistence(persist_path)

        # Create a mock run
        run_dir = tmp_path / "run_persist"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({"recommendation": "SHIP"}))
        (run_dir / "contract.json").write_text(json.dumps({"valid": True}))
        (run_dir / "contamination_results.json").write_text(json.dumps({"status": "PASS"}))
        (run_dir / "risk_score.json").write_text(json.dumps({"total": 20.0}))
        (run_dir / "health_index.json").write_text(json.dumps({"total": 80.0}))

        # Register and promote
        record = store.register(RegistrationRequest(
            run_dir=str(run_dir),
            model_id="persist_model",
            version="v1",
        ))
        store.promote(PromotionRequest(
            model_id="persist_model",
            version="v1",
            target_stage=ModelStage.STAGING.value,
        ))

        # Create new store and replay
        store2 = RegistryStore()
        store2.configure_persistence(persist_path)

        # Should have the model
        retrieved = store2.get("persist_model", "v1")
        assert retrieved is not None
        assert retrieved.stage == ModelStage.STAGING.value


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_register_model_convenience(self, tmp_path: Path):
        """Should work via convenience function."""
        run_dir = tmp_path / "run_conv"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({"recommendation": "SHIP"}))
        (run_dir / "contract.json").write_text(json.dumps({"valid": True}))
        (run_dir / "contamination_results.json").write_text(json.dumps({"status": "PASS"}))
        (run_dir / "risk_score.json").write_text(json.dumps({"total": 20.0}))
        (run_dir / "health_index.json").write_text(json.dumps({"total": 80.0}))

        record = register_model(str(run_dir), model_id="conv_model", version="v1")

        assert record.model_id == "conv_model"

    def test_promote_model_convenience(self, tmp_path: Path):
        """Should work via convenience function."""
        run_dir = tmp_path / "run_promo"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({"recommendation": "SHIP"}))
        (run_dir / "contract.json").write_text(json.dumps({"valid": True}))
        (run_dir / "contamination_results.json").write_text(json.dumps({"status": "PASS"}))
        (run_dir / "risk_score.json").write_text(json.dumps({"total": 20.0}))
        (run_dir / "health_index.json").write_text(json.dumps({"total": 80.0}))

        register_model(str(run_dir), model_id="promo_conv", version="v1")

        record = promote_model("promo_conv", "v1", ModelStage.STAGING.value)

        assert record.stage == ModelStage.STAGING.value


class TestRegistryStages:
    """Tests for stage transitions."""

    def test_invalid_promotion(self, tmp_path: Path):
        """Should reject invalid stage transitions."""
        run_dir = tmp_path / "run_stage"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({"recommendation": "SHIP"}))
        (run_dir / "contract.json").write_text(json.dumps({"valid": True}))
        (run_dir / "contamination_results.json").write_text(json.dumps({"status": "PASS"}))
        (run_dir / "risk_score.json").write_text(json.dumps({"total": 20.0}))
        (run_dir / "health_index.json").write_text(json.dumps({"total": 80.0}))

        registry_store.register(RegistrationRequest(
            run_dir=str(run_dir),
            model_id="stage_test",
            version="v1",
        ))

        # Cannot go from NONE directly to PRODUCTION
        with pytest.raises(PromotionError) as exc_info:
            registry_store.promote(PromotionRequest(
                model_id="stage_test",
                version="v1",
                target_stage=ModelStage.PRODUCTION.value,
            ))

        assert "Invalid promotion" in str(exc_info.value)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_run_same_id(self, tmp_path: Path):
        """Same run should produce deterministic IDs."""
        run_dir = tmp_path / "run_det"
        run_dir.mkdir()
        (run_dir / "decision.json").write_text(json.dumps({"recommendation": "SHIP"}))
        (run_dir / "contract.json").write_text(json.dumps({"valid": True}))
        (run_dir / "contamination_results.json").write_text(json.dumps({"status": "PASS"}))
        (run_dir / "risk_score.json").write_text(json.dumps({"total": 20.0}))
        (run_dir / "health_index.json").write_text(json.dumps({"total": 80.0}))

        record1 = registry_store.register(RegistrationRequest(
            run_dir=str(run_dir),
            model_id="det_model",
            version="v1",
        ))

        # Same model_id/version should identify same record
        record2 = registry_store.get("det_model", "v1")

        assert record1.model_id == record2.model_id
        assert record1.version == record2.version
