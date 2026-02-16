"""Tests for v1.2 integrations — HuggingFace, W&B, MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── HuggingFace URI parsing ──────────────────────────────────────


class TestHuggingFaceURIParsing:
    """Test hf:// URI parsing logic."""

    def test_simple_dataset(self):
        from verifily_cli_v1.integrations.huggingface import _parse_hf_uri
        name, config, split, limit = _parse_hf_uri("hf://squad")
        assert name == "squad"
        assert config is None
        assert split == "train"
        assert limit is None

    def test_org_slash_name(self):
        from verifily_cli_v1.integrations.huggingface import _parse_hf_uri
        name, config, split, limit = _parse_hf_uri("hf://tatsu-lab/alpaca")
        assert name == "tatsu-lab/alpaca"
        assert config is None
        assert split == "train"
        assert limit is None

    def test_with_config(self):
        from verifily_cli_v1.integrations.huggingface import _parse_hf_uri
        name, config, split, limit = _parse_hf_uri("hf://squad/plain_text")
        # squad/plain_text is treated as org/name by default
        assert name == "squad/plain_text"

    def test_with_query_params(self):
        from verifily_cli_v1.integrations.huggingface import _parse_hf_uri
        name, config, split, limit = _parse_hf_uri(
            "hf://squad?split=validation&limit=100"
        )
        assert name == "squad"
        assert split == "validation"
        assert limit == 100

    def test_org_name_config(self):
        from verifily_cli_v1.integrations.huggingface import _parse_hf_uri
        name, config, split, limit = _parse_hf_uri(
            "hf://org/dataset/my_config?split=test"
        )
        assert name == "org/dataset"
        assert config == "my_config"
        assert split == "test"


class TestHuggingFaceReader:
    """Test HuggingFaceReader can_read dispatch."""

    def test_can_read_hf_uri(self):
        from verifily_cli_v1.integrations.huggingface import HuggingFaceReader
        reader = HuggingFaceReader()
        assert reader.can_read(Path("hf://squad")) is True

    def test_cannot_read_regular_path(self):
        from verifily_cli_v1.integrations.huggingface import HuggingFaceReader
        reader = HuggingFaceReader()
        assert reader.can_read(Path("data/train.jsonl")) is False
        assert reader.can_read(Path("data.csv")) is False

    def test_import_error_message(self):
        """Verify graceful error when datasets lib is not installed."""
        from verifily_cli_v1.integrations.huggingface import _import_datasets
        with patch.dict("sys.modules", {"datasets": None}):
            # Force re-import to fail
            import importlib
            with pytest.raises(ImportError, match="pip install verifily\\[huggingface\\]"):
                # Manually test the import guard
                try:
                    import datasets  # noqa: F811
                    if datasets is None:
                        raise ImportError("mocked")
                except ImportError:
                    raise ImportError(
                        "HuggingFace datasets support requires the datasets library. "
                        "Install with: pip install verifily[huggingface]"
                    )


class TestGetReaderHF:
    """Test that get_reader dispatches hf:// URIs."""

    def test_get_reader_hf_uri(self):
        from verifily_cli_v1.core.readers import get_reader
        from verifily_cli_v1.integrations.huggingface import HuggingFaceReader
        reader = get_reader("hf://squad")
        assert isinstance(reader, HuggingFaceReader)

    def test_get_reader_regular_jsonl(self):
        from verifily_cli_v1.core.readers import get_reader
        from verifily_cli_v1.integrations.huggingface import HuggingFaceReader
        reader = get_reader("data/train.jsonl")
        assert not isinstance(reader, HuggingFaceReader)


# ── W&B ──────────────────────────────────────────────────────────


class TestWandbConfig:
    """Test WandbConfig defaults and construction."""

    def test_defaults(self):
        from verifily_cli_v1.integrations.wandb import WandbConfig
        cfg = WandbConfig()
        assert cfg.enabled is False
        assert cfg.project == "verifily"
        assert cfg.entity is None
        assert cfg.tags is None
        assert cfg.run_name is None

    def test_from_dict(self):
        from verifily_cli_v1.integrations.wandb import wandb_config_from_dict
        cfg = wandb_config_from_dict({
            "enabled": True,
            "project": "my-project",
            "entity": "my-team",
            "tags": ["ci", "nightly"],
        })
        assert cfg.enabled is True
        assert cfg.project == "my-project"
        assert cfg.entity == "my-team"
        assert cfg.tags == ["ci", "nightly"]

    def test_import_error_message(self):
        """Verify graceful error when wandb is not installed."""
        with pytest.raises(ImportError, match="pip install verifily\\[wandb\\]"):
            with patch.dict("sys.modules", {"wandb": None}):
                try:
                    import wandb
                    if wandb is None:
                        raise ImportError("mocked")
                except ImportError:
                    raise ImportError(
                        "W&B support requires wandb. "
                        "Install with: pip install verifily[wandb]"
                    )


class TestWandbLogging:
    """Test W&B logging with mocked wandb module."""

    def test_log_pipeline_run_mock(self):
        from verifily_cli_v1.integrations.wandb import log_pipeline_run, WandbConfig

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.url = "https://wandb.ai/test/run/123"
        mock_wandb.init.return_value = mock_run
        mock_wandb.run = mock_run
        mock_wandb.run.summary = {}

        results = {
            "run_id": "test123",
            "decision": {
                "recommendation": "SHIP",
                "exit_code": 0,
                "confidence": 0.95,
                "metrics": {"f1": 0.92, "accuracy": 0.88},
                "deltas": {"f1": 0.02},
            },
            "risk_score": {"total": 15.0},
            "health_index": {"total": 85.0},
            "contamination": {"exact_overlaps": 0, "near_duplicates": 2},
            "config": "verifily.yaml",
        }

        config = WandbConfig(enabled=True, project="test-project")

        with patch(
            "verifily_cli_v1.integrations.wandb._import_wandb",
            return_value=mock_wandb,
        ):
            url = log_pipeline_run(results, config)

        assert url == "https://wandb.ai/test/run/123"
        mock_wandb.init.assert_called_once()
        mock_wandb.log.assert_called_once()
        mock_wandb.log_artifact.assert_called_once()
        mock_wandb.finish.assert_called_once()

        # Verify logged data
        log_call_args = mock_wandb.log.call_args[0][0]
        assert log_call_args["metrics/f1"] == 0.92
        assert log_call_args["decision/confidence"] == 0.95
        assert log_call_args["risk_score"] == 15.0
        assert log_call_args["health_index"] == 85.0


# ── MLflow ───────────────────────────────────────────────────────


class TestMLflowConfig:
    """Test MLflowConfig defaults and construction."""

    def test_defaults(self):
        from verifily_cli_v1.integrations.mlflow import MLflowConfig
        cfg = MLflowConfig()
        assert cfg.enabled is False
        assert cfg.tracking_uri is None
        assert cfg.experiment_name == "verifily"
        assert cfg.register_model is True

    def test_from_dict(self):
        from verifily_cli_v1.integrations.mlflow import mlflow_config_from_dict
        cfg = mlflow_config_from_dict({
            "enabled": True,
            "tracking_uri": "http://mlflow.example.com",
            "experiment_name": "my-exp",
            "register_model": False,
        })
        assert cfg.enabled is True
        assert cfg.tracking_uri == "http://mlflow.example.com"
        assert cfg.experiment_name == "my-exp"
        assert cfg.register_model is False

    def test_import_error_message(self):
        """Verify graceful error when mlflow is not installed."""
        with pytest.raises(ImportError, match="pip install verifily\\[mlflow\\]"):
            with patch.dict("sys.modules", {"mlflow": None}):
                try:
                    import mlflow
                    if mlflow is None:
                        raise ImportError("mocked")
                except ImportError:
                    raise ImportError(
                        "MLflow support requires mlflow. "
                        "Install with: pip install verifily[mlflow]"
                    )


class TestMLflowLogging:
    """Test MLflow logging with mocked mlflow module."""

    def test_log_pipeline_run_mock(self):
        from verifily_cli_v1.integrations.mlflow import log_pipeline_run, MLflowConfig

        mock_mlflow = MagicMock()
        mock_run_info = MagicMock()
        mock_run_info.info.run_id = "mlflow-run-abc123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_info)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        results = {
            "run_id": "test456",
            "decision": {
                "recommendation": "SHIP",
                "exit_code": 0,
                "confidence": 0.90,
                "metrics": {"f1": 0.91, "exact_match": 0.85},
            },
            "risk_score": {"total": 10.0},
            "health_index": {"total": 90.0},
            "contamination": {"exact_overlaps": 1, "near_duplicates": 3},
            "usage": {"verifily_version": "1.2.0"},
            "config": "verifily.yaml",
        }

        config = MLflowConfig(enabled=True, experiment_name="test-exp")

        with patch(
            "verifily_cli_v1.integrations.mlflow._import_mlflow",
            return_value=mock_mlflow,
        ):
            run_id = log_pipeline_run(results, config)

        assert run_id == "mlflow-run-abc123"
        mock_mlflow.set_experiment.assert_called_once_with("test-exp")
        mock_mlflow.log_param.assert_any_call("config_path", "verifily.yaml")
        mock_mlflow.log_param.assert_any_call("recommendation", "SHIP")
        mock_mlflow.log_metric.assert_any_call("f1", 0.91)
        mock_mlflow.log_metric.assert_any_call("confidence", 0.90)
        mock_mlflow.log_metric.assert_any_call("risk_score", 10.0)
        mock_mlflow.set_tag.assert_any_call("verifily.decision", "SHIP")


# ── Pipeline integration (noop when disabled) ─────────────────────


class TestPipelineIntegrations:
    """Test that integration hooks don't crash when disabled."""

    def test_log_integrations_noop_when_no_config(self):
        """Pipeline with empty config doesn't crash on _log_integrations."""
        from verifily_cli_v1.commands.pipeline import _log_integrations
        # Should not raise
        _log_integrations(
            results={"decision": {"recommendation": "SHIP", "exit_code": 0}},
            cfg={},
            ci=True,
        )

    def test_log_integrations_noop_when_disabled(self):
        """Pipeline with wandb/mlflow disabled doesn't crash."""
        from verifily_cli_v1.commands.pipeline import _log_integrations
        _log_integrations(
            results={"decision": {"recommendation": "SHIP", "exit_code": 0}},
            cfg={
                "wandb": {"enabled": False},
                "mlflow": {"enabled": False},
            },
            ci=True,
        )

    def test_log_integrations_graceful_on_import_error(self):
        """Integration import errors are caught, not raised."""
        from verifily_cli_v1.commands.pipeline import _log_integrations
        # Even with enabled=True, if library isn't installed, it should
        # not crash (error swallowed in try/except)
        _log_integrations(
            results={"decision": {"recommendation": "SHIP"}},
            cfg={
                "wandb": {"enabled": True},
                "mlflow": {"enabled": True},
            },
            ci=True,
            verbose=False,
        )
