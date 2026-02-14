"""Verifily Train v1.1 — dataset-aware fine-tuning in one command."""

__version__ = "1.1.0"

from verifily_train.config import TrainConfig
from verifily_train.dataset import DatasetVersion
from verifily_train.run import Run
from verifily_train.trainer import train
from verifily_train.evaluator import evaluate, EvalResult
from verifily_train.compare import compare, CompareResult
from verifily_train.reproduce import verify, ReproduceResult
from verifily_train.billing import BillingTracker, UsageRecord
from verifily_train.tuner import run_tuning
from verifily_train.errors import (
    VerifilyTrainError,
    ConfigError,
    DataError,
    TrainingError,
    EvalError,
    ReproduceError,
)

__all__ = [
    "TrainConfig",
    "DatasetVersion",
    "Run",
    "train",
    "evaluate",
    "EvalResult",
    "compare",
    "CompareResult",
    "verify",
    "ReproduceResult",
    "BillingTracker",
    "UsageRecord",
    "run_tuning",
    "VerifilyTrainError",
    "ConfigError",
    "DataError",
    "TrainingError",
    "EvalError",
    "ReproduceError",
]
