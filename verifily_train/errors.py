"""Verifily Train error hierarchy."""


class VerifilyTrainError(Exception):
    """Base error for all Verifily Train errors."""
    pass


class ConfigError(VerifilyTrainError):
    """Invalid or missing configuration."""
    pass


class DataError(VerifilyTrainError):
    """Dataset loading, validation, or hash mismatch."""
    pass


class TrainingError(VerifilyTrainError):
    """Training failure (OOM, NaN loss, CUDA error, etc.)."""
    pass


class EvalError(VerifilyTrainError):
    """Evaluation failure."""
    pass


class ReproduceError(VerifilyTrainError):
    """Reproducibility verification failure."""
    pass
