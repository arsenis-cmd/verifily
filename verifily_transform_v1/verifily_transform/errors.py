"""Verifily Transform error hierarchy."""


class TransformError(Exception):
    """Base error for all Verifily Transform operations."""


class ConfigError(TransformError):
    """Invalid or missing configuration."""


class IngestError(TransformError):
    """Failed to read or parse input data."""


class NormalizeError(TransformError):
    """Schema normalization failed."""


class LabelError(TransformError):
    """Labeling pipeline failed."""


class SynthesisError(TransformError):
    """Synthetic data generation failed."""


class DedupeError(TransformError):
    """Deduplication failed."""


class FilterError(TransformError):
    """Data filtering failed."""


class ArtifactError(TransformError):
    """Artifact packaging or hashing failed."""
