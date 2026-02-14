# Verifily Train v1 -- Python API

## Overview

The Python API mirrors the CLI. Every CLI command corresponds to a top-level function. The API is the implementation layer; the CLI is a thin wrapper.

```
verifily_train/
  __init__.py          # Public API: train(), evaluate(), compare(), reproduce()
  config.py            # Config loading, validation, merging
  dataset.py           # DatasetVersion, manifest parsing, hash verification
  trainer.py           # TrainJob creation and execution
  evaluator.py         # Evaluation, slicing, hard examples
  compare.py           # Cross-run comparison
  reproduce.py         # Reproducibility verification
  run.py               # Run artifact reading/writing
  utils.py             # Hashing, device detection, logging
  cli.py               # Click/Typer CLI entry points
```

## Core API

### `verifily_train.train()`

```python
def train(
    config: Union[str, dict],     # Path to train.yaml or dict
    dataset: Optional[str] = None,  # "ds_abc123@v3" override
    base_model: Optional[str] = None,
    output_dir: str = "runs/",
    device: str = "auto",
    seed: int = 42,
    dry_run: bool = False,
    resume_from: Optional[str] = None,
) -> Run:
    """
    Execute a training job.

    Returns a Run object with metrics, artifact path, and reproducibility hash.
    Raises ConfigError, DataError, or TrainingError on failure.
    """
```

**Usage**:

```python
from verifily_train import train

run = train(
    config="train.yaml",
    dataset="ds_abc123@v3",
    base_model="mistralai/Mistral-7B-v0.3",
)

print(run.run_id)            # "run_20260208_143022_a1b2c3"
print(run.metrics)           # {"train_loss": 0.312, "eval_loss": 0.387}
print(run.artifact_path)     # "runs/run_20260208_143022_a1b2c3/"
print(run.reproducibility_hash)  # "sha256:c4d1..."
```

### `verifily_train.evaluate()`

```python
def evaluate(
    run_path: str,                          # Path to run artifact directory
    test_data: Optional[str] = None,        # Override test data path
    metrics: Optional[List[str]] = None,    # Override metric list
    slice_by: Optional[List[str]] = None,   # Tag keys to slice by
    hard_examples: int = 50,                # Number of hard examples to surface
    device: str = "auto",
    batch_size: int = 16,
) -> EvalResult:
    """
    Evaluate a trained model from a run.

    Returns EvalResult with overall metrics, sliced metrics, and hard examples.
    """
```

**Usage**:

```python
from verifily_train import evaluate

result = evaluate(
    run_path="runs/run_20260208_143022_a1b2c3/",
    slice_by=["source", "difficulty"],
    hard_examples=100,
)

print(result.overall)         # {"f1": 0.7139, "exact_match": 0.5945}
print(result.slices)          # {"source=human": {"f1": 0.7312}, ...}
print(len(result.hard_examples))  # 100
```

### `verifily_train.compare()`

```python
def compare(
    run_paths: List[str],
    metric: str = "f1",
    slice_by: Optional[str] = None,
) -> CompareResult:
    """
    Compare metrics across multiple runs.

    Returns CompareResult with per-run metrics, deltas, and optional slice breakdown.
    """
```

### `verifily_train.reproduce()`

```python
def reproduce(
    run_path: str,
    verify_only: bool = False,
    output_dir: Optional[str] = None,
    device: str = "auto",
) -> ReproduceResult:
    """
    Verify or reproduce a run.

    If verify_only=True, checks hashes without retraining.
    Otherwise, retrains and compares metrics.
    """
```

## Data Classes

### `TrainConfig`

```python
@dataclass
class TrainConfig:
    task: str                              # "sft" | "classification"
    base_model: str                        # HF model ID
    dataset_version: Optional[str]         # "ds_abc123@v3"
    data_paths: Optional[DataPaths]        # explicit local paths
    training: TrainingParams
    lora: LoraParams
    compute: ComputeParams
    eval: EvalParams
    output: OutputParams
    seed: int

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load and validate from YAML file."""

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Load and validate from dict."""

    def merge_cli_overrides(self, **kwargs) -> "TrainConfig":
        """Return new config with CLI overrides applied."""

    def validate(self) -> None:
        """Raise ConfigError if config is invalid."""

    def to_hash(self) -> str:
        """SHA-256 of normalized, sorted config dict."""
```

### `DataPaths`

```python
@dataclass
class DataPaths:
    train: str
    val: Optional[str] = None
    test: Optional[str] = None
```

### `TrainingParams`

```python
@dataclass
class TrainingParams:
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
```

### `LoraParams`

```python
@dataclass
class LoraParams:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, List[str]] = "auto"
    quantization: str = "none"           # "none", "4bit", "8bit"
    bnb_4bit_compute_dtype: str = "bfloat16"
```

### `EvalResult`

```python
@dataclass
class EvalResult:
    run_id: str
    test_data_path: str
    num_examples: int
    overall: Dict[str, float]            # {"f1": 0.71, "exact_match": 0.59, ...}
    slices: Dict[str, Dict[str, float]]  # {"source=human": {"f1": 0.73}, ...}
    hard_examples: List[Dict]            # worst-performing examples with tags
    eval_duration_seconds: float
```

### `CompareResult`

```python
@dataclass
class CompareResult:
    runs: List[str]                      # run IDs
    metric: str                          # compared metric name
    overall: Dict[str, float]            # {run_id: metric_value}
    deltas: Dict[str, float]             # {run_id: delta_vs_first_run}
    slices: Optional[Dict[str, Dict[str, float]]]  # per-slice comparison
```

## Error Hierarchy

```python
class VerifilyTrainError(Exception): pass
class ConfigError(VerifilyTrainError): pass
class DataError(VerifilyTrainError): pass
class TrainingError(VerifilyTrainError): pass
class EvalError(VerifilyTrainError): pass
class ReproduceError(VerifilyTrainError): pass
```

## Integration Points

### HuggingFace Trainer

`verifily_train.trainer` constructs a `transformers.Trainer` (or `Seq2SeqTrainer`) internally. It does not expose the Trainer object. Users who need custom Trainer behavior should use the HF stack directly and Verifily Train for eval/compare only.

### PEFT

LoRA/QLoRA config is translated to `peft.LoraConfig` internally. `target_modules: "auto"` delegates to PEFT's auto-detection.

### Dataset Pull (v1: local only)

In v1, `DatasetVersion` is resolved from local files. The `dataset_version` string in config is matched against a local `manifest.json`. Future versions will pull from the Verifily API.

```python
# v1 resolution order:
# 1. Check data_paths in config (explicit local paths)
# 2. Check dataset_version against local manifest files in output.dir
# 3. (future) Pull from Verifily API
```
