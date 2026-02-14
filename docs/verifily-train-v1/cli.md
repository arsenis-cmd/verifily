# Verifily Train v1 -- CLI Design

## Installation

```bash
pip install verifily-train
```

This installs the `verifily` CLI entry point. Dependencies: `transformers`, `peft`, `accelerate`, `datasets`, `evaluate`, `bitsandbytes` (optional, for QLoRA).

## Commands

### `verifily train`

Run a training job.

```
verifily train [OPTIONS]
```

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `str` | `train.yaml` | Path to training config file |
| `--dataset` | `str` | (from config) | Verifily dataset version ref, e.g., `ds_abc123@v3` |
| `--base-model` | `str` | (from config) | HuggingFace model ID, overrides config |
| `--task` | `str` | (from config) | `sft` or `classification`, overrides config |
| `--output-dir` | `str` | `runs/` | Directory to write run artifacts |
| `--device` | `str` | `auto` | `auto`, `cuda`, `cuda:0`, `mps`, `cpu` |
| `--mode` | `str` | `local` | `local` or `managed` |
| `--epochs` | `int` | (from config) | Override `training.num_epochs` |
| `--batch-size` | `int` | (from config) | Override `training.batch_size` |
| `--lr` | `float` | (from config) | Override `training.learning_rate` |
| `--lora-r` | `int` | (from config) | Override `lora.r` |
| `--quantization` | `str` | `none` | `none`, `4bit`, `8bit` |
| `--seed` | `int` | `42` | Random seed |
| `--name` | `str` | (auto) | Human-readable run name |
| `--dry-run` | `flag` | `false` | Validate config and print plan, do not train |
| `--resume` | `str` | (none) | Path to a previous run to resume from checkpoint |
| `--verbose` | `flag` | `false` | Enable DEBUG logging |

**Precedence**: CLI flags > config file > defaults.

**Examples**:

```bash
# Minimal: use config file for everything
verifily train --config train.yaml

# Override model and dataset on CLI
verifily train --config train.yaml \
  --base-model mistralai/Mistral-7B-v0.3 \
  --dataset ds_abc123@v3

# Quick local test with small settings
verifily train --config train.yaml \
  --epochs 1 --batch-size 2 --dry-run

# QLoRA 4-bit on a large model
verifily train --config train.yaml \
  --base-model meta-llama/Llama-3.1-8B \
  --quantization 4bit --device cuda

# Train with explicit local data paths (no Verifily dataset ref)
verifily train --config train.yaml \
  --task sft \
  --base-model google/flan-t5-base

# Managed mode (v1: stub)
verifily train --config train.yaml --mode managed
# Output: "Managed mode is not yet available. Coming soon."
```

**Output on success**:
```
Run completed: run_20260208_143022_a1b2c3
  Duration:    42m 18s
  Train loss:  0.312
  Eval loss:   0.387
  Artifacts:   runs/run_20260208_143022_a1b2c3/
  Config hash: sha256:9f2a...
  Data hash:   sha256:af3b...
  Repro hash:  sha256:c4d1...

Run `verifily eval --run runs/run_20260208_143022_a1b2c3/` for full evaluation.
```

---

### `verifily eval`

Evaluate a trained model on a test set, with optional tag-based slicing.

```
verifily eval [OPTIONS]
```

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run` | `str` | (required) | Path to run artifact directory |
| `--test-data` | `str` | (from run config) | Path to test JSONL file, overrides run config |
| `--metrics` | `str` | (task default) | Comma-separated metric names |
| `--slice-by` | `str` | (from config) | Comma-separated tag keys to slice metrics by |
| `--hard-examples` | `int` | `50` | Number of worst-performing examples to output |
| `--output` | `str` | (run dir) | Directory to write eval results |
| `--device` | `str` | `auto` | Device for inference |
| `--batch-size` | `int` | `16` | Inference batch size |
| `--verbose` | `flag` | `false` | Enable DEBUG logging |

**Examples**:

```bash
# Evaluate a run using test set from config
verifily eval --run runs/run_20260208_143022_a1b2c3/

# Evaluate with specific test data and slicing
verifily eval \
  --run runs/run_20260208_143022_a1b2c3/ \
  --test-data data/test.jsonl \
  --slice-by source,difficulty \
  --hard-examples 100

# Classification-specific metrics
verifily eval \
  --run runs/run_20260208_143022_a1b2c3/ \
  --metrics accuracy,macro_f1,confusion_matrix
```

**Output**:
```
Evaluation: run_20260208_143022_a1b2c3
  Test set: 2000 examples

  Overall:
    exact_match:  0.5945
    f1:           0.7139
    perplexity:   4.21

  Sliced by [source]:
    source=human        (n=800):  f1=0.7312
    source=synthetic    (n=700):  f1=0.7089
    source=contaminated (n=500):  f1=0.6841

  Hard examples written to: runs/run_.../eval/hard_examples.jsonl (50 examples)
  Full results: runs/run_.../eval/eval_results.json

  Note: Slice metrics show correlation, not causation.
```

---

### `verifily compare`

Compare metrics across two or more runs. Surfaces the impact of dataset changes.

```
verifily compare [OPTIONS]
```

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--runs` | `str` | (required) | Comma-separated paths or run IDs |
| `--metric` | `str` | `f1` | Primary metric to compare |
| `--slice-by` | `str` | (none) | Tag key to break down comparison |
| `--format` | `str` | `table` | `table`, `json`, `csv` |
| `--output` | `str` | (stdout) | Write comparison to file |

**Examples**:

```bash
# Compare two runs
verifily compare \
  --runs runs/run_human_v2/,runs/run_synthetic_v3/ \
  --metric f1

# Compare with tag slicing
verifily compare \
  --runs runs/run_a/,runs/run_b/,runs/run_c/ \
  --metric f1 \
  --slice-by source

# Export as CSV
verifily compare \
  --runs runs/run_a/,runs/run_b/ \
  --format csv --output comparison.csv
```

**Output**:
```
Comparison: f1

  Run                      Dataset Version   Overall   source=human   source=synthetic
  run_human_v2             ds_abc123@v2      0.7139    0.7312         0.6950
  run_synthetic_v3         ds_abc123@v3      0.7285    0.7298         0.7271

  Delta (v3 vs v2):                         +0.0146   -0.0014        +0.0321

  Interpretation:
    - Overall F1 improved by +1.46 points.
    - Synthetic-source slice improved most (+3.21 points).
    - Human-source slice slightly decreased (-0.14 points).
    Note: Correlation, not causation.
```

---

### `verifily reproduce`

Verify that a run can be reproduced from its artifacts.

```
verifily reproduce [OPTIONS]
```

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run` | `str` | (required) | Path to run artifact directory |
| `--verify-only` | `flag` | `false` | Check hashes without retraining |
| `--output-dir` | `str` | (auto) | Where to write the reproduced run |
| `--device` | `str` | `auto` | Device for retraining |

**Examples**:

```bash
# Verify reproducibility hashes without retraining
verifily reproduce --run runs/run_20260208_143022_a1b2c3/ --verify-only

# Full reproduction: retrain and compare
verifily reproduce --run runs/run_20260208_143022_a1b2c3/
```

**Output (`--verify-only`)**:
```
Reproducibility check: run_20260208_143022_a1b2c3

  Config hash:   sha256:9f2a... MATCH
  Data hash:     sha256:af3b... MATCH
  Env hash:      sha256:c4d1... MISMATCH (torch 2.2.0 vs 2.3.1)

  Verdict: PARTIAL MATCH (environment differs)
  Note: Metric differences may occur due to environment mismatch.
```

**Output (full reproduction)**:
```
Reproduction complete.
  Original f1:    0.7139
  Reproduced f1:  0.7141
  Delta:          +0.0002 (within expected variance)
  Verdict: REPRODUCIBLE
```

---

## Global Flags

These flags apply to all commands:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hf-token` | `str` | `$HF_TOKEN` | HuggingFace API token |
| `--verifily-token` | `str` | `$VERIFILY_TOKEN` | Verifily API token (for dataset pull / managed mode) |
| `--log-level` | `str` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--no-color` | `flag` | `false` | Disable colored output |
| `--version` | `flag` | -- | Print version and exit |
| `--help` | `flag` | -- | Print help and exit |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error (config invalid, model not found, etc.) |
| `2` | Data error (dataset missing, format invalid, hash mismatch) |
| `3` | Training error (OOM, NaN loss, CUDA error) |
| `4` | Evaluation error |
| `10` | Managed mode not available (v1 stub) |
