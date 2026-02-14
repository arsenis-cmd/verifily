# Verifily Train — Complete User Guide

**Version 1.1.0**

---

## Table of Contents

1. [What is Verifily Train?](#1-what-is-verifily-train)
2. [Installation](#2-installation)
3. [Quick Start (5 minutes)](#3-quick-start)
4. [Configuration Reference](#4-configuration-reference)
5. [CLI Commands](#5-cli-commands)
6. [Python API](#6-python-api)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Comparing Runs](#9-comparing-runs)
10. [Reproducibility](#10-reproducibility)
11. [Multi-Dataset Training](#11-multi-dataset-training)
12. [Distributed Training](#12-distributed-training)
13. [Hyperparameter Tuning](#13-hyperparameter-tuning)
14. [WandB Integration](#14-wandb-integration)
15. [Web Dashboard](#15-web-dashboard)
16. [Billing Hooks](#16-billing-hooks)
17. [Run Artifacts](#17-run-artifacts)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. What is Verifily Train?

Verifily Train is a dataset-aware fine-tuning toolkit that wraps HuggingFace Transformers and PEFT to provide:

- **One-command training** — point it at a YAML config and go.
- **Tag-based evaluation** — slice metrics by any tag in your data (source, difficulty, domain) to see *which data helps which outcomes*.
- **Reproducibility by default** — every run produces a hash chain (config + data + environment) so you can prove what produced what.
- **Dataset attribution** — compare runs trained on different data splits and see per-slice metric deltas.

It supports two task types:
- **SFT** (Supervised Fine-Tuning) — instruction/output pairs for generative models.
- **Classification** — text/label pairs for sequence classification.

---

## 2. Installation

### Basic install

```bash
cd /path/to/your/repo
pip3 install -e .
```

### With optional extras

```bash
# QLoRA (4-bit/8-bit quantization)
pip3 install -e ".[qlora]"

# Weights & Biases logging
pip3 install -e ".[wandb]"

# Web dashboard
pip3 install -e ".[web]"

# ROUGE metrics
pip3 install -e ".[rouge]"

# Everything
pip3 install -e ".[all]"
```

### Verify installation

```bash
python3 -m verifily_train --version
# Output: verifily, version 1.1.0
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Transformers >= 4.35.0
- PEFT >= 0.6.0
- Accelerate >= 0.24.0

---

## 3. Quick Start

### Step 1: Prepare your data

Create a JSONL file. For SFT, each line needs `instruction` and `output`:

```jsonl
{"instruction": "Summarise this article", "input": "The economy grew 3%...", "output": "The economy experienced 3% growth.", "tags": {"source": "human", "difficulty": "easy"}}
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde", "tags": {"source": "synthetic", "difficulty": "easy"}}
```

For classification, each line needs `text` and `label`:

```jsonl
{"text": "I love this product!", "label": "positive", "tags": {"source": "reviews"}}
{"text": "Terrible experience.", "label": "negative", "tags": {"source": "reviews"}}
```

The `tags` field is optional but enables slice-based evaluation.

### Step 2: Write a config

Create `train.yaml`:

```yaml
task: sft
base_model: google/flan-t5-base
seed: 42

data_paths:
  train: data/train.jsonl
  val: data/val.jsonl        # optional
  test: data/test.jsonl      # optional

training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 2e-4
  max_seq_length: 512

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05

output:
  dir: runs/
  save_adapter_only: true
```

### Step 3: Train

```bash
verifily train --config train.yaml
```

Output:
```
Run completed: run_20260208_143022_a1b2c3
  Duration:    342s
  Train loss:  0.0821
  Artifacts:   runs/run_20260208_143022_a1b2c3
  Repro hash:  sha256:abc123...

Run `verifily eval --run runs/run_20260208_143022_a1b2c3` for full evaluation.
```

### Step 4: Evaluate

```bash
verifily eval-cmd --run runs/run_20260208_143022_a1b2c3 \
  --test-data data/test.jsonl \
  --slice-by source,difficulty
```

### Step 5: Compare runs

```bash
verifily compare-cmd \
  --runs runs/run_A,runs/run_B \
  --metric f1 \
  --slice-by source
```

---

## 4. Configuration Reference

The full YAML schema with all defaults:

```yaml
# ─── Required ───────────────────────────────────────────────
task: sft                        # "sft" or "classification"
base_model: ""                   # HuggingFace model ID (required)

# ─── Data ───────────────────────────────────────────────────
dataset_version: null            # Verifily platform dataset ref (e.g. ds_abc@v3)
data_paths:
  train: ""                      # Path to training JSONL (or list of paths for multi-dataset)
  val: null                      # Path to validation JSONL (optional)
  test: null                     # Path to test JSONL (optional)
  weights: null                  # Per-dataset sampling weights for multi-dataset (optional)

# ─── Training ───────────────────────────────────────────────
training:
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2e-4            # Peak learning rate
  lr_scheduler: cosine           # cosine, linear, constant, etc.
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_grad_norm: 1.0
  max_seq_length: 2048           # Max token length for inputs
  eval_steps: 500                # Evaluate every N steps (if val set provided)
  save_steps: 500                # Checkpoint every N steps
  logging_steps: 10              # Log metrics every N steps
  save_total_limit: 2            # Keep only N most recent checkpoints

# ─── LoRA / QLoRA ──────────────────────────────────────────
lora:
  enabled: true
  r: 16                          # LoRA rank
  alpha: 32                      # LoRA alpha (scaling factor)
  dropout: 0.05
  target_modules: auto           # "auto" or list: ["q_proj", "v_proj"]
  quantization: none             # "none", "4bit", "8bit"
  bnb_4bit_compute_dtype: bfloat16

# ─── Compute ───────────────────────────────────────────────
compute:
  mode: local                    # "local" or "managed" (managed = future)
  device: auto                   # "auto", "cuda", "cuda:0", "mps", "cpu"
  fp16: false
  bf16: true
  distributed: false             # Enable multi-GPU via Accelerate
  num_gpus: null                 # null = all available

# ─── Evaluation ─────────────────────────────────────────────
eval:
  enabled: true
  metrics: null                  # null = task defaults. Override: ["exact_match", "f1", "perplexity"]
  gold_set: null                 # Reserved for future gold-set evaluation
  slice_by_tags: null            # Tag keys to slice by: ["source", "difficulty"]
  hard_examples: 50              # Number of worst examples to surface
  batch_size: 16
  generation_max_new_tokens: 128
  generation_num_beams: 4

# ─── Output ─────────────────────────────────────────────────
output:
  dir: runs/                     # Base directory for run artifacts
  save_adapter_only: true        # Save only LoRA adapter (not full model)
  push_to_hub: false             # Reserved for future HF Hub push

# ─── WandB ──────────────────────────────────────────────────
wandb:
  enabled: false
  project: verifily-train        # WandB project name
  entity: null                   # WandB team/entity
  tags: null                     # List of tags: ["experiment-1", "sft"]
  log_model: false               # Upload model artifact to WandB

# ─── Meta ───────────────────────────────────────────────────
seed: 42
name: null                       # Human-readable run name
```

---

## 5. CLI Commands

### `verifily train`

Run a training job.

```bash
verifily train --config train.yaml [OPTIONS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--config` | PATH | Training YAML config (default: `train.yaml`) |
| `--base-model` | TEXT | Override HuggingFace model ID |
| `--task` | sft/classification | Override task type |
| `--dataset` | TEXT | Verifily dataset version ref |
| `--output-dir` | PATH | Override output directory |
| `--device` | TEXT | auto / cuda / mps / cpu |
| `--epochs` | INT | Override num_epochs |
| `--batch-size` | INT | Override batch_size |
| `--lr` | FLOAT | Override learning_rate |
| `--lora-r` | INT | Override LoRA rank |
| `--quantization` | none/4bit/8bit | Override quantization |
| `--seed` | INT | Override random seed |
| `--name` | TEXT | Human-readable run name |
| `--distributed` | FLAG | Enable multi-GPU training |
| `--num-gpus` | INT | Number of GPUs for distributed |
| `--dry-run` | FLAG | Validate config only, don't train |
| `--verbose` | FLAG | Enable DEBUG logging |

**Examples:**

```bash
# Basic training
verifily train --config train.yaml

# Override learning rate and epochs
verifily train --config train.yaml --lr 5e-5 --epochs 5

# QLoRA 4-bit training
verifily train --config train.yaml --quantization 4bit

# Dry run (validate config, create run dir, stop)
verifily train --config train.yaml --dry-run

# Distributed on 4 GPUs
verifily train --config train.yaml --distributed --num-gpus 4
```

---

### `verifily eval-cmd`

Evaluate a trained model on a test set.

```bash
verifily eval-cmd --run <RUN_PATH> [OPTIONS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--run` | PATH | Path to run artifact directory (required) |
| `--test-data` | PATH | Override test JSONL path |
| `--metrics` | TEXT | Comma-separated metric names |
| `--slice-by` | TEXT | Comma-separated tag keys |
| `--hard-examples` | INT | Number of worst examples (default: 50) |
| `--device` | TEXT | Device for inference |
| `--batch-size` | INT | Inference batch size |
| `--verbose` | FLAG | Enable DEBUG logging |

**Examples:**

```bash
# Basic evaluation
verifily eval-cmd --run runs/run_20260208_143022_a1b2c3

# With specific test data and slicing
verifily eval-cmd --run runs/run_abc \
  --test-data data/test.jsonl \
  --slice-by source,difficulty \
  --hard-examples 100

# Only specific metrics
verifily eval-cmd --run runs/run_abc \
  --metrics exact_match,f1,perplexity,rouge1,rougeL
```

**Available SFT metrics:** `exact_match`, `f1`, `perplexity`, `rouge1`, `rougeL`

**Available classification metrics:** `accuracy`, `macro_f1`, `precision_per_class`, `recall_per_class`, `confusion_matrix`

---

### `verifily compare-cmd`

Compare a metric across multiple runs.

```bash
verifily compare-cmd --runs <RUN1,RUN2,...> [OPTIONS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--runs` | TEXT | Comma-separated run paths (required) |
| `--metric` | TEXT | Metric to compare (default: `f1`) |
| `--slice-by` | TEXT | Tag key for slice breakdown |
| `--format` | table/json/csv | Output format (default: `table`) |
| `--output` | PATH | Write to file instead of stdout |

**Examples:**

```bash
# Compare F1 across two runs
verifily compare-cmd --runs runs/run_A,runs/run_B --metric f1

# Compare with slice breakdown, output as JSON
verifily compare-cmd \
  --runs runs/run_A,runs/run_B,runs/run_C \
  --metric exact_match \
  --slice-by source \
  --format json

# Save CSV to file
verifily compare-cmd \
  --runs runs/run_A,runs/run_B \
  --metric f1 \
  --format csv \
  --output comparison.csv
```

**Sample table output:**

```
Comparison: f1

  Run                                  Overall
  ----------------------------------- ----------
  run_20260208_model_a                    0.7139
  run_20260208_model_b                    0.7057
  run_20260208_model_c                    0.7217

  Delta (vs first)                       +0.0000    -0.0082    +0.0078

  Config differences:
    data_paths.train:
      run_20260208_model_a: data/human.jsonl
      run_20260208_model_b: data/contaminated.jsonl
      run_20260208_model_c: data/synthetic.jsonl

  Note: Slice metrics show correlation, not causation.
```

---

### `verifily reproduce`

Verify the reproducibility hash chain of a run.

```bash
verifily reproduce --run <RUN_PATH> --verify-only [OPTIONS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--run` | PATH | Path to run artifact directory (required) |
| `--verify-only` | FLAG | Check hashes without retraining |
| `--verbose` | FLAG | Enable DEBUG logging |

**Example:**

```bash
verifily reproduce --run runs/run_abc --verify-only
```

**Output:**

```
Reproducibility check: run_20260208_143022_a1b2c3
  Config hash:      PASS (sha256:abc123...)
  Data hash:        PASS (sha256:def456...)
  Environment hash: WARN (sha256:789... != sha256:012...)
  Overall:          PASS (with warnings)
```

---

### `verifily tune`

Auto-tune hyperparameters via grid or random search.

```bash
verifily tune --config <CONFIG> --param <SPEC> [OPTIONS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--config` | PATH | Base training YAML config (required) |
| `--param` | TEXT | Parameter search spec (repeatable). Format: `dotted.key=val1,val2,val3` |
| `--strategy` | grid/random | Search strategy (default: `grid`) |
| `--max-trials` | INT | Max trials for random search |
| `--metric` | TEXT | Metric to minimise (default: `eval_loss`) |
| `--verbose` | FLAG | Enable DEBUG logging |

**Examples:**

```bash
# Grid search over learning rate and LoRA rank
verifily tune \
  --config train.yaml \
  --param "training.learning_rate=1e-5,5e-5,2e-4,5e-4" \
  --param "lora.r=8,16,32" \
  --metric eval_loss

# Random search, 10 trials
verifily tune \
  --config train.yaml \
  --param "training.learning_rate=1e-5,5e-5,1e-4,2e-4,5e-4" \
  --param "training.num_epochs=2,3,5" \
  --param "lora.r=8,16,32,64" \
  --strategy random \
  --max-trials 10
```

**Output:**

```
Best trial: run_20260208_tune_trial_7
  eval_loss: 0.082341
  Params: {'training.learning_rate': 5e-05, 'lora.r': 32}
  Artifacts: runs/run_20260208_tune_trial_7
```

Each trial is a full training run saved in the `runs/` directory.

---

### `verifily dashboard`

Launch the web UI.

```bash
verifily dashboard [OPTIONS]
```

| Flag | Type | Description |
|------|------|-------------|
| `--runs-dir` | PATH | Directory containing run artifacts (default: `runs/`) |
| `--port` | INT | Port number (default: `7860`) |
| `--share` | FLAG | Create a public Gradio share link |

**Example:**

```bash
verifily dashboard --port 8080
```

Opens a browser with 4 tabs:
- **Runs** — list of all runs with status, model, loss, duration
- **Run Detail** — full metrics, slices, hashes for a single run
- **Hard Examples** — browse the worst-performing predictions
- **Compare** — side-by-side metric comparison

---

## 6. Python API

All features are accessible programmatically:

```python
from verifily_train import (
    TrainConfig, train, evaluate, compare, verify,
    EvalResult, CompareResult, ReproduceResult,
    BillingTracker, run_tuning,
)
```

### Training

```python
config = TrainConfig.from_yaml("train.yaml")
run = train(config)

print(run.run_id)              # "run_20260208_143022_a1b2c3"
print(run.metrics)             # {"train_loss": 0.0821}
print(run.artifact_path)       # "runs/run_20260208_143022_a1b2c3"
print(run.reproducibility_hash)# "sha256:..."
```

### Evaluation

```python
result = evaluate(
    run_path=run.artifact_path,
    test_data="data/test.jsonl",
    metrics=["exact_match", "f1", "perplexity", "rouge1", "rougeL"],
    slice_by=["source", "difficulty"],
    hard_examples_n=50,
)

print(result.overall)
# {"exact_match": 0.6095, "f1": 0.7217, "perplexity": 4.2301, "rouge1": 0.68, "rougeL": 0.65}

print(result.slices)
# {"source": {"human": {"n": 500, "f1": 0.73}, "synthetic": {"n": 500, "f1": 0.71}}}

print(result.hard_examples[0])
# {"rank": 1, "f1": 0.0, "prediction": "...", "reference": "...", "input": "...", "tags": {...}}
```

### Comparison

```python
from verifily_train.compare import compare, format_comparison

result = compare(
    run_paths=["runs/run_A", "runs/run_B", "runs/run_C"],
    metric="f1",
    slice_by="source",
)

print(result.overall)      # {"run_A": 0.7139, "run_B": 0.7057, "run_C": 0.7217}
print(result.deltas)       # {"run_A": 0.0, "run_B": -0.0082, "run_C": 0.0078}
print(result.config_diff)  # {"data_paths.train": {"run_A": "human.jsonl", "run_B": "contam.jsonl", ...}}

# Pretty-print
print(format_comparison(result, fmt="table"))
```

### Reproducibility

```python
result = verify("runs/run_abc")
print(result.verdict)      # "PASS" or "FAIL"
print(result.checks)       # {"config_hash": "PASS", "data_hash": "PASS", ...}
```

### Hyperparameter Tuning

```python
from verifily_train.tuner import run_tuning

best = run_tuning(
    config_path="train.yaml",
    param_grid={
        "training.learning_rate": [1e-5, 5e-5, 2e-4],
        "lora.r": [8, 16, 32],
    },
    strategy="grid",       # or "random"
    max_trials=None,       # None = all combos for grid
    metric="eval_loss",    # minimised
)

print(best["run_id"])
print(best["score"])
print(best["params"])
```

### Dataset Version

```python
from verifily_train.dataset import DatasetVersion

dsv = DatasetVersion.from_local_paths(
    train_path="data/train.jsonl",
    task="sft",
    val_path="data/val.jsonl",
)

print(dsv.num_rows)            # 5000
print(dsv.tag_keys)            # ["source", "difficulty"]
print(dsv.tag_distribution)    # {"source": {"human": 2500, "synthetic": 2500}, ...}
print(dsv.content_hash)        # "sha256:..."

# Export as manifest
manifest = dsv.to_manifest_dict()
```

### Multi-Dataset Loading

```python
from verifily_train.dataset import load_multi_jsonl

rows = load_multi_jsonl(
    paths=["data/set_a.jsonl", "data/set_b.jsonl", "data/set_c.jsonl"],
    weights=[0.5, 0.3, 0.2],  # optional proportional sampling
)
# Each row has tags._source_dataset = "set_a" / "set_b" / "set_c"
```

### Billing

```python
from verifily_train.billing import BillingTracker

tracker = BillingTracker(run_id="run_abc", artifact_path="runs/run_abc")
tracker.add_hook(lambda record: print(f"BILLING: {record.event} gpu_hours={record.gpu_hours}"))

tracker.start(event="train_start")
# ... do work ...
usage = tracker.end(event="train_end", tokens_processed=1_000_000, train_rows=5000)

print(usage.gpu_hours)       # 0.5
print(usage.storage_bytes)   # 1234567
# Saves usage_summary.json to artifact dir
```

---

## 7. Training

### Task Types

**SFT (Supervised Fine-Tuning)**

For generative models. Data format:

```jsonl
{"instruction": "...", "output": "...", "input": "...(optional)", "tags": {...}}
```

- `instruction` (required) — the task prompt
- `output` (required) — the expected response
- `input` (optional) — additional context
- `tags` (optional) — key-value metadata for slicing

Supports both CausalLM (GPT-style) and Seq2Seq (T5-style) architectures. Auto-detected from model name.

**Classification**

For sequence classification. Data format:

```jsonl
{"text": "...", "label": "...", "tags": {...}}
```

- `text` (required) — input text
- `label` (required) — class label (string)
- `tags` (optional) — metadata for slicing

### LoRA

LoRA is enabled by default. It trains a small adapter (~0.5-2% of model parameters) instead of the full model.

```yaml
lora:
  enabled: true
  r: 16          # rank — higher = more capacity, more memory
  alpha: 32      # scaling — typically 2x the rank
  dropout: 0.05
  target_modules: auto  # or ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Choosing rank (r):**
- r=8: lightweight, good for small models
- r=16: default, good balance
- r=32: higher capacity, good for larger models
- r=64: maximum capacity, use with large models + complex tasks

### QLoRA (Quantised LoRA)

Reduces memory by loading the base model in 4-bit or 8-bit:

```yaml
lora:
  enabled: true
  r: 16
  quantization: 4bit    # "none", "4bit", "8bit"
  bnb_4bit_compute_dtype: bfloat16
```

Requires: `pip3 install bitsandbytes` and a CUDA GPU.

### Device Selection

```yaml
compute:
  device: auto    # auto-detects: CUDA > MPS (Apple Silicon) > CPU
```

Or override via CLI: `--device cuda:1`

---

## 8. Evaluation

### Default Metrics

**SFT:** `exact_match`, `f1`, `perplexity`, `rouge1`, `rougeL`

| Metric | Description |
|--------|-------------|
| `exact_match` | Fraction of predictions that exactly match references (after normalisation) |
| `f1` | Token-level F1 score (precision/recall of overlapping words) |
| `perplexity` | `exp(cross_entropy_loss)` on test data — lower is better |
| `rouge1` | ROUGE-1 F-measure (unigram overlap). Requires `pip install rouge-score` |
| `rougeL` | ROUGE-L F-measure (longest common subsequence). Requires `pip install rouge-score` |

**Classification:** `accuracy`, `macro_f1`, `precision_per_class`, `recall_per_class`, `confusion_matrix`

| Metric | Description |
|--------|-------------|
| `accuracy` | Fraction of correct predictions |
| `macro_f1` | Unweighted mean F1 across all classes |
| `precision_per_class` | Per-class precision, keyed by label name |
| `recall_per_class` | Per-class recall, keyed by label name |
| `confusion_matrix` | NxN matrix of predicted vs actual labels |

### Tag-Based Slicing

If your data has a `tags` field, you can slice evaluation by any tag key:

```bash
verifily eval-cmd --run runs/run_abc --slice-by source,difficulty
```

This computes all metrics *per tag value*:

```
Sliced by [source]:
  source=human (n=2500): exact_match=0.6200, f1=0.7300
  source=synthetic (n=2500): exact_match=0.5990, f1=0.7134

Sliced by [difficulty]:
  difficulty=easy (n=1500): exact_match=0.7800, f1=0.8500
  difficulty=medium (n=2000): exact_match=0.5900, f1=0.7100
  difficulty=hard (n=1500): exact_match=0.4200, f1=0.5900
```

This is the core of Verifily's **dataset attribution**: see which data slices correlate with better or worse model performance.

### Hard Examples

The evaluator surfaces the N worst-performing examples (lowest F1), saved to `eval/hard_examples.jsonl`:

```json
{
  "rank": 1,
  "f1": 0.0,
  "prediction": "The answer is unclear",
  "reference": "42",
  "input": "What is the meaning of life according to the passage?...",
  "tags": {"source": "synthetic", "difficulty": "hard"}
}
```

Use these to diagnose failure modes and improve your training data.

---

## 9. Comparing Runs

The `compare` command is designed for A/B testing training data:

```bash
# Did synthetic data help?
verifily compare-cmd \
  --runs runs/human_baseline,runs/synthetic_augmented \
  --metric f1 \
  --slice-by source
```

### Config Diff

When comparing runs, Verifily automatically detects differences between their `config.yaml` files and includes them in the output:

```
Config differences:
  data_paths.train:
    run_A: data/human_5k.jsonl
    run_B: data/synthetic_15k.jsonl
  training.num_epochs:
    run_A: 3
    run_B: 5
```

This makes it easy to see *what changed* between runs.

---

## 10. Reproducibility

Every run produces a **reproducibility hash**:

```
reproducibility_hash = SHA256(config_hash | data_hash | environment_hash | seed)
```

Where:
- `config_hash` = SHA256 of the full training config
- `data_hash` = SHA256 of the training data file
- `environment_hash` = SHA256 of Python version, package versions, GPU info

This is saved in `hashes.json` alongside the run artifacts. Use `verifily reproduce --verify-only` to check that nothing has changed since training.

### Files saved per run

```
runs/run_20260208_143022_a1b2c3/
  run_meta.json          # Run ID, status, metrics, timestamps
  config.yaml            # Frozen training config
  hashes.json            # Hash chain for reproducibility
  environment.json       # Python/package/GPU snapshot
  training_info.json     # Base model, task, paths
  train_summary.json     # Steps, epochs, final loss, wall time
  usage_summary.json     # GPU-hours, tokens, storage (billing)
  adapter/               # LoRA adapter weights
  tokenizer/             # Saved tokenizer
  eval/
    eval_results.json    # All evaluation metrics
    hard_examples.jsonl  # Worst-performing examples
  checkpoints/           # Intermediate checkpoints
  logs/                  # TensorBoard logs
```

---

## 11. Multi-Dataset Training

Train on multiple JSONL files simultaneously:

```yaml
data_paths:
  train:
    - data/human_curated.jsonl
    - data/synthetic_v1.jsonl
    - data/synthetic_v2.jsonl
  weights: [0.5, 0.3, 0.2]    # optional: proportional sampling
```

Without `weights`, all files are concatenated and shuffled. With `weights`, rows are sampled proportionally.

Every row automatically receives a `_source_dataset` tag with the filename stem, so you can slice evaluation by data source:

```bash
verifily eval-cmd --run runs/run_multi --slice-by _source_dataset
```

---

## 12. Distributed Training

### Via CLI

```bash
verifily train --config train.yaml --distributed --num-gpus 4
```

This auto-launches via `accelerate launch` with the correct mixed-precision settings from your config.

### Via Config

```yaml
compute:
  distributed: true
  num_gpus: 4    # null = all available
  bf16: true
```

### Via Accelerate directly

You can also use Accelerate's launcher directly:

```bash
accelerate launch -m verifily_train.cli train --config train.yaml
```

### Requirements

- Multiple CUDA GPUs
- `accelerate` >= 0.24.0 (included in base install)

---

## 13. Hyperparameter Tuning

### Grid Search

Tries every combination:

```bash
verifily tune \
  --config train.yaml \
  --param "training.learning_rate=1e-5,5e-5,2e-4" \
  --param "lora.r=8,16,32" \
  --strategy grid \
  --metric eval_loss
```

This runs 3 x 3 = 9 trials. Each trial is a full training run.

### Random Search

Samples random combinations:

```bash
verifily tune \
  --config train.yaml \
  --param "training.learning_rate=1e-5,5e-5,1e-4,2e-4,5e-4" \
  --param "training.num_epochs=2,3,5" \
  --param "lora.r=8,16,32,64" \
  --param "lora.dropout=0.0,0.05,0.1" \
  --strategy random \
  --max-trials 15 \
  --metric eval_loss
```

### Tuneable Parameters

Any config value can be tuned using dotted notation:

| Parameter | Example values |
|-----------|---------------|
| `training.learning_rate` | `1e-5,5e-5,2e-4,5e-4` |
| `training.num_epochs` | `2,3,5,10` |
| `training.batch_size` | `4,8,16` |
| `training.warmup_ratio` | `0.0,0.03,0.1` |
| `training.weight_decay` | `0.0,0.01,0.1` |
| `lora.r` | `4,8,16,32,64` |
| `lora.alpha` | `16,32,64` |
| `lora.dropout` | `0.0,0.05,0.1` |
| `training.max_seq_length` | `256,512,1024,2048` |

### Important Notes

- Each trial is a full training run. Grid search with many params can be expensive.
- The metric is **minimised** (lower is better). Use `eval_loss` or `train_loss`.
- All trial runs are saved in the `runs/` directory for inspection.
- A validation set (`data_paths.val`) is required for `eval_loss`.

---

## 14. WandB Integration

### Setup

```bash
pip3 install wandb
wandb login
```

### Enable in Config

```yaml
wandb:
  enabled: true
  project: my-project
  entity: my-team           # optional
  tags: ["sft", "exp-1"]   # optional
  log_model: false          # set true to upload model artifact
```

### What Gets Logged

- Training loss, eval loss, learning rate (every `logging_steps`)
- Final metrics (train_loss, eval_loss)
- Full training config as WandB config
- Model artifact (if `log_model: true`)

### Disable

Set `wandb.enabled: false` (the default) or omit the section entirely.

---

## 15. Web Dashboard

### Launch

```bash
verifily dashboard
# Opens http://localhost:7860
```

### Tabs

1. **Runs** — overview of all runs with status, model, loss, duration. Click Refresh to rescan.

2. **Run Detail** — enter a run ID to see:
   - Training metrics
   - Evaluation metrics (if eval has been run)
   - Slice breakdowns
   - Reproducibility hashes

3. **Hard Examples** — browse the worst predictions for any run.

4. **Compare** — enter comma-separated run IDs and a metric to get a side-by-side comparison.

### Requirements

```bash
pip3 install gradio
# or
pip3 install -e ".[web]"
```

### Public Sharing

```bash
verifily dashboard --share
```

Generates a temporary public URL via Gradio's tunnelling service.

---

## 16. Billing Hooks

Verifily tracks resource usage for every training run.

### Automatic Tracking

Every `verifily train` run automatically:
- Records GPU type and wall-clock time
- Computes GPU-hours
- Estimates tokens processed
- Measures artifact storage size
- Saves `usage_summary.json` to the run directory

### Webhook Notifications

Set the `VERIFILY_BILLING_WEBHOOK` environment variable to POST billing events to your endpoint:

```bash
export VERIFILY_BILLING_WEBHOOK=https://your-api.com/billing/events
verifily train --config train.yaml
```

Events are POSTed as JSON:

```json
{
  "run_id": "run_20260208_143022_a1b2c3",
  "event": "train_end",
  "timestamp": "2026-02-08T14:35:00Z",
  "gpu_type": "NVIDIA A100",
  "gpu_hours": 0.5234,
  "tokens_processed": 1048576,
  "train_rows": 5000,
  "storage_bytes": 52428800,
  "metadata": {}
}
```

### File Logging

Append all billing events to a JSONL file:

```bash
export VERIFILY_BILLING_LOG=/var/log/verifily_billing.jsonl
verifily train --config train.yaml
```

### Custom Hooks (Python API)

```python
from verifily_train.billing import BillingTracker

tracker = BillingTracker(run_id="run_abc", artifact_path="runs/run_abc")

# Custom hook: send to your billing system
def my_hook(record):
    requests.post("https://billing.internal/ingest", json={
        "team": "ml-team",
        "gpu_hours": record.gpu_hours,
        "run_id": record.run_id,
    })

tracker.add_hook(my_hook)
```

### Usage Summary File

Saved automatically to `runs/<run_id>/usage_summary.json`:

```json
{
  "run_id": "run_20260208_143022_a1b2c3",
  "total_gpu_hours": 0.5234,
  "total_tokens_processed": 1048576,
  "storage_bytes": 52428800,
  "storage_mb": 50.0,
  "events": [...]
}
```

---

## 17. Run Artifacts

Every run creates an immutable artifact directory:

```
runs/<run_id>/
├── run_meta.json           # Run metadata (ID, status, metrics, timestamps)
├── config.yaml             # Frozen copy of training config
├── hashes.json             # Reproducibility hash chain
├── environment.json        # Python, packages, GPU snapshot
├── training_info.json      # Base model, task, data paths
├── train_summary.json      # Steps, epochs, final loss, wall time
├── usage_summary.json      # Billing: GPU-hours, tokens, storage
├── adapter/                # LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── tokenizer/              # Saved tokenizer files
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── ...
├── eval/                   # Evaluation outputs
│   ├── eval_results.json   # All metrics, slices
│   └── hard_examples.jsonl # Worst predictions
├── checkpoints/            # Intermediate training checkpoints
└── logs/                   # TensorBoard event logs
```

### Key Files

**`run_meta.json`** — the source of truth for a run:

```json
{
  "run_id": "run_20260208_143022_a1b2c3",
  "status": "completed",
  "task": "sft",
  "base_model": "google/flan-t5-base",
  "dataset_version": "local",
  "device": "mps",
  "started_at": "2026-02-08T14:30:22Z",
  "completed_at": "2026-02-08T14:35:44Z",
  "duration_seconds": 322.5,
  "metrics": {"train_loss": 0.0821},
  "config_hash": "sha256:...",
  "data_hash": "sha256:...",
  "reproducibility_hash": "sha256:...",
  "seed": 42
}
```

**`eval_results.json`** — full evaluation output:

```json
{
  "run_id": "run_20260208_143022_a1b2c3",
  "test_data_path": "data/test.jsonl",
  "num_examples": 1000,
  "overall": {
    "exact_match": 0.6095,
    "f1": 0.7217,
    "perplexity": 4.2301,
    "rouge1": 0.6843,
    "rougeL": 0.6512
  },
  "slices": {
    "source": {
      "human": {"n": 500, "exact_match": 0.62, "f1": 0.73},
      "synthetic": {"n": 500, "exact_match": 0.599, "f1": 0.713}
    }
  },
  "hard_examples": [...],
  "eval_duration_seconds": 45.2
}
```

---

## 18. Troubleshooting

### `command not found: verifily`

The install went to a user directory not on your PATH. Use the module syntax:

```bash
python3 -m verifily_train --version
python3 -m verifily_train train --config train.yaml
```

### `pip: command not found`

Use `pip3` on macOS:

```bash
pip3 install -e .
```

### `CUDA out of memory`

Options:
1. Reduce `training.batch_size` (try 4, 2, or 1)
2. Increase `training.gradient_accumulation_steps` to compensate
3. Enable QLoRA: `lora.quantization: 4bit`
4. Reduce `training.max_seq_length`

### `rouge-score not installed, skipping ROUGE metrics`

```bash
pip3 install rouge-score
# or
pip3 install -e ".[rouge]"
```

### `wandb not installed, disabling WandB logging`

```bash
pip3 install wandb
wandb login
```

### `Gradio is required for the web UI`

```bash
pip3 install gradio
# or
pip3 install -e ".[web]"
```

### Evaluation: `training_info.json not found`

The run may have been interrupted before saving this file. Check that training completed (`status: completed` in `run_meta.json`).

### MPS (Apple Silicon) issues

MPS is supported but some operations may fall back to CPU. If you get MPS errors:
```yaml
compute:
  device: cpu    # fallback
```

### Reproducibility hash mismatch

The environment has changed since the original training. Common causes:
- Different Python version
- Different package versions (especially torch, transformers)
- Different GPU hardware

This is expected when moving between machines. The `WARN` status on environment hash is informational — config and data hashes are the critical checks.

---

## Quick Reference Card

```bash
# Train
verifily train --config train.yaml
verifily train --config train.yaml --lr 5e-5 --epochs 5 --name experiment_1
verifily train --config train.yaml --quantization 4bit
verifily train --config train.yaml --distributed --num-gpus 4
verifily train --config train.yaml --dry-run

# Evaluate
verifily eval-cmd --run runs/<ID> --test-data test.jsonl --slice-by source

# Compare
verifily compare-cmd --runs runs/<A>,runs/<B> --metric f1 --slice-by source

# Verify
verifily reproduce --run runs/<ID> --verify-only

# Tune
verifily tune --config train.yaml --param "training.learning_rate=1e-5,5e-5,2e-4"

# Dashboard
verifily dashboard --port 8080

# Help
verifily --help
verifily train --help
```
