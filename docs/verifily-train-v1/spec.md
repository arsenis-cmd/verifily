# Verifily Train v1 -- Core Specification

## 1. Product Goal

**One sentence**: Given a versioned Verifily dataset, produce a fine-tuned model with full reproducibility and dataset-to-metric attribution, in one command.

**Expanded**:
- Accept a Verifily `DatasetVersion` (a pinned, immutable snapshot of training data with per-row tags).
- Run supervised fine-tuning (SFT) or classification fine-tuning using LoRA/QLoRA on open-source HuggingFace CausalLM or Seq2SeqLM models.
- Produce an immutable `Run` artifact containing adapter weights, metrics, config, and reproducibility hashes.
- Evaluate the resulting model with metrics sliced by dataset tags (e.g., `source:human` vs `source:synthetic`), surfacing which data segments help or hurt.

## 2. Non-Goals (v1)

- **Not a training framework**. We wrap HuggingFace Trainer / PEFT. We do not implement custom optimizers, schedulers, or distributed strategies.
- **Not a model registry**. Runs produce local artifacts. Model deployment, serving, and registry integration are out of scope for v1.
- **Not an AutoML system**. We do not search hyperparameter spaces. Users provide a config; we execute it.
- **Not a data labeling tool**. Data creation and annotation happen upstream in the Verifily platform.
- **No RLHF / DPO / preference tuning**. v1 supports SFT and classification only.
- **No multi-node distributed training**. v1 targets single-machine (single GPU or CPU) execution.

## 3. Target Users

### Primary: ML Engineer at a data-centric AI team (5-50 people)

- Has 1-10 GPUs (or uses cloud instances).
- Already uses HuggingFace models.
- Manages training data across multiple sources (human-annotated, vendor-labeled, synthetic, scraped).
- Wants to understand which data sources improve or degrade model quality.
- Currently tracks experiments manually (spreadsheets, WandB, ad-hoc scripts).

### Buyer Persona: Head of ML / CTO at Series A-B startup

- Cares about data ROI: "We spent $50k on labeling -- did it help?"
- Wants reproducibility for compliance and auditing.
- Evaluating build-vs-buy for training infrastructure.

## 4. Supported Tasks (v1)

### 4.1 Supervised Fine-Tuning (SFT)

- **Input format**: JSONL with `instruction`, `input` (optional), `output` fields.
- **Models**: Any HuggingFace CausalLM (e.g., Llama-3, Mistral, Phi-3) or Seq2SeqLM (e.g., Flan-T5).
- **Method**: LoRA or QLoRA via PEFT.
- **Metrics**: loss, perplexity, format compliance rate, gold-set exact match.

### 4.2 Classification

- **Input format**: JSONL with `text`, `label` fields.
- **Models**: Any HuggingFace model with a classification head, or CausalLM with label-token mapping.
- **Method**: LoRA fine-tuning of base model + linear classification head.
- **Metrics**: accuracy, macro-F1, per-class precision/recall, confusion matrix.

## 5. Supported Base Models

### Requirements
- Must be hosted on HuggingFace Hub (public or gated with token).
- Must be compatible with `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, or `AutoModelForSequenceClassification`.
- Must be compatible with PEFT LoRA.

### Tested Models (v1 launch)

| Model | Size | Type | QLoRA 4-bit VRAM |
|-------|------|------|------------------|
| `meta-llama/Llama-3.1-8B` | 8B | CausalLM | ~6 GB |
| `mistralai/Mistral-7B-v0.3` | 7B | CausalLM | ~5 GB |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | CausalLM | ~3 GB |
| `google/flan-t5-base` | 250M | Seq2SeqLM | ~1 GB |
| `google/flan-t5-large` | 780M | Seq2SeqLM | ~2 GB |

### Untested but Expected to Work
Any HuggingFace model that works with `peft.get_peft_model()` and `transformers.Trainer`.

## 6. Compute Modes

### 6.1 Local Mode (v1 -- fully implemented)

- Training runs on the user's machine.
- Supports: CUDA GPU, Apple MPS, CPU fallback.
- Device auto-detection (same pattern as existing `src/train.py`).
- All data stays local. Nothing leaves the machine.

### 6.2 Managed Mode (v1 -- placeholder, stub CLI)

- Verifily-hosted GPU runners.
- User runs `verifily train --mode managed`, job is submitted to Verifily infra.
- v1 scope: CLI accepts the flag, prints "Managed mode coming soon", exits.
- Data boundary: only synthetic/processed data + config sent to Verifily. Raw customer data never leaves customer boundary. See [security_privacy.md](security_privacy.md).

### 6.3 Self-Host Runner (v1 -- placeholder, documented)

- Customer deploys a Verifily runner agent on their own infra.
- Accepts jobs from the Verifily API; runs training locally.
- v1 scope: documented interface only. No implementation.

## 7. Core Primitives

### 7.1 DatasetVersion

An immutable, pinned snapshot of a training dataset.

```python
@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str          # e.g., "ds_abc123"
    version: str             # e.g., "v3" or content-hash "sha256:af3b..."
    format: str              # "jsonl"
    task: str                # "sft" | "classification"
    num_rows: int
    tags: List[str]          # available per-row tag keys, e.g., ["source", "difficulty"]
    manifest_path: str       # local path to manifest.json
    data_path: str           # local path to data file(s)
    content_hash: str        # SHA-256 of sorted data file bytes
    created_at: str          # ISO 8601 timestamp
```

**Manifest schema** (`manifest.json`):

```json
{
  "dataset_id": "ds_abc123",
  "version": "v3",
  "format": "jsonl",
  "task": "sft",
  "num_rows": 20000,
  "content_hash": "sha256:af3b2c...",
  "created_at": "2026-02-08T12:00:00Z",
  "tag_keys": ["source", "difficulty", "length_bucket"],
  "files": [
    {
      "path": "train.jsonl",
      "split": "train",
      "num_rows": 16000,
      "sha256": "abc123..."
    },
    {
      "path": "val.jsonl",
      "split": "val",
      "num_rows": 2000,
      "sha256": "def456..."
    },
    {
      "path": "test.jsonl",
      "split": "test",
      "num_rows": 2000,
      "sha256": "ghi789..."
    }
  ],
  "row_schema": {
    "sft": ["instruction", "input", "output"],
    "classification": ["text", "label"]
  },
  "tag_schema": {
    "source": {"type": "categorical", "values": ["human", "synthetic", "ai_contaminated"]},
    "difficulty": {"type": "categorical", "values": ["easy", "medium", "hard"]},
    "length_bucket": {"type": "categorical", "values": ["short", "medium", "long"]}
  }
}
```

**Per-row tag encoding** (inline in JSONL):

```json
{
  "instruction": "Summarize the following...",
  "input": "Long article text...",
  "output": "Summary here.",
  "tags": {"source": "human", "difficulty": "medium", "length_bucket": "long"}
}
```

### 7.2 TrainJob

A training job specification. Created from CLI args + config file. Validated before execution.

```python
@dataclass
class TrainJob:
    job_id: str               # auto-generated UUID
    dataset_version: DatasetVersion
    base_model: str           # HF model ID
    task: str                 # "sft" | "classification"
    config: TrainConfig       # parsed from train.yaml
    compute_mode: str         # "local" | "managed" | "self_host"
    device: str               # "cuda" | "mps" | "cpu" | "auto"
    created_at: str
    config_hash: str          # SHA-256 of normalized config
```

### 7.3 Run

An immutable record of a completed training job. Contains everything needed to reproduce the result.

```python
@dataclass(frozen=True)
class Run:
    run_id: str               # auto-generated, e.g., "run_20260208_143022_a1b2c3"
    job: TrainJob
    status: str               # "completed" | "failed"
    started_at: str
    completed_at: str
    duration_seconds: float
    metrics: Dict[str, float]
    artifact_path: str        # path to run artifact directory
    reproducibility_hash: str # SHA-256(config_hash + data_hash + env_hash)
```

See [run_artifacts.md](run_artifacts.md) for the exact artifact directory layout.

## 8. Config Schema: `train.yaml`

```yaml
# -- Required --
task: "sft"                           # str: "sft" | "classification"
base_model: "meta-llama/Llama-3.1-8B" # str: HuggingFace model ID

# -- Dataset (one of dataset_version OR data_paths required) --
dataset_version: "ds_abc123@v3"       # str: Verifily dataset version ref
# OR explicit local paths:
data_paths:
  train: "data/train.jsonl"           # str: path to training split
  val: "data/val.jsonl"               # str: path to validation split (optional)
  test: "data/test.jsonl"             # str: path to test split (optional)

# -- Training --
training:
  num_epochs: 3                       # int, default: 3
  batch_size: 8                       # int, default: 8 (per device)
  gradient_accumulation_steps: 4      # int, default: 4
  learning_rate: 2.0e-4               # float, default: 2e-4
  lr_scheduler: "cosine"              # str, default: "cosine". Options: "linear", "cosine", "constant"
  warmup_ratio: 0.03                  # float, default: 0.03 (fraction of total steps)
  weight_decay: 0.01                  # float, default: 0.01
  max_grad_norm: 1.0                  # float, default: 1.0
  max_seq_length: 2048                # int, default: 2048
  eval_steps: 500                     # int, default: 500
  save_steps: 500                     # int, default: 500
  logging_steps: 10                   # int, default: 10

# -- LoRA / QLoRA --
lora:
  enabled: true                       # bool, default: true
  r: 16                               # int, default: 16
  alpha: 32                           # int, default: 32
  dropout: 0.05                       # float, default: 0.05
  target_modules: "auto"              # str | list[str], default: "auto" (PEFT auto-detect)
  quantization: "none"                # str, default: "none". Options: "none", "4bit", "8bit"
  bnb_4bit_compute_dtype: "bfloat16"  # str, default: "bfloat16". Used when quantization: "4bit"

# -- Compute --
compute:
  mode: "local"                       # str, default: "local". Options: "local", "managed"
  device: "auto"                      # str, default: "auto". Options: "auto", "cuda", "mps", "cpu"
  fp16: false                         # bool, default: false
  bf16: true                          # bool, default: true (if supported)

# -- Evaluation (run automatically at end of training) --
eval:
  enabled: true                       # bool, default: true
  metrics: ["loss", "perplexity"]     # list[str], task-dependent defaults
  gold_set: null                      # str: path to gold evaluation set (optional)
  slice_by_tags: ["source"]           # list[str]: tag keys to slice metrics by

# -- Output --
output:
  dir: "runs/"                        # str, default: "runs/"
  save_adapter_only: true             # bool, default: true (save LoRA adapter, not full model)
  push_to_hub: false                  # bool, default: false

# -- Reproducibility --
seed: 42                              # int, default: 42
```

### Config Validation Rules

1. `task` must be `"sft"` or `"classification"`.
2. Either `dataset_version` or `data_paths.train` must be provided.
3. If `lora.quantization` is `"4bit"` or `"8bit"`, `bitsandbytes` must be installed.
4. If `compute.device` is `"auto"`, detect in order: CUDA > MPS > CPU.
5. `base_model` must be resolvable on HuggingFace Hub or be a local path.
6. If `eval.slice_by_tags` is set, the dataset must contain a `tags` field with matching keys.

## 9. Dataset-to-Metric Attribution (v1 Approach)

### Philosophy
Full causal attribution (e.g., data Shapley) is prohibitively expensive. v1 uses a practical approach that gives 80% of the insight at 1% of the cost.

### v1 Method: Tag-Sliced Evaluation + Hard Examples

1. **Tag-sliced metrics**: After training, evaluate on the test set. For each tag key (e.g., `source`), compute metrics separately for test examples matching each tag value.
   - Example: F1 on `source:human` test examples vs F1 on `source:synthetic` test examples.
   - This tells you: "the model trained on dataset version X performs well on human-sourced questions but poorly on synthetic ones."

2. **Hard examples list**: For each run, output the N worst-performing test examples (lowest F1 / confidence). Tag these with their dataset tags. This surfaces patterns like: "most hard examples come from `source:ai_contaminated`."

3. **Cross-version delta**: `verifily compare` diffs metrics between two runs trained on different dataset versions. The delta, sliced by tags, shows which data changes caused which metric changes.

### Limitations (acknowledged)
- This is observational, not causal. Adding synthetic data and seeing improved synthetic-slice metrics doesn't prove the synthetic data caused it.
- Confounders exist (dataset size changes, distribution shifts).
- Documented clearly in CLI output: "Correlation, not causation. For causal attribution, see v2 roadmap."

## 10. Open Questions

1. **Auth flow for managed mode**: OAuth2 device flow? API key? Both? Decision needed before managed mode implementation.
2. **Dataset pull mechanism**: Should `verifily train` pull the dataset from the Verifily API on-the-fly, or require a prior `verifily dataset pull` step? On-the-fly is simpler UX; explicit pull is more transparent.
3. **Multi-GPU**: v1 explicitly excludes multi-node, but should we support single-node multi-GPU via Accelerate in v1? Cost: ~2 days of engineering. Benefit: covers the 80% use case for teams with a single 4xA100 box.
4. **Wandb / MLflow integration**: Should `verifily train` emit metrics to external trackers? v1 could support `--report-to wandb` as a pass-through to HF Trainer. Low effort, high perceived value.
5. **Test split requirement**: Should we require a test split in every DatasetVersion, or allow eval-only on validation? Requiring test enforces good practice but adds friction.
6. **Adapter weight upload**: In managed mode, should adapter weights be uploaded to Verifily by default? Privacy implications -- see [security_privacy.md](security_privacy.md).
7. **Classification head architecture**: For classification tasks on CausalLM, do we use the last-token hidden state + linear head, or fine-tune with label tokens? Former is standard; latter is simpler for LoRA. Needs benchmarking.
