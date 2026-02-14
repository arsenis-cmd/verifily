# Verifily — Product Overview

**The release gate for machine learning.**

Verifily is infrastructure for deciding whether a model should ship. It sits between training and production — transforming raw data into versioned datasets, validating run contracts, detecting eval contamination, and producing a machine-readable ship/don't-ship decision. One command. Runs in CI. Exit code 0 means ship.

---

## The Problem

ML teams ship models without a structured gate between training and production. Eval metrics look good on paper, but nobody checks whether the eval set leaked into training data. Config files go missing between runs. Metric regressions go unnoticed because there is no enforced baseline comparison. The decision to ship lives in a Slack thread or a spreadsheet — not in the pipeline.

**Common failure modes Verifily catches:**

| Failure | What happens | What Verifily does |
|---------|-------------|-------------------|
| Eval contamination | Training data overlaps with eval set; metrics are inflated | SHA-256 exact matching + n-gram Jaccard similarity with configurable thresholds |
| Silent regression | F1 drops 0.08 between runs; nobody notices | Automated delta computation against a pinned baseline with hard thresholds |
| Missing artifacts | config.yaml or hashes.json absent from run directory | Contract validation before any evaluation begins |
| Irreproducible runs | "It worked on my machine" | SHA-256 hash chain over config, data, environment, and seed |
| No audit trail | Cannot explain why a model was shipped or held | Decision summary (JSON + plaintext) persisted alongside run artifacts |

---

## How It Works

Verifily runs as a single pipeline — four checks, one decision:

```
Raw Data
  │
  ▼
┌─────────────┐
│  Transform   │  Ingest → Normalize → Deduplicate → Label → Synthesize
│              │  Output: versioned dataset with lineage + content-addressed ID
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Contract    │  Validate run directory: config.yaml, hashes.json,
│              │  environment.json, eval_results.json all present and valid
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Contamination │  SHA-256 exact match + n-gram Jaccard similarity
│              │  between training data and eval set
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Decision    │  Compare metrics against baseline, apply ship_if criteria,
│              │  produce SHIP / DON'T SHIP / INVESTIGATE + exit code
└─────────────┘
```

Every step reads artifacts from disk and writes artifacts to disk. No hosted service. No vendor lock-in.

---

## Core Capabilities

### 1. Dataset Transformation

Clean data in, versioned artifacts out. A single reproducible pipeline that ingests raw data (JSONL, CSV, or plaintext), normalizes formats, removes duplicates via exact hash and MinHash LSH fuzzy matching, applies labels, optionally synthesizes additional training rows, and redacts PII.

Every dataset gets a **content-addressed version ID** and a **lineage record**:

```
datasets/<name>/<version_id>/
  dataset.jsonl        # training-ready rows
  manifest.json        # metadata, schema, row count, file hashes, chain hash
  hashes.json          # SHA-256 for every file
  lineage.json         # parent datasets, transform config hash, tag distribution
```

The version ID is deterministic: `sha256(content_hash | schema | sorted_parent_hashes)[:12]`. Same input always produces the same version.

**Pipeline steps:**

| Step | What it does |
|------|-------------|
| Ingest | Load JSONL, CSV, or TXT with encoding detection |
| Normalize | Map to canonical schema (SFT or classification) via field auto-detection |
| Deduplicate | Exact (SHA-256) + fuzzy (MinHash LSH, default Jaccard threshold 0.85) |
| Label | Heuristic or LLM-based labeling |
| Synthesize | LLM-based expansion (configurable model, temperature, expansion factor) |
| Filter | Length bounds, repetition detection, leakage check against seed data |
| Package | Write manifest, compute file hashes, generate chain hash |

### 2. Run Contract Validation

Before evaluation begins, Verifily validates that a run directory contains everything needed for a reproducible, auditable decision:

- `config.yaml` — the exact training configuration
- `hashes.json` — SHA-256 integrity chain
- `environment.json` — Python, torch, transformers, device info
- `eval_results.json` — metrics from evaluation

Schema-level validation for dataset rows is also available:

```bash
verifily contract-check --dataset data/train.jsonl --schema sft
```

SFT requires `{instruction, output}`. Classification requires `{text, label}`. QA-style `{question, answer, context}` is accepted and mapped automatically.

### 3. Contamination Detection

Verifily detects when eval set rows have leaked into training data — the most common source of inflated metrics in production ML.

**Two detection methods:**

- **Exact overlap**: SHA-256 hash of each row; any collision is a verbatim leak
- **Near-duplicate**: N-gram Jaccard similarity catches paraphrased or lightly edited copies

Thresholds are configurable per project. Both methods run in linear time over the dataset.

### 4. Decision Surface

Every pipeline run produces a single, machine-readable decision:

```json
{
  "dataset_version": "a3f8c1d20b47",
  "recommendation": "DONT_SHIP",
  "exit_code": 1,
  "metrics": { "f1": 0.6841, "exact_match": 0.5600 },
  "baseline_metrics": { "f1": 0.7139, "exact_match": 0.5945 },
  "deltas": { "f1": -0.0298, "exact_match": -0.0345 },
  "risk_flags": ["f1_below_threshold", "contamination_detected"],
  "reproducibility_verified": true,
  "reasoning": [
    "F1 (0.6841) is below minimum threshold (0.70)",
    "5 exact overlaps detected between train and eval sets"
  ]
}
```

Three possible outcomes:

| Recommendation | Meaning | Exit code |
|---------------|---------|-----------|
| **SHIP** | All criteria pass | 0 |
| **DONT_SHIP** | At least one hard blocker | 1 |
| **INVESTIGATE** | Ambiguous — metrics pass but risk flags present | 2 |

A human-readable `.txt` summary is written alongside the JSON. Both are persisted in the run's `eval/` directory.

---

## Reproducibility Infrastructure

Every artifact Verifily produces is backed by a SHA-256 hash chain:

```
config.yaml ──→ sha256(config)
                    │
dataset.jsonl ──→ sha256(data)
                    │
environment ────→ sha256(env)
                    │
seed ───────────────┤
                    ▼
        reproducibility_hash = sha256(config | data | env | seed)
```

Verification is one command:

```bash
verifily reproduce --run runs/my_run
```

This recomputes every file hash, compares against recorded values, and reports VERIFIED or BROKEN. No network, no GPU, runs in under a second.

---

## CLI Reference

```
verifily init              Generate a config template (transform or train)
verifily doctor            Environment health check (Python, CUDA, MPS, disk, deps)
verifily transform         Run the data transformation pipeline
verifily train             Launch a training run
verifily eval              Evaluate a run and compute metrics
verifily compare           Compare metrics across multiple runs
verifily reproduce         Verify reproducibility via hash chain
verifily pipeline          End-to-end: transform → train → eval → decision
verifily contract-check    Validate dataset schema and run artifacts
```

Every command supports `--plan` for dry-run mode and `--verbose` for debug output.

---

## Evaluation Framework

Verifily evaluates models with tag-aware metric slicing — not just aggregate numbers, but per-source, per-domain, and per-difficulty breakdowns.

**Supported metrics:**

| Task | Metrics |
|------|---------|
| SFT (generative) | exact_match, token F1, perplexity, ROUGE-1, ROUGE-L |
| Classification | accuracy, macro F1, per-class precision/recall, confusion matrix |

**Sliced evaluation example:**

```
Overall:
  f1: 0.7139  |  exact_match: 0.5945

By source:
  human:        f1=0.7312  (n=800)
  synthetic:    f1=0.7089  (n=700)
  contaminated: f1=0.6841  (n=500)   ← worst performer

By difficulty:
  easy:   f1=0.8201  (n=900)
  medium: f1=0.6832  (n=700)
  hard:   f1=0.5510  (n=400)
```

Hard example analysis surfaces the N worst predictions with their tags, scores, and references — so you know exactly where the model fails and which data source is responsible.

---

## Proof

Controlled experiments across three training configurations. Human-derived synthetic data, transformed and validated through Verifily's pipeline.

| Comparison | Delta | What it means |
|-----------|-------|--------------|
| **+1.60 F1** vs AI-contaminated data | Human-derived synthetic training data outperformed AI-generated training data on a clean, uncontaminated eval set |
| **+0.78 F1** vs raw human baseline | The same synthetic data surpassed the raw human-only baseline, showing that transformation and validation improve training quality |

**Experiment details:**
- Base model: FLAN-T5-base (250M parameters)
- Fine-tuning: LoRA (r=16, alpha=32)
- 20,000 training rows per configuration
- Eval set verified clean via Verifily's contamination gate
- 217 deterministic tests, no network, no GPU

---

## Architecture

Verifily is three packages that compose into a single pipeline:

```
┌──────────────────────────────────────────────────┐
│                  verifily_cli_v1                  │
│                                                  │
│  commands/    core/                              │
│  ├─ init      ├─ hashing    (SHA-256 chain)      │
│  ├─ doctor    ├─ manifest   (dataset manifests)  │
│  ├─ transform ├─ env        (device detection)   │
│  ├─ train     ├─ io         (file I/O)           │
│  ├─ eval      ├─ lineage    (dataset versioning) │
│  ├─ compare   ├─ decision   (ship/don't ship)    │
│  ├─ reproduce └─ subprocess (run management)     │
│  ├─ pipeline                                     │
│  └─ contract-check                               │
└─────────┬─────────────────────┬──────────────────┘
          │                     │
          ▼                     ▼
┌─────────────────┐   ┌─────────────────────┐
│verifily_transform│   │   verifily_train     │
│                 │   │                     │
│ ingest          │   │ trainer   (HF/LoRA) │
│ normalize       │   │ evaluator (metrics) │
│ deduplicate     │   │ dataset   (loading) │
│ label           │   │ run       (artifacts)│
│ synthesize      │   │ compare   (multi-run)│
│ filter          │   │ reproduce (hashing) │
│ package         │   │ tuner     (HP search)│
└─────────────────┘   └─────────────────────┘
```

**Key design decisions:**

- **No hosted service.** Reads artifacts from disk, writes artifacts to disk. Runs anywhere Python runs.
- **No vendor lock-in.** Works with GitHub Actions, GitLab CI, or any CI system that respects exit codes.
- **Graceful degradation.** Optional dependencies (openai, datasketch, wandb) are checked at runtime. Missing packages skip the step with a warning — they never crash the pipeline.
- **LoRA-only storage.** Adapter weights are 10-100 MB, not multi-GB full model checkpoints.
- **Canonical JSON hashing.** Sorted keys, no whitespace. Identical configs always produce identical hashes.

---

## Training Infrastructure

Verifily wraps HuggingFace Transformers and PEFT for fine-tuning. It is not a training framework — it is the infrastructure around training that makes runs reproducible and auditable.

**Supported models:**

| Model | Size | Type | QLoRA 4-bit VRAM |
|-------|------|------|------------------|
| meta-llama/Llama-3.1-8B | 8B | CausalLM | ~6 GB |
| mistralai/Mistral-7B-v0.3 | 7B | CausalLM | ~5 GB |
| microsoft/Phi-3-mini-4k-instruct | 3.8B | CausalLM | ~3 GB |
| google/flan-t5-base | 250M | Seq2SeqLM | ~1 GB |
| google/flan-t5-large | 780M | Seq2SeqLM | ~2 GB |

**Training features:**
- LoRA and QLoRA (4-bit/8-bit) fine-tuning
- Multi-dataset training with sampling weights
- Device auto-detection (CUDA > MPS > CPU)
- Gradient accumulation for large effective batch sizes
- Optional WandB experiment tracking
- One-command training: `verifily train --config train.yaml`

---

## Run Artifacts

Every training run produces a self-contained directory:

```
runs/run_20260208_143022_a1b2c3/
  run_meta.json              # run ID, status, timestamps, hashes
  config.yaml                # exact config used
  environment.json           # Python, torch, GPU snapshot
  hashes.json                # SHA-256 integrity chain
  adapter/
    adapter_config.json      # PEFT config
    adapter_model.safetensors
  train_log.jsonl            # per-step loss and metrics
  eval/
    eval_results.json        # overall + sliced metrics
    hard_examples.jsonl      # worst predictions with tags
    decision_summary.json    # SHIP / DONT_SHIP / INVESTIGATE
    decision_summary.txt     # human-readable version
```

Typical run size: 50-400 MB (adapter weights dominate).

---

## Who It's For

Verifily is built for ML engineers and platform teams who ship models on a regular cadence and need a gate that is not a spreadsheet.

**You should use Verifily if:**

- You ship models regularly and need a repeatable, auditable release process
- You have been burned by eval contamination or silent metric regression
- You want every release decision traceable to a specific dataset version, config, and set of results
- You are building internal ML infrastructure and need a decision layer that runs in CI

**Verifily is not:**

- A training framework (it wraps HuggingFace, not replaces it)
- A model registry (artifacts are local, not hosted)
- A compliance product (it produces auditable artifacts, not compliance reports)
- A monitoring dashboard (it makes shipping decisions, not observability)

---

## Quick Start

```bash
# 1. Check your environment
verifily doctor

# 2. Transform raw data into a versioned dataset
verifily transform --in data/raw/ --out datasets/ --schema sft

# 3. Validate the dataset contract
verifily contract-check --dataset datasets/my_data/dataset.jsonl --schema sft

# 4. Train a model
verifily train --config train.yaml

# 5. Evaluate
verifily eval --run runs/my_run --slice-by source

# 6. Compare against baseline
verifily compare --runs runs/baseline,runs/my_run --metric f1

# 7. Verify reproducibility
verifily reproduce --run runs/my_run

# 8. Or run the full pipeline end-to-end
verifily pipeline --config pipeline.yaml
```

---

## CI Integration

Verifily is designed to run in CI. The `pipeline` command exits with code 0 (ship), 1 (don't ship), or 2 (investigate).

**GitHub Actions:**

```yaml
- name: Verifily gate
  run: |
    verifily pipeline --config pipeline.yaml
```

**GitLab CI:**

```yaml
verifily-gate:
  script:
    - verifily pipeline --config pipeline.yaml
  allow_failure: false
```

**Any CI with exit codes:**

```bash
verifily pipeline --config pipeline.yaml || exit 1
```

No tokens, no API keys, no hosted service. Just a Python package that reads files and returns an exit code.

---

*Verifily is the missing step between training and production.*
