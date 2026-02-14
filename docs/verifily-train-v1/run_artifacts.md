# Verifily Train v1 -- Run Artifact Layout

## Directory Structure

Every completed run produces the following artifact directory. This directory is self-contained: it has everything needed to understand, evaluate, and reproduce the run.

```
runs/
  run_20260208_143022_a1b2c3/         # {run_id}

    # -- Metadata --
    run_meta.json                      # Run metadata (IDs, timestamps, hashes, status)
    config.yaml                        # Exact config used (after CLI overrides applied)
    manifest.json                      # Copy of input DatasetVersion manifest
    environment.json                   # Python version, package versions, OS, GPU info

    # -- Model Weights --
    adapter/                           # LoRA adapter weights (default: adapter only)
      adapter_config.json              # PEFT adapter config
      adapter_model.safetensors        # Adapter weights (safetensors format)
    tokenizer/                         # Tokenizer files (copied from base model)
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json

    # -- Training Logs --
    train_log.jsonl                    # Per-step training metrics (loss, lr, grad_norm)
    train_summary.json                 # Final training metrics (total loss, duration, steps)
    checkpoints/                       # Intermediate checkpoints (pruned to save_total_limit)
      checkpoint-500/
      checkpoint-1000/

    # -- Evaluation (populated by `verifily eval`) --
    eval/
      eval_results.json                # Overall metrics + slice metrics
      hard_examples.jsonl              # Worst-performing examples with tags and scores
      predictions.jsonl                # All test predictions (optional, can be large)
      confusion_matrix.json            # Classification only: full confusion matrix

    # -- Reproducibility --
    hashes.json                        # All reproducibility hashes
```

## File Specifications

### `run_meta.json`

```json
{
  "run_id": "run_20260208_143022_a1b2c3",
  "job_id": "job_8f2a1b3c",
  "status": "completed",
  "task": "sft",
  "base_model": "meta-llama/Llama-3.1-8B",
  "dataset_version": "ds_abc123@v3",
  "compute_mode": "local",
  "device": "cuda",
  "started_at": "2026-02-08T14:30:22Z",
  "completed_at": "2026-02-08T15:12:40Z",
  "duration_seconds": 2538,
  "seed": 42,
  "verifily_train_version": "1.0.0",
  "run_name": "mistral-sft-v3-data"
}
```

### `hashes.json`

```json
{
  "config_hash": "sha256:9f2a4b...",
  "data_hash": "sha256:af3b2c...",
  "environment_hash": "sha256:c4d1e5...",
  "reproducibility_hash": "sha256:7a8b9c...",
  "adapter_weights_hash": "sha256:d3e4f5...",
  "hash_algorithm": "sha256",
  "hash_inputs": {
    "config_hash": "SHA-256 of normalized, JSON-serialized config (sorted keys, no whitespace)",
    "data_hash": "SHA-256 of concatenated sorted data file bytes",
    "environment_hash": "SHA-256 of (python_version + torch_version + transformers_version + peft_version)",
    "reproducibility_hash": "SHA-256 of (config_hash + data_hash + environment_hash + seed)",
    "adapter_weights_hash": "SHA-256 of adapter_model.safetensors bytes"
  }
}
```

### `environment.json`

```json
{
  "python_version": "3.11.7",
  "platform": "Linux-6.2.0-x86_64",
  "os": "Ubuntu 22.04.3 LTS",
  "gpu": {
    "name": "NVIDIA A100-SXM4-80GB",
    "count": 1,
    "vram_gb": 80,
    "cuda_version": "12.1",
    "driver_version": "535.129.03"
  },
  "packages": {
    "torch": "2.2.0",
    "transformers": "4.38.0",
    "peft": "0.8.2",
    "accelerate": "0.27.0",
    "bitsandbytes": "0.42.0",
    "datasets": "2.17.0",
    "verifily-train": "1.0.0"
  }
}
```

### `train_log.jsonl`

One JSON object per logging step:

```json
{"step": 10, "epoch": 0.02, "loss": 2.451, "learning_rate": 1.2e-5, "grad_norm": 0.83, "timestamp": "2026-02-08T14:31:05Z"}
{"step": 20, "epoch": 0.04, "loss": 2.103, "learning_rate": 2.4e-5, "grad_norm": 0.71, "timestamp": "2026-02-08T14:31:48Z"}
```

### `train_summary.json`

```json
{
  "total_steps": 1500,
  "total_epochs": 3,
  "final_train_loss": 0.312,
  "best_eval_loss": 0.387,
  "best_eval_step": 1200,
  "total_train_samples": 48000,
  "tokens_processed": 98304000,
  "wall_time_seconds": 2538,
  "gpu_hours": 0.705
}
```

### `eval/eval_results.json`

```json
{
  "run_id": "run_20260208_143022_a1b2c3",
  "test_data_path": "data/test.jsonl",
  "num_test_examples": 2000,
  "eval_duration_seconds": 342,
  "overall": {
    "exact_match": 0.5945,
    "f1": 0.7139,
    "perplexity": 4.21
  },
  "slices": {
    "source": {
      "human": {"n": 800, "f1": 0.7312, "exact_match": 0.6100},
      "synthetic": {"n": 700, "f1": 0.7089, "exact_match": 0.5857},
      "contaminated": {"n": 500, "f1": 0.6841, "exact_match": 0.5600}
    },
    "difficulty": {
      "easy": {"n": 900, "f1": 0.8201, "exact_match": 0.7400},
      "medium": {"n": 700, "f1": 0.6832, "exact_match": 0.5571},
      "hard": {"n": 400, "f1": 0.5510, "exact_match": 0.4000}
    }
  }
}
```

### `eval/hard_examples.jsonl`

```json
{"rank": 1, "f1": 0.0, "exact_match": 0, "prediction": "New York", "reference": "Albany", "question": "What is the capital of New York state?", "tags": {"source": "synthetic", "difficulty": "hard"}, "input_hash": "sha256:abc..."}
{"rank": 2, "f1": 0.05, "exact_match": 0, "prediction": "1492", "reference": "October 12, 1492", "question": "When did Columbus arrive?", "tags": {"source": "human", "difficulty": "medium"}, "input_hash": "sha256:def..."}
```

## Size Estimates

| Component | Typical Size | Notes |
|-----------|-------------|-------|
| `adapter/` | 10-100 MB | Depends on LoRA rank and model size |
| `tokenizer/` | 1-5 MB | Copied from base model |
| `train_log.jsonl` | 0.1-2 MB | ~100 bytes per step |
| `eval/predictions.jsonl` | 5-50 MB | Optional; 2000 test examples ~5 MB |
| `eval/hard_examples.jsonl` | 0.1-1 MB | 50-100 examples |
| `checkpoints/` | 20-200 MB | 2 checkpoints, adapter-only |
| **Total (typical)** | **50-400 MB** | Without full model weights |

## Immutability Contract

Once a run status is `"completed"`:
- No files in the run directory may be modified or deleted by Verifily Train.
- `verifily eval` writes **only** to the `eval/` subdirectory.
- `verifily reproduce` writes to a **new** run directory, never modifies the original.
- Users may delete runs manually. Verifily Train does not garbage-collect.
