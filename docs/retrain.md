# RETRAIN Job Type — User Guide

Run mock (or real) fine-tuning with full Verifily artifact production and automatic SHIP / DONT_SHIP decisions.

## Flow

```
dataset artifact (dataset.jsonl + hashes.json)
  → RETRAIN (mock mode, default)
    → run directory with full contract artifacts
    → eval/eval_results.json (deterministic metrics)
    → decision/decision.json (SHIP / DONT_SHIP / INVESTIGATE)
    → hashes.json, usage.json, audit_log.jsonl
```

## Quick Start

```bash
bash scripts/demo_api_jobs_retrain.sh
```

## Modes

| Mode | Default | Behaviour |
|------|---------|-----------|
| `mock` | Yes | Generates deterministic metrics from `sha256(dataset_hash:seed)`. No GPU, no real training. |
| `real` | No | Gated behind `VERIFILY_ENABLE_REAL_TRAIN=1` env var. Not yet implemented (raises 501). |

## API Endpoints

### Sync retrain

```bash
curl -s -X POST http://localhost:8080/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "/path/to/artifact",
    "output_dir": "/path/to/output",
    "seed": 42
  }'
```

Returns `RetrainResponse` with `run_dir`, `decision`, `exit_code`, `eval_summary`, `artifacts`.

### Async retrain (job queue)

```bash
# Submit
curl -s -X POST http://localhost:8080/v1/jobs/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_dir": "/path/to/artifact",
    "output_dir": "/path/to/output"
  }'

# Poll status
curl -s http://localhost:8080/v1/jobs/{job_id}

# Get result
curl -s http://localhost:8080/v1/jobs/{job_id}/result
```

### Request fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_dir` | string | required | Path to ingested dataset artifact |
| `output_dir` | string | `/tmp/verifily_retrain/...` | Where to write run directory |
| `project_id` | string | `"default"` | Project identifier |
| `base_run_dir` | string | null | Baseline run for regression comparison |
| `contaminated_run_dir` | string | null | Run with contamination results |
| `mode` | string | `"mock"` | `mock` or `real` |
| `seed` | int | 42 | Seed for deterministic mock metrics |
| `metric` | string | `"f1"` | Primary metric name |
| `notes` | string | null | Free-text annotation |

## SDK

```python
from verifily_sdk import VerifilyClient

client = VerifilyClient()

# Sync
resp = client.retrain(dataset_dir="/path/to/artifact", seed=42)
print(resp.decision)  # {"recommendation": "SHIP", ...}

# Async
sub = client.submit_retrain_job(dataset_dir="/path/to/artifact")
print(sub.job_id, sub.status)  # "abc123" "QUEUED"
```

## CLI

```bash
# Sync retrain via CLI
verifily retrain --dataset /path/to/artifact --out /path/to/output --seed 42

# With baseline comparison
verifily retrain --dataset /path/to/artifact --baseline /path/to/baseline_run
```

## Monitor Integration

Monitors can trigger automatic retrains when all gates pass:

```python
MonitorConfig(
    monitor_id="m1",
    project_id="prod",
    config_path="pipeline.yaml",
    allow_retrain=True,           # Enable retrain trigger
    retrain_dataset_dir="/data",  # Dataset for retrain
)
```

When a tick produces `decision == "SHIP"` and contamination + contract pass, a mock retrain is submitted automatically. The tick result includes `retrain_submitted` and `retrain_run_dir`.

## Artifacts Produced

A successful mock retrain creates:

```
runs/retrain_0001/
  config.yaml           # Training configuration
  environment.json      # Python/platform snapshot
  hashes.json           # SHA-256 of all artifacts + chain_hash
  run_meta.json         # status=completed, timestamps, seed
  eval/
    eval_results.json   # Deterministic metrics {f1, exact_match}
  decision/
    decision.json       # Full decision with recommendation
    decision.txt        # Human-readable summary
  usage.json            # Usage tracking
  audit_log.jsonl       # Audit trail
```

All artifacts pass `validate_run_contract()`.

## Determinism

Mock mode guarantees: same `dataset_dir` + same `seed` → identical `eval_summary` and `decision`. This enables reproducible CI testing without real training infrastructure.
