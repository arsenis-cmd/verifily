# Verifily Train v1 -- Roadmap

## 2-Week MVP Plan

**Goal**: A working `verifily train` and `verifily eval` that a senior engineer can demo to YC partners in 2-3 minutes.

### Week 1: Core Training Loop

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Project scaffold: package layout, CLI skeleton (Typer), config loading/validation | `verifily --version` works, `verifily train --dry-run` validates config |
| 2 | `DatasetVersion` primitive: manifest parsing, JSONL loading, hash computation, tag extraction | Can load a local dataset with tags, verify hashes |
| 3 | `TrainJob` + training execution: wire up HF Trainer + PEFT LoRA for SFT (CausalLM) | `verifily train` runs LoRA SFT on Llama-3.1-8B with a test dataset |
| 4 | Run artifact writing: `run_meta.json`, `config.yaml`, `hashes.json`, `environment.json`, adapter saving | Complete run directory produced after training |
| 5 | QLoRA support (4-bit via bitsandbytes), Seq2SeqLM support (Flan-T5), device auto-detection | Both CausalLM and Seq2SeqLM paths work; 4-bit quantization works on CUDA |

### Week 2: Evaluation + Compare + Polish

| Day | Task | Deliverable |
|-----|------|-------------|
| 6 | `verifily eval`: load run, batch inference, compute overall metrics (SFT + classification) | `verifily eval --run <path>` prints metrics |
| 7 | Tag-sliced evaluation: group-by tag keys, compute per-slice metrics, hard examples extraction | Slice output in `eval_results.json`, `hard_examples.jsonl` written |
| 8 | `verifily compare`: load multiple eval results, compute deltas, format table output | `verifily compare --runs a/,b/ --slice-by source` prints comparison table |
| 9 | `verifily reproduce --verify-only`: hash verification, classification task path (text+label JSONL, classification head) | Reproduce verification works; classification fine-tuning works end-to-end |
| 10 | Integration testing, error handling, CLI help text, README, one-command demo script | `./demo.sh` runs train -> eval -> compare in <5 minutes on a small dataset |

### What the MVP Proves

1. **One command**: `verifily train --config train.yaml` -- from dataset to evaluated model.
2. **Dataset awareness**: Metrics sliced by `source:human` vs `source:synthetic` out of the box.
3. **Reproducibility**: Hash chain in every run. `verifily reproduce --verify-only` checks integrity.
4. **Practical attribution**: "Synthetic data improved F1 by 1.5 points on this slice."

### YC Demo Script (2-3 minutes)

```
0:00 - "Verifily Train makes dataset-version-aware fine-tuning one command."
0:15 - Show train.yaml (10 lines). Show dataset with tags.
0:30 - Run `verifily train --config train.yaml --dataset ds_demo@v1`
       (pre-trained, show cached result completing in seconds)
0:45 - Run `verifily eval --run runs/demo_v1/ --slice-by source`
       Show: human slice F1 = 0.73, synthetic slice F1 = 0.68
1:15 - Run `verifily eval --run runs/demo_v2/ --slice-by source`
       Show: after adding better synthetic data, synthetic slice F1 = 0.72
1:30 - Run `verifily compare --runs runs/demo_v1/,runs/demo_v2/ --slice-by source`
       Show: +4 points on synthetic slice, flat on human slice.
1:45 - "This is the feedback loop. You see which data helps, which hurts."
2:00 - Show hard_examples.jsonl: "80% of failures are from AI-contaminated source."
2:15 - "Reproducible. Every run has a hash chain. verifily reproduce --verify-only."
2:30 - "Pricing: free tier for 5 jobs/month, $99/mo for unlimited, managed GPU add-on."
```

## Post-MVP: v1.1 - v1.x

### v1.1 (Week 3-4)

- **Managed mode (basic)**: Submit jobs to Verifily-hosted GPU runners. Requires auth flow, job queue, result retrieval.
- **WandB / MLflow integration**: `--report-to wandb` pass-through.
- **Resume from checkpoint**: `verifily train --resume <run_path>`.

### v1.2 (Month 2)

- **Multi-GPU (single node)**: Accelerate-based data parallelism. No code change in user-facing API.
- **Streaming datasets**: For datasets too large to fit in memory.
- **Classification: CausalLM path**: Fine-tune CausalLM with label tokens instead of a classification head.

### v1.3 (Month 3)

- **Self-host runner**: Dockerized runner agent that customers deploy on their infra.
- **Dataset pull from API**: `verifily train --dataset ds_abc@v3` pulls data from Verifily API.
- **Cross-tag slicing**: Evaluate on `source=human AND difficulty=hard`.

## v2 Vision (6-12 months)

### Data Attribution (Causal)

- **Leave-one-out evaluation**: Train N models, each excluding one data segment. Measure impact.
- **Data Shapley (approximate)**: Use TMC-Shapley or KNN-Shapley for tractable attribution scores.
- **Influence functions**: Per-example influence estimation without retraining.

### Advanced Training

- **DPO / RLHF**: Preference tuning with Verifily-versioned preference datasets.
- **Continued pre-training**: Domain-adaptive pre-training before SFT.
- **Multi-node distributed**: For >70B models.

### Platform Integration

- **Verifily Dashboard**: View runs, metrics, and comparisons in a web UI.
- **CI/CD integration**: `verifily train` as a GitHub Action / GitLab CI step.
- **Model registry**: Push adapter weights to a Verifily-managed registry with version tags.

## Pricing Hooks (v1)

### Tiers

| Tier | Price | Included | Overage |
|------|-------|----------|---------|
| Free | $0 | 5 local jobs/month, eval + compare unlimited | N/A (local jobs are free; limit is for managed mode readiness) |
| Pro | $99/month | Unlimited local jobs, 20 managed GPU-hours/month | $3/GPU-hour |
| Enterprise | Custom | Unlimited everything, self-host runner, SLA, SSO | Negotiated |

### What Counts as a "Job"

- A `verifily train` invocation that produces a completed run.
- `verifily eval` and `verifily compare` are always free (no GPU needed beyond the user's own).
- Failed jobs do not count.

### Managed GPU Pricing

- Billed per GPU-hour, rounded up to the nearest minute.
- GPU types: A100-40GB ($3/hr), A100-80GB ($4/hr), H100 ($6/hr).
- Spot pricing: 50% discount, job may be preempted and resumed.

### Implementation

v1 CLI is fully functional without any Verifily account. Pricing is enforced only when:
1. Managed mode is used (requires Verifily API token).
2. (Future) Usage telemetry for free-tier limits.

Local mode is completely free and unrestricted. No license key, no telemetry, no phone-home.

## Technical Debt / Known Shortcuts in MVP

1. **No async job submission**: `verifily train` blocks until completion. Background execution is a v1.1 feature.
2. **No progress callback API**: Training progress is only visible in terminal output and `train_log.jsonl`.
3. **No model merging**: Adapter-only saving. Full merged model export is a future feature.
4. **Classification path is basic**: Single linear head on CausalLM hidden states. No label-token fine-tuning.
5. **No data validation beyond schema**: We check JSONL structure and required fields, but don't validate data quality (e.g., empty strings, nonsensical text).
