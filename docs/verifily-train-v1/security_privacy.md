# Verifily Train v1 -- Security and Privacy

## Threat Model

Verifily Train handles customer training data, model weights, and evaluation metrics. The primary concerns:

1. **Data confidentiality**: Customer training data must not leak outside authorized boundaries.
2. **Model weight confidentiality**: Fine-tuned adapter weights may encode proprietary data.
3. **Reproducibility integrity**: Run artifacts must be tamper-evident.
4. **Credential safety**: HuggingFace tokens, Verifily API tokens must not be logged or persisted in artifacts.

## Data Boundary by Compute Mode

### Local Mode

| Data Type | Leaves Customer Machine? | Notes |
|-----------|-------------------------|-------|
| Raw training data | No | Never transmitted. |
| Processed training data | No | Never transmitted. |
| Synthetic data | No | Never transmitted. |
| Model adapter weights | No | Stored locally in run artifacts. |
| Training metrics (loss, lr) | No | Stored locally in run artifacts. |
| Evaluation metrics | No | Stored locally. |
| Config / environment info | No | Stored locally. |
| HuggingFace model downloads | Inbound only | Downloaded from HF Hub. No customer data sent. |

**Summary**: In local mode, nothing leaves the customer's machine. Verifily Train operates as a local CLI tool with no network calls except HF model downloads.

### Managed Mode (v1: stub / future)

When managed mode is implemented, the following data boundary applies:

| Data Type | Leaves Customer Boundary? | Destination | Justification |
|-----------|--------------------------|-------------|---------------|
| Raw human training data | **No** | -- | Never sent to Verifily infra. |
| Synthetic training data | **Yes** (opt-in) | Verifily runner | Required for training on Verifily infra. Customer must explicitly opt in. |
| Config (train.yaml) | **Yes** | Verifily API | Needed to configure the training job. |
| Aggregate metrics | **Yes** | Verifily API | Loss, F1, accuracy -- no individual predictions. |
| Adapter weights | **Opt-in** | Verifily storage | Customer chooses whether to store adapter on Verifily. Default: no. |
| Individual predictions | **No** | -- | Predictions stay on runner, returned to customer, then deleted. |
| HF tokens | **Yes** (encrypted) | Verifily runner | Used to download base model. Encrypted in transit and at rest. Deleted after job. |

#### Managed Mode Data Flow

```
Customer machine                    Verifily infra
  |                                    |
  |-- config.yaml ------------------>  |
  |-- synthetic_data.jsonl (opt-in) -> | Runner (ephemeral GPU instance)
  |                                    |   |-- downloads base model from HF
  |                                    |   |-- trains adapter
  |                                    |   |-- evaluates
  |                                    |   |-- returns: adapter + metrics
  |<-- adapter_weights.safetensors --- |
  |<-- metrics.json ------------------|
  |                                    |   |-- deletes all data
  |                                    |   |-- instance terminated
```

### Self-Host Runner (future)

- All data stays within customer infrastructure.
- Verifily runner agent runs on customer's machines.
- Only job status and aggregate metrics are sent to Verifily API (for dashboard/billing).
- Customer controls network egress; can firewall all outbound traffic except Verifily API calls.

## Credential Handling

### Tokens Never Persisted in Artifacts

- `HF_TOKEN` and `VERIFILY_TOKEN` are read from environment variables or CLI flags.
- They are **never** written to `config.yaml`, `run_meta.json`, `environment.json`, or any artifact file.
- Log files redact any string matching the pattern `hf_[a-zA-Z0-9]{20,}` or `vfy_[a-zA-Z0-9]{20,}`.

### Token Precedence

1. CLI flag (`--hf-token`, `--verifily-token`)
2. Environment variable (`HF_TOKEN`, `VERIFILY_TOKEN`)
3. HuggingFace CLI cache (`~/.cache/huggingface/token`)
4. Verifily CLI cache (`~/.config/verifily/token`) -- future

### Token Scoping

- HF tokens: read-only access to model repositories is sufficient. Write access is only needed if `push_to_hub: true`.
- Verifily tokens: scoped to dataset read + job submit. No admin/delete permissions needed for training.

## Reproducibility and Integrity

### Hash Chain

Every run produces a `hashes.json` file containing:

- `config_hash`: SHA-256 of the normalized training config.
- `data_hash`: SHA-256 of the input data files (byte-level).
- `environment_hash`: SHA-256 of key package versions.
- `reproducibility_hash`: SHA-256 of (config_hash + data_hash + environment_hash + seed).
- `adapter_weights_hash`: SHA-256 of the output adapter weights.

These hashes enable:
- **Tamper detection**: If any input or output is modified, the hash chain breaks.
- **Reproduction verification**: `verifily reproduce --verify-only` checks all hashes.
- **Audit trail**: Hashes can be stored externally (e.g., in a compliance database) and verified later.

### What Reproducibility Does NOT Guarantee

- Bit-exact training results across different hardware (GPU non-determinism).
- Identical metrics when environment versions differ (library behavior changes).
- Reproducibility is documented as "best effort" -- same config + same data + same environment = same results within statistical noise.

## Audit Logging

### Local Mode

Training events are logged to `train_log.jsonl` within the run directory. This includes:
- Job start/end timestamps.
- Config used (with tokens redacted).
- Data file paths and hashes.
- Training steps and metrics.
- Any errors or warnings.

No external audit log service in v1. The run artifact directory **is** the audit log.

### Managed Mode (future)

- All job submissions logged server-side with: customer ID, job ID, timestamps, config hash, data hash, status.
- No training data content is logged.
- Logs retained for the duration specified in the customer's data retention policy (default: 90 days).
- Customers can request log export or deletion.

## Dependency Security

### Supply Chain

- All dependencies are pinned in `requirements.txt` with version ranges (minimum versions).
- For production use, we recommend pinning exact versions with a lockfile.
- No custom C extensions or binary blobs. All dependencies are from PyPI.

### Model Downloads

- Base models are downloaded from HuggingFace Hub over HTTPS.
- Model integrity is verified by HuggingFace's built-in hash verification.
- Users can pre-download models and point to local paths to avoid runtime network access.

## Recommendations for Customers

1. **Use local mode for sensitive data**. If training data contains PII, trade secrets, or regulated data, use local mode exclusively.
2. **Pre-download models**. Avoid runtime HF Hub access by downloading models ahead of time: `huggingface-cli download <model-id>`.
3. **Pin dependencies**. Use `pip freeze > requirements.lock` and install from lockfile in production.
4. **Rotate tokens**. If HF or Verifily tokens are shared in logs or chat (it happens), rotate them promptly.
5. **Back up run artifacts**. Run directories are the only record of training. Back them up to durable storage.
