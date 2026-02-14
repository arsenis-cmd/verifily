# Verifily Python SDK

Typed Python client for the Verifily API.

## Install

```bash
pip install -e verifily_sdk/
```

## Quick start

```python
from verifily_sdk import VerifilyClient

c = VerifilyClient(base_url="http://localhost:8080", api_key="my-key")

# Health check
print(c.health())

# Run pipeline
r = c.pipeline(config_path="/path/to/verifily.yaml", plan=True)
print(r.decision)
print(r.exit_code)

# Contamination check
r = c.contamination(train_path="train.jsonl", eval_path="eval.jsonl")
print(r.status, r.exact_overlaps)

# Dataset report
r = c.report(dataset_path="dataset.jsonl", sample=3)
print(r.row_count, r.pii_clean)
```

## Authentication

Pass `api_key` to the constructor, or set the `VERIFILY_API_KEY` environment variable.
If no key is configured client-side and the server has auth disabled, requests work without auth.

## Retry

Enable retry for transient errors (502/503/504):

```python
c = VerifilyClient(base_url="http://localhost:8080", retries=2)
```

## Errors

All API errors raise typed exceptions:

- `AuthError` (401)
- `ValidationError` (422)
- `NotFoundError` (404)
- `ServerError` (500+)
- `ApiError` (any other 4xx)
