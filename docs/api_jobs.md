# Verifily Async Jobs API — 5-Minute Guide

Run pipeline, contamination, or report checks **without holding an HTTP connection open**.
Submit a job, poll for status, retrieve the result when ready.

## Quick Start

```bash
# 1. Start the server
verifily serve --port 8080

# 2. Submit an async pipeline job
curl -s -X POST http://localhost:8080/v1/jobs/pipeline \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/verifily.yaml", "plan": true, "ci": true}' | jq .

# Response:
# { "job_id": "a1b2c3d4e5f6g7h8", "status": "QUEUED", "request_id": "...", "project_id": "default" }

# 3. Poll for status
curl -s http://localhost:8080/v1/jobs/a1b2c3d4e5f6g7h8 | jq .status
# "RUNNING" ... then "SUCCEEDED"

# 4. Get the result (same shape as POST /v1/pipeline)
curl -s http://localhost:8080/v1/jobs/a1b2c3d4e5f6g7h8/result | jq .
```

## Endpoints

### Submit Jobs

| Endpoint | Body | Returns |
|----------|------|---------|
| `POST /v1/jobs/pipeline` | Same as `POST /v1/pipeline` | `{job_id, status, request_id, project_id}` |
| `POST /v1/jobs/contamination` | Same as `POST /v1/contamination` | `{job_id, status, request_id, project_id}` |
| `POST /v1/jobs/report` | Same as `POST /v1/report` | `{job_id, status, request_id, project_id}` |
| `POST /v1/jobs/classify` | `{dataset_path, max_rows_scan?, output_dir?}` | `{job_id, status, request_id, project_id}` |

### Query Jobs

| Endpoint | Description |
|----------|-------------|
| `GET /v1/jobs/{job_id}` | Job metadata (status, timestamps, error) |
| `GET /v1/jobs/{job_id}/result` | Full result (same schema as sync endpoint + `job_id`) |
| `GET /v1/jobs?status=...&project_id=...&limit=50` | List jobs with optional filters |

### Status Codes

| Status | Meaning |
|--------|---------|
| `QUEUED` | Job accepted, waiting for worker |
| `RUNNING` | Worker is executing the job |
| `SUCCEEDED` | Done — result available at `/result` |
| `FAILED` | Error — check `error` field in metadata |

### Result Endpoint Behavior

| Job Status | `/result` returns |
|------------|-------------------|
| `SUCCEEDED` | `200` with full result JSON |
| `FAILED` | `500` with error message |
| `QUEUED` / `RUNNING` | `409` — try again later |

## Authentication

When `VERIFILY_API_KEY` is set, all `/v1/jobs/*` endpoints require `Authorization: Bearer <key>`, just like the synchronous endpoints.

```bash
curl -s -X POST http://localhost:8080/v1/jobs/pipeline \
  -H "Authorization: Bearer $VERIFILY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"config_path": "verifily.yaml", "plan": true}'
```

## Persistence

By default, job state is in-memory only (lost on restart). To persist:

```bash
export VERIFILY_JOBS_PERSIST=1
# Optional: custom path (default: /tmp/verifily_jobs_events.jsonl)
export VERIFILY_JOBS_LOG_PATH=/var/log/verifily/jobs_events.jsonl
verifily serve
```

On restart, the store replays the JSONL events file to rebuild state.

## SDK Usage

```python
from verifily_sdk import VerifilyClient

c = VerifilyClient(base_url="http://localhost:8080")

# Submit
sub = c.submit_pipeline_job(config_path="verifily.yaml", plan=True, ci=True)
print(sub.job_id, sub.status)  # "a1b2..." "QUEUED"

# Wait (blocking poll)
meta = c.wait_for_job(sub.job_id, timeout_s=120, poll_s=1.0)
print(meta.status)  # "SUCCEEDED"

# Get result
result = c.get_job_result(sub.job_id)
print(result["decision"]["recommendation"])  # "SHIP"

# List jobs
listing = c.list_jobs(status="SUCCEEDED", project_id="my-project")
for j in listing.jobs:
    print(j.job_id, j.status)
```

## CLASSIFY Job Type

Classify a messy dataset into structured buckets and tags using heuristics (no model calls).

### Submit

```bash
curl -s -X POST http://localhost:8080/v1/jobs/classify \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/path/to/data.csv", "max_rows_scan": 500}' | jq .
```

### Result Shape

```json
{
  "job_id": "...",
  "classification": {
    "row_count": 16,
    "rows_scanned": 16,
    "suggested_schema": "qa",
    "dataset_buckets": [
      {"schema": "qa", "category": "support", "count": 6, "fraction": 0.375}
    ],
    "tags_summary": {
      "language": {"en": 16, "non_en": 0},
      "pii_risk": {"emails": 4, "phones": 3, "rows_with_pii": 4},
      "duplicate_rate": 0.125,
      "categories": {"support": 6, "qa_geography": 4, "qa_tech": 4, "qa_science": 2}
    },
    "warnings": ["High duplicate rate: 12.5%", "PII detected in 4/16 rows"]
  },
  "artifacts": {"classification_json": "/path/to/classification.json"},
  "elapsed_ms": 5
}
```

### Detected Schemas

| Schema | Key fields |
|--------|-----------|
| `sft` | prompt/completion, input/output, instruction/response |
| `qa` | question/context/answer, passage/question/answer |
| `classification` | text/label, body/category, input/label |
| `chat` | messages (list of role/content dicts) |
| `unknown` | No recognized pattern |

### SDK

```python
sub = c.submit_classify_job(dataset_path="/path/to/data.csv")
meta = c.wait_for_job(sub.job_id, timeout_s=60)
result = c.get_job_result(sub.job_id)
print(result["classification"]["suggested_schema"])
```

### Bucket Exports (v1.1)

When `export_buckets: true` and `output_dir` is set, the classify job writes per-bucket JSONL files
and a `suggested_next_steps.json` routing file.

#### Submit with exports

```bash
curl -s -X POST http://localhost:8080/v1/jobs/classify \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/path/to/data.csv",
    "output_dir": "/tmp/classify_out",
    "export_buckets": true,
    "min_bucket_rows": 2
  }' | jq .
```

#### Output directory structure

```
/tmp/classify_out/classification/
  classification.json          # Full classification result
  suggested_next_steps.json    # Routing recommendations
  buckets/
    qa__support.jsonl           # One JSONL file per (schema, category)
    qa__qa_geography.jsonl
    qa__qa_tech.jsonl
    qa__qa_science.jsonl
```

#### Export result shape

When `export_buckets=true`, the result includes an `export_summary` key:

```json
{
  "job_id": "...",
  "classification": { ... },
  "artifacts": {
    "classification_json": "/tmp/.../classification.json",
    "suggested_next_steps": "/tmp/.../suggested_next_steps.json",
    "bucket_qa__support.jsonl": "/tmp/.../buckets/qa__support.jsonl"
  },
  "export_summary": {
    "bucket_paths": {"qa__support.jsonl": "/tmp/.../buckets/qa__support.jsonl"},
    "rows_per_bucket": {"qa__support.jsonl": 6},
    "schemas_per_bucket": {"qa__support.jsonl": "qa"},
    "total_rows_written": 16,
    "suggested_next_steps_path": "/tmp/.../suggested_next_steps.json"
  },
  "elapsed_ms": 8
}
```

#### `min_bucket_rows` filtering

Set `min_bucket_rows` to skip small buckets. For example, `min_bucket_rows: 3` will only export
buckets with 3+ rows.

#### SDK with exports

```python
sub = c.submit_classify_job(
    dataset_path="/path/to/data.csv",
    output_dir="/tmp/classify_out",
    export_buckets=True,
    min_bucket_rows=2,
)
meta = c.wait_for_job(sub.job_id, timeout_s=60)
result = c.get_job_result(sub.job_id)
for name, path in result["export_summary"]["bucket_paths"].items():
    print(f"{name}: {result['export_summary']['rows_per_bucket'][name]} rows")
```

## Architecture

- **In-memory queue** with a single worker thread (serial execution).
- No external services (no Redis, no Celery).
- Worker calls the same runner functions as synchronous endpoints.
- Produces the same artifacts (audit_log.jsonl, usage.json, outputs).
- Thread-safe via `threading.Lock`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFILY_JOBS_PERSIST` | `0` | Set to `1` to enable JSONL persistence |
| `VERIFILY_JOBS_LOG_PATH` | `/tmp/verifily_jobs_events.jsonl` | Custom persistence file path |
