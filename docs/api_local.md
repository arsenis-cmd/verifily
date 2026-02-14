# Verifily Local API

Run Verifily pipeline, contamination checks, and dataset reports over HTTP from your own stack. The server binds to `127.0.0.1` by default and never leaves localhost.

## Install

```bash
pip install fastapi uvicorn pydantic httpx
```

(These are already listed in `verifily_cli_v1/requirements.txt`.)

## Start the server

```bash
# Default: 127.0.0.1:8000
verifily serve

# Custom port
verifily serve --port 8099

# With project directory
verifily serve --project /path/to/my/project --port 8099
```

Interactive API docs are available at `http://127.0.0.1:8000/docs` while the server is running.

## Endpoints

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "time": "2026-02-08T12:00:00Z",
  "mode": "local"
}
```

### POST /v1/pipeline

Run the full pipeline gate: contract check, dataset report, contamination check, and decision.

```bash
# Using a config file
curl -X POST http://127.0.0.1:8000/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "/path/to/verifily.yaml",
    "plan": false,
    "ci": true
  }'

# Using project auto-discovery
curl -X POST http://127.0.0.1:8000/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "project_path": "/path/to/project",
    "plan": true,
    "ci": true
  }'
```

Response includes:
- `exit_code` — same as CLI: 0=SHIP, 1=DONT_SHIP, 2=INVESTIGATE, 3=CONTRACT_FAIL, 4=TOOL_ERROR
- `decision` — recommendation, confidence, reasons, risk_flags, metrics
- `contamination` — status, overlap counts and fractions
- `contract` — valid, checks, has_eval
- `output_dir` — where artifacts were written (null if plan=true)
- `elapsed_ms` — wall-clock time

When `plan=true`, no files are written to disk.

### POST /v1/contamination

Check for train/eval data leakage.

```bash
curl -X POST http://127.0.0.1:8000/v1/contamination \
  -H "Content-Type: application/json" \
  -d '{
    "train_path": "/path/to/train.jsonl",
    "eval_path": "/path/to/eval.jsonl",
    "no_write": true
  }'
```

Response: `status` (PASS/FAIL/WARN), `exit_code` (0/1/2), overlap counts.

### POST /v1/report

Generate dataset statistics and PII scan.

```bash
curl -X POST http://127.0.0.1:8000/v1/report \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/path/to/dataset.jsonl",
    "schema": "sft",
    "sample": 3
  }'
```

Response: row_count, field_stats, tag_distribution, pii_summary (counts only, never raw PII).

## Example: Customer drill via API

```bash
# 1. Start server
verifily serve --port 8099 &

# 2. Ingest messy CSV
verifily ingest \
  --in examples/customer_drill/raw/support_tickets.csv \
  --schema sft \
  --map question:subject --map answer:resolution --map context:body \
  --tag source:customer_drill \
  --out /tmp/verifily_api_demo/datasets/customer_train_artifact

# 3. Call pipeline API (clean path)
curl -s -X POST http://127.0.0.1:8099/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/tmp/verifily_api_demo/pipeline_clean.yaml", "plan": true}'

# 4. Call pipeline API (leaked path)
curl -s -X POST http://127.0.0.1:8099/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/tmp/verifily_api_demo/pipeline_leaked.yaml", "plan": true}'
```

Or run the full automated demo:

```bash
bash scripts/demo_api_customer_drill.sh
```

## CI exit codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | SHIP | All checks passed, safe to deploy |
| 1 | DONT_SHIP | Hard blocker found (contamination, metric below threshold) |
| 2 | INVESTIGATE | Risk flags present but no hard blockers |
| 3 | CONTRACT_FAIL | Run directory missing required artifacts |
| 4 | TOOL_ERROR | Verifily internal error |

## Security: local-only

The API server is designed for **local use only**:

- Default bind: `127.0.0.1` (localhost)
- Non-local hosts are rejected unless `--allow-nonlocal` is passed
- No authentication, no user accounts, no database
- PII is never returned in raw form — only redacted counts
- No data leaves your machine
