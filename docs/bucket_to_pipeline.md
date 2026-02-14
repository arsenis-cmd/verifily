# Bucket to Pipeline — End-to-End Guide

Use CLASSIFY bucket exports as direct inputs to the Verifily pipeline.

## Flow

```
mixed_dump.csv
  → classify (export_buckets=true)
    → buckets/qa__support.jsonl
    → buckets/qa__qa_geography.jsonl
    → ...
  → pick bucket
  → verifily ingest --in bucket.jsonl --out datasets/selected
  → verifily pipeline --config pipeline.yaml --ci
  → SHIP / DONT_SHIP
```

## Quick Start

```bash
bash scripts/demo_bucket_to_pipeline.sh
```

## Step-by-Step

### 1. Classify with bucket export

```python
from verifily_cli_v1.core.api.runners import run_classify_api

result = run_classify_api(
    dataset_path="data.csv",
    output_dir="/tmp/classify_out",
    export_buckets=True,
)

for name, path in result["export_summary"]["bucket_paths"].items():
    rows = result["export_summary"]["rows_per_bucket"][name]
    print(f"{name}: {rows} rows -> {path}")
```

Or via the API:

```bash
curl -s -X POST http://localhost:8080/v1/jobs/classify \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "/path/to/data.csv",
    "output_dir": "/tmp/classify_out",
    "export_buckets": true
  }'
```

### 2. Pick a bucket

Each bucket is a valid JSONL file at `<output_dir>/classification/buckets/<schema>__<category>.jsonl`.

### 3. Ingest the bucket

```bash
verifily ingest \
  --in /tmp/classify_out/classification/buckets/qa__support.jsonl \
  --out /tmp/datasets/qa_support \
  --schema sft \
  --map question:prompt --map answer:completion
```

### 4. Run pipeline

```bash
verifily pipeline --config pipeline.yaml --ci
```

Where `pipeline.yaml` points `train_data` to the ingested dataset:

```yaml
run_dir: runs/my_run
train_data: /tmp/datasets/qa_support/dataset.jsonl
eval_data: data/eval.jsonl
ship_if:
  min_f1: 0.50
  max_pii_hits: 0
```

## Output

```
CONTRACT PASS
CONTAMINATION PASS
Decision: SHIP
Exit code: 0
```
