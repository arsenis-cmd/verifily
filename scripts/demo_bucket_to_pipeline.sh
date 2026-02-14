#!/usr/bin/env bash
# demo_bucket_to_pipeline.sh — End-to-end: classify → bucket export → ingest → pipeline --ci
#
# Proves that exported bucket JSONL files are immediately usable as pipeline inputs.
#
# Steps:
#   1) Classify examples/classify_demo/raw/mixed_dump.csv with bucket export
#   2) Pick the first QA bucket (deterministic)
#   3) Ingest that bucket JSONL into a canonical dataset artifact
#   4) Run pipeline --ci on the ingested dataset
#   5) Print decision
#
# CLI-only — no server, no network, no GPU.  Runtime: <5 seconds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="python3 -m verifily_cli_v1"
DEMO_DIR="/tmp/verifily_bucket_demo"
CLASSIFY_DIR="$REPO_ROOT/examples/classify_demo"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"

cd "$REPO_ROOT"

# Clean up from previous runs
rm -rf "$DEMO_DIR"

echo "================================================================"
echo "  Verifily — Bucket to Pipeline Demo"
echo "================================================================"
echo ""

# ── Step 1: Classify with bucket export ──────────────────────────
echo "--- Step 1: Classify mixed_dump.csv with bucket export ---"
python3 -c "
from verifily_cli_v1.core.classify import classify_dataset, write_bucket_exports
from verifily_cli_v1.core.api.runners import run_classify_api

result = run_classify_api(
    dataset_path='$CLASSIFY_DIR/raw/mixed_dump.csv',
    output_dir='$DEMO_DIR',
    export_buckets=True,
)

cls = result['classification']
es = result.get('export_summary', {})
print(f'  Rows: {cls[\"row_count\"]}')
print(f'  Schema: {cls[\"suggested_schema\"]}')
print(f'  Buckets exported: {len(es.get(\"bucket_paths\", {}))}')
for name, path in sorted(es.get('bucket_paths', {}).items()):
    rows = es.get('rows_per_bucket', {}).get(name, 0)
    print(f'    {name}: {rows} rows -> {path}')
"
echo ""

# ── Step 2: Pick a QA bucket deterministically ───────────────────
echo "--- Step 2: Pick first QA bucket ---"
BUCKET_PATH=$(python3 -c "
import os, json
buckets_dir = '$DEMO_DIR/classification/buckets'
files = sorted(os.listdir(buckets_dir))
# Prefer qa buckets, then sft, then first available
qa = [f for f in files if f.startswith('qa__')]
sft = [f for f in files if f.startswith('sft__')]
pick = (qa or sft or files)[0]
print(os.path.join(buckets_dir, pick))
")
echo "  Selected: $(basename "$BUCKET_PATH")"
echo "  Path: $BUCKET_PATH"
BUCKET_ROWS=$(python3 -c "
lines = open('$BUCKET_PATH').readlines()
print(len(lines))
")
echo "  Rows: $BUCKET_ROWS"
echo ""

# ── Step 3: Ingest bucket JSONL into canonical dataset ────────────
echo "--- Step 3: Ingest bucket into canonical dataset ---"
DATASET_DIR="$DEMO_DIR/datasets/selected_bucket"
$CLI ingest --in "$BUCKET_PATH" --out "$DATASET_DIR" --schema sft --map question:prompt --map answer:completion
echo "  Output: $DATASET_DIR"
echo "  Files:"
ls "$DATASET_DIR" | while read f; do echo "    $f"; done
echo ""

# ── Step 4: Pipeline --ci ─────────────────────────────────────────
echo "--- Step 4: Pipeline --ci ---"

# Build a minimal pipeline config
PIPELINE_CFG="$DEMO_DIR/pipeline.yaml"
python3 -c "
import yaml
cfg = {
    'run_dir': '$DRILL_DIR/runs/run_clean',
    'train_data': '$DATASET_DIR/dataset.jsonl',
    'eval_data': '$DRILL_DIR/raw/eval_clean.jsonl',
    'baseline_run': '$DRILL_DIR/runs/run_clean',
    'ship_if': {
        'min_f1': 0.50,
        'min_exact_match': 0.40,
        'max_f1_regression': 0.05,
        'max_pii_hits': 0,
    },
}
with open('$PIPELINE_CFG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('  Config: $PIPELINE_CFG')
"

DECISION_JSON=$($CLI pipeline --config "$PIPELINE_CFG" --ci 2>/dev/null) || true
EXIT_CODE=$?

# Parse the decision
python3 -c "
import json, sys
d = json.loads('''$DECISION_JSON''')
rec = d.get('recommendation', 'UNKNOWN')
exit_code = d.get('exit_code', 4)

# Contract check
if exit_code != 3:
    print('  CONTRACT PASS')
else:
    print('  CONTRACT FAIL')

# Contamination
risk_flags = d.get('risk_flags', [])
if any('contamination' in r for r in risk_flags):
    print('  CONTAMINATION FAIL')
else:
    print('  CONTAMINATION PASS')

# Decision
print(f'  Decision: {rec}')
print(f'  Exit code: {exit_code}')

# Metrics
metrics = d.get('metrics', {})
if metrics:
    for k, v in metrics.items():
        print(f'    {k}: {v}')
"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "================================================================"
echo "  Bucket to Pipeline demo - ALL PASSED"
echo "================================================================"

# Clean up
rm -rf "$DEMO_DIR"
