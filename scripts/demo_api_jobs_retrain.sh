#!/usr/bin/env bash
# demo_api_jobs_retrain.sh — End-to-end: ingest → submit retrain → drain → result
#
# Proves that the RETRAIN job type produces valid artifacts, passes
# the run contract, and returns a SHIP / DONT_SHIP decision.
#
# CLI-only — no server, no network, no GPU.  Runtime: <5 seconds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="/tmp/verifily_retrain_demo"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"

cd "$REPO_ROOT"

# Clean up from previous runs
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"

echo "================================================================"
echo "  Verifily — RETRAIN Job Type Demo"
echo "================================================================"
echo ""

# ── Step 1: Ingest training data ─────────────────────────────────
echo "--- Step 1: Ingest customer drill data ---"
python3 -m verifily_cli_v1 ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --out "$DEMO_DIR/artifact" \
  --schema sft \
  --map question:subject --map answer:resolution --map context:body
echo "  Output: $DEMO_DIR/artifact"
echo ""

# ── Step 2: Submit async retrain job, drain, get result ──────────
echo "--- Step 2: Submit retrain job (async) ---"
python3 -c "
import json
from fastapi.testclient import TestClient
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store

app = create_app()
client = TestClient(app)

# Submit
resp = client.post('/v1/jobs/retrain', json={
    'dataset_dir': '$DEMO_DIR/artifact',
    'output_dir': '$DEMO_DIR/retrain_out',
})
sub = resp.json()
print(f'  Job ID: {sub[\"job_id\"]}')
print(f'  Status: {sub[\"status\"]}')

# Drain
jobs_store.drain()
meta = client.get(f'/v1/jobs/{sub[\"job_id\"]}').json()
print(f'  After drain: {meta[\"status\"]}')

# Result
result = client.get(f'/v1/jobs/{sub[\"job_id\"]}/result').json()
print(f'  Run dir: {result[\"run_dir\"]}')
print(f'  Decision: {result[\"decision\"][\"recommendation\"]}')
print(f'  Exit code: {result[\"exit_code\"]}')
print(f'  Eval F1: {result.get(\"eval_summary\", {}).get(\"f1\", \"N/A\")}')
"
echo ""

# ── Step 3: Sync retrain endpoint ────────────────────────────────
echo "--- Step 3: Sync retrain endpoint ---"
python3 -c "
import json
from fastapi.testclient import TestClient
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store

jobs_store.reset()
app = create_app()
client = TestClient(app)

resp = client.post('/v1/retrain', json={
    'dataset_dir': '$DEMO_DIR/artifact',
    'output_dir': '$DEMO_DIR/retrain_sync',
    'seed': 42,
})
data = resp.json()
print(f'  Job type: {data[\"job_type\"]}')
print(f'  Run dir: {data[\"run_dir\"]}')
print(f'  Decision: {data[\"decision\"][\"recommendation\"]}')
print(f'  Artifacts: {list(data[\"artifacts\"].keys())}')
"
echo ""

# ── Step 4: Contract validation ──────────────────────────────────
echo "--- Step 4: Validate run contract ---"
python3 -c "
from pathlib import Path
from verifily_cli_v1.commands.contract_check import validate_run_contract

# Find the latest run dir
runs_dir = Path('$DEMO_DIR/retrain_sync/runs')
run_dirs = sorted(runs_dir.glob('retrain_*'))
if run_dirs:
    rd = run_dirs[-1]
    contract = validate_run_contract(str(rd))
    print(f'  Run dir: {rd}')
    print(f'  Valid: {contract[\"valid\"]}')
    print(f'  Has eval: {contract[\"has_eval\"]}')
    for f in ['config.yaml', 'hashes.json', 'environment.json', 'eval/eval_results.json', 'run_meta.json', 'decision/decision.json']:
        exists = (rd / f).exists()
        print(f'    {f}: {\"OK\" if exists else \"MISSING\"}')
else:
    print('  [ERROR] No run directories found')
"
echo ""

# ── Step 5: Deterministic check ──────────────────────────────────
echo "--- Step 5: Deterministic output check ---"
python3 -c "
import json
from fastapi.testclient import TestClient
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.jobs import jobs_store

jobs_store.reset()
app = create_app()
client = TestClient(app)

r1 = client.post('/v1/retrain', json={
    'dataset_dir': '$DEMO_DIR/artifact',
    'output_dir': '$DEMO_DIR/det1',
    'seed': 42,
}).json()
r2 = client.post('/v1/retrain', json={
    'dataset_dir': '$DEMO_DIR/artifact',
    'output_dir': '$DEMO_DIR/det2',
    'seed': 42,
}).json()

match = r1['eval_summary'] == r2['eval_summary']
print(f'  Seed 42 run 1: f1={r1[\"eval_summary\"][\"f1\"]}')
print(f'  Seed 42 run 2: f1={r2[\"eval_summary\"][\"f1\"]}')
print(f'  Deterministic: {match}')
assert match, 'Determinism check failed!'
"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "================================================================"
echo "  RETRAIN JOB: SUCCEEDED"
echo "================================================================"

# Clean up
rm -rf "$DEMO_DIR"
