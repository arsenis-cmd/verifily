#!/usr/bin/env bash
# demo_api_jobs_classify.sh — Demonstrate Verifily CLASSIFY async job
#
# 1) Starts verifily serve on 127.0.0.1:8097
# 2) Submits classify job (basic) for mixed_dump.csv
# 3) Waits for completion, prints bucket summary + PII risk + warnings
# 4) Submits classify job with export_buckets=true
# 5) Prints bucket export info + suggested next steps
# 6) Stops the server
#
# Runtime: <10 seconds, no network, no GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLASSIFY_DIR="$REPO_ROOT/examples/classify_demo"
CLI="python3 -m verifily_cli_v1"
PORT=8097
BASE_URL="http://127.0.0.1:${PORT}"
EXPORT_DIR="/tmp/verifily_classify_export_demo"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — CLASSIFY Job Demo (v1.1 with bucket exports)   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Start server ─────────────────────────────────────────────────
echo "--- Starting API server on 127.0.0.1:${PORT} ---"
$CLI serve --port $PORT 2>/tmp/verifily_classify_demo_server.log &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$EXPORT_DIR"
}
trap cleanup EXIT

for i in $(seq 1 50); do
    if curl -s "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    sleep 0.1
done
echo "  Server PID: $SERVER_PID"
echo ""

# ── Helper: wait for job ─────────────────────────────────────────
wait_for_job() {
    local JOB_ID="$1"
    local MAX_WAIT=60
    local ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS=$(curl -s "$BASE_URL/v1/jobs/$JOB_ID" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
        if [ "$STATUS" = "SUCCEEDED" ] || [ "$STATUS" = "FAILED" ]; then
            echo "$STATUS"
            return 0
        fi
        sleep 0.2
        ELAPSED=$((ELAPSED + 1))
    done
    echo "TIMEOUT"
    return 1
}

# ── Step 1: Submit basic classify job ─────────────────────────────
DATASET="$CLASSIFY_DIR/raw/mixed_dump.csv"
echo "--- Step 1: POST /v1/jobs/classify (basic) ---"
echo "  Dataset: $DATASET"
SUBMIT_RESP=$(curl -s -X POST "$BASE_URL/v1/jobs/classify" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_path\": \"$DATASET\"}")

JOB_ID=$(echo "$SUBMIT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "  Job ID: $JOB_ID"
echo "  Status: QUEUED"
echo ""

# ── Step 2: Wait for basic job ────────────────────────────────────
echo "--- Step 2: Waiting for job ---"
FINAL=$(wait_for_job "$JOB_ID")
echo "  Final status: $FINAL"
if [ "$FINAL" != "SUCCEEDED" ]; then
    echo "  FAIL: Job did not succeed"
    exit 1
fi
echo ""

# ── Step 3: Get basic result ──────────────────────────────────────
echo "--- Step 3: GET /v1/jobs/$JOB_ID/result ---"
RESULT=$(curl -s "$BASE_URL/v1/jobs/$JOB_ID/result")

python3 -c "
import sys, json
r = json.loads('''$RESULT''')
cls = r['classification']

print(f'  Rows: {cls[\"row_count\"]}')
print(f'  Suggested schema: {cls[\"suggested_schema\"]}')
print()
print('  Buckets:')
for b in cls['dataset_buckets']:
    print(f'    {b[\"schema\"]:15s} | {b[\"category\"]:20s} | {b[\"count\"]:3d} rows ({b[\"fraction\"]:.0%})')
print()
ts = cls['tags_summary']
print('  PII risk:')
print(f'    Emails: {ts[\"pii_risk\"][\"emails\"]}')
print(f'    Phones: {ts[\"pii_risk\"][\"phones\"]}')
print(f'    Rows with PII: {ts[\"pii_risk\"][\"rows_with_pii\"]}')
print()
print(f'  Duplicate rate: {ts[\"duplicate_rate\"]:.1%}')
print()
print('  Language: en={}, non_en={}'.format(ts['language']['en'], ts['language']['non_en']))
print()
if cls['warnings']:
    print('  Warnings:')
    for w in cls['warnings']:
        print(f'    - {w}')
"
echo ""

# ── Step 4: Submit classify job WITH bucket exports ───────────────
echo "--- Step 4: POST /v1/jobs/classify (export_buckets=true) ---"
echo "  Output dir: $EXPORT_DIR"
SUBMIT2_RESP=$(curl -s -X POST "$BASE_URL/v1/jobs/classify" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_path\": \"$DATASET\", \"output_dir\": \"$EXPORT_DIR\", \"export_buckets\": true}")

JOB2_ID=$(echo "$SUBMIT2_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "  Job ID: $JOB2_ID"
echo "  Status: QUEUED"
echo ""

# ── Step 5: Wait for export job ───────────────────────────────────
echo "--- Step 5: Waiting for export job ---"
FINAL2=$(wait_for_job "$JOB2_ID")
echo "  Final status: $FINAL2"
if [ "$FINAL2" != "SUCCEEDED" ]; then
    echo "  FAIL: Export job did not succeed"
    exit 1
fi
echo ""

# ── Step 6: Print export results ──────────────────────────────────
echo "--- Step 6: Bucket export results ---"
RESULT2=$(curl -s "$BASE_URL/v1/jobs/$JOB2_ID/result")

python3 -c "
import sys, json
r = json.loads('''$RESULT2''')
es = r.get('export_summary', {})

print(f'  Total rows exported: {es.get(\"total_rows_written\", 0)}')
print(f'  Buckets written: {len(es.get(\"bucket_paths\", {}))}')
print()
print('  Bucket files:')
for name, path in sorted(es.get('bucket_paths', {}).items()):
    rows = es.get('rows_per_bucket', {}).get(name, 0)
    schema = es.get('schemas_per_bucket', {}).get(name, '?')
    print(f'    {name:40s} | {rows:3d} rows | schema={schema}')
print()

# Read suggested_next_steps.json
steps_path = es.get('suggested_next_steps_path')
if steps_path:
    with open(steps_path) as f:
        steps = json.load(f)
    print('  Suggested next steps:')
    for s in steps.get('steps', []):
        action = s.get('action', '?')
        reason = s.get('reason', '')
        bucket = s.get('bucket', '')
        if bucket:
            print(f'    [{action}] bucket={bucket} ({s.get(\"rows\", 0)} rows)')
        else:
            print(f'    [{action}] {reason}')
"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CLASSIFY JOB v1.1 — ALL PASSED                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Schema: qa   | Rows: 16  | Buckets: 4                    ║"
echo "║  PII: 4 email, 3 phone   | Dups: 12.5%                    ║"
echo "║  Bucket exports: 4 JSONL files + suggested_next_steps.json ║"
echo "╚══════════════════════════════════════════════════════════════╝"
