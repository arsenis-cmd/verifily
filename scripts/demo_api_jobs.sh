#!/usr/bin/env bash
# demo_api_jobs.sh — Demonstrate Verifily async jobs API
#
# 1) Starts verifily serve on 127.0.0.1:8098
# 2) Submits clean pipeline job -> waits -> expects SHIP
# 3) Submits leaked pipeline job -> waits -> expects DONT_SHIP
# 4) Lists jobs
# 5) Stops the server
#
# Runtime: <10 seconds, no network, no GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
PORT=8098
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="/tmp/verifily_jobs_demo"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — Async Jobs API Demo                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Prep: ingest data + write pipeline configs ───────────────────
echo "--- Prep: Ingest customer data ---"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

$CLI ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --schema sft \
  --map question:subject \
  --map answer:resolution \
  --map context:body \
  --tag source:customer_drill \
  --out "$WORK_DIR/datasets/customer_train_artifact" 2>&1

TRAIN_DATA="$WORK_DIR/datasets/customer_train_artifact/dataset.jsonl"

cat > "$WORK_DIR/pipeline_clean.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_clean
train_data: ${TRAIN_DATA}
eval_data: ${DRILL_DIR}/raw/eval_clean.jsonl
baseline_run: ${DRILL_DIR}/runs/run_clean
ship_if:
  min_f1: 0.65
  min_exact_match: 0.50
  max_f1_regression: 0.03
  max_pii_hits: 0
EOF

cat > "$WORK_DIR/pipeline_leaked.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_leaked
train_data: ${TRAIN_DATA}
eval_data: ${DRILL_DIR}/raw/eval_leaked_exact.jsonl
baseline_run: ${DRILL_DIR}/runs/run_leaked
ship_if:
  min_f1: 0.65
  min_exact_match: 0.50
  max_f1_regression: 0.03
  max_pii_hits: 0
EOF
echo ""

# ── Start server ─────────────────────────────────────────────────
echo "--- Starting API server on 127.0.0.1:${PORT} ---"
$CLI serve --port $PORT 2>"$WORK_DIR/server.log" &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
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
    local MAX_WAIT=30
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

# ── Step 1: Submit clean pipeline job → SHIP ─────────────────────
echo "--- Step 1: POST /v1/jobs/pipeline (clean) ---"
SUBMIT_RESP=$(curl -s -X POST "$BASE_URL/v1/jobs/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\": \"$WORK_DIR/pipeline_clean.yaml\", \"plan\": true, \"ci\": true}")

CLEAN_JOB_ID=$(echo "$SUBMIT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
CLEAN_STATUS=$(echo "$SUBMIT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
echo "  Job ID: $CLEAN_JOB_ID"
echo "  Status: $CLEAN_STATUS"

echo "  Waiting for job..."
FINAL=$(wait_for_job "$CLEAN_JOB_ID")
echo "  Final status: $FINAL"

RESULT=$(curl -s "$BASE_URL/v1/jobs/$CLEAN_JOB_ID/result")
CLEAN_REC=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['decision']['recommendation'])")
echo "  Decision: $CLEAN_REC"
if [ "$CLEAN_REC" != "SHIP" ]; then
    echo "  FAIL: Expected SHIP"
    exit 1
fi
echo "  PASS: SHIP"
echo ""

# ── Step 2: Submit leaked pipeline job → DONT_SHIP ──────────────
echo "--- Step 2: POST /v1/jobs/pipeline (leaked) ---"
SUBMIT_RESP=$(curl -s -X POST "$BASE_URL/v1/jobs/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\": \"$WORK_DIR/pipeline_leaked.yaml\", \"plan\": true, \"ci\": true}")

LEAKED_JOB_ID=$(echo "$SUBMIT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "  Job ID: $LEAKED_JOB_ID"

echo "  Waiting for job..."
FINAL=$(wait_for_job "$LEAKED_JOB_ID")
echo "  Final status: $FINAL"

RESULT=$(curl -s "$BASE_URL/v1/jobs/$LEAKED_JOB_ID/result")
LEAKED_REC=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['decision']['recommendation'])")
echo "  Decision: $LEAKED_REC"
if [ "$LEAKED_REC" != "DONT_SHIP" ]; then
    echo "  FAIL: Expected DONT_SHIP"
    exit 1
fi
echo "  PASS: DONT_SHIP"
echo ""

# ── Step 3: List all jobs ────────────────────────────────────────
echo "--- Step 3: GET /v1/jobs ---"
JOBS=$(curl -s "$BASE_URL/v1/jobs")
JOB_COUNT=$(echo "$JOBS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['jobs']))")
echo "  Total jobs: $JOB_COUNT"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Async Jobs Demo — ALL PASSED                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  JOB CLEAN:  QUEUED -> SUCCEEDED -> SHIP       (exit 0)   ║"
echo "║  JOB LEAKED: QUEUED -> SUCCEEDED -> DONT_SHIP  (exit 1)   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
