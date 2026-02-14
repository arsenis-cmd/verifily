#!/usr/bin/env bash
# demo_api_customer_drill.sh — Demonstrate Verifily local API with customer drill data
#
# 1) Starts verifily serve on 127.0.0.1:8099
# 2) Calls /health
# 3) Calls /v1/pipeline (clean → SHIP, leaked → DONT_SHIP)
# 4) Stops the server
#
# Runtime: <5 seconds, no network, no GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
PORT=8099
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="/tmp/verifily_api_demo"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — API Customer Drill Demo                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Prep: ingest data + write pipeline configs ───────────────────
echo "━━━ Prep: Ingest customer data ━━━"
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

# Write clean pipeline config
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

# Write leaked pipeline config
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

# ── Start server in background ───────────────────────────────────
echo "━━━ Starting API server on 127.0.0.1:${PORT} ━━━"
$CLI serve --port $PORT 2>"$WORK_DIR/server.log" &
SERVER_PID=$!

# Ensure server is stopped on exit
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server to be ready (max 5 seconds)
for i in $(seq 1 50); do
    if curl -s "$BASE_URL/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
echo "  Server PID: $SERVER_PID"
echo ""

# ── Step 1: Health check ─────────────────────────────────────────
echo "━━━ Step 1: GET /health ━━━"
HEALTH=$(curl -s "$BASE_URL/health")
echo "  $HEALTH"
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
echo "  Status: $STATUS"
if [ "$STATUS" != "ok" ]; then
    echo "  ✗ Health check failed"
    exit 1
fi
echo "  ✓ Health OK"
echo ""

# ── Step 2: Clean pipeline (expect SHIP) ─────────────────────────
echo "━━━ Step 2: POST /v1/pipeline (clean → expect SHIP) ━━━"
CLEAN_RESP=$(curl -s -X POST "$BASE_URL/v1/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\": \"$WORK_DIR/pipeline_clean.yaml\", \"plan\": true, \"ci\": true}")

CLEAN_REC=$(echo "$CLEAN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['decision']['recommendation'])")
CLEAN_EXIT=$(echo "$CLEAN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['exit_code'])")
echo "  Decision: $CLEAN_REC"
echo "  Exit code: $CLEAN_EXIT"
if [ "$CLEAN_EXIT" != "0" ]; then
    echo "  ✗ Expected exit_code 0 (SHIP)"
    exit 1
fi
echo "  ✓ SHIP"
echo ""

# ── Step 3: Leaked pipeline (expect DONT_SHIP) ──────────────────
echo "━━━ Step 3: POST /v1/pipeline (leaked → expect DONT_SHIP) ━━━"
LEAKED_RESP=$(curl -s -X POST "$BASE_URL/v1/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\": \"$WORK_DIR/pipeline_leaked.yaml\", \"plan\": true, \"ci\": true}")

LEAKED_REC=$(echo "$LEAKED_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['decision']['recommendation'])")
LEAKED_EXIT=$(echo "$LEAKED_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['exit_code'])")
echo "  Decision: $LEAKED_REC"
echo "  Exit code: $LEAKED_EXIT"
if [ "$LEAKED_EXIT" != "1" ]; then
    echo "  ✗ Expected exit_code 1 (DONT_SHIP)"
    exit 1
fi
echo "  ✓ DONT_SHIP"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  API Customer Drill — ALL PASSED ✓                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  API CLEAN:  SHIP       exit 0                             ║"
echo "║  API LEAKED: DONT_SHIP  exit 1                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
