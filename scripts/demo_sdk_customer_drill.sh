#!/usr/bin/env bash
# demo_sdk_customer_drill.sh — Run Verifily SDK demo against a local server.
#
# 1) Starts verifily serve on 127.0.0.1:8099
# 2) Ingests customer drill data
# 3) Runs SDK Python demo (health, pipeline clean, pipeline leaked)
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
WORK_DIR="/tmp/verifily_sdk_demo"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — SDK Customer Drill Demo                        ║"
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

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server ready (max 5 seconds)
for i in $(seq 1 50); do
    if curl -s "$BASE_URL/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
echo "  Server PID: $SERVER_PID"
echo ""

# ── Run SDK demo ─────────────────────────────────────────────────
echo "━━━ Running SDK demo ━━━"
python3 "$SCRIPT_DIR/demo_sdk_customer_drill.py" \
  --base-url "$BASE_URL" \
  --clean-config "$WORK_DIR/pipeline_clean.yaml" \
  --leaked-config "$WORK_DIR/pipeline_leaked.yaml"
