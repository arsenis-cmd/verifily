#!/usr/bin/env bash
# smoke_hosted.sh — Verify Verifily server works in deployed-like mode.
#
# Starts server, hits core endpoints, runs a pipeline, stops server.
# Runtime target: <6 seconds. No network, no GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
PORT=8098
BASE_URL="http://127.0.0.1:${PORT}"
WORK_DIR="/tmp/verifily_smoke_hosted"

cd "$REPO_ROOT"

echo "━━━ Hosted smoke test ━━━"
echo ""

# ── Prep: ingest demo data ──────────────────────────────────────
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

$CLI ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --schema sft \
  --map question:subject \
  --map answer:resolution \
  --map context:body \
  --tag source:smoke \
  --out "$WORK_DIR/datasets/train_artifact" >/dev/null 2>&1

TRAIN_DATA="$WORK_DIR/datasets/train_artifact/dataset.jsonl"

cat > "$WORK_DIR/pipeline_clean.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_clean
train_data: ${TRAIN_DATA}
eval_data: ${DRILL_DIR}/raw/eval_clean.jsonl
baseline_run: ${DRILL_DIR}/runs/run_clean
ship_if:
  min_f1: 0.65
  min_exact_match: 0.50
  max_f1_regression: 0.03
  max_pii_hits: 10
EOF

# ── Start server in background ──────────────────────────────────
$CLI serve --port $PORT --allow-nonlocal >"$WORK_DIR/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server ready (max 5s)
for i in $(seq 1 50); do
    if curl -s "$BASE_URL/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

PASS=0
FAIL=0

check() {
    local label="$1"
    local ok="$2"
    if [ "$ok" = "true" ]; then
        echo "  [PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. GET /health ──────────────────────────────────────────────
HEALTH=$(curl -s "$BASE_URL/health")
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
check "GET /health status=ok" "$([ "$STATUS" = "ok" ] && echo true || echo false)"

# ── 2. GET /ready ───────────────────────────────────────────────
READY=$(curl -s "$BASE_URL/ready")
RSTATUS=$(echo "$READY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
check "GET /ready status=ready" "$([ "$RSTATUS" = "ready" ] && echo true || echo false)"

# ── 3. GET /metrics ─────────────────────────────────────────────
METRICS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/metrics")
check "GET /metrics returns 200" "$([ "$METRICS_CODE" = "200" ] && echo true || echo false)"

# ── 4. POST /v1/report ──────────────────────────────────────────
REPORT_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/report" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_path\": \"$TRAIN_DATA\", \"schema\": \"sft\"}")
check "POST /v1/report returns 200" "$([ "$REPORT_CODE" = "200" ] && echo true || echo false)"

# ── 5. POST /v1/pipeline (clean → SHIP) ─────────────────────────
PIPELINE=$(curl -s -X POST "$BASE_URL/v1/pipeline" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\": \"$WORK_DIR/pipeline_clean.yaml\", \"plan\": true, \"ci\": true}" 2>/dev/null)
DECISION=$(echo "$PIPELINE" | python3 -c "import sys,json; print(json.load(sys.stdin)['decision']['recommendation'])" 2>/dev/null)
check "POST /v1/pipeline decision=SHIP" "$([ "$DECISION" = "SHIP" ] && echo true || echo false)"

# ── Summary ─────────────────────────────────────────────────────
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Hosted smoke test — ALL PASSED ($PASS/$PASS)              ║"
    echo "╚══════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Hosted smoke test — FAILED ($FAIL failures)            ║"
    echo "╚══════════════════════════════════════════════════╝"
    exit 1
fi
