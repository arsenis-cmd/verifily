#!/usr/bin/env bash
# demo_customer_drill_clean.sh — Customer drill: clean path (expect SHIP)
#
# Simulates a customer bringing messy CSV data through the full Verifily pipeline.
# Expected: ingest → contract PASS → contamination PASS → SHIP
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
WORK_DIR="/tmp/verifily_customer_drill_clean"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — Customer Drill (Clean Path)                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Ingest messy CSV ─────────────────────────────────────
echo "━━━ Step 1: Ingest support_tickets.csv ━━━"
echo ""
rm -rf "$WORK_DIR"
$CLI ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --schema sft \
  --map question:subject \
  --map answer:resolution \
  --map context:body \
  --tag source:customer_drill \
  --out "$WORK_DIR/datasets/customer_train_artifact"
echo ""

# ── Step 2: Contract check (dataset schema) ──────────────────────
echo "━━━ Step 2: Contract Check (dataset schema) ━━━"
echo ""
$CLI contract-check \
  --dataset "$WORK_DIR/datasets/customer_train_artifact/dataset.jsonl" \
  --schema sft
EXIT_CONTRACT=$?
echo "  → Exit code: $EXIT_CONTRACT (expected: 0)"
if [ "$EXIT_CONTRACT" -ne 0 ]; then
    echo "  ✗ UNEXPECTED: contract check should PASS"
    exit 1
fi
echo "  ✓ Contract PASS"
echo ""

# ── Step 3: Contamination — clean eval (expect PASS, exit 0) ────
echo "━━━ Step 3: Contamination — Clean Eval (expect PASS) ━━━"
echo ""
set +e
$CLI contamination \
  --train "$WORK_DIR/datasets/customer_train_artifact/dataset.jsonl" \
  --eval "$DRILL_DIR/raw/eval_clean.jsonl"
EXIT_CONTAM=$?
set -e
echo "  → Exit code: $EXIT_CONTAM (expected: 0)"
if [ "$EXIT_CONTAM" -ne 0 ]; then
    echo "  ✗ UNEXPECTED: clean eval should PASS (exit 0)"
    exit 1
fi
echo "  ✓ Contamination PASS"
echo ""

# ── Step 4: Pipeline CI gate (expect SHIP, exit 0) ──────────────
echo "━━━ Step 4: Pipeline CI Gate (expect SHIP) ━━━"
echo ""

# Generate pipeline config with absolute paths
cat > "$WORK_DIR/pipeline.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_clean
train_data: ${WORK_DIR}/datasets/customer_train_artifact/dataset.jsonl
eval_data: ${DRILL_DIR}/raw/eval_clean.jsonl
baseline_run: ${DRILL_DIR}/runs/run_clean
ship_if:
  min_f1: 0.65
  min_exact_match: 0.50
  max_f1_regression: 0.03
  max_pii_hits: 0
EOF

set +e
$CLI pipeline --config "$WORK_DIR/pipeline.yaml" --ci
EXIT_PIPELINE=$?
set -e
echo ""
echo "  → Exit code: $EXIT_PIPELINE (expected: 0 = SHIP)"
if [ "$EXIT_PIPELINE" -ne 0 ]; then
    echo "  ✗ UNEXPECTED: clean pipeline should SHIP (exit 0)"
    exit 1
fi
echo "  ✓ Pipeline SHIP"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RESULT: SHIP ✓                                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Ingest:         30 in → 28 out (2 dropped)               ║"
echo "║  Contract:       PASS                                      ║"
echo "║  Contamination:  PASS  (exit 0)                            ║"
echo "║  Pipeline:       SHIP  (exit 0)                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
