#!/usr/bin/env bash
# demo_customer_drill_leaked.sh — Customer drill: leaked path (expect DONT_SHIP)
#
# Same messy CSV data but eval set contains exact leaks from training.
# Expected: ingest → contract PASS → contamination FAIL → DONT_SHIP
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
WORK_DIR="/tmp/verifily_customer_drill_leaked"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — Customer Drill (Leaked Path)                   ║"
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

# ── Step 3: Contamination — exact leaks (expect FAIL, exit 1) ───
echo "━━━ Step 3: Contamination — Exact Leaks (expect FAIL) ━━━"
echo ""
set +e
$CLI contamination \
  --train "$WORK_DIR/datasets/customer_train_artifact/dataset.jsonl" \
  --eval "$DRILL_DIR/raw/eval_leaked_exact.jsonl"
EXIT_CONTAM=$?
set -e
echo "  → Exit code: $EXIT_CONTAM (expected: 1)"
if [ "$EXIT_CONTAM" -ne 1 ]; then
    echo "  ✗ UNEXPECTED: exact leak eval should FAIL (exit 1)"
    exit 1
fi
echo "  ✓ Contamination FAIL (exact leaks detected)"
echo ""

# ── Step 4: Pipeline CI gate (expect DONT_SHIP, exit 1) ─────────
echo "━━━ Step 4: Pipeline CI Gate (expect DONT_SHIP) ━━━"
echo ""

# Generate pipeline config with absolute paths
cat > "$WORK_DIR/pipeline.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_leaked
train_data: ${WORK_DIR}/datasets/customer_train_artifact/dataset.jsonl
eval_data: ${DRILL_DIR}/raw/eval_leaked_exact.jsonl
baseline_run: ${DRILL_DIR}/runs/run_leaked
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
echo "  → Exit code: $EXIT_PIPELINE (expected: 1 = DONT_SHIP)"
if [ "$EXIT_PIPELINE" -ne 1 ]; then
    echo "  ✗ UNEXPECTED: leaked pipeline should DONT_SHIP (exit 1)"
    exit 1
fi
echo "  ✓ Pipeline DONT_SHIP (contamination blocker)"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RESULT: DONT_SHIP ✓                                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Ingest:         30 in → 28 out (2 dropped)               ║"
echo "║  Contract:       PASS                                      ║"
echo "║  Contamination:  FAIL  (exit 1) — 4 exact leaks           ║"
echo "║  Pipeline:       DONT_SHIP (exit 1)                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
