#!/usr/bin/env bash
# demo_customer_drill_ci.sh — Customer drill: CI-style (both clean + leaked)
#
# Runs both scenarios plus near-duplicate WARN check.  Shows all exit codes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"
CLI="python3 -m verifily_cli_v1"
WORK_CLEAN="/tmp/verifily_customer_drill_clean"
WORK_LEAKED="/tmp/verifily_customer_drill_leaked"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — Customer Drill CI Suite                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Ingest (both paths use same data) ────────────────────────────
echo "━━━ Ingest support_tickets.csv (clean workspace) ━━━"
rm -rf "$WORK_CLEAN"
$CLI ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --schema sft \
  --map question:subject \
  --map answer:resolution \
  --map context:body \
  --tag source:customer_drill \
  --out "$WORK_CLEAN/datasets/customer_train_artifact"
echo ""

echo "━━━ Ingest support_tickets.csv (leaked workspace) ━━━"
rm -rf "$WORK_LEAKED"
$CLI ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --schema sft \
  --map question:subject \
  --map answer:resolution \
  --map context:body \
  --tag source:customer_drill \
  --out "$WORK_LEAKED/datasets/customer_train_artifact"
echo ""

TRAIN_CLEAN="$WORK_CLEAN/datasets/customer_train_artifact/dataset.jsonl"
TRAIN_LEAKED="$WORK_LEAKED/datasets/customer_train_artifact/dataset.jsonl"

# ── Scenario 1: Clean pipeline (expect SHIP = 0) ────────────────
echo "━━━ Scenario 1: Clean Pipeline ━━━"
echo ""

cat > "$WORK_CLEAN/pipeline.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_clean
train_data: ${TRAIN_CLEAN}
eval_data: ${DRILL_DIR}/raw/eval_clean.jsonl
baseline_run: ${DRILL_DIR}/runs/run_clean
ship_if:
  min_f1: 0.65
  min_exact_match: 0.50
  max_f1_regression: 0.03
  max_pii_hits: 0
EOF

set +e
DECISION=$($CLI pipeline --config "$WORK_CLEAN/pipeline.yaml" --ci 2>/dev/null)
EXIT_CLEAN=$?
set -e
echo "  Decision: $(echo "$DECISION" | python3 -c "import sys,json; print(json.load(sys.stdin)['recommendation'])")"
echo "  Exit code: $EXIT_CLEAN (expected: 0 = SHIP)"
if [ "$EXIT_CLEAN" -ne 0 ]; then
    echo "  ✗ FAIL"
    exit 1
fi
echo "  ✓ PASS"
echo ""

# ── Scenario 2: Leaked pipeline (expect DONT_SHIP = 1) ──────────
echo "━━━ Scenario 2: Leaked Pipeline (exact contamination) ━━━"
echo ""

cat > "$WORK_LEAKED/pipeline.yaml" << EOF
run_dir: ${DRILL_DIR}/runs/run_leaked
train_data: ${TRAIN_LEAKED}
eval_data: ${DRILL_DIR}/raw/eval_leaked_exact.jsonl
baseline_run: ${DRILL_DIR}/runs/run_leaked
ship_if:
  min_f1: 0.65
  min_exact_match: 0.50
  max_f1_regression: 0.03
  max_pii_hits: 0
EOF

set +e
DECISION=$($CLI pipeline --config "$WORK_LEAKED/pipeline.yaml" --ci 2>/dev/null)
EXIT_LEAKED=$?
set -e
echo "  Decision: $(echo "$DECISION" | python3 -c "import sys,json; print(json.load(sys.stdin)['recommendation'])")"
echo "  Exit code: $EXIT_LEAKED (expected: 1 = DONT_SHIP)"
if [ "$EXIT_LEAKED" -ne 1 ]; then
    echo "  ✗ FAIL"
    exit 1
fi
echo "  ✓ PASS"
echo ""

# ── Scenario 3: Near-duplicate contamination (expect WARN = 2) ──
echo "━━━ Scenario 3: Contamination — Near Duplicates (expect WARN) ━━━"
echo ""
set +e
$CLI contamination \
  --train "$TRAIN_CLEAN" \
  --eval "$DRILL_DIR/raw/eval_leaked_near.jsonl" 2>/dev/null
EXIT_NEAR=$?
set -e
echo "  Exit code: $EXIT_NEAR (expected: 2 = WARN)"
if [ "$EXIT_NEAR" -ne 2 ]; then
    echo "  ✗ FAIL"
    exit 1
fi
echo "  ✓ PASS"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Customer Drill CI Suite — ALL PASSED ✓                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Clean pipeline:       SHIP       (exit 0)                 ║"
echo "║  Leaked pipeline:      DONT_SHIP  (exit 1)                 ║"
echo "║  Near-dup contamination: WARN     (exit 2)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
