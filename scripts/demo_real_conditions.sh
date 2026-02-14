#!/usr/bin/env bash
# demo_real_conditions.sh — End-to-end demo of Verifily under real conditions
#
# Exercises: report, contamination (clean/exact/near), pipeline (CI gate)
# Expected: deterministic, no network, no GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_DIR="$REPO_ROOT/examples/real_conditions"
CLI="python3 -m verifily_cli_v1"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — Real Conditions Demo                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Dataset report ──────────────────────────────────────
echo "━━━ Step 1: Dataset Report (train.jsonl) ━━━"
echo ""
$CLI report --dataset "$RC_DIR/data/train.jsonl"
echo ""

# ── Step 2: Contamination — clean eval (expect PASS, exit 0) ───
echo "━━━ Step 2: Contamination Check — Clean Eval (expect PASS) ━━━"
echo ""
set +e
$CLI contamination --train "$RC_DIR/data/train.jsonl" --eval "$RC_DIR/data/eval_clean.jsonl"
EXIT_CLEAN=$?
set -e
echo "  → Exit code: $EXIT_CLEAN (expected: 0)"
if [ "$EXIT_CLEAN" -ne 0 ]; then
    echo "  ✗ UNEXPECTED: clean eval should exit 0"
    exit 1
fi
echo "  ✓ Clean eval passed as expected"
echo ""

# ── Step 3: Contamination — exact leaks (expect FAIL, exit 1) ──
echo "━━━ Step 3: Contamination Check — Exact Leaks (expect FAIL) ━━━"
echo ""
set +e
$CLI contamination --train "$RC_DIR/data/train.jsonl" --eval "$RC_DIR/data/eval_leaked_exact.jsonl"
EXIT_EXACT=$?
set -e
echo "  → Exit code: $EXIT_EXACT (expected: 1)"
if [ "$EXIT_EXACT" -ne 1 ]; then
    echo "  ✗ UNEXPECTED: exact leak eval should exit 1"
    exit 1
fi
echo "  ✓ Exact leak detected as expected"
echo ""

# ── Step 4: Contamination — near leaks (expect WARN, exit 2) ───
echo "━━━ Step 4: Contamination Check — Near Leaks (expect WARN) ━━━"
echo ""
set +e
$CLI contamination --train "$RC_DIR/data/train.jsonl" --eval "$RC_DIR/data/eval_leaked_near.jsonl"
EXIT_NEAR=$?
set -e
echo "  → Exit code: $EXIT_NEAR (expected: 2)"
if [ "$EXIT_NEAR" -ne 2 ]; then
    echo "  ✗ UNEXPECTED: near leak eval should exit 2"
    exit 1
fi
echo "  ✓ Near-duplicate leak detected as expected"
echo ""

# ── Step 5: Pipeline CI gate — leaked eval (expect DONT_SHIP) ──
echo "━━━ Step 5: Pipeline CI Gate — Leaked Eval (expect DONT_SHIP) ━━━"
echo ""
set +e
$CLI pipeline --config "$RC_DIR/verifily.yaml" --ci
EXIT_PIPELINE=$?
set -e
echo ""
echo "  → Exit code: $EXIT_PIPELINE (expected: 1 = DONT_SHIP)"
if [ "$EXIT_PIPELINE" -ne 1 ]; then
    echo "  ✗ UNEXPECTED: leaked pipeline should exit 1"
    exit 1
fi
echo "  ✓ Pipeline correctly returned DONT_SHIP"
echo ""

# ── Step 6: Pipeline CI gate — clean eval (expect SHIP) ────────
echo "━━━ Step 6: Pipeline CI Gate — Clean Eval (expect SHIP) ━━━"
echo ""
set +e
$CLI pipeline --config "$RC_DIR/verifily_clean.yaml" --ci
EXIT_CLEAN_PIPE=$?
set -e
echo ""
echo "  → Exit code: $EXIT_CLEAN_PIPE (expected: 0 = SHIP)"
if [ "$EXIT_CLEAN_PIPE" -ne 0 ]; then
    echo "  ✗ UNEXPECTED: clean pipeline should exit 0"
    exit 1
fi
echo "  ✓ Pipeline correctly returned SHIP"
echo ""

# ── Summary ─────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RESULT: Real conditions demo complete ✓                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Contamination clean:   PASS  (exit 0)                    ║"
echo "║  Contamination exact:   FAIL  (exit 1)                    ║"
echo "║  Contamination near:    WARN  (exit 2)                    ║"
echo "║  Pipeline (leaked):     DONT_SHIP (exit 1)                ║"
echo "║  Pipeline (clean):      SHIP      (exit 0)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
