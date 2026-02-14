#!/usr/bin/env bash
# demo_regression.sh — Demonstrate metric regression detection across runs
#
# Shows verifily history tracking F1 across 4 runs and detecting the regression.
# Returns exit 2 (WARN) when a regression is found.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_DIR="$REPO_ROOT/examples/real_conditions"
CLI="python3 -m verifily_cli_v1"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — Regression Detection Demo                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Run history:"
echo "  run_01_good       → baseline  (f1 = 0.7139)"
echo "  run_02_good       → stable    (f1 = 0.7201, Δ = +0.0062)"
echo "  run_03_regression → regressed (f1 = 0.6650, Δ = -0.0551)"
echo "  run_04_recovery   → recovered (f1 = 0.7050, Δ = +0.0400)"
echo ""
echo "Regression threshold: 0.02 (any drop > 0.02 is flagged)"
echo ""

# ── Run history ─────────────────────────────────────────────────
echo "━━━ verifily history ━━━"
echo ""

set +e
$CLI history --runs "$RC_DIR/runs" --metric f1 --threshold 0.02
EXIT_CODE=$?
set -e

echo "  Exit code: $EXIT_CODE"
echo ""

if [ "$EXIT_CODE" -eq 2 ]; then
    echo "  ✓ Regression correctly detected on run_03_regression"
    echo "    The f1 drop of 0.0551 exceeds the 0.02 threshold."
elif [ "$EXIT_CODE" -eq 0 ]; then
    echo "  ✗ UNEXPECTED: should have detected regression (exit 2)"
else
    echo "  ✗ UNEXPECTED: exit code $EXIT_CODE"
fi

echo ""

# ── Also show individual run comparison ─────────────────────────
echo "━━━ Side-by-side comparison (compare command) ━━━"
echo ""
$CLI compare \
    --runs "$RC_DIR/runs/run_01_good,$RC_DIR/runs/run_02_good,$RC_DIR/runs/run_03_regression,$RC_DIR/runs/run_04_recovery" \
    --metric f1
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Regression demo complete ✓                                ║"
echo "║  Exit code: $EXIT_CODE (2 = regression detected)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

exit $EXIT_CODE
