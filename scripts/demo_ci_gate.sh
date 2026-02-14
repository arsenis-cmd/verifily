#!/usr/bin/env bash
# demo_ci_gate.sh — Demonstrate pipeline --ci behavior and exit code semantics
#
# Shows how a CI system would consume Verifily's exit codes.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RC_DIR="$REPO_ROOT/examples/real_conditions"
CLI="python3 -m verifily_cli_v1"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verifily — CI Gate Demo                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Exit code reference ─────────────────────────────────────────
echo "Exit Code Reference:"
echo "  0 = SHIP           Model passes all checks, safe to deploy"
echo "  1 = DONT_SHIP      Hard blocker found (contamination, metric below threshold)"
echo "  2 = INVESTIGATE    Risk flags present but no hard blockers"
echo "  3 = CONTRACT_FAIL  Run directory missing required artifacts"
echo "  4 = TOOL_ERROR     Verifily itself encountered an internal error"
echo ""

# ── Run 1: Clean pipeline (expect SHIP = 0) ────────────────────
echo "━━━ Scenario 1: Clean pipeline (expect SHIP) ━━━"
echo ""

set +e
DECISION_JSON=$($CLI pipeline --config "$RC_DIR/verifily_clean.yaml" --ci 2>/dev/null)
EXIT_CODE=$?
set -e

echo "$DECISION_JSON"
echo ""
echo "  Exit code: $EXIT_CODE"

case $EXIT_CODE in
    0) echo "  Meaning:   SHIP — all checks passed" ;;
    1) echo "  Meaning:   DONT_SHIP — hard blocker found" ;;
    2) echo "  Meaning:   INVESTIGATE — risk flags present" ;;
    3) echo "  Meaning:   CONTRACT_FAIL — missing artifacts" ;;
    4) echo "  Meaning:   TOOL_ERROR — internal error" ;;
    *) echo "  Meaning:   UNKNOWN ($EXIT_CODE)" ;;
esac
echo ""

# ── Run 2: Leaked pipeline (expect DONT_SHIP = 1) ──────────────
echo "━━━ Scenario 2: Leaked pipeline (expect DONT_SHIP) ━━━"
echo ""

set +e
DECISION_JSON=$($CLI pipeline --config "$RC_DIR/verifily.yaml" --ci 2>/dev/null)
EXIT_CODE=$?
set -e

echo "$DECISION_JSON"
echo ""
echo "  Exit code: $EXIT_CODE"

case $EXIT_CODE in
    0) echo "  Meaning:   SHIP — all checks passed" ;;
    1) echo "  Meaning:   DONT_SHIP — hard blocker found" ;;
    2) echo "  Meaning:   INVESTIGATE — risk flags present" ;;
    3) echo "  Meaning:   CONTRACT_FAIL — missing artifacts" ;;
    4) echo "  Meaning:   TOOL_ERROR — internal error" ;;
    *) echo "  Meaning:   UNKNOWN ($EXIT_CODE)" ;;
esac
echo ""

# ── How to use in CI ────────────────────────────────────────────
echo "━━━ CI Integration Examples ━━━"
echo ""
echo "  GitHub Actions:"
echo "    - name: Verifily gate"
echo "      run: python3 -m verifily_cli_v1 pipeline --config verifily.yaml --ci"
echo ""
echo "  GitLab CI:"
echo "    verifily-gate:"
echo "      script: python3 -m verifily_cli_v1 pipeline --config verifily.yaml --ci"
echo "      allow_failure: false"
echo ""
echo "  Generic (capture exit code):"
echo "    python3 -m verifily_cli_v1 pipeline --config verifily.yaml --ci"
echo '    if [ $? -eq 0 ]; then echo "SHIP"; else echo "BLOCKED"; fi'
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CI gate demo complete ✓                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
