#!/usr/bin/env bash
# Verifily pre-commit hook — runs pipeline gate and blocks on FAIL or CONTRACT_FAIL.
#
# Exit behavior:
#   SHIP (0)           → allow commit
#   DONT_SHIP (1)      → block commit
#   INVESTIGATE (2)    → allow commit (investigation doesn't block)
#   CONTRACT_FAIL (3)  → block commit
#   TOOL_ERROR (4)     → allow commit (don't block on tool issues)
#
# Set VERIFILY_CONFIG to override the config path (default: verifily.yaml).

set -euo pipefail

CONFIG="${VERIFILY_CONFIG:-verifily.yaml}"

if [ ! -f "$CONFIG" ]; then
  # No config found — skip silently.
  exit 0
fi

set +e
verifily pipeline --ci --config "$CONFIG" > /dev/null 2>&1
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 1 ] || [ "$EXIT_CODE" -eq 3 ]; then
  echo "verifily: pipeline gate FAILED (exit $EXIT_CODE) — commit blocked" >&2
  exit 1
fi

exit 0
