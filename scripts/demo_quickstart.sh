#!/usr/bin/env bash
# demo_quickstart.sh — Scaffold a quickstart project and run the demo script
#
# Proves that verifily quickstart produces a working project
# that passes the full pipeline gate.
#
# CLI-only — no server, no network, no GPU.  Runtime: <5 seconds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="/tmp/verifily_quickstart_demo"
CLI="python3 -m verifily_cli_v1"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Clean up from previous runs
rm -rf "$DEMO_DIR"

echo "================================================================"
echo "  Verifily — Quickstart Demo"
echo "================================================================"
echo ""

# ── Step 1: Scaffold project ─────────────────────────────────────
echo "--- Step 1: Scaffold quickstart project ---"
$CLI quickstart "$DEMO_DIR" --json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  Project: {data[\"project_dir\"]}')
print(f'  Files: {len(data[\"created_paths\"])}')
"
echo ""

# ── Step 2: Run the generated demo script ────────────────────────
echo "--- Step 2: Run generated demo script ---"
bash "$DEMO_DIR/scripts/run_demo.sh"
echo ""

# ── Step 3: Verify artifacts ─────────────────────────────────────
echo "--- Step 3: Verify artifacts ---"
python3 -c "
from pathlib import Path
from verifily_cli_v1.commands.contract_check import validate_run_contract

project = Path('$DEMO_DIR')
assert (project / 'data' / 'artifact' / 'dataset.jsonl').exists(), 'Missing dataset.jsonl'
assert (project / 'data' / 'artifact' / 'report.json').exists(), 'Missing report.json'
contract = validate_run_contract(str(project / 'runs' / 'baseline'))
assert contract['valid'], 'Contract invalid'
print('  dataset.jsonl: OK')
print('  report.json: OK')
print('  Contract: VALID')
"
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "================================================================"
echo "  Quickstart demo -- ALL PASSED"
echo "================================================================"

# Clean up
rm -rf "$DEMO_DIR"
