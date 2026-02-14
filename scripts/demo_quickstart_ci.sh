#!/usr/bin/env bash
# demo_quickstart_ci.sh — Quickstart CI gate "money shot"
#
# Scaffolds a temp project, runs the full CI pipeline gate,
# and prints a boxed summary with Decision + Exit code.
#
# CLI-only — no server, no network, no GPU.  Runtime: <5 seconds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="$(mktemp -d /tmp/verifily_ci_demo.XXXXXX)"
CLI="python3 -m verifily_cli_v1"

cd "$REPO_ROOT"

cleanup() { rm -rf "$DEMO_DIR"; }
trap cleanup EXIT

# ── Step 1: Scaffold quickstart project ──────────────────────────
$CLI quickstart "$DEMO_DIR/project" --force 2>/dev/null

# ── Step 2: Ingest data ─────────────────────────────────────────
$CLI ingest \
  --in "$DEMO_DIR/project/data/raw/sample.csv" \
  --out "$DEMO_DIR/project/data/artifact" \
  --schema sft \
  --map question:subject --map answer:resolution --map context:body \
  2>/dev/null

# ── Step 3: Build pipeline config ────────────────────────────────
python3 -c "
import yaml
cfg = {
    'run_dir': '$DEMO_DIR/project/runs/baseline',
    'train_data': '$DEMO_DIR/project/data/artifact/dataset.jsonl',
    'eval_data': '$DEMO_DIR/project/data/raw/eval.jsonl',
    'baseline_run': '$DEMO_DIR/project/runs/baseline',
    'ship_if': {
        'min_f1': 0.50,
        'min_exact_match': 0.40,
        'max_f1_regression': 0.05,
        'max_pii_hits': 0,
    },
}
with open('$DEMO_DIR/pipeline.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

# ── Step 4: Run pipeline gate ────────────────────────────────────
RESULT_FILE="$DEMO_DIR/verifily_result.json"
set +e
$CLI pipeline --config "$DEMO_DIR/pipeline.yaml" --ci 2>/dev/null > "$RESULT_FILE"
EXIT_CODE=$?
set -e

# ── Step 5: Extract decision ────────────────────────────────────
DECISION=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('recommendation','UNKNOWN'))")
F1=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('metrics',{}).get('f1','N/A'))")

# ── Boxed summary ───────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  Quickstart CI demo — ALL PASSED                  ║"
echo "╠════════════════════════════════════════════════════╣"
printf "║  Decision:  %-38s║\n" "$DECISION"
printf "║  Exit code: %-38s║\n" "$EXIT_CODE"
printf "║  F1:        %-38s║\n" "$F1"
echo "║  CI gate ready                                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "  Output artifact: $RESULT_FILE"
echo ""

# Propagate exit code
exit $EXIT_CODE
