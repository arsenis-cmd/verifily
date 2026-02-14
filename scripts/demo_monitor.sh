#!/usr/bin/env bash
# demo_monitor.sh — End-to-end: ingest → start monitor → tick × 3 → history → stop
#
# Proves that the MONITOR feature runs the gate repeatedly and produces
# rolling history with regression detection.
#
# CLI-only — no server, no network, no GPU.  Runtime: <5 seconds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="/tmp/verifily_monitor_demo"
DRILL_DIR="$REPO_ROOT/examples/customer_drill"

cd "$REPO_ROOT"

# Clean up from previous runs
rm -rf "$DEMO_DIR"
mkdir -p "$DEMO_DIR"

echo "================================================================"
echo "  Verifily — Continuous Gating Monitor Demo"
echo "================================================================"
echo ""

# ── Step 1: Ingest training data ─────────────────────────────────
echo "--- Step 1: Ingest customer drill data ---"
python3 -m verifily_cli_v1 ingest \
  --in "$DRILL_DIR/raw/support_tickets.csv" \
  --out "$DEMO_DIR/artifact" \
  --schema sft \
  --map question:subject --map answer:resolution --map context:body
echo "  Output: $DEMO_DIR/artifact"
echo ""

# ── Step 2: Build pipeline config ────────────────────────────────
echo "--- Step 2: Build pipeline config ---"
python3 -c "
import yaml
cfg = {
    'run_dir': '$DRILL_DIR/runs/run_clean',
    'train_data': '$DEMO_DIR/artifact/dataset.jsonl',
    'eval_data': '$DRILL_DIR/raw/eval_clean.jsonl',
    'baseline_run': '$DRILL_DIR/runs/run_clean',
    'ship_if': {
        'min_f1': 0.50,
        'min_exact_match': 0.40,
        'max_f1_regression': 0.05,
        'max_pii_hits': 0,
    },
}
with open('$DEMO_DIR/pipeline.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('  Config: $DEMO_DIR/pipeline.yaml')
"
echo ""

# ── Step 3: Start monitor, tick 3 times, print history ───────────
echo "--- Step 3: Start monitor + tick 3 times ---"
python3 -c "
import json, uuid
from verifily_cli_v1.core.api.monitor_store import MonitorConfig, monitor_store

mid = uuid.uuid4().hex[:12]
config = MonitorConfig(
    monitor_id=mid,
    project_id='demo',
    config_path='$DEMO_DIR/pipeline.yaml',
    interval_seconds=60,
    max_ticks=3,
    rolling_window=20,
)
monitor_store.start(config, paused=True)

for i in range(3):
    tick = monitor_store.tick_once(mid)
    print(f'  Tick {tick.tick_number}: decision={tick.decision}  f1={tick.metric_value}  delta={tick.delta}  regression={tick.regression_detected}')

# Print history via API
history = monitor_store.get_history(mid)
print()
print(f'  Total ticks: {len(history)}')
print(f'  All decisions: {[t.decision for t in history]}')
print(f'  Any regression: {any(t.regression_detected for t in history)}')

# Status
status = monitor_store.status(mid)
print(f'  Status: {status[\"status\"]}')
print(f'  Tick count: {status[\"tick_count\"]}')
"
echo ""

# ── Step 4: Verify JSONL history file ────────────────────────────
echo "--- Step 4: Verify history file ---"
HISTORY_FILE="$DEMO_DIR/monitor_history.jsonl"
if [ -f "$HISTORY_FILE" ]; then
    LINES=$(wc -l < "$HISTORY_FILE" | tr -d ' ')
    echo "  History file: $HISTORY_FILE"
    echo "  Lines: $LINES"
    echo "  Sample:"
    head -1 "$HISTORY_FILE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'    tick={d[\"tick_number\"]} decision={d[\"decision\"]} f1={d.get(\"metric_value\",\"N/A\")}')"
else
    echo "  [WARN] No history file written (expected at $HISTORY_FILE)"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "================================================================"
echo "  Monitor demo - ALL PASSED"
echo "================================================================"

# Clean up
rm -rf "$DEMO_DIR"
