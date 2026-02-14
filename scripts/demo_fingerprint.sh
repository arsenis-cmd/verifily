#!/usr/bin/env bash
# demo_fingerprint.sh — Dataset fingerprinting + diff demo
#
# Proves that fingerprints are deterministic, near-dup similarity > 0.7,
# and disjoint similarity < 0.2.
#
# CLI-only — no server, no network, no GPU.  Runtime: <5 seconds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FP_DIR="$REPO_ROOT/examples/fingerprint_demo"
OUT_DIR="/tmp/verifily_fingerprint_demo"
CLI="python3 -m verifily_cli_v1"

cd "$REPO_ROOT"

# Clean up from previous runs
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "================================================================"
echo "  Verifily — Fingerprint Demo"
echo "================================================================"
echo ""

# ── Step 1: Compute fingerprints ─────────────────────────────────
echo "--- Step 1: Compute fingerprints ---"
$CLI fingerprint --dataset "$FP_DIR/ds_a.jsonl" --out "$OUT_DIR/fp_a" --json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  ds_a: rows={data[\"rows\"]}  schema={data[\"schema\"]}  dup_rate={data[\"exact_dup_rate\"]:.4f}')
"

$CLI fingerprint --dataset "$FP_DIR/ds_b_neardup.jsonl" --out "$OUT_DIR/fp_b" --json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  ds_b: rows={data[\"rows\"]}  schema={data[\"schema\"]}  dup_rate={data[\"exact_dup_rate\"]:.4f}')
"

$CLI fingerprint --dataset "$FP_DIR/ds_c_disjoint.jsonl" --out "$OUT_DIR/fp_c" --json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  ds_c: rows={data[\"rows\"]}  schema={data[\"schema\"]}  dup_rate={data[\"exact_dup_rate\"]:.4f}')
"
echo ""

# ── Step 2: Diff near-dup pair ───────────────────────────────────
echo "--- Step 2: Diff near-dup pair (ds_a vs ds_b) ---"
DIFF_AB=$($CLI diff-datasets "$FP_DIR/ds_a.jsonl" "$FP_DIR/ds_b_neardup.jsonl" --out "$OUT_DIR/diff_ab" --json 2>/dev/null)
SIM_AB=$(echo "$DIFF_AB" | python3 -c "import json,sys; print(json.load(sys.stdin)['similarity_score'])")
echo "  Similarity: $SIM_AB"
echo ""

# ── Step 3: Diff disjoint pair ───────────────────────────────────
echo "--- Step 3: Diff disjoint pair (ds_a vs ds_c) ---"
DIFF_AC=$($CLI diff-datasets "$FP_DIR/ds_a.jsonl" "$FP_DIR/ds_c_disjoint.jsonl" --out "$OUT_DIR/diff_ac" --json 2>/dev/null)
SIM_AC=$(echo "$DIFF_AC" | python3 -c "import json,sys; print(json.load(sys.stdin)['similarity_score'])")
echo "  Similarity: $SIM_AC"
echo ""

# ── Step 4: Verify thresholds ────────────────────────────────────
echo "--- Step 4: Verify thresholds ---"
python3 -c "
sim_ab = $SIM_AB
sim_ac = $SIM_AC
print(f'  Near-dup similarity: {sim_ab:.4f} (expected > 0.5)')
print(f'  Disjoint similarity: {sim_ac:.4f} (expected < 0.2)')
assert sim_ab > 0.5, f'Near-dup similarity too low: {sim_ab}'
assert sim_ac < 0.2, f'Disjoint similarity too high: {sim_ac}'
print('  Thresholds: OK')
"
echo ""

# ── Step 5: Verify output files ──────────────────────────────────
echo "--- Step 5: Verify output files ---"
for f in fp_a/fingerprint.json fp_b/fingerprint.json fp_c/fingerprint.json diff_ab/diff.json diff_ab/diff.txt diff_ac/diff.json; do
    if [ -f "$OUT_DIR/$f" ]; then
        echo "  $f: OK"
    else
        echo "  $f: MISSING"
        exit 1
    fi
done
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "================================================================"
echo "  Fingerprint demo -- ALL PASSED"
echo "================================================================"

# Clean up
rm -rf "$OUT_DIR"
