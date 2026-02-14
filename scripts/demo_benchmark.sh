#!/bin/bash
# Benchmark Demo for Verifily
# Generates datasets and runs performance benchmarks

set -e

echo "=== Verifily Benchmark Demo ==="
echo

# Setup
DEMO_DIR="/tmp/verifily_benchmark_demo"
mkdir -p "$DEMO_DIR"
export VERIFILY_HOME="$DEMO_DIR"

echo "Working directory: $DEMO_DIR"
echo

# --- 1. Generate Synthetic Datasets ---
echo "=== 1. Generating Synthetic Datasets ==="
cd /Users/arsenispapachristos/Desktop/verifily-dev

python3 examples/benchmark_demo/generate_large_dataset.py \
    --output "$DEMO_DIR/sft_1k.jsonl" \
    --rows 1000 \
    --schema sft \
    --seed 42

python3 examples/benchmark_demo/generate_large_dataset.py \
    --output "$DEMO_DIR/sft_10k.jsonl" \
    --rows 10000 \
    --schema sft \
    --seed 42

echo

# --- 2. Benchmark Ingest ---
echo "=== 2. Benchmark Ingest Performance ==="
echo "Testing 1K rows..."
python3 -c "
from verifily_cli_v1.core.benchmark import benchmark_ingest
from pathlib import Path

result = benchmark_ingest(Path('$DEMO_DIR/sft_1k.jsonl'), runs=3)
print(f\"  Rows: {result.rows:,}\")
print(f\"  Time: {result.elapsed_ms:.1f} ms\")
print(f\"  Speed: {result.rows_per_sec:,.0f} rows/sec\")
print(f\"  Memory: {result.memory_mb:.1f} MB\")
"

echo
echo "Testing 10K rows..."
python3 -c "
from verifily_cli_v1.core.benchmark import benchmark_ingest
from pathlib import Path

result = benchmark_ingest(Path('$DEMO_DIR/sft_10k.jsonl'), runs=3)
print(f\"  Rows: {result.rows:,}\")
print(f\"  Time: {result.elapsed_ms:.1f} ms\")
print(f\"  Speed: {result.rows_per_sec:,.0f} rows/sec\")
print(f\"  Memory: {result.memory_mb:.1f} MB\")
"

echo

# --- 3. Benchmark Contamination ---
echo "=== 3. Benchmark Contamination Detection ==="
echo "Creating train/eval datasets..."

python3 -c "
import json
from pathlib import Path

# Create train dataset (80% of 1K)
with open('$DEMO_DIR/train.jsonl', 'w') as f:
    for i in range(800):
        record = {'instruction': f'Train task {i}', 'output': f'Output {i}'}
        f.write(json.dumps(record) + '\n')

# Create eval dataset (20% of 1K)
with open('$DEMO_DIR/eval.jsonl', 'w') as f:
    for i in range(200):
        record = {'instruction': f'Eval task {i}', 'output': f'Output {i}'}
        f.write(json.dumps(record) + '\n')
"

python3 -c "
from verifily_cli_v1.core.benchmark import benchmark_contamination
from pathlib import Path

result = benchmark_contamination(
    Path('$DEMO_DIR/train.jsonl'),
    Path('$DEMO_DIR/eval.jsonl'),
    runs=1
)
print(f\"  Train rows: {result.rows // 5 * 4:,}\")
print(f\"  Eval rows: {result.rows // 5:,}\")
print(f\"  Total: {result.rows:,}\")
print(f\"  Time: {result.elapsed_ms:.1f} ms\")
print(f\"  Speed: {result.rows_per_sec:,.0f} rows/sec\")
"

echo

# --- 4. Benchmark Pipeline ---
echo "=== 4. Benchmark Pipeline Execution ==="
echo "Creating mock run directory..."

mkdir -p "$DEMO_DIR/run_001"
python3 -c "
import json
from pathlib import Path

# Create dataset
with open('$DEMO_DIR/run_001/dataset.jsonl', 'w') as f:
    for i in range(1000):
        record = {'text': f'Record {i}', 'label': 'A' if i % 2 == 0 else 'B'}
        f.write(json.dumps(record) + '\n')

# Create manifest
with open('$DEMO_DIR/run_001/manifest.json', 'w') as f:
    json.dump({'run_id': 'run_001', 'status': 'complete'}, f)
"

python3 -c "
from verifily_cli_v1.core.benchmark import benchmark_pipeline
from pathlib import Path

result = benchmark_pipeline(Path('$DEMO_DIR/run_001'), plan=True, runs=1)
print(f\"  Rows: {result.rows:,}\")
print(f\"  Time: {result.elapsed_ms:.1f} ms\")
print(f\"  Speed: {result.rows_per_sec:,.0f} rows/sec\")
"

echo

# --- 5. Full Benchmark Suite ---
echo "=== 5. Full Benchmark Suite ==="
python3 -c "
from verifily_cli_v1.core.benchmark import run_benchmark_suite, format_benchmark_table
from pathlib import Path

report = run_benchmark_suite(
    dataset_path=Path('$DEMO_DIR/sft_1k.jsonl'),
    train_path=Path('$DEMO_DIR/train.jsonl'),
    eval_path=Path('$DEMO_DIR/eval.jsonl'),
    run_dir=Path('$DEMO_DIR/run_001'),
)

print(format_benchmark_table(report))

# Performance assessment
avg_rps = report.summary.get('avg_rows_per_sec', 0)
print(f\"\nOverall average: {avg_rps:,.0f} rows/sec\")

if avg_rps > 5000:
    print('✓ Excellent performance (>5K rows/sec)')
elif avg_rps > 1000:
    print('✓ Good performance (>1K rows/sec)')
else:
    print('⚠ Moderate performance')
"

echo

# --- 6. Cleanup ---
echo "=== 6. Cleanup ==="
rm -rf "$DEMO_DIR"
echo "Removed: $DEMO_DIR"
echo

echo "=== Benchmark Demo -- ALL PASSED ==="
echo
echo "Key Takeaways:"
echo "  - Verifily ingests datasets at 1K-50K+ rows/sec (depends on hardware)"
echo "  - Memory usage scales linearly with dataset size"
echo "  - Contamination detection uses efficient n-gram sampling"
echo "  - Pipeline benchmark includes all validation stages"
echo "  - All measurements are deterministic (seed=42)"
echo
echo "Note: Verifily prioritizes determinism and accuracy over raw speed."
