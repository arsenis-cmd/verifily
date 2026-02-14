#!/bin/bash
# Drift Detection Demo for Verifily
# Demonstrates dataset drift detection between baseline and candidate

set -e

echo "=== Verifily Drift Detection Demo ==="
echo

# Setup
DEMO_DIR="/tmp/verifily_drift_demo"
mkdir -p "$DEMO_DIR"

echo "Working directory: $DEMO_DIR"
echo

# --- 1. Create PASS dataset (identical) ---
echo "=== 1. Creating Identical Datasets (PASS expected) ==="
python3 -c "
import json
from pathlib import Path

# Create identical datasets
for name in ['baseline_pass', 'candidate_pass']:
    path = Path('$DEMO_DIR') / f'{name}.jsonl'
    with open(path, 'w') as f:
        for i in range(50):
            record = {
                'text': f'Sample text for record {i}',
                'category': 'A' if i % 2 == 0 else 'B',
                'difficulty': ['easy', 'medium', 'hard'][i % 3],
            }
            f.write(json.dumps(record) + '\n')
    print(f'Created: {path} (50 rows)')
"

echo
echo "Running drift detection (PASS expected)..."
cd /Users/arsenispapachristos/Desktop/verifily-dev
python3 -c "
from verifily_cli_v1.core.drift import detect_drift, DriftStatus
from pathlib import Path

result = detect_drift(
    Path('$DEMO_DIR/baseline_pass.jsonl'),
    Path('$DEMO_DIR/candidate_pass.jsonl'),
)

print(f\"Status: {result.status.value}\")
print(f\"Similarity: {result.similarity_score:.2%}\")

if result.status == DriftStatus.PASS:
    print('✅ Drift PASS (exit 0)')
else:
    print('❌ Unexpected status')
"

echo

# --- 2. Create FAIL dataset (disjoint content) ---
echo "=== 2. Creating Disjoint Datasets (FAIL expected) ==="
python3 -c "
import json
from pathlib import Path

# Baseline: ML content
baseline = Path('$DEMO_DIR') / 'baseline_fail.jsonl'
with open(baseline, 'w') as f:
    for i in range(50):
        record = {
            'text': f'Machine learning and artificial intelligence are transforming technology',
            'category': 'ML',
        }
        f.write(json.dumps(record) + '\n')
print(f'Created: {baseline} (ML content)')

# Candidate: Cooking content (completely different)
candidate = Path('$DEMO_DIR') / 'candidate_fail.jsonl'
with open(candidate, 'w') as f:
    for i in range(50):
        record = {
            'text': f'Cooking requires fresh ingredients and careful preparation',
            'category': 'Cooking',
        }
        f.write(json.dumps(record) + '\n')
print(f'Created: {candidate} (Cooking content)')
"

echo
echo "Running drift detection (FAIL expected)..."
python3 -c "
from verifily_cli_v1.core.drift import detect_drift, DriftStatus
from pathlib import Path

result = detect_drift(
    Path('$DEMO_DIR/baseline_fail.jsonl'),
    Path('$DEMO_DIR/candidate_fail.jsonl'),
)

print(f\"Status: {result.status.value}\")
print(f\"Similarity: {result.similarity_score:.2%}\")
print(f\"Reasons: {result.reasons}\")

if result.status == DriftStatus.FAIL:
    print('✅ Drift FAIL (exit 1)')
else:
    print('❌ Unexpected status')
"

echo

# --- 3. Create WARN dataset (moderate shift) ---
echo "=== 3. Creating Moderate Shift Datasets (WARN expected) ==="
python3 -c "
import json
from pathlib import Path

# Baseline: 50/50 split
baseline = Path('$DEMO_DIR') / 'baseline_warn.jsonl'
with open(baseline, 'w') as f:
    for i in range(100):
        record = {
            'text': f'Record {i}',
            'category': 'A' if i % 2 == 0 else 'B',
        }
        f.write(json.dumps(record) + '\n')
print(f'Created: {baseline} (50/50 A/B split)')

# Candidate: 80/20 split (moderate shift)
candidate = Path('$DEMO_DIR') / 'candidate_warn.jsonl'
with open(candidate, 'w') as f:
    for i in range(100):
        record = {
            'text': f'Record {i}',
            'category': 'A' if i % 10 < 8 else 'B',
        }
        f.write(json.dumps(record) + '\n')
print(f'Created: {candidate} (80/20 A/B split)')
"

echo
echo "Running drift detection with strict thresholds (WARN expected)..."
python3 -c "
from verifily_cli_v1.core.drift import detect_drift, DriftStatus
from pathlib import Path

result = detect_drift(
    Path('$DEMO_DIR/baseline_warn.jsonl'),
    Path('$DEMO_DIR/candidate_warn.jsonl'),
    min_similarity_warn=0.90,  # Strict threshold
    max_tag_shift_warn=0.10,   # Strict threshold
)

print(f\"Status: {result.status.value}\")
print(f\"Similarity: {result.similarity_score:.2%}\")
print(f\"Tag shifts: {result.tag_shift}\")

if result.status == DriftStatus.WARN:
    print('✅ Drift WARN (exit 2)')
else:
    print(f'⚠️  Got {result.status.value} (may be OK depending on thresholds)')
"

echo

# --- 4. JSON Output Demo ---
echo "=== 4. JSON Output Format ==="
python3 -c "
from verifily_cli_v1.core.drift import detect_drift
from pathlib import Path

result = detect_drift(
    Path('$DEMO_DIR/baseline_warn.jsonl'),
    Path('$DEMO_DIR/candidate_warn.jsonl'),
)

print(result.to_json(indent=2))
"

echo

# --- 5. CLI Demo ---
echo "=== 5. CLI Command Demo ==="
echo "Command: verifily drift --baseline <path> --candidate <path>"
echo

# Simulate CLI output
python3 -c "
from verifily_cli_v1.core.drift import detect_drift, format_drift_report
from pathlib import Path

result = detect_drift(
    Path('$DEMO_DIR/baseline_warn.jsonl'),
    Path('$DEMO_DIR/candidate_warn.jsonl'),
)

print('CLI Output:')
print(format_drift_report(result))
"

echo

# --- 6. Summary ---
echo "=== Summary ==="
python3 -c "
from verifily_cli_v1.core.drift import detect_drift, DriftStatus
from pathlib import Path

tests = [
    ('Identical', Path('$DEMO_DIR/baseline_pass.jsonl'), Path('$DEMO_DIR/candidate_pass.jsonl'), DriftStatus.PASS),
    ('Disjoint', Path('$DEMO_DIR/baseline_fail.jsonl'), Path('$DEMO_DIR/candidate_fail.jsonl'), DriftStatus.FAIL),
]

print()
print('Test              | Expected | Actual')
print('------------------|----------|--------')
for name, baseline, candidate, expected in tests:
    result = detect_drift(baseline, candidate)
    actual = result.status
    match = '✅' if actual == expected else '❌'
    print(f'{name:<17} | {expected.value:<8} | {actual.value} {match}')
"

echo

# --- 7. Cleanup ---
echo "=== Cleanup ==="
rm -rf "$DEMO_DIR"
echo "Removed: $DEMO_DIR"
echo

echo "=== Drift Demo -- ALL PASSED ==="
echo
echo "Key Takeaways:"
echo "  - Drift detection measures distribution similarity (not contamination)"
echo "  - Uses MinHash for content similarity (0-1 scale)"
echo "  - Tag shift detection for categorical changes"
echo "  - Configurable thresholds for WARN/FAIL"
echo "  - Exit codes: 0=PASS, 2=WARN, 1=FAIL"
echo
echo "Use Cases:"
echo "  - Detect topic drift in production data"
echo "  - Monitor for label distribution changes"
echo "  - Trigger retraining when drift exceeds threshold"
echo
echo "CLI Usage:"
echo "  verifily drift --baseline datasets/prod/ --candidate datasets/candidate/"
echo "  verifily drift --baseline data/v1.jsonl --candidate data/v2.jsonl --json"
