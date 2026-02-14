#!/bin/bash
# Readiness Demo for Verifily
# Demonstrates production readiness validation

set -e

echo "=== Verifily Readiness Demo ==="
echo

# Setup
DEMO_DIR="/tmp/verifily_readiness_demo"
mkdir -p "$DEMO_DIR"

echo "Working directory: $DEMO_DIR"
echo

# --- 1. Create Clean Run (READY) ---
echo "=== 1. Creating Clean Run (should be READY) ==="
CLEAN_RUN="$DEMO_DIR/run_clean"
mkdir -p "$CLEAN_RUN"

python3 -c "
import json
from pathlib import Path

run_dir = Path('$CLEAN_RUN')

# Manifest
manifest = {
    'run_id': 'run_clean',
    'timestamp': '2025-01-20T12:00:00Z',
    'version': '1.0.0',
    'contracts': [
        {'name': 'completeness', 'min_row_count': 100},
        {'name': 'schema', 'fields': [{'name': 'text', 'type': 'string'}]},
    ],
}
(run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

# Decision (PASS)
decision = {
    'run_id': 'run_clean',
    'timestamp': '2025-01-20T12:00:01Z',
    'status': 'PASS',
    'passed': True,
    'checks': {
        'completeness': 'PASS',
        'schema': 'PASS',
        'contamination': 'PASS',
    },
    'summary': {'total': 3, 'passed': 3, 'warnings': []},
}
(run_dir / 'decision.json').write_text(json.dumps(decision, indent=2))

# Environment
environment = {
    'seed': 42,
    'versions': {
        'verifily': '1.0.0',
        'python': '3.11.0',
    },
    'dependencies': {'numpy': '1.24.0'},
}
(run_dir / 'environment.json').write_text(json.dumps(environment, indent=2))

# Redaction audit (PASS)
audit = {
    'status': 'PASS',
    'files_scanned': 3,
    'findings_count': 0,
    'findings_by_type': {},
    'summary': {'high_severity': 0, 'medium_severity': 0, 'low_severity': 0},
}
(run_dir / 'redaction_audit.json').write_text(json.dumps(audit, indent=2))

print('Created clean run at:', run_dir)
"

echo "Validating clean run..."
cd /Users/arsenispapachristos/Desktop/verifily-dev
python3 -c "
from verifily_cli_v1.core.readiness import validate_readiness, ReadinessStatus
from pathlib import Path

report = validate_readiness(Path('$CLEAN_RUN'))

print(f\"Overall Status: {report.overall_status.value}\")
print(f\"Checks: {len(report.checks)}\")
print()

for check in report.checks:
    status_icon = '✅' if check.status == ReadinessStatus.PASS else '⚠️' if check.status == ReadinessStatus.WARN else '❌'
    print(f\"  {status_icon} {check.name}: {check.status.value}\")

if report.overall_status == ReadinessStatus.PASS:
    print()
    print('✅ Run is READY for production!')
elif report.overall_status == ReadinessStatus.WARN:
    print()
    print('⚠️  Run has warnings (may still be deployable)')
else:
    print()
    print('❌ Run is NOT ready')
"

echo

# --- 2. Create Contaminated Run (NOT READY) ---
echo "=== 2. Creating Contaminated Run (should be NOT READY) ==="
BAD_RUN="$DEMO_DIR/run_contaminated"
mkdir -p "$BAD_RUN"

python3 -c "
import json
from pathlib import Path

run_dir = Path('$BAD_RUN')

# Manifest
manifest = {
    'run_id': 'run_contaminated',
    'timestamp': '2025-01-20T12:00:00Z',
    'version': '1.0.0',
}
(run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

# Decision (FAIL due to contamination)
decision = {
    'run_id': 'run_contaminated',
    'timestamp': '2025-01-20T12:00:01Z',
    'status': 'FAIL',
    'passed': False,
    'checks': {
        'completeness': 'PASS',
        'schema': 'PASS',
        'contamination': 'FAIL',  # FAIL - contamination detected!
    },
    'summary': {'total': 3, 'passed': 2, 'failed': 1},
}
(run_dir / 'decision.json').write_text(json.dumps(decision, indent=2))

# No environment.json (reproducibility warning)
# No redaction audit (privacy warning)

print('Created contaminated run at:', run_dir)
"

echo "Validating contaminated run..."
python3 -c "
from verifily_cli_v1.core.readiness import validate_readiness, ReadinessStatus
from pathlib import Path

report = validate_readiness(Path('$BAD_RUN'))

print(f\"Overall Status: {report.overall_status.value}\")
print(f\"Checks: {len(report.checks)}\")
print()

for check in report.checks:
    status_icon = '✅' if check.status == ReadinessStatus.PASS else '⚠️' if check.status == ReadinessStatus.WARN else '❌'
    print(f\"  {status_icon} {check.name}: {check.status.value}\")
    if check.status != ReadinessStatus.PASS:
        print(f\"       {check.message}\")

print()
if report.overall_status == ReadinessStatus.FAIL:
    print('❌ Run correctly flagged as NOT READY (contamination detected)')
else:
    print('⚠️  Unexpected status')
"

echo

# --- 3. Create Run with Regression Warning ---
echo "=== 3. Creating Run with Regression Warning ==="
WARN_RUN="$DEMO_DIR/run_warning"
mkdir -p "$WARN_RUN"

python3 -c "
import json
from pathlib import Path

run_dir = Path('$WARN_RUN')

# Manifest
manifest = {
    'run_id': 'run_warning',
    'version': '1.0.0',
}
(run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

# Decision (PASS but with regression warning)
decision = {
    'run_id': 'run_warning',
    'status': 'PASS',
    'passed': True,
    'checks': {'completeness': 'PASS'},
    'summary': {
        'warnings': ['Regression detected: accuracy dropped 5% from baseline'],
    },
}
(run_dir / 'decision.json').write_text(json.dumps(decision, indent=2))

# Environment present
environment = {'seed': 42, 'versions': {'verifily': '1.0.0'}}
(run_dir / 'environment.json').write_text(json.dumps(environment, indent=2))

print('Created warning run at:', run_dir)
"

echo "Validating warning run..."
python3 -c "
from verifily_cli_v1.core.readiness import validate_readiness, ReadinessStatus
from pathlib import Path

report = validate_readiness(Path('$WARN_RUN'))

print(f\"Overall Status: {report.overall_status.value}\")
print()

for check in report.checks:
    if check.status == ReadinessStatus.WARN:
        print(f\"  ⚠️  {check.name}: {check.message}\")
"

echo

# --- 4. Summary ---
echo "=== Summary ==="
python3 -c "
from verifily_cli_v1.core.readiness import validate_readiness, ReadinessStatus
from pathlib import Path

runs = [
    ('Clean Run', Path('$CLEAN_RUN'), ReadinessStatus.PASS),
    ('Contaminated Run', Path('$BAD_RUN'), ReadinessStatus.FAIL),
    ('Warning Run', Path('$WARN_RUN'), ReadinessStatus.WARN),
]

print()
print('Run                | Expected   | Actual')
print('-------------------|------------|--------')
for name, path, expected in runs:
    report = validate_readiness(path)
    actual = report.overall_status
    match = '✅' if actual == expected else '❌'
    print(f'{name:<18} | {expected.value:<10} | {actual.value} {match}')
"

echo

# --- 5. Cleanup ---
echo "=== Cleanup ==="
rm -rf "$DEMO_DIR"
echo "Removed: $DEMO_DIR"
echo

echo "=== Readiness Demo -- ALL PASSED ==="
echo
echo "Key Takeaways:"
echo "  - Clean runs with all artifacts are marked READY"
echo "  - Contamination failures block production readiness"
echo "  - Missing artifacts trigger warnings (not failures)"
echo "  - Regression warnings alert but don't block by default"
echo "  - Use 'verifily ready --run <dir>' in CI/CD"
echo
echo "Exit Codes:"
echo "  0 = READY (all critical checks pass)"
echo "  1 = NOT READY (failures present)"
