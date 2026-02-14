# Production Readiness Guide

This guide explains how to validate that a Verifily run is ready for production deployment.

## What is "Readiness"?

**Readiness** is a comprehensive validation that goes beyond the decision gate. While the decision gate checks if your data meets quality contracts, readiness checks if the entire run artifact is suitable for production.

### Readiness vs Decision Gate

| Aspect | Decision Gate | Readiness Check |
|--------|---------------|-----------------|
| **Purpose** | Validate data quality | Validate production deployability |
| **Scope** | Data contracts only | Artifacts, security, reproducibility |
| **Timing** | After each validation | Before production deployment |
| **Blocking** | Blocks bad data | Blocks unsafe deployments |
| **Checks** | Schema, completeness, contamination | Integrity, privacy, config, reproducibility |

## Readiness Checks

### 1. Contract Integrity ✅

Verifies all required files are present and valid.

**Required Files:**
- `manifest.json` - Run metadata
- `decision.json` - Decision gate results

**Checks:**
- Files exist
- Valid JSON
- Required fields present

**Failure Example:**
```json
{
  "name": "contract_integrity",
  "status": "FAIL",
  "message": "Missing required files: decision.json"
}
```

### 2. Hash Chain Validity 🔒

Verifies no tampering occurred.

**Validates:**
- Manifest integrity
- Dataset hashes
- Decision consistency

**Failure Example:**
```json
{
  "name": "hash_chain",
  "status": "FAIL",
  "message": "Tampering detected: dataset hash mismatch"
}
```

### 3. Privacy Safety 🛡️

Checks for PII/secrets leaks.

**Validates:**
- Redaction audit performed
- No high-severity findings
- Email/phone/API keys not present

**Warning Example:**
```json
{
  "name": "privacy_safety",
  "status": "WARN",
  "message": "No redaction audit performed"
}
```

**Failure Example:**
```json
{
  "name": "privacy_safety",
  "status": "FAIL",
  "message": "Redaction audit failed: 5 findings",
  "details": {"email": 3, "api_key": 2}
}
```

### 4. Contamination Risk ⚠️

Checks train/eval overlap status.

**Validates:**
- Contamination check passed
- No data leakage

**Failure Example:**
```json
{
  "name": "contamination_risk",
  "status": "FAIL",
  "message": "Contamination check failed"
}
```

### 5. Regression Risk 📉

Checks for metric degradation.

**Validates:**
- No accuracy drops
- No performance regressions
- Warnings reviewed

**Warning Example:**
```json
{
  "name": "regression_risk",
  "status": "WARN",
  "message": "Regression detected: accuracy dropped 5%"
}
```

### 6. Config Sanity ⚙️

Checks for absurd configuration values.

**Validates:**
- Reasonable thresholds
- Sane limits
- No dangerous settings

**Warning Example:**
```json
{
  "name": "config_sanity",
  "status": "WARN",
  "message": "Very high max_null_ratio (0.9) in completeness contract"
}
```

### 7. Reproducibility 📋

Checks for environment capture.

**Validates:**
- `environment.json` present
- Random seed recorded
- Version information captured

**Warning Example:**
```json
{
  "name": "reproducibility",
  "status": "WARN",
  "message": "No reproducibility artifacts found"
}
```

## Using the Readiness Validator

### CLI Command

```bash
verifily ready --run ./runs/run_20250120_120000
```

Output:
```
Readiness Report for run_20250120_120000
==================================================
Overall Status: ✅ PASS

✅ Contract Integrity
   Status: PASS
   All required files present and valid

✅ Hash Chain
   Status: PASS
   Hash chain verified successfully

✅ Privacy Safety
   Status: PASS
   Redaction audit passed

✅ Contamination Risk
   Status: PASS
   No contamination detected

✅ Regression Risk
   Status: PASS
   No regression detected

⚠️ Config Sanity
   Status: WARN
   Configuration warnings: Very low min_row_count in completeness

✅ Reproducibility
   Status: PASS
   Reproducibility artifacts present

--------------------------------------------------
Summary: 6 passed, 1 warnings, 0 failed

✓ This run is READY for production deployment
```

### JSON Output

```bash
verifily ready --run ./runs/run_001 --json
```

Output:
```json
{
  "run_id": "run_001",
  "overall_status": "PASS",
  "checks": [
    {
      "name": "contract_integrity",
      "status": "PASS",
      "message": "All required files present and valid",
      "details": {"files": ["manifest.json", "decision.json"]}
    }
  ],
  "summary": {
    "total_checks": 7,
    "passed": 6,
    "warnings": 1,
    "failed": 0
  }
}
```

### Python API

```python
from verifily_cli_v1.core.readiness import validate_readiness, ReadinessStatus

report = validate_readiness("./runs/run_001")

if report.overall_status == ReadinessStatus.PASS:
    print("✅ Ready for production!")
    deploy_model(run_dir="./runs/run_001")
elif report.overall_status == ReadinessStatus.WARN:
    print("⚠️ Review warnings before deployment")
    for check in report.checks:
        if check.status == ReadinessStatus.WARN:
            print(f"  - {check.name}: {check.message}")
else:
    print("❌ Not ready for production")
    for check in report.checks:
        if check.status == ReadinessStatus.FAIL:
            print(f"  - {check.name}: {check.message}")
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Check Production Readiness
  run: |
    verifily ready --run ./runs/latest
    if [ $? -ne 0 ]; then
      echo "❌ Not ready for production"
      exit 1
    fi
```

### GitLab CI

```yaml
readiness_check:
  script:
    - verifily ready --run ./runs/latest
  allow_failure: false
  only:
    - main
```

### Pipeline Integration

```bash
verifily pipeline --config verifily.yaml --ready-check
```

This runs readiness validation after the decision gate and blocks deployment if not ready.

## Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | READY - All critical checks pass |
| 1 | NOT READY - Failures present |

## Readiness Levels

### Strict Mode (CI/CD)

Fail on any warning:

```bash
verifily ready --run ./runs/latest
if [ $? -ne 0 ]; then
  exit 1
fi
```

### Permissive Mode (Development)

Allow warnings:

```python
from verifily_cli_v1.core.readiness import validate_readiness, ReadinessStatus

report = validate_readiness("./runs/latest")

# Deploy if not FAIL
if report.overall_status != ReadinessStatus.FAIL:
    deploy()
```

## Best Practices

### 1. Run Before Every Production Deployment

```bash
#!/bin/bash
set -e

# Validate
verifily ready --run "$RUN_DIR"

# Deploy only if ready
if [ $? -eq 0 ]; then
  kubectl apply -f deployment.yaml
fi
```

### 2. Include in Model Registry

```python
# When registering model
readiness_report = validate_readiness(run_dir)

model_registry.register(
    model_path=run_dir,
    readiness=readiness_report.to_dict(),
)
```

### 3. Track Readiness History

```python
# Store readiness reports
reports_db.store({
    "run_id": report.run_id,
    "status": report.overall_status.value,
    "timestamp": datetime.utcnow(),
    "checks": [c.to_dict() for c in report.checks],
})
```

### 4. Alert on Readiness Failures

```python
if report.overall_status == ReadinessStatus.FAIL:
    slack.send_message(
        channel="#ml-alerts",
        text=f"🚨 Run {report.run_id} failed readiness check",
        attachments=[{"fields": [
            {"title": c.name, "value": c.message, "short": False}
            for c in report.checks if c.status == ReadinessStatus.FAIL
        ]}],
    )
```

## Troubleshooting

### "No redaction audit performed"

**Solution:**
```bash
# Run with audit
verifily verify-run --run ./runs/latest --audit
```

### "Hash chain status: INCOMPLETE"

**Solution:**
```bash
# Regenerate hashes
verifily verify-run --run ./runs/latest --integrity
```

### "Missing required files"

**Solution:**
Check that the run completed successfully:
```bash
ls -la ./runs/latest/
# Should have: manifest.json, decision.json
```

## Example Workflows

### Production Deployment Workflow

```bash
# 1. Run validation
verifily pipeline --config verifily.yaml

# 2. Check readiness
verifily ready --run ./runs/latest

# 3. If ready, deploy
if [ $? -eq 0 ]; then
  deploy_to_production ./runs/latest
fi
```

### Model Registry Workflow

```python
from verifily_cli_v1.core.readiness import validate_readiness

for run_dir in get_candidate_runs():
    report = validate_readiness(run_dir)
    
    if report.overall_status == ReadinessStatus.PASS:
        registry.promote(run_dir, stage="production")
    elif report.overall_status == ReadinessStatus.WARN:
        registry.promote(run_dir, stage="staging")
    else:
        registry.reject(run_dir)
```

## See Also

- [Decision Gate](./decision_gate.md)
- [Security Model](./security_model.md)
- [CI/CD Integration](./ci_integration.md)
