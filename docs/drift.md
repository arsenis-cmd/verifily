# Dataset Drift Detection

This guide explains how to detect and handle dataset drift in Verifily.

## What is Drift?

**Dataset drift** occurs when the distribution of your data changes over time. Unlike contamination (train/test leakage), drift represents legitimate but potentially problematic changes:

- New topics or categories appearing
- Label distribution shifts
- Text length changes
- Source data changes

## Drift vs Contamination

| Aspect | Contamination | Drift |
|--------|--------------|-------|
| **Definition** | Train/test overlap | Distribution change |
| **Cause** | Data leakage | Natural evolution |
| **Detection** | Jaccard similarity | Statistical comparison |
| **Action** | Clean data | Retrain model |
| **Urgency** | Immediate fix | Monitor and plan |

## How Drift Detection Works

### 1. MinHash Similarity

Measures content overlap using MinHash (same as fingerprinting):
- 1.0 = identical datasets
- 0.0 = completely different
- 0.7+ = high similarity (normal)
- 0.5-0.7 = moderate drift (investigate)
- <0.5 = significant drift (retrain)

### 2. Tag Distribution Analysis

Compares categorical distributions:
- Category mix changes
- Source distribution shifts
- Difficulty balance changes

### 3. Length Statistics

Tracks text/token length changes:
- Median length shift
- P95 length changes

## Using Drift Detection

### CLI Command

```bash
# Compare two datasets
verifily drift --baseline datasets/production/ --candidate datasets/candidate/

# Direct JSONL files
verifily drift --baseline data/v1.jsonl --candidate data/v2.jsonl

# Custom thresholds
verifily drift \
  --baseline data/baseline.jsonl \
  --candidate data/candidate.jsonl \
  --warn-sim 0.70 \
  --fail-sim 0.50 \
  --warn-tag 0.20 \
  --fail-tag 0.40

# JSON output
verifily drift --baseline data/baseline.jsonl --candidate data/candidate.jsonl --json
```

### Python API

```python
from verifily_cli_v1.core.drift import detect_drift, DriftStatus

result = detect_drift(
    baseline_dataset="datasets/production/dataset.jsonl",
    candidate_dataset="datasets/candidate/dataset.jsonl",
    min_similarity_warn=0.65,
    min_similarity_fail=0.45,
    max_tag_shift_warn=0.15,
    max_tag_shift_fail=0.30,
)

print(f"Status: {result.status.value}")
print(f"Similarity: {result.similarity_score:.2%}")
print(f"Tag shifts: {result.tag_shift}")

if result.status == DriftStatus.FAIL:
    print("Significant drift detected - retraining recommended")
    for action in result.recommended_actions:
        print(f"  - {action}")
```

### Configuration

Add to your `verifily.yaml`:

```yaml
drift:
  enabled: true
  baseline_dataset: datasets/production_last/dataset.jsonl
  warn_similarity: 0.65
  fail_similarity: 0.45
  warn_tag_shift: 0.15
  fail_tag_shift: 0.30
```

## Thresholds Guide

### Similarity Thresholds

| Threshold | Use Case | Action |
|-----------|----------|--------|
| 0.80+ | Strict | Warn if <0.80 |
| 0.65+ | Standard (default) | Warn if <0.65 |
| 0.50+ | Lenient | Fail if <0.50 |
| <0.50 | Critical | Always fail |

### Tag Shift Thresholds

| Threshold | Interpretation |
|-----------|----------------|
| 0.10 | Minor shift (10% distribution change) |
| 0.15 | Moderate shift (default WARN) |
| 0.30 | Major shift (default FAIL) |
| 0.50 | Severe shift (category dominance change) |

## Pipeline Integration

Drift detection runs automatically if configured:

```yaml
pipeline:
  steps:
    - contract
    - report
    - drift  # Optional drift check
    - contamination
    - decision
```

If drift FAILs:
- Pipeline exits with code 1
- Decision becomes DONT_SHIP
- Notification sent (if configured)

If drift WARNs:
- Pipeline continues
- Decision becomes INVESTIGATE
- Warning logged

## Interpreting Results

### Example Output

```
Drift Detection Report
==================================================
Status: ⚠️ WARN
Similarity Score: 58.45%

Tag Distribution Shifts:
  category: 22.50%
  source: 8.00%

Length Statistic Shifts:
  median: +15.2%
  p95: +8.5%

Reasons:
  • Similarity 0.5845 below warn threshold 0.65
  • Tag shift 0.2250 exceeds warn threshold 0.15

Recommended Actions:
  • Monitor for continued drift
  • Investigate tag distribution changes
```

### Decision Matrix

| Similarity | Tag Shift | Status | Action |
|------------|-----------|--------|--------|
| >0.65 | <0.15 | PASS | Deploy normally |
| 0.45-0.65 | <0.30 | WARN | Review before deploy |
| <0.45 | <0.30 | FAIL | Retrain required |
| Any | >0.30 | FAIL | Retrain required |

## Best Practices

### 1. Establish Baseline

```python
# Use production data as baseline
result = detect_drift(
    baseline_dataset="datasets/production_v1.jsonl",
    candidate_dataset="datasets/candidate.jsonl",
)
```

### 2. Monitor Over Time

```python
# Track drift trends
from datetime import datetime

drift_log = []
for candidate in daily_datasets:
    result = detect_drift(baseline, candidate)
    drift_log.append({
        "date": datetime.now(),
        "similarity": result.similarity_score,
        "status": result.status.value,
    })

# Alert on trend
if drift_log[-1]["similarity"] < drift_log[0]["similarity"] * 0.8:
    alert("Significant drift trend detected")
```

### 3. Set Up Alerts

```yaml
# verifily.yaml
alerts:
  - name: drift_alert
    condition: drift.status == "FAIL"
    channels: [slack, email]
    
  - name: drift_warning
    condition: drift.status == "WARN"
    channels: [slack]
```

### 4. Automate Retraining

```python
if result.status == DriftStatus.FAIL:
    # Trigger retraining pipeline
    retrain_job = submit_retrain_job(
        base_dataset=baseline,
        new_dataset=candidate,
        reason="Drift detected: similarity={result.similarity_score:.2%}"
    )
```

## Troubleshooting

### "No dataset found in artifact directory"

**Solution:** Ensure artifact directory contains `dataset.jsonl`:

```bash
ls runs/run_001/
# Should show: dataset.jsonl, manifest.json, decision.json
```

### Similarity always 1.0

**Cause:** Datasets are identical or too small.

**Solution:** Increase sample size or verify datasets differ.

### Similarity always 0.0

**Cause:** Datasets use completely different vocabulary.

**Solution:** This is correct - indicates severe drift.

## Demo

Run the drift detection demo:

```bash
./scripts/demo_drift.sh
```

Expected output:
```
=== Verifily Drift Detection Demo ===
...
Drift PASS (exit 0)
Drift FAIL (exit 1)
Drift WARN (exit 2)
=== Drift Demo -- ALL PASSED ===
```

## See Also

- [Contamination Detection](./contamination.md)
- [Monitor](./monitor.md)
- [Retrain](./retrain.md)
