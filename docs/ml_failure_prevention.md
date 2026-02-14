# ML Failure Prevention

This guide explains how Verifily prevents common ML pipeline failures.

## Common ML Failures

### 1. Data Contamination

**Problem**: Training data leaks into evaluation/test sets, causing inflated metrics.

**Detection**: Jaccard similarity analysis

```bash
verifily contamination \
  --train data/train.jsonl \
  --eval data/eval.jsonl \
  --jaccard-cutoff 0.70
```

**Prevention**:
- Automated contamination checks on every data change
- Block training if contamination detected
- Report overlap statistics

**Example Output**:
```json
{
  "contaminated": true,
  "jaccard_similarity": 0.85,
  "overlap_count": 150,
  "recommendation": "Remove overlapping samples before training"
}
```

### 2. Train/Eval Regression

**Problem**: Model performs worse on evaluation than training.

**Detection**: Monitor metrics comparison

```yaml
contracts:
  - name: no_regression
    type: threshold
    metric: f1_score
    min_eval_improvement: 0.05  # Eval must be within 5% of train
```

**Prevention**:
- Compare train/eval metrics
- Alert on significant gaps
- Block deployment if regression detected

### 3. Silent Data Corruption

**Problem**: Data changes break model without obvious errors.

**Detection**: Schema validation + statistical checks

```yaml
contracts:
  - name: schema
    fields:
      - name: instruction
        type: string
        min_length: 10
        max_length: 2000
      - name: response
        type: string
        required: true
```

**Prevention**:
- Strict schema validation
- Field type checking
- Range validation

### 4. Irreproducibility

**Problem**: Results vary between runs with same data.

**Detection**: Deterministic validation

**Prevention**:
- Fixed random seeds
- Deterministic algorithms
- Versioned contracts
- Immutable run artifacts

```bash
# Same input → same output
cat data/train.jsonl | verifily validate --seed 42
```

### 5. Silent Degradation

**Problem**: Model quality slowly degrades over time.

**Detection**: Continuous monitoring

```bash
verifily monitor start \
  --config verifily.yaml \
  --interval 3600 \
  --alert-threshold 0.02
```

**Prevention**:
- Scheduled quality checks
- Drift detection
- Automatic alerting
- Retrain triggers

### 6. Schema Drift

**Problem**: Production data differs from training data schema.

**Detection**: Schema validation

```yaml
contracts:
  - name: schema_stability
    forbid_new_fields: true
    forbid_missing_fields: true
```

**Prevention**:
- Strict schema enforcement
- Version-controlled schemas
- Migration detection

### 7. Distribution Shift

**Problem**: Production data distribution differs from training.

**Detection**: Statistical analysis

```yaml
contracts:
  - name: distribution
    reference: data/train_stats.json
    max_ks_distance: 0.1
```

**Prevention**:
- Statistical tests (KS, Chi-square)
- Reference distribution comparison
- Drift alerts

### 8. Annotation Errors

**Problem**: Incorrect labels in training data.

**Detection**: Quality checks

```yaml
contracts:
  - name: annotation_quality
    min_agreement: 0.8
    max_blank_labels: 0.01
```

**Prevention**:
- Inter-annotator agreement
- Outlier detection
- Consistency checks

## Failure Response Matrix

| Failure Type | Detection | Response | Auto-Fix |
|--------------|-----------|----------|----------|
| Contamination | Jaccard check | Block training | No |
| Regression | Metric compare | Alert + Block | No |
| Data Corruption | Schema validation | Block + Report | No |
| Irreproducibility | Deterministic test | Alert | No |
| Silent Degradation | Monitoring | Alert | Trigger retrain |
| Schema Drift | Schema check | Block | No |
| Distribution Shift | Statistics | Alert | No |
| Annotation Errors | Quality check | Report | No |

## Best Practices

### 1. Fail Fast

Catch issues early in the pipeline:

```
Data Ingest → Validation → Cleaning → Training → Deployment
                ↑
         Block here if issues
```

### 2. Immutable Artifacts

Never modify raw data:

```
raw_data/      # Original, immutable
cleaned_data/  # Processed, versioned
runs/          # Validation results, signed
```

### 3. Version Everything

- Data versions
- Contract versions
- Model versions
- Validation runs

### 4. Continuous Validation

Validate at every stage:

```yaml
stages:
  - name: ingest
    validations: [schema, completeness]
  - name: clean
    validations: [contamination, quality]
  - name: train
    validations: [regression, metrics]
  - name: deploy
    validations: [performance, drift]
```

### 5. Automated Recovery

```yaml
recovery:
  on_failure:
    - notify: slack/ml-alerts
    - rollback: true
    - retrain: if_drift_detected
```

## Detection Metrics

### Contamination Metrics

- **Jaccard Similarity**: Intersection / Union
- **Overlap Count**: Number of duplicate samples
- **Leakage Percentage**: % of eval in train

### Quality Metrics

- **Completeness**: % non-null values
- **Validity**: % schema-compliant records
- **Consistency**: % consistent records
- **Uniqueness**: % unique records

### Drift Metrics

- **KS Distance**: Kolmogorov-Smirnov test
- **PSI**: Population Stability Index
- **Chi-square**: Categorical distribution
- **KL Divergence**: Information-theoretic distance

## Alerting Rules

```yaml
alerts:
  - name: contamination_alert
    condition: contamination.jaccard > 0.1
    severity: critical
    channels: [slack, email]
    
  - name: drift_alert
    condition: drift.ks_distance > 0.05
    severity: warning
    channels: [slack]
    
  - name: quality_alert
    condition: quality.completeness < 0.95
    severity: warning
    channels: [email]
```

## Case Studies

### Case 1: Contamination in Production

A team deployed a model that appeared to have 95% accuracy. Verifily detected 30% contamination between train and test. After cleaning, actual accuracy was 72%.

**Lesson**: Always validate contamination before deployment.

### Case 2: Silent Schema Drift

API response format changed, adding new fields. Model training continued but performance degraded. Verifily schema validation detected the drift on day 1.

**Lesson**: Strict schema validation prevents silent failures.

### Case 3: Distribution Shift

Seasonal data shift caused model accuracy to drop from 90% to 75% over 3 months. Verifily monitoring detected drift at week 2, triggering automatic retraining.

**Lesson**: Continuous monitoring catches drift early.

## See Also

- [Contracts Reference](./contracts.md)
- [Monitoring Guide](./monitoring.md)
- [CI/CD Integration](./ci_integration.md)
