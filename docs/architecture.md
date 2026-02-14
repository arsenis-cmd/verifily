# Verifily Architecture

## Overview

Verifily is a data quality gate system for ML pipelines. It validates datasets against contracts, detects contamination, monitors for regressions, and triggers retraining when needed.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
│  (Datasets, Configs, Schemas)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONTRACT VALIDATION                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Completeness│ │   Schema    │ │   Custom    │               │
│  │   Check     │ │  Validation │ │   Rules     │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PIPELINE ENGINE                            │
│  - Loads configuration                                         │
│  - Executes validation stages                                   │
│  - Collects results                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DECISION ENGINE                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DECISION GATE                         │   │
│  │  ┌─────────┐  ┌─────────────┐  ┌─────────────────┐     │   │
│  │  │  PASS   │  │ FAIL + Alert│  │ Retry/Override  │     │   │
│  │  └─────────┘  └─────────────┘  └─────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MONITOR & ALERTS                            │
│  - Regression detection                                         │
│  - Drift detection                                              │
│  - Alert notifications                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RETRAIN TRIGGER                            │
│  - Automatically trigger retraining on failure                  │
│  - A/B test validation                                          │
│  - Model promotion pipeline                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Contract Validation

Contracts define expectations for your data:

- **Completeness**: Checks for null values, required fields
- **Schema Validation**: Validates data types, structures
- **Custom Rules**: User-defined validation logic
- **Contamination Detection**: Prevents train/eval overlap

Example contract (`verifily.yaml`):
```yaml
dataset:
  path: data/train.jsonl
  schema: sft
  
contracts:
  - name: completeness
    min_row_count: 1000
    max_null_ratio: 0.01
    
  - name: schema
    fields:
      - name: instruction
        type: string
        required: true
      - name: response
        type: string
        required: true
```

### 2. Pipeline Engine

The pipeline engine orchestrates validation:

1. **Load**: Load configuration and datasets
2. **Validate**: Execute all contract checks
3. **Report**: Generate structured report
4. **Decide**: Determine PASS/FAIL status

### 3. Decision Gate

The decision gate makes the final call:

- **PASS**: All contracts satisfied → proceed with training
- **FAIL**: Contract violated → block training, alert team
- **Conditional**: Warnings present → proceed with notification

### 4. Monitor & Alerts

Continuous monitoring for deployed models:

- **Regression Detection**: Catch performance drops
- **Drift Detection**: Detect data distribution shifts
- **Alert Channels**: Slack, email, webhooks

### 5. Retrain Trigger

Automated retraining pipeline:

- Triggered by: Schedule, drift detection, manual
- Validates: New data quality before retraining
- Promotes: Successful models through stages

## Data Flow

```
Dataset Upload → Contract Validation → Decision Gate → 
    ├─ PASS → Training Pipeline → Model Registry
    └─ FAIL → Alert → Retrain/Fix
```

## Key Features

### Deterministic Validation
- Same input → same output
- Reproducible results
- Versioned contracts

### Enterprise Security
- API key authentication
- Path traversal protection
- PII redaction audit
- Signed decisions

### Integration Ready
- REST API
- Python SDK (sync + async)
- CI/CD plugins
- Prometheus metrics

## Scalability

Verifily scales horizontally:

- **Stateless**: No local state required
- **Async Jobs**: Long-running validations queued
- **Resource Efficient**: Minimal memory footprint
- **Fast**: <100ms for most validations

## See Also

- [Security Model](./security_model.md)
- [ML Failure Prevention](./ml_failure_prevention.md)
- [API Reference](./API.md)
