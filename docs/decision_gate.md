# Decision Gate

The decision gate is the core of Verifily's CI pipeline. It evaluates contract validity, data quality, contamination, and metric thresholds to produce a deterministic SHIP / INVESTIGATE / DONT_SHIP recommendation.

**Guarantee: SHIP means no blockers.** If any blocker is present, the decision is always DONT_SHIP.

## Blocker Precedence

Blockers are evaluated in strict order. If any blocker fires, the decision is DONT_SHIP.

| # | Blocker | Risk flag | When |
|---|---------|-----------|------|
| 1 | Contract invalid | `contract_fail` | Required artifacts (config.yaml, hashes.json, environment.json) missing or run incomplete |
| 2 | PII violation | `pii_violation` | `pii_total_hits` exceeds `max_pii_hits` threshold |
| 3 | Contamination FAIL | `contamination_fail` | Exact overlaps between train and eval data |
| 4 | Eval missing | `eval_missing` | Metric thresholds are configured but eval results are absent |
| 5 | F1 below threshold | `f1_below_threshold` | `f1 < min_f1` |
| 6 | Exact match below threshold | `exact_match_below_threshold` | `exact_match < min_exact_match` |
| 7 | F1 regression exceeded | `regression_exceeded` | `f1_delta < -max_f1_regression` vs baseline |

Contract failure (blocker #1) short-circuits immediately with exit code 3 — no further checks are run.

## Decision Semantics

| Decision | Meaning | Exit code |
|----------|---------|-----------|
| **SHIP** | All gates passed, no blockers, no risk flags | `0` |
| **INVESTIGATE** | No blockers, but risk flags present (e.g., contamination WARN) | `2` |
| **DONT_SHIP** | One or more blockers detected | `1` |

## CI Exit Codes

| Code | Label | Meaning |
|------|-------|---------|
| `0` | SHIP | All quality gates passed |
| `1` | DONT_SHIP | One or more blockers failed |
| `2` | INVESTIGATE | Risk flags present, no hard blockers |
| `3` | CONTRACT_FAIL | Run contract invalid (missing artifacts) |
| `4` | TOOL_ERROR | Invalid config, unexpected exception, or threshold validation failure |

## Confidence Scoring

Confidence is deterministic — same inputs always produce the same score.

| Decision | Range | Rule |
|----------|-------|------|
| SHIP | 0.90 - 1.00 | Base 0.90, +0.02 per passing criterion |
| INVESTIGATE | 0.50 - 0.80 | Base 0.80, -0.10 per risk flag |
| DONT_SHIP | 0.10 - 0.30 | 0.10 per blocker, capped at 0.30 |
| CONTRACT_FAIL | 1.00 | Always absolute |

## Threshold Validation

Thresholds in `ship_if` are validated at load time before any pipeline steps run. Invalid thresholds produce EXIT_TOOL_ERROR (4).

| Threshold | Valid range |
|-----------|------------|
| `min_f1` | 0.0 - 1.0 |
| `min_exact_match` | 0.0 - 1.0 |
| `max_f1_regression` | >= 0.0 |
| `max_pii_hits` | >= 0 |

## Examples

### Contamination FAIL forces DONT_SHIP

```bash
$ verifily pipeline --config verifily.yaml --ci 2>/dev/null
{
  "recommendation": "DONT_SHIP",
  "exit_code": 1,
  "confidence": 0.1,
  "reasons": [
    "Contamination FAIL: 5 exact overlaps, 0 near-duplicates"
  ],
  "risk_flags": ["contamination_fail"],
  "blocker_precedence": ["Contamination FAIL: 5 exact overlaps, 0 near-duplicates"],
  ...
}
```

Even if F1 is excellent, contamination FAIL is a hard blocker.

### Contamination WARN produces INVESTIGATE

```bash
$ verifily pipeline --config verifily_warn.yaml --ci 2>/dev/null
{
  "recommendation": "INVESTIGATE",
  "exit_code": 2,
  "confidence": 0.7,
  "reasons": [
    "Contamination WARN: 2 near-duplicates",
    "F1 (0.8500) meets threshold (0.65)",
    "Risk flags present but no hard blockers"
  ],
  "risk_flags": ["contamination_warn"],
  "blocker_precedence": [],
  ...
}
```

### Contract fail produces CONTRACT_FAIL

```bash
$ verifily pipeline --config bad_contract.yaml --ci 2>/dev/null
{
  "recommendation": "DONT_SHIP",
  "exit_code": 3,
  "confidence": 1.0,
  "reasons": ["Run contract failed — required artifacts missing"],
  "risk_flags": ["contract_fail"],
  "blocker_precedence": ["CONTRACT_INVALID"],
  "missing_files": ["config.yaml", "hashes.json"],
  ...
}
```

### PII violation blocks SHIP

```bash
$ verifily pipeline --config pii_strict.yaml --ci 2>/dev/null
{
  "recommendation": "DONT_SHIP",
  "exit_code": 1,
  "reasons": ["PII hits (3) exceed max allowed (0)"],
  "risk_flags": ["pii_violation"],
  ...
}
```

## CI Output Format

In `--ci` mode:
- **stdout**: Raw JSON decision object (no ANSI codes, no Rich formatting)
- **stderr**: Deterministic final line: `Decision: <RECOMMENDATION> (exit <CODE>)`

```bash
# Parse JSON decision in CI
DECISION=$(verifily pipeline --config pipeline.yaml --ci 2>/dev/null | jq -r .recommendation)
```

## Decision Artifacts

When `--output` is specified, the pipeline writes:
- `pipeline_result.json` — full pipeline results
- `decision_summary.json` — decision object only

These artifacts are written even on contract failure, providing a minimal decision record with `missing_files` listing what was absent.
