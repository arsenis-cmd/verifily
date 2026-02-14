# Real Conditions Test Harness

Verifily ships with a self-contained test harness that validates the full pipeline under production-like conditions — without network access, GPU, or cloud dependencies.

## What "Real Conditions" Means

The harness exercises every gate in the Verifily pipeline using realistic fixture data:

| Check | What it validates |
|-------|------------------|
| **Contract** | Run directory has config.yaml, hashes.json, environment.json, eval_results.json |
| **Report** | Dataset field statistics and PII scan (email, phone, SSN, IP, credit card) |
| **Contamination** | SHA-256 exact matching + n-gram Jaccard similarity between train and eval |
| **Decision** | SHIP / DONT_SHIP / INVESTIGATE with confidence, risk flags, and exit code |
| **Regression** | F1 trend tracking across ordered runs with configurable thresholds |

All fixture data is deterministic (seeded), hand-written, and committed to the repo.

## Fixture Data

```
examples/real_conditions/
  verifily.yaml              # Pipeline config (leaked eval → DONT_SHIP)
  verifily_clean.yaml        # Pipeline config (clean eval → SHIP)
  generate_fixtures.py       # Regenerate all fixture data (seed=42)
  data/
    train.jsonl              # 60 SFT/QA rows (38 human, 22 synthetic)
    eval_clean.jsonl         # 20 rows, fully disjoint from train
    eval_leaked_exact.jsonl  # 20 rows, 4 exact copies from train (20% overlap)
    eval_leaked_near.jsonl   # 20 rows, 6 near-duplicates from train (30% near-overlap)
  runs/
    run_01_good/             # Baseline: f1=0.7139
    run_02_good/             # Stable:   f1=0.7201 (+0.0062)
    run_03_regression/       # Regressed: f1=0.6650 (-0.0551)
    run_04_recovery/         # Recovered: f1=0.7050 (+0.0400)
```

## Three Demo Commands

### 1. Full pipeline demo

```bash
bash scripts/demo_real_conditions.sh
```

Runs report, all three contamination scenarios, and both pipeline configs. Validates every exit code. Expected output ends with:

```
╔══════════════════════════════════════════════════════════════╗
║  RESULT: Real conditions demo complete ✓                   ║
╠══════════════════════════════════════════════════════════════╣
║  Contamination clean:   PASS  (exit 0)                    ║
║  Contamination exact:   FAIL  (exit 1)                    ║
║  Contamination near:    WARN  (exit 2)                    ║
║  Pipeline (leaked):     DONT_SHIP (exit 1)                ║
║  Pipeline (clean):      SHIP      (exit 0)                ║
╚══════════════════════════════════════════════════════════════╝
```

### 2. CI gate demo

```bash
bash scripts/demo_ci_gate.sh
```

Shows `pipeline --ci` behavior with JSON output and exit code semantics:

```
Exit Code Reference:
  0 = SHIP           Model passes all checks, safe to deploy
  1 = DONT_SHIP      Hard blocker found
  2 = INVESTIGATE    Risk flags present but no hard blockers
  3 = CONTRACT_FAIL  Run directory missing required artifacts
  4 = TOOL_ERROR     Verifily internal error
```

### 3. Regression detection demo

```bash
bash scripts/demo_regression.sh
```

Tracks F1 across 4 runs and flags the regression:

```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Run               ┃ Metric ┃   Delta ┃ Status     ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩
│ run_01_good       │ 0.7139 │       — │ OK         │
│ run_02_good       │ 0.7201 │ +0.0062 │ OK         │
│ run_03_regression │ 0.6650 │ -0.0551 │ REGRESSION │
│ run_04_recovery   │ 0.7050 │ +0.0400 │ IMPROVED   │
└───────────────────┴────────┴─────────┴────────────┘

REGRESSIONS DETECTED: 1
  run_03_regression: f1 dropped 0.7201 → 0.6650 (Δ = -0.0551)
```

Returns exit code 2 (regression detected).

## Running Tests

```bash
python3 -m pytest verifily_cli_v1/tests/test_real_conditions_fixtures.py -v
```

56 tests covering:
- Fixture data shape (row counts, schemas)
- Overlap expectations (clean=0, exact>=15%, near>=30%)
- Run contract compliance (all 4 runs)
- Regression detection (threshold sensitivity, ordering, trend)
- CI exit codes (contamination PASS/FAIL/WARN, pipeline SHIP/DONT_SHIP)
- Contract-check and report output correctness

All tests are deterministic, require no network or GPU, and run in < 1 second.

## Adapting to Your Data

To use this harness with your own data:

1. **Replace the fixture data** — put your train and eval JSONL files in `data/`
2. **Edit `verifily.yaml`** — point `train_data`, `eval_data`, and `run_dir` to your paths
3. **Set thresholds** — adjust `ship_if.min_f1`, `ship_if.max_f1_regression`, etc.
4. **Run the pipeline**:
   ```bash
   python3 -m verifily_cli_v1 pipeline --config verifily.yaml --ci
   echo $?  # 0=SHIP, 1=DONT_SHIP, 2=INVESTIGATE
   ```

### Example `verifily.yaml` for your project

```yaml
run_dir: runs/latest
train_data: data/train.jsonl
eval_data: data/eval.jsonl
baseline_run: runs/baseline

ship_if:
  min_f1: 0.70
  min_exact_match: 0.55
  max_f1_regression: 0.03
  max_pii_hits: 0
```

### CI integration

```yaml
# GitHub Actions
- name: Verifily gate
  run: python3 -m verifily_cli_v1 pipeline --config verifily.yaml --ci

# GitLab CI
verifily-gate:
  script: python3 -m verifily_cli_v1 pipeline --config verifily.yaml --ci
  allow_failure: false
```

The pipeline reads artifacts from disk, writes results to disk, and communicates via exit code. No hosted service, no API keys, no vendor lock-in.
