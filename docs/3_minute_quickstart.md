# 3-Minute Quickstart

Get from zero to a SHIP/DONT_SHIP decision in 3 commands.

## Install

```bash
pip install -e /path/to/verifily-dev
```

## Scaffold

```bash
verifily quickstart
```

This creates `./verifily_quickstart/` with:
- Sample training data (12 rows, no PII)
- Evaluation data (5 rows)
- A baseline run that passes the contract check
- A demo script that runs the full pipeline

## Run

```bash
cd verifily_quickstart
bash scripts/run_demo.sh
```

## Expected output

```
--- Step 1: Ingest sample data ---
  ...
--- Step 2: Build pipeline config ---
  ...
--- Step 3: Run pipeline gate ---
  Decision:      SHIP
  Exit code:     0
  F1:            0.72

================================================================
  Quickstart -- ALL PASSED
================================================================
```

## What happened

1. **Ingest** converted `data/raw/sample.csv` into canonical `dataset.jsonl` + `report.json`
2. **Pipeline** ran CONTRACT → REPORT → CONTAMINATION → DECISION against `runs/baseline/`
3. **Decision** was SHIP because the baseline metrics (f1=0.72) pass the gate thresholds

## Next steps

- Edit `verifily.yaml` to point at your own data and runs
- Run `verifily pipeline --config verifily.yaml --ci` in CI
- Use `verifily ci-init --github` to generate a GitHub Actions workflow

## Options

```bash
verifily quickstart [path] [--force] [--json]
```

| Flag | Default | Description |
|------|---------|-------------|
| `path` | `./verifily_quickstart` | Project directory to create |
| `--force` | false | Overwrite existing directory |
| `--json` | false | Output JSON instead of Rich panel |
