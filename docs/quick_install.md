# Quick Install

Verifily is a CLI toolkit for ML data quality — ingest, validate, fingerprint, and gate datasets before they ship.

It runs locally with no network, no GPU, and no external services required.

## Install

```bash
pip install -e .

# With API server and SDK
pip install -e ".[all]"
```

## Verify

```bash
verifily version
```

```
       Verifily
┌──────────┬──────────────┐
│ Version  │ 1.0.0        │
│ Python   │ 3.x.x        │
│ Platform │ Darwin arm64  │
│ Extras   │ api, sdk      │
└──────────┴──────────────┘
```

## Minimal Example

```bash
verifily quickstart my_project
verifily pipeline --config my_project/verifily.yaml --ci
```

## CI Gate Usage

```bash
verifily pipeline --config pipeline.yaml --ci 2>/dev/null > verifily_result.json
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "SHIP"
else
  echo "BLOCKED (exit $EXIT_CODE)"
fi
```

| Exit code | Label | Meaning |
|-----------|-------|---------|
| `0` | **SHIP** | All quality gates passed |
| `1` | **DONT_SHIP** | One or more blockers failed |
| `2` | **INVESTIGATE** | Risk flags present, no hard blockers |
| `3` | **CONTRACT_FAIL** | Run contract invalid (missing artifacts) |
| `4` | **TOOL_ERROR** | Invalid config or unexpected error |

## Commands

| Command | Purpose |
|---------|---------|
| `verifily quickstart <path>` | Scaffold a working project |
| `verifily ingest` | Normalize raw data to artifact format |
| `verifily pipeline --ci` | Run full quality gate |
| `verifily report` | Dataset quality report with PII scan |
| `verifily contamination` | Detect train/eval overlap |
| `verifily fingerprint` | Privacy-safe dataset summary |
| `verifily diff-datasets` | Compare two datasets without raw data |
| `verifily ci-init` | Generate GitHub/GitLab CI config |
| `verifily serve` | Start API server |
| `verifily version` | Show version, Python, platform |

## Further Reading

- [3-Minute Quickstart](3_minute_quickstart.md)
- [Decision Gate](decision_gate.md)
- [Dataset Fingerprints](fingerprints.md)
- [CI Init](ci/quick_ci_init.md)
- [Ingest](ingest.md)
- [API & Jobs](api_jobs.md)
- [Monitor](monitor.md)
- [Versioning & Stability](../VERSIONING.md)
