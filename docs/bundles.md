# Shareable Report Bundles

Verifily can package run artifacts into a shareable zip bundle that contains no raw data.

## Create a Bundle

```bash
verifily bundle --run verifily_artifacts/ --out report.zip
verifily bundle --run runs/my_run --out report.zip --json
```

## What's Included

| File | Description |
|------|-------------|
| `decision_summary.json` | Final SHIP/DONT_SHIP decision with reasons |
| `pipeline_result.json` | Full pipeline execution results |
| `audit_log.jsonl` / `audit.jsonl` | Structured audit trail |
| `usage.json` | Timing and billing metadata |
| `eval/eval_results.json` | Evaluation metrics (F1, exact match, slices) |
| `hashes.json` | SHA-256 file hash chain |
| `environment.json` | Python/platform/package versions |
| `config.yaml` | Pipeline or training configuration |
| `run_meta.json` | Run metadata (timestamps, status) |
| `report.json` | Dataset report (statistics, PII summary) |
| `bundle_manifest.json` | File listing with SHA-256 checksums |

Only files that exist in the run directory are included.

## What's Excluded

| Pattern | Reason |
|---------|--------|
| `dataset.jsonl` | Raw training/eval data — may contain PII |
| `*.csv` | Raw data files |
| `*.parquet` | Raw data files |
| `*.bin`, `*.pt`, `*.safetensors` | Model weights — large, proprietary |

Bundles are safe to share with reviewers, auditors, or stakeholders. No raw dataset content or model weights are included.

## Manifest

Every bundle contains `bundle_manifest.json` with SHA-256 checksums:

```json
{
  "bundle_version": 1,
  "source_dir": "/absolute/path/to/run",
  "files": {
    "decision_summary.json": "a1b2c3...",
    "hashes.json": "d4e5f6...",
    "config.yaml": "789abc..."
  }
}
```

## Verify Integrity

After extracting, verify file integrity against the manifest:

```bash
unzip report.zip -d report/
python3 -c "
import json, hashlib, pathlib
manifest = json.load(open('report/bundle_manifest.json'))
for name, expected in manifest['files'].items():
    actual = hashlib.sha256(pathlib.Path(f'report/{name}').read_bytes()).hexdigest()
    status = 'OK' if actual == expected else 'MISMATCH'
    print(f'{status}: {name}')
"
```
