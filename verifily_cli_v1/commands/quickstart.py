"""verifily quickstart — scaffold a working project in 3 minutes."""

from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List

from verifily_cli_v1.core.hashing import (
    compute_chain_hash,
    compute_file_hashes,
    sha256_file,
)
from verifily_cli_v1.core.io import ensure_dir, write_json, write_yaml

# ── Sample data (no PII, deterministic) ───────────────────────────

SAMPLE_CSV_HEADER = "subject,resolution,body"
SAMPLE_CSV_ROWS = [
    "Cannot log in,Reset your password via Settings > Security,User reports login failure after update",
    "Slow dashboard,Clear browser cache and reload,Dashboard takes 10 seconds to load",
    "Missing report,Reports refresh daily at midnight UTC,Quarterly report not visible",
    "Export fails,Ensure at least 10 records exist,Export button is grayed out",
    "Wrong balance,Go to Billing > Refresh Balance,Account shows incorrect amount",
    "Search broken,Rebuild index via Admin > Search,Search returns zero results",
    "Email alerts,Check Settings > Notifications,No email notifications received",
    "Rate limited,Use exponential backoff,API returns 429 during batch jobs",
    "Stale data,Add nocache parameter to query,Report shows yesterday data",
    "Webhook error,Check logs and ensure 10s timeout,Webhook returns 500",
    "Upload limit,Contact admin to increase to 50MB,File upload fails above 5MB",
    "Widget broken,Reset dashboard layout in Settings,Widgets not rendering",
]

EVAL_ROWS = [
    {"input": "Context:\nUser locked out\n\nQuestion:\nHow to unlock?", "output": "Reset via admin panel."},
    {"input": "Context:\nPage timeout\n\nQuestion:\nHow to fix?", "output": "Check server load and increase timeout."},
    {"input": "Context:\nMissing data\n\nQuestion:\nWhere is my data?", "output": "Data syncs every hour. Wait and refresh."},
    {"input": "Context:\nPermission denied\n\nQuestion:\nHow to get access?", "output": "Request access from your project admin."},
    {"input": "Context:\nDuplicate entry\n\nQuestion:\nHow to resolve?", "output": "Merge duplicates via the dedup tool."},
]

RUN_DEMO_SCRIPT = r'''#!/usr/bin/env bash
# Quickstart demo — ingest + pipeline gate
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use verifily if installed, otherwise fall back to python3 -m
if command -v verifily &>/dev/null; then
    CLI="verifily"
else
    CLI="python3 -m verifily_cli_v1"
fi

echo "================================================================"
echo "  Verifily Quickstart Demo"
echo "================================================================"
echo ""

# Step 1: Ingest
echo "--- Step 1: Ingest sample data ---"
$CLI ingest \
  --in "$PROJECT_DIR/data/raw/sample.csv" \
  --out "$PROJECT_DIR/data/artifact" \
  --schema sft \
  --map question:subject --map answer:resolution --map context:body
echo ""

# Step 2: Build pipeline config with absolute paths
echo "--- Step 2: Build pipeline config ---"
python3 -c "
import yaml
cfg = {
    'run_dir': '$PROJECT_DIR/runs/baseline',
    'train_data': '$PROJECT_DIR/data/artifact/dataset.jsonl',
    'eval_data': '$PROJECT_DIR/data/raw/eval.jsonl',
    'baseline_run': '$PROJECT_DIR/runs/baseline',
    'ship_if': {
        'min_f1': 0.50,
        'min_exact_match': 0.40,
        'max_f1_regression': 0.05,
        'max_pii_hits': 0,
    },
}
with open('$PROJECT_DIR/.pipeline_run.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('  Config written to .pipeline_run.yaml')
"
echo ""

# Step 3: Run pipeline gate
echo "--- Step 3: Run pipeline gate ---"
set +e
OUTPUT=$($CLI pipeline --config "$PROJECT_DIR/.pipeline_run.yaml" --ci 2>/dev/null)
EXIT_CODE=$?
set -e

echo "$OUTPUT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    # Pipeline --ci outputs the decision dict directly
    rec = data.get('recommendation', 'N/A')
    ec = data.get('exit_code', 'N/A')
    metrics = data.get('metrics', {})
    print(f'  Decision:      {rec}')
    print(f'  Exit code:     {ec}')
    print(f'  F1:            {metrics.get(\"f1\", \"N/A\")}')
except Exception:
    print('  (raw output)')
" 2>/dev/null || echo "  Exit code: $EXIT_CODE"

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "================================================================"
    echo "  Quickstart -- ALL PASSED"
    echo "================================================================"
else
    echo "================================================================"
    echo "  Quickstart -- FAILED (exit code $EXIT_CODE)"
    echo "================================================================"
    exit $EXIT_CODE
fi
'''

README_CONTENT = """# Verifily Quickstart Project

## 3 commands to get started

```bash
# 1. Install verifily (if not already)
pip install -e /path/to/verifily-dev

# 2. Run the demo
bash scripts/run_demo.sh

# 3. Check the results
cat data/artifact/report.json | python3 -m json.tool
```

## What this does

1. **Ingest** — converts `data/raw/sample.csv` into canonical dataset artifacts
2. **Pipeline** — runs CONTRACT → REPORT → CONTAMINATION → DECISION gate
3. **Result** — prints SHIP/DONT_SHIP decision with exit code

## Project structure

```
verifily.yaml           # Pipeline configuration
data/raw/sample.csv     # Sample training data (12 rows)
data/raw/eval.jsonl     # Evaluation data (5 rows)
runs/baseline/          # Pre-built baseline run (passes contract)
scripts/run_demo.sh     # One-command demo
```
"""


# ── Public API ─────────────────────────────────────────────────────

def scaffold(path: str, *, force: bool = False) -> Dict[str, Any]:
    """Scaffold a quickstart project directory.

    Returns:
        Dict with project_dir, created_paths, next_steps.
    """
    project = Path(path).resolve()

    if project.exists() and not force:
        raise FileExistsError(
            f"Directory already exists: {project}\n"
            "Use --force to overwrite."
        )

    if project.exists() and force:
        shutil.rmtree(project)

    # Create directories
    ensure_dir(project / "data" / "raw")
    ensure_dir(project / "runs" / "baseline" / "eval")
    ensure_dir(project / "scripts")

    created: List[str] = []

    # 1. Sample CSV
    csv_path = project / "data" / "raw" / "sample.csv"
    csv_content = SAMPLE_CSV_HEADER + "\n" + "\n".join(SAMPLE_CSV_ROWS) + "\n"
    csv_path.write_text(csv_content, encoding="utf-8")
    created.append("data/raw/sample.csv")

    # 2. Eval JSONL
    eval_path = project / "data" / "raw" / "eval.jsonl"
    eval_content = "\n".join(json.dumps(r, separators=(",", ":")) for r in EVAL_ROWS) + "\n"
    eval_path.write_text(eval_content, encoding="utf-8")
    created.append("data/raw/eval.jsonl")

    # 3. Baseline run artifacts
    _write_baseline_run(project / "runs" / "baseline")
    created.extend([
        "runs/baseline/config.yaml",
        "runs/baseline/environment.json",
        "runs/baseline/run_meta.json",
        "runs/baseline/eval/eval_results.json",
        "runs/baseline/hashes.json",
    ])

    # 4. verifily.yaml
    config = {
        "schema": "sft",
        "ingest": {
            "mapping": {"question": "subject", "answer": "resolution", "context": "body"},
            "tags": {"source": "quickstart"},
        },
        "ship_if": {
            "min_f1": 0.50,
            "min_exact_match": 0.40,
            "max_f1_regression": 0.05,
            "max_pii_hits": 0,
        },
    }
    write_yaml(project / "verifily.yaml", config)
    created.append("verifily.yaml")

    # 5. Demo script
    script_path = project / "scripts" / "run_demo.sh"
    script_path.write_text(RUN_DEMO_SCRIPT, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    created.append("scripts/run_demo.sh")

    # 6. README
    readme_path = project / "README.md"
    readme_path.write_text(README_CONTENT, encoding="utf-8")
    created.append("README.md")

    next_steps = [
        f"cd {project}",
        "bash scripts/run_demo.sh",
        "cat data/artifact/report.json | python3 -m json.tool",
    ]

    return {
        "project_dir": str(project),
        "created_paths": created,
        "next_steps": next_steps,
    }


def _write_baseline_run(run_dir: Path) -> None:
    """Write a minimal contract-passing baseline run."""
    # config.yaml
    config = {
        "task": "sft",
        "base_model": "mock/quickstart-base",
        "seed": 42,
        "data_paths": {"train": "data/artifact/dataset.jsonl", "test": "data/raw/eval.jsonl"},
        "training": {"num_epochs": 3, "batch_size": 8, "learning_rate": 0.0002},
    }
    write_yaml(run_dir / "config.yaml", config)

    # environment.json
    env = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "gpu": "none",
        "verifily_version": "quickstart",
    }
    write_json(run_dir / "environment.json", env)

    # eval/eval_results.json
    eval_results = {
        "run_id": "baseline",
        "test_data_path": "data/raw/eval.jsonl",
        "num_examples": 5,
        "overall": {"exact_match": 0.60, "f1": 0.72},
        "slices": {},
        "hard_examples": [],
        "eval_duration_seconds": 0.01,
    }
    write_json(run_dir / "eval" / "eval_results.json", eval_results)

    # run_meta.json
    run_meta = {
        "run_id": "baseline",
        "status": "completed",
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": 42,
    }
    write_json(run_dir / "run_meta.json", run_meta)

    # hashes.json — computed from the files we just wrote
    file_hashes = {}
    for fp in sorted(run_dir.rglob("*")):
        if fp.is_file() and fp.name != "hashes.json":
            rel = str(fp.relative_to(run_dir))
            file_hashes[rel] = sha256_file(str(fp))

    chain_hash = compute_chain_hash(file_hashes)
    hashes = {"files": file_hashes, "chain_hash": chain_hash}
    write_json(run_dir / "hashes.json", hashes)
