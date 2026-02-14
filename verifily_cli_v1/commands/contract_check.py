"""verifily contract-check — validate run directory and dataset schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table

from verifily_cli_v1.core.io import read_json, read_jsonl
from verifily_cli_v1.core.schemas import SCHEMA_REGISTRY, get_schema, schema_names

console = Console(stderr=True)

# ── Schema definitions (from centralized registry) ────────────────

# Build SCHEMAS dict from registry — each required_group is treated as an
# independent valid combination (either "required" or "alternates").
# The original v1.0 SCHEMAS for SFT expected `instruction + output` as primary
# with `question + answer` and `input + output` as alternates.
SCHEMAS: Dict[str, Dict[str, Any]] = {
    "sft": {
        "required": ["instruction", "output"],
        "optional": ["input", "tags"],
        "alternates": [
            {"question": "instruction", "answer": "output", "context": "input"},
            {"input": "instruction", "output": "output"},
        ],
    },
    "classification": {
        "required": ["text", "label"],
        "optional": ["tags"],
        "alternates": [],
    },
}

# Add remaining schemas from registry (new in v1.1)
for _name, _schema_def in SCHEMA_REGISTRY.items():
    if _name in SCHEMAS:
        continue
    _required = list(_schema_def.required_groups[0]) if _schema_def.required_groups else []
    _alternates = []
    for _group in _schema_def.required_groups[1:]:
        _alternates.append({field: field for field in _group})
    for _alt in _schema_def.alternates:
        if isinstance(_alt, dict):
            _alternates.append(_alt)
    SCHEMAS[_name] = {
        "required": _required,
        "optional": list(_schema_def.optional),
        "alternates": _alternates,
    }

# Files required for a valid run contract
RUN_CONTRACT_FILES = [
    "config.yaml",
    "hashes.json",
    "environment.json",
]

# At least one of these must exist for eval completeness
EVAL_RESULT_CANDIDATES = [
    "eval/eval_results.json",
    "eval_results.json",
]


# ── Public API ──────────────────────────────────────────────────

def validate_run_contract(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Validate a run directory satisfies the Verifily run contract.

    Returns:
        {
            "valid": bool,
            "checks": [{"file": str, "status": "PASS"|"FAIL", "detail": str}],
            "has_eval": bool,
        }
    """
    d = Path(run_dir)
    checks: List[Dict[str, str]] = []
    valid = True

    if not d.is_dir():
        return {"valid": False, "checks": [{"file": str(d), "status": "FAIL", "detail": "not a directory"}], "has_eval": False}

    # Check required files
    for fname in RUN_CONTRACT_FILES:
        p = d / fname
        if p.exists():
            checks.append({"file": fname, "status": "PASS", "detail": "present"})
        else:
            checks.append({"file": fname, "status": "FAIL", "detail": "missing"})
            valid = False

    # Check eval results (at least one candidate)
    has_eval = False
    for candidate in EVAL_RESULT_CANDIDATES:
        if (d / candidate).exists():
            has_eval = True
            checks.append({"file": candidate, "status": "PASS", "detail": "present"})
            break
    if not has_eval:
        checks.append({"file": "eval/eval_results.json", "status": "FAIL", "detail": "missing (no eval results found)"})
        valid = False

    # If run_meta.json exists, validate status
    meta_path = d / "run_meta.json"
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
            status = meta.get("status", "unknown")
            if status == "completed":
                checks.append({"file": "run_meta.json", "status": "PASS", "detail": f"status={status}"})
            else:
                checks.append({"file": "run_meta.json", "status": "FAIL", "detail": f"status={status} (expected completed)"})
                valid = False
        except Exception as e:
            checks.append({"file": "run_meta.json", "status": "FAIL", "detail": f"parse error: {e}"})
            valid = False

    return {"valid": valid, "checks": checks, "has_eval": has_eval}


def validate_dataset_schema(
    dataset_path: Union[str, Path], schema: str
) -> Dict[str, Any]:
    """Validate every row of a JSONL dataset against a schema.

    Returns:
        {
            "valid": bool,
            "schema": str,
            "total_rows": int,
            "valid_rows": int,
            "errors": [{"row": int, "error": str}],
        }
    """
    if schema not in SCHEMAS:
        return {"valid": False, "schema": schema, "total_rows": 0, "valid_rows": 0,
                "errors": [{"row": 0, "error": f"unknown schema: {schema}"}]}

    spec = SCHEMAS[schema]
    rows = read_jsonl(dataset_path)
    errors: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        row_errors = _validate_row(row, spec, i)
        errors.extend(row_errors)

    return {
        "valid": len(errors) == 0,
        "schema": schema,
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(set(e["row"] for e in errors)),
        "errors": errors[:50],  # cap reported errors
    }


def _validate_row(row: Dict, spec: Dict, row_index: int) -> List[Dict]:
    """Validate a single row against schema spec. Returns error dicts."""
    errors = []
    required = spec["required"]

    # Check if direct fields present
    direct_ok = all(k in row and row[k] for k in required)
    if direct_ok:
        return []

    # Check alternates
    for alt_map in spec.get("alternates", []):
        alt_ok = all(k in row and row[k] for k in alt_map)
        if alt_ok:
            return []

    # Report what's missing
    missing = [k for k in required if k not in row or not row[k]]
    if missing:
        alt_fields = []
        for alt_map in spec.get("alternates", []):
            alt_fields.append(", ".join(alt_map.keys()))
        alt_msg = f" (or alternates: {'; '.join(alt_fields)})" if alt_fields else ""
        errors.append({
            "row": row_index,
            "error": f"missing required fields: {missing}{alt_msg}",
        })
    return errors


# ── CLI entry point ─────────────────────────────────────────────

def run(
    *,
    dataset: Optional[str] = None,
    schema: str = "sft",
    run_dir: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Run contract check. Returns True if all checks pass."""
    all_ok = True

    if run_dir:
        console.print(f"\n[bold]Run contract check:[/bold] {run_dir}\n")
        result = validate_run_contract(run_dir)
        tbl = Table(title="Run Contract", show_lines=False)
        tbl.add_column("File", style="cyan")
        tbl.add_column("Status")
        tbl.add_column("Detail", style="dim")
        for c in result["checks"]:
            status_style = "green" if c["status"] == "PASS" else "red bold"
            tbl.add_row(c["file"], f"[{status_style}]{c['status']}[/{status_style}]", c["detail"])
        console.print(tbl)

        if result["valid"]:
            console.print("\n[green bold]CONTRACT: PASS[/green bold]\n")
        else:
            console.print("\n[red bold]CONTRACT: FAIL[/red bold]\n")
            all_ok = False

    if dataset:
        console.print(f"\n[bold]Schema validation:[/bold] {dataset} (schema={schema})\n")
        result = validate_dataset_schema(dataset, schema)
        if result["valid"]:
            console.print(f"[green bold]SCHEMA: PASS[/green bold] — {result['total_rows']} rows, all valid\n")
        else:
            console.print(f"[red bold]SCHEMA: FAIL[/red bold] — {result['valid_rows']}/{result['total_rows']} valid\n")
            if verbose:
                for e in result["errors"][:20]:
                    console.print(f"  row {e['row']}: {e['error']}")
            all_ok = False

    return all_ok
