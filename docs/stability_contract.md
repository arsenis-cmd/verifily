# Stability Contract — v1.x

This document defines what is stable in the v1.x release line and what may change in minor versions.

## Stable (will not break in v1.x)

### CLI commands and flags

| Command | Stable flags |
|---------|-------------|
| `verifily pipeline` | `--config`, `--ci`, `--output`, `--verbose` |
| `verifily contamination` | `--train`, `--eval`, `--jaccard`, `--output` |
| `verifily report` | `--dataset`, `--schema`, `--output` |
| `verifily ingest` | `--in`, `--out`, `--schema`, `--map`, `--tag` |
| `verifily fingerprint` | `--dataset`, `--out`, `--json`, `--seed` |
| `verifily diff-datasets` | positional args, `--out`, `--json` |
| `verifily doctor` | `--config`, `--verbose` |
| `verifily version` | (no flags) |
| `verifily serve` | `--host`, `--port`, `--allow-nonlocal` |
| `verifily badge` | `--decision`, `--from-run`, `--out`, `--json` |
| `verifily bundle` | `--run`, `--out`, `--json` |
| `verifily --version` | (global flag) |

New flags may be added. Existing flags will not be renamed or removed.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | SHIP |
| 1 | DONT_SHIP |
| 2 | INVESTIGATE |
| 3 | CONTRACT_FAIL |
| 4 | TOOL_ERROR |

These codes are frozen for v1.x.

### Decision JSON schema

The following fields in `decision_summary.json` are stable:

```
recommendation: str    (SHIP | DONT_SHIP | INVESTIGATE)
exit_code: int         (0 | 1 | 2 | 3 | 4)
confidence: float      (0.0–1.0)
reasons: list[str]
risk_flags: list[str]
blocker_precedence: list[str]
metrics: dict
```

New fields may be added. Existing fields will not be renamed or removed.

### API endpoints

| Endpoint | Method | Stable |
|----------|--------|--------|
| `/health` | GET | Yes |
| `/ready` | GET | Yes |
| `/v1/pipeline` | POST | Yes |
| `/v1/contamination` | POST | Yes |
| `/v1/report` | POST | Yes |
| `/v1/usage` | GET | Yes |
| `/v1/jobs/*` | GET/POST | Yes |

### API error envelope

```json
{"error": {"type": "...", "message": "...", "request_id": "..."}}
```

This format is frozen for v1.x.

### Badge JSON schema

```json
{"schemaVersion": 1, "label": "...", "message": "...", "color": "..."}
```

### Bundle manifest schema

```json
{"bundle_version": 1, "source_dir": "...", "files": {"name": "sha256hex"}}
```

## May change in minor versions (1.x → 1.y)

- New CLI commands may be added
- New flags may be added to existing commands
- New fields may be added to JSON schemas (additive only)
- New API endpoints may be added
- Doctor checks may be expanded
- Confidence scoring formula may be tuned (ranges preserved)
- Default threshold values may change
- Rich output formatting may change (tables, colors, layout)
- Log message wording may change

## Will never change without a major version bump

- Removal of a CLI command or flag
- Removal of a JSON schema field
- Change of exit code semantics
- Change of API endpoint URL
- Change of error envelope format
- Removal of an API endpoint
