# Versioning & Stability Policy

This document defines how Verifily is versioned and what stability guarantees users can rely on.

## Semantic Versioning Policy

Verifily follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH`.

| Component | Incremented when |
|-----------|-----------------|
| **MAJOR** | Breaking changes to API response schema, run contract structure, CLI exit code semantics, decision semantics, or artifact format |
| **MINOR** | New backward-compatible features — new commands, new optional fields, new report sections, new job types |
| **PATCH** | Bug fixes, performance improvements, documentation updates, internal refactors with no user-visible behavior change |

The current version is in the `VERSION` file at the repository root.

## Stability Guarantees (v1)

The following interfaces are **stable** within the v1.x release line. They will not change in backward-incompatible ways without a MAJOR version bump.

### Run Contract v1

- The `run_contract_v1` schema (required fields, hash chain format, validation rules) is frozen.
- New optional fields may be added in MINOR releases. Existing fields will not be removed or retyped.
- `validate_run_contract()` will continue to accept all valid v1 contracts.

### API Response Envelope

- The `/v1` error envelope `{"error": {"type": "...", "message": "...", "request_id": "..."}}` is frozen.
- Success response shapes for existing endpoints are frozen. New fields may be added; existing fields will not be removed.
- Error types (`AUTH_ERROR`, `VALIDATION_ERROR`, `NOT_FOUND`, `RATE_LIMITED`, `INTERNAL_ERROR`) are stable.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | **SHIP** — all quality gates passed |
| `1` | **NO-SHIP** — one or more gates failed |
| `2` | **ERROR** — pipeline could not complete |

These exit code semantics are frozen for v1.x.

### Decision Semantics

The three decision values — `SHIP`, `INVESTIGATE`, `DONT_SHIP` — and their definitions are frozen. New decision values will not be added in MINOR releases.

### Artifact Structure

The following artifact file names and their formats are stable:

- `dataset.jsonl` — ingested dataset rows
- `fingerprint.json` — dataset fingerprint (format v1)
- `report.json` — quality report
- `hashes.json` — file hash manifest
- `run_meta.json` — run metadata
- `eval/eval_results.json` — evaluation results

New files may be added to artifact directories in MINOR releases. Existing files will not be renamed or restructured.

## Backward Compatibility Rules

### Contract Version Evolution

- Each contract version (e.g., `run_contract_v1`) is independently stable.
- If a new contract version is introduced (e.g., `run_contract_v2`), the previous version remains supported for at least two MINOR releases.
- Verifily will validate older contracts without error. The contract version is recorded in the artifact and used to select the appropriate validation logic.

### Handling Older Runs

- `verifily report` and `verifily pipeline` will accept artifacts produced by any v1.x release.
- If an artifact is missing a field added in a later MINOR release, Verifily will use a sensible default rather than fail.

### Deprecation vs. Failure

- Deprecated features produce a warning on stderr: `[DEPRECATED] <feature> will be removed in v<version>.`
- Deprecated features continue to work without error for the duration of the support window.
- Removal only happens in MAJOR releases, after the deprecation window has elapsed.

## Deprecation Policy

| Rule | Detail |
|------|--------|
| **Minimum support window** | Deprecated features are supported for at least **2 MINOR releases** after the deprecation notice |
| **Warning strategy** | Deprecation warnings are emitted to stderr on first use per session, not repeated |
| **Documentation** | Deprecated features are marked in CHANGELOG.md, CLI `--help` output, and relevant docs |
| **Removal** | Only in a MAJOR version bump, after the support window has elapsed |

### Example Timeline

1. v1.3.0 — Feature X deprecated, warning added
2. v1.4.0 — Feature X still works, warning persists
3. v1.5.0 — Feature X still works (minimum window met)
4. v2.0.0 — Feature X may be removed
