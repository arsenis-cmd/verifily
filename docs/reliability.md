# Reliability

This document defines Verifily's reliability guarantees — what the system promises about exit codes, failure behavior, artifacts, determinism, and subsystem safety.

## Exit Codes Contract

Verifily CLI exit codes are a stable API. They do not change within a major version.

| Code | Meaning | When |
|------|---------|------|
| 0 | SHIP | No blockers, all criteria met |
| 1 | DONT_SHIP | One or more blockers present |
| 2 | INVESTIGATE | Risk flags present, no hard blockers |
| 3 | CONTRACT_FAIL | Run contract validation failed |
| 4 | TOOL_ERROR | Internal error, invalid config, or unhandled exception |

CI pipelines can rely on these codes for branching logic. A non-zero exit code always means "do not deploy without human review."

## Pipeline Failure Behavior

### Fail-fast on invalid config

If `ship_if` thresholds are invalid (negative values, out-of-range, wrong types), the pipeline returns `TOOL_ERROR` (exit 4) immediately — before any pipeline steps execute.

### Step failure isolation

Each pipeline step (contract, report, contamination, decision) is independent. A failure in one step does not prevent subsequent steps from running. The decision gate aggregates all results and produces a single verdict.

Exception: if the run contract is invalid, the pipeline short-circuits with `CONTRACT_FAIL` (exit 3).

### No partial writes

Artifacts are written atomically after the decision is made. If the pipeline fails before the decision step, no artifacts are written.

### Unhandled exceptions

Any unhandled exception in the pipeline command produces `TOOL_ERROR` (exit 4). The exception message is printed to stderr. No partial state is left behind.

## Artifact Write Guarantees

When `--output` is specified, the pipeline writes:

| File | Contents | Always written |
|------|----------|----------------|
| `pipeline_result.json` | Full pipeline result including decision, contamination, report | Yes |
| `decision_summary.json` | Decision recommendation, exit code, confidence, reasons, risk flags, blocker precedence | Yes |
| `audit.jsonl` | Structured audit log of all pipeline steps | Yes |
| `usage.json` | Timing and billing data | Yes |

All files are valid JSON. All files are written after the decision is made — never before.

### Contract failure artifacts

Even when the run contract is invalid, `decision_summary.json` is written (if `--output` is set). This ensures CI systems always have a machine-readable decision artifact.

## Determinism Guarantees

### Decision determinism

Given identical inputs (contract result, contamination result, report result, eval results, baseline results, ship criteria), `make_decision()` always produces the same output. No randomness, no time-based behavior, no external state.

Only `timestamps` and `run_ids` vary between runs.

### Fingerprint determinism

With a fixed seed (default: 42), identical datasets always produce identical fingerprints. The MinHash signature, exact hash sketch, and all statistics are reproducible.

### CI output determinism

In CI mode (`--ci`), the pipeline writes:
- Raw JSON to stdout (no ANSI codes, no Rich markup)
- A deterministic final line to stderr: `Decision: <REC> (exit <CODE>)`

This format is stable and machine-parseable.

## Monitor Safety

### Tick isolation

Each monitor tick runs a full pipeline independently. A failure in one tick does not affect subsequent ticks.

### Rolling window

Monitor history is bounded by `rolling_window` (default: 20 ticks). Old ticks are evicted. Memory usage is bounded.

### Stop semantics

`monitor-stop` halts the tick loop but does not discard history. History remains queryable after stop.

## Retrain Safety

### Mock mode is default

`verifily retrain` defaults to `--mode mock`. Mock mode produces deterministic metrics from a fixed seed — no real training occurs, no GPU required, no side effects.

### Real mode requires opt-in

Real training requires `VERIFILY_ENABLE_REAL_TRAIN=1` environment variable. Without it, real mode is refused.

### Retrain artifacts

Retrain always writes a complete run directory with `config.yaml`, `hashes.json`, `environment.json`, and `eval_results.json`. These artifacts are compatible with the standard pipeline contract.

## API Server Safety

### Localhost by default

The API server binds to `127.0.0.1` by default. Binding to non-localhost requires explicit `--allow-nonlocal`.

### No raw data in responses

All API responses contain only computed results (scores, counts, statistics). No raw dataset rows appear in any response body.

### Request isolation

Each API request is independent. No shared mutable state leaks between requests. Singleton stores (jobs, usage, monitor) use thread-safe access patterns.

## What Verifily Does Not Guarantee

- **Filesystem safety**: Verifily trusts the local filesystem. It does not sandbox reads or writes.
- **Encryption at rest**: Artifacts are plain JSON. Use OS-level encryption if needed.
- **Network security**: The API server is HTTP-only. Use a reverse proxy for TLS.
- **Availability**: Verifily is a CLI tool, not a service. There is no uptime SLA.
