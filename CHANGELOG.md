# Changelog

All notable changes to Verifily are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — see [VERSIONING.md](VERSIONING.md).

## [1.0.0] — Initial Stable Release

### Added

#### Core Pipeline
- **Dataset ingestion** with schema detection (SFT, classification, chat), field mapping, and JSONL artifact output
- **Contamination and leakage detection** via n-gram overlap, exact-match hashing, and configurable thresholds
- **Run contracts** (`run_contract_v1`) — reproducibility metadata with SHA-256 chain hashes, environment snapshots, and structural validation
- **Decision gate** producing `SHIP`, `INVESTIGATE`, or `DONT_SHIP` recommendations with deterministic exit codes (0, 1, 2)
- **Pipeline command** (`verifily pipeline --ci`) for single-command CI gating with JSON output

#### Quality & Comparison
- **Dataset fingerprinting** — privacy-safe MinHash signatures, length statistics, exact-hash sketches, and tag distributions with no raw text stored
- **Diff-datasets** — Jaccard similarity estimation, exact-duplicate overlap, row/length/tag deltas between any two datasets
- **Reports** with contract validation, contamination results, and optional fingerprint summaries

#### API & SDK
- **REST API** (`verifily serve`) on `/v1` namespace with structured error envelope (`{error: {type, message, request_id}}`)
- **Python SDK** (`verifily_sdk`) for programmatic access to all API endpoints
- **Async jobs** for long-running operations (classify, retrain) with submit/poll/drain lifecycle

#### Operations
- **Monitor** — continuous gating that watches for drift and triggers alerts
- **Retrain** — job type for retraining workflows with environment validation
- **Audit logging** — append-only JSONL event log for all API operations
- **Usage metering** — per-key request accounting with optional persistence
- **Rate limiting** — per-key fixed-window RPM limiter
- **Structured JSON logging** — machine-readable request logs for observability

#### Developer Experience
- **Quickstart** (`verifily quickstart`) — scaffold a working project in seconds, no network or GPU required
- **CI init** (`verifily ci-init`) — generate GitHub Actions or GitLab CI config files
- **Demo scripts** for quickstart, CI gate, and fingerprinting workflows

### Stability Guarantees

The following are stable within the v1.x line and will not change without a MAJOR version bump:

- Run contract v1 schema and validation rules
- API `/v1` response envelope format
- CLI exit codes (0 = SHIP, 1 = NO-SHIP, 2 = ERROR)
- Decision semantics (SHIP / INVESTIGATE / DONT_SHIP)
- Artifact directory structure (dataset.jsonl, fingerprint.json, report.json, etc.)
- Fingerprint format v1 (deterministic with fixed seed)

See [VERSIONING.md](VERSIONING.md) for the full stability policy.
