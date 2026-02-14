# Guarantees and Invariants

These are the guarantees Verifily makes. They are tested, enforced, and stable within the v1.x release line.

## Data Privacy

**Verifily never logs raw dataset rows.**
Audit logs, fingerprints, and decision artifacts contain only hashes, statistical summaries, and operation metadata. No raw text from datasets appears in any Verifily output.

**Fingerprints contain no reversible content.**
MinHash signatures use SHA-256-derived universal hashing. Exact hash sketches are truncated to 8-byte prefixes. Neither can be reversed to recover original text.

**PII detection is enforced in the decision gate.**
When `max_pii_hits` is configured, PII violations are hard blockers. SHIP is impossible with PII above the threshold.

## Determinism

**All decisions are reproducible given the same inputs and seed.**
The decision gate is a pure function of its inputs: contract result, contamination result, report result, eval results, baseline results, and ship criteria. No randomness, no time-based behavior, no external state.

**Fingerprints are deterministic with a fixed seed.**
The default seed is 42. Identical datasets with identical seeds always produce identical fingerprints.

**Decision artifacts are deterministic.**
Given the same pipeline config and data, the decision JSON (recommendation, exit code, confidence, reasons, risk flags, blocker precedence) is identical across runs. Only timestamps and run IDs vary.

## Contamination Detection

**Contamination detection is content-based, not metadata-based.**
Verifily compares actual text content between train and eval datasets using exact SHA-256 matching and character n-gram Jaccard similarity. Metadata fields (tags, IDs) are excluded from comparison.

**Exact overlaps always trigger FAIL.**
Any exact text match between train and eval data produces contamination status FAIL. There is no threshold below which exact overlaps are ignored (default `exact_threshold=0.0`).

## Decision Gate

**SHIP means no blockers.**
A SHIP decision is only possible when zero blockers are present. Any blocker — contract failure, PII violation, contamination FAIL, missing eval, or metric threshold violation — forces DONT_SHIP.

**Blockers follow strict precedence.**
Contract invalid > PII violation > contamination FAIL > eval missing > metric thresholds. This order is deterministic and documented.

**INVESTIGATE requires no blockers.**
INVESTIGATE is only issued when risk flags exist but no hard blockers are present. It is never used to soften a DONT_SHIP.

## Exit Codes

**Exit codes are a stable API.**

| Code | Meaning |
|------|---------|
| 0 | SHIP |
| 1 | DONT_SHIP |
| 2 | INVESTIGATE |
| 3 | CONTRACT_FAIL |
| 4 | TOOL_ERROR |

These semantics are frozen for the v1.x release line.

## Contracts

**Run contracts are validated structurally.**
Required files (config.yaml, hashes.json, environment.json) must exist. Eval results must be present. Run status must be "completed". Hash chain integrity is verified.

**Contract validation does not execute arbitrary code.**
Validation reads and parses JSON/YAML files. No code from the run directory is imported or executed.

## API

**Error responses use a stable envelope.**
All API errors return `{"error": {"type": "...", "message": "...", "request_id": "..."}}`. This format is frozen for v1.x.

**No raw data in API responses.**
Pipeline, contamination, and report API responses contain only computed results (scores, counts, statistics). No raw dataset rows are included in any response body.
