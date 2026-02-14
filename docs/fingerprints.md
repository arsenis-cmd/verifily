# Dataset Fingerprinting

Privacy-safe dataset comparison without exposing raw data.

## What it is

A **dataset fingerprint** is a compact summary that captures the statistical shape of a dataset — without storing any raw text. Fingerprints enable:

- Detecting near-duplicate datasets
- Tracking dataset drift over time
- Comparing training datasets across teams without sharing data

## What it contains

| Field | Description |
|-------|-------------|
| `version` | Fingerprint format version ("v1") |
| `schema` | Detected schema (sft, classification, chat, unknown) |
| `rows` | Number of rows |
| `tag_distribution` | Tag counts (e.g., `{"source": {"support": 10}}`) |
| `length_stats` | Token length statistics: min, median, p95, max |
| `exact_dup_rate` | Fraction of rows that are exact duplicates |
| `minhash_signature` | 128-value MinHash for Jaccard similarity estimation |
| `exact_hash_sketch` | Top 200 truncated SHA-256 hashes for overlap estimation |
| `created_at` | ISO timestamp |

**No raw text** is stored. The fingerprint is safe to share across teams.

## Commands

### Compute a fingerprint

```bash
verifily fingerprint --dataset data/artifact

# Output to specific directory
verifily fingerprint --dataset data.jsonl --out /tmp/fp

# JSON to stdout
verifily fingerprint --dataset data/artifact --json
```

Writes `fingerprint.json` alongside the dataset (or to `--out` directory).

### Compare two datasets

```bash
verifily diff-datasets data/artifact_a data/artifact_b

# With output files
verifily diff-datasets ds1.jsonl ds2.jsonl --out /tmp/diff

# JSON output
verifily diff-datasets ds1.jsonl ds2.jsonl --json
```

Produces:
- `similarity_score` (0..1): MinHash Jaccard estimate of content overlap
- `exact_dup_overlap_estimate` (0..1): Jaccard on exact hash sketches
- `row_delta`: Difference in row counts
- `dup_rate_delta`: Difference in duplicate rates
- `length_deltas`: Per-stat token length differences
- `tag_deltas`: Tag distribution changes

## Examples

### Near-duplicate detection

```bash
$ verifily diff-datasets examples/fingerprint_demo/ds_a.jsonl \
                         examples/fingerprint_demo/ds_b_neardup.jsonl --json

{
  "similarity_score": 0.6953,
  "exact_dup_overlap_estimate": 0.5,
  "row_delta": 0,
  ...
}
```

High similarity (> 0.5) indicates the datasets share significant content.

### Disjoint datasets

```bash
$ verifily diff-datasets examples/fingerprint_demo/ds_a.jsonl \
                         examples/fingerprint_demo/ds_c_disjoint.jsonl --json

{
  "similarity_score": 0.0,
  "exact_dup_overlap_estimate": 0.0,
  "row_delta": 0,
  ...
}
```

Low similarity (< 0.1) confirms the datasets are unrelated.

## Privacy guarantees

- **No raw text is stored or logged.** Fingerprints contain only hashes, statistical summaries, and numeric sketches.
- **Only hashes and sketches are produced.** Hash sketches are truncated SHA-256 prefixes (8 bytes) — irreversible.
- **Diff outputs contain only numeric similarity scores and deltas.** No example rows or text snippets are included.
- **Deterministic with fixed seed.** Default seed is 42; identical inputs always produce identical fingerprints.
- **Stable normalization.** Text is lowercased and whitespace-split before hashing. Word-level 3-gram shingles are used for MinHash.

## How it works

1. **Text extraction**: Each row's text fields are concatenated (same pattern as contamination detection)
2. **Normalization**: Text is lowercased and whitespace-split for stable tokenization
3. **Token counting**: Whitespace-split token counts produce length statistics
4. **Exact hashing**: SHA-256 of each row's text identifies exact duplicates; top 200 truncated hashes (8-byte prefixes) form the overlap sketch
5. **MinHash**: Universal hash family `h_i(x) = (a_i * hash(x) + b_i) mod p` with 128 permutations produces a compact Jaccard signature. Each shingle is hashed once with SHA-256, then mapped through 128 hash functions.
6. **Determinism**: Fixed seed (default: 42) ensures identical inputs always produce identical fingerprints

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `VERIFILY_FINGERPRINT_PERMS` | `128` | Number of MinHash permutations (lower = faster, less accurate) |

## Demo

```bash
bash scripts/demo_fingerprint.sh
```
