# verifily ingest

Convert messy CSV or JSONL files into canonical Verifily dataset artifacts.

## What it does

`verifily ingest` reads a raw data file, validates and canonicalizes every row,
generates deterministic IDs and integrity hashes, and writes a versioned artifact
directory containing:

| File | Purpose |
|------|---------|
| `dataset.jsonl` | Canonical rows (one JSON object per line) |
| `manifest.json` | Metadata: schema, row counts, drop reasons, tags, version |
| `hashes.json` | SHA-256 for every output file + dataset chain hash |
| `report.json` | Field-level stats, PII scan, exact-duplicate rate |

## Schemas

| Schema | Required fields | Auto-detect trigger |
|--------|----------------|---------------------|
| `sft` | `input` + `output` (or `question` + `answer`) | columns contain input/output or question/answer |
| `classification` | `text` + `label` | columns contain text + label |
| `auto` | — | tries sft then classification |

## Examples

### 1. CSV with column mapping (SFT)

Your CSV has columns `prompt`, `completion`, `passage` but Verifily expects
`question`, `answer`, `context`. Use `--map` to bridge them:

```bash
verifily ingest \
  --in examples/ingest_demo/data_sft.csv \
  --schema sft \
  --map question:prompt \
  --map answer:completion \
  --map context:passage \
  --out /tmp/verifily_ingest_demo
```

Expected output:

```
Ingest complete

  Output:  /tmp/verifily_ingest_demo
  Schema:  sft
  Rows in: 13
  Rows out: 12
  Dropped: 1
  Hash:    2ea3efc0e10d277ad45078fc…
```

### 2. JSONL identity ingest with extra tags

When your JSONL already has `input`/`output` fields, no mapping is needed.
Inject tags with `--tag`:

```bash
verifily ingest \
  --in examples/ingest_demo/data_sft.jsonl \
  --schema sft \
  --tag project:demo \
  --tag batch:2025-01 \
  --out /tmp/verifily_ingest_jsonl
```

### 3. Classification CSV

```bash
verifily ingest \
  --in examples/ingest_demo/data_cls.csv \
  --schema classification \
  --map text:body \
  --map label:category \
  --out /tmp/verifily_ingest_cls
```

## Flags

| Flag | Description |
|------|-------------|
| `--in PATH` | Input CSV or JSONL file (required) |
| `--out PATH` | Output directory for artifacts (required) |
| `--schema` | `sft`, `classification`, or `auto` (default: `auto`) |
| `--map canonical:source` | Column mapping, repeatable |
| `--tag key:value` | Extra tags to inject, repeatable |
| `--id-col NAME` | Use this column as row ID instead of generating one |
| `--limit N` | Only ingest the first N rows |
| `--dry-run` | Validate and preview without writing files |
| `--strict` | Fail on first invalid row instead of dropping |
| `--verbose` | Show detailed progress |

## Canonicalization rules

**SFT with question/answer/context:**
- `input` = `"Context:\n{context}\n\nQuestion:\n{question}"` (context omitted if empty)
- `output` = answer text

**SFT with input/output:**
- Passed through as-is

**Classification:**
- `text` and `label` passed through as-is

**All schemas:**
- Deterministic ID: `row_` + first 16 hex chars of SHA-256(canonical JSON)
- Tags: extra `--tag` values merged as defaults (existing row tags take priority)
- Empty required fields → row dropped (or error in `--strict` mode)

## Determinism

Running `verifily ingest` twice on the same input with the same flags produces
identical `dataset.jsonl` files, identical row IDs, and identical dataset hashes.
This is verified by the test suite (`TestIngestDeterministicIDs`).
