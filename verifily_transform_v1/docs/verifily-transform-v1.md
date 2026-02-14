# Verifily Transform v1 — Specification

## Overview

Verifily Transform is the data preparation layer of the Verifily platform. It takes raw, messy, potentially unlabeled data and produces clean, versioned, training-ready datasets.

## Architecture

```
Raw Data (JSONL/CSV/TXT)
    |
    v
[Ingest] --- read files, detect format
    |
    v
[Normalize] --- map to canonical schema (SFT or CLS)
    |
    v
[Label] --- heuristic rules + LLM weak supervision
    |
    v
[Synthesize] --- LLM-generated variations (optional)
    |
    v
[Deduplicate] --- exact hash + fuzzy MinHash LSH
    |
    v
[Filter] --- quality, length, leakage, PII, toxicity
    |
    v
[Package] --- dataset.jsonl + manifest.json + hashes.json
```

## Supported Input Formats

| Format | Description |
|--------|-------------|
| JSONL  | One JSON object per line. Any schema. |
| CSV    | Header row + data rows. Auto-maps columns. |
| TXT    | One text sample per line. |

## Output Schema

### SFT Task
```json
{"instruction": "...", "output": "...", "input": "...(optional)", "tags": {"_origin": "human"}}
```

### Classification Task
```json
{"text": "...", "label": "...", "tags": {"_origin": "human"}}
```

## Pipeline Stages

### 1. Ingest
- Reads single files or directories
- Supports JSONL, CSV, TXT
- Configurable encoding

### 2. Normalize
- Auto-detects field mapping (e.g., "question" -> "instruction", "answer" -> "output")
- Maps to canonical schema
- Drops rows missing required fields

### 3. Label
- **Heuristic:** regex-based Q&A extraction, keyword classification
- **LLM:** API calls to GPT-4o-mini or similar for weak supervision
- **Heuristic+LLM:** heuristic first, LLM for uncertain cases

### 4. Synthesize
- LLM-backed generation of new examples from seed data
- Configurable expansion factor (e.g., 5x)
- Each synthetic row tagged with `_origin: synthetic` and `_model: gpt-4o-mini`

### 5. Deduplicate
- **Exact:** MD5 hash of content
- **Fuzzy:** MinHash LSH with configurable similarity threshold
- Falls back to brute-force Jaccard if datasketch not installed

### 6. Filter
- Empty content
- Length bounds (min/max)
- Bad characters / encoding artifacts
- Repetitive content (word frequency, trigram analysis)
- Leakage detection (overlap with seed data)
- PII detection and redaction

### 7. Package
- Writes immutable artifact directory
- `dataset.jsonl` — the final training data
- `manifest.json` — metadata, tag distribution, file hashes
- `hashes.json` — integrity chain
- `transform_meta.json` — full pipeline stats
- `transform_config.yaml` — frozen config snapshot

## Privacy

- Regex-based PII detection: email, phone, SSN, credit card, IP address
- Configurable redaction with `[TYPE_REDACTED]` placeholders
- Optional audit log tracking all redactions

## Configuration

See `configs/transform_example.yaml` for the full schema.

## Contract with Train v1

Transform outputs are **drop-in compatible** with Train v1:
- JSONL format with `instruction`/`output` (SFT) or `text`/`label` (CLS)
- `manifest.json` compatible with `DatasetVersion.from_manifest()`
- Content hashes for reproducibility verification

Transform does NOT import, call, or depend on Train v1 code.
