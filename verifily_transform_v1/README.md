# Verifily Transform v1

Raw data to training-ready datasets in one command.

## What it does

Verifily Transform takes messy, raw, unlabeled human data and produces clean, deduplicated, labeled dataset artifacts ready for fine-tuning with Verifily Train.

**Pipeline:** Ingest -> Normalize -> Label -> Synthesize -> Dedupe -> Filter -> Package

## Quick Start

```bash
cd verifily_transform_v1
pip3 install -e .

# Run a transform pipeline
python3 -m verifily_transform run --config configs/transform_example.yaml
```

## Output

```
datasets/customer_support_v1/
├── dataset.jsonl          # Training-ready JSONL
├── manifest.json          # Dataset metadata + hashes
├── hashes.json            # Integrity chain
└── transform_meta.json    # Full run metadata
```

Drop directly into Verifily Train:

```bash
verifily train --config train.yaml --data datasets/customer_support_v1/dataset.jsonl
```

## Relationship to Train

Transform is a **sibling** to Train, not a dependency. It produces dataset artifacts that Train consumes, but the two modules are completely independent.
