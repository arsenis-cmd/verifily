# Classify Demo — 60-Second Run

Demonstrates the Verifily CLASSIFY job type on a messy mixed dataset.

## Dataset

`raw/mixed_dump.csv` contains 16 rows mixing:
- QA-style rows (question/context/answer)
- Support tickets with PII-like patterns (emails, phone numbers)
- Duplicate rows (3 exact copies of one entry)
- Multiple categories (qa_geography, qa_tech, qa_science, support)

## Run via API

```bash
bash scripts/demo_api_jobs_classify.sh
```

## Run via pytest

```bash
pytest verifily_cli_v1/tests/test_api_jobs_classify.py -v
pytest verifily_sdk/tests/test_sdk_jobs_classify.py -v
```

## Expected Output

- Suggested schema: `qa`
- Buckets: qa_geography, qa_tech, support, qa_science
- PII risk: emails and phone patterns detected (counts only, never raw)
- Duplicate rate: ~12.5% (2 exact duplicates out of 16 rows)
