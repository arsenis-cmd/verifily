# Customer Drill

End-to-end demo that simulates a real customer bringing messy CSV data through the full Verifily pipeline: ingest, contract-check, contamination gate, and ship/don't-ship decision.

## Data

`raw/support_tickets.csv` — 30 rows of realistic customer support tickets with:
- 2 exact duplicate tickets (same content, different ticket IDs)
- 2 near-duplicate tickets (small word swaps)
- 2 rows with empty required fields (dropped during ingest)
- 2 rows containing PII patterns (email, phone number)

After ingest: 28 valid rows, 2 dropped.

## Eval sets

| File | Rows | Purpose |
|------|------|---------|
| `eval_clean.jsonl` | 10 | No overlap with training data |
| `eval_leaked_exact.jsonl` | 10 | 4 exact copies of training rows |
| `eval_leaked_near.jsonl` | 10 | 3 near-duplicates (Jaccard > 0.70) |

## Drill scenarios

### Clean drill (SHIP)
```bash
bash scripts/demo_customer_drill_clean.sh
```
Ingest → contract PASS → contamination PASS (exit 0) → pipeline SHIP (exit 0)

### Leaked drill (DONT_SHIP)
```bash
bash scripts/demo_customer_drill_leaked.sh
```
Ingest → contract PASS → contamination FAIL (exit 1) → pipeline DONT_SHIP (exit 1)

### CI drill (both scenarios)
```bash
bash scripts/demo_customer_drill_ci.sh
```
Runs both drills plus near-duplicate WARN check. Shows all exit codes.

## Ingest mapping

```
--map question:subject --map answer:resolution --map context:body --tag source:customer_drill
```

Canonical format: `input = "Context:\n{body}\n\nQuestion:\n{subject}"`, `output = "{resolution}"`
