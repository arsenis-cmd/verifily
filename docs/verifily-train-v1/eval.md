# Verifily Train v1 -- Evaluation Framework

## Design Principles

1. **Eval is automatic**. Every `verifily train` run evaluates at the end if a test/val split is available.
2. **Slicing is the insight**. Aggregate metrics are table stakes. Tag-sliced metrics show which data segments matter.
3. **Hard examples are actionable**. The worst-performing examples, tagged with their data source, tell users where to invest in better data.
4. **Metrics are task-appropriate**. Classification and SFT have different default metric sets.

## Metrics by Task

### Classification Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| Accuracy | `accuracy` | Fraction of correct predictions |
| Macro F1 | `macro_f1` | F1 averaged across all classes (equal class weight) |
| Weighted F1 | `weighted_f1` | F1 weighted by class support |
| Per-class precision | `precision_per_class` | Dict of class -> precision |
| Per-class recall | `recall_per_class` | Dict of class -> recall |
| Confusion matrix | `confusion_matrix` | Full NxN matrix as nested list |

**Default set**: `accuracy`, `macro_f1`, `confusion_matrix`

### SFT Metrics

| Metric | Key | Description |
|--------|-----|-------------|
| Eval loss | `eval_loss` | Cross-entropy loss on test set |
| Perplexity | `perplexity` | exp(eval_loss) |
| Exact match | `exact_match` | Fraction of predictions exactly matching reference |
| Token F1 | `f1` | Token-level F1 between prediction and reference |
| ROUGE-1 | `rouge1` | Unigram overlap |
| ROUGE-L | `rougeL` | Longest common subsequence overlap |
| Format compliance | `format_compliance` | Fraction of outputs matching expected format (see below) |
| Gold set accuracy | `gold_set_accuracy` | Exact match on a small curated gold set (optional) |

**Default set**: `eval_loss`, `perplexity`, `f1`, `exact_match`

### Format Compliance (SFT)

For SFT tasks, format compliance checks whether the model output follows the expected structure. Users define format rules in eval config:

```yaml
eval:
  format_rules:
    - name: "ends_with_period"
      pattern: "\\.$"        # regex
    - name: "no_preamble"
      pattern: "^(?!Sure|Of course|I'd be happy)"  # must NOT match
      invert: true
    - name: "max_length"
      max_tokens: 256
```

Format compliance rate = fraction of test examples where **all** rules pass.

If no format rules are defined, `format_compliance` is not computed.

### Gold Set Accuracy (SFT)

For SFT tasks where exact string metrics are unreliable (e.g., open-ended generation), users can provide a small (50-200 example) gold evaluation set with unambiguous expected outputs.

```yaml
eval:
  gold_set: "data/gold_eval.jsonl"
```

Gold set examples follow the same JSONL format as training data. Exact match is computed against gold references.

## Tag-Based Slicing

### How It Works

1. Every row in the dataset has a `tags` field: `{"source": "human", "difficulty": "hard"}`.
2. During evaluation, predictions are grouped by tag values.
3. Metrics are computed per group.

### Configuration

```yaml
eval:
  slice_by_tags: ["source", "difficulty", "length_bucket"]
```

### Slice Output Format

```json
{
  "slices": {
    "source": {
      "human":        {"n": 800, "f1": 0.7312, "exact_match": 0.6100},
      "synthetic":    {"n": 700, "f1": 0.7089, "exact_match": 0.5857},
      "contaminated": {"n": 500, "f1": 0.6841, "exact_match": 0.5600}
    },
    "difficulty": {
      "easy":   {"n": 900, "f1": 0.8201},
      "medium": {"n": 700, "f1": 0.6832},
      "hard":   {"n": 400, "f1": 0.5510}
    }
  }
}
```

### Cross-Tag Slicing

v1 supports single-key slicing only. Cross-tag slicing (e.g., `source=human AND difficulty=hard`) is a v2 feature.

### Missing Tags

If a test example lacks a tag key specified in `slice_by_tags`, it is placed in a `_untagged` bucket for that key.

## Hard Examples

### Definition

Hard examples are the N test examples with the **lowest primary metric score** (F1 for SFT, confidence for classification).

### Purpose

Surface patterns in failure cases. When hard examples are tagged, users can see:
- "80% of hard examples are `source:ai_contaminated`" -- the contaminated data is hurting.
- "Most failures are `difficulty:hard` + `length_bucket:long`" -- the model struggles on long, hard inputs.

### Output

Written to `eval/hard_examples.jsonl`. Each entry contains:

```json
{
  "rank": 1,
  "primary_metric": 0.0,
  "primary_metric_name": "f1",
  "prediction": "the model's output",
  "reference": "the expected output",
  "input": "the model's input (truncated to 500 chars)",
  "tags": {"source": "synthetic", "difficulty": "hard"},
  "input_hash": "sha256:abc123..."
}
```

### Configuration

```yaml
eval:
  hard_examples: 50   # number of hard examples to output (default: 50)
```

## Dataset Attribution (v1)

### Approach: Practical, Not Causal

v1 does not attempt causal attribution (data Shapley, influence functions). These methods require O(n) retraining runs and are impractical for most users.

Instead, v1 provides three observational tools:

#### 1. Slice Metrics
Metrics broken down by dataset tags. Shows how the model performs on different data segments.

#### 2. Hard Examples with Tags
The worst predictions, annotated with their data source tags. Shows which data sources correlate with failures.

#### 3. Cross-Version Comparison
`verifily compare` shows metric deltas between runs trained on different dataset versions. The diff, sliced by tags, shows which data changes are associated with which metric changes.

### Example Attribution Workflow

```bash
# Train on dataset v2 (human only)
verifily train --config train.yaml --dataset ds_abc@v2 --name human-only

# Train on dataset v3 (human + synthetic)
verifily train --config train.yaml --dataset ds_abc@v3 --name human-plus-synthetic

# Compare
verifily compare \
  --runs runs/human-only/,runs/human-plus-synthetic/ \
  --metric f1 \
  --slice-by source
```

Output:
```
  Run                   Overall   source=human   source=synthetic
  human-only            0.7139    0.7312         N/A
  human-plus-synthetic  0.7285    0.7298         0.7271

  Delta:               +0.0146   -0.0014        N/A
```

Interpretation: Adding synthetic data improved overall F1, with the gain concentrated in the synthetic slice. Human-slice performance held steady.

### Caveats (Displayed to User)

Every attribution output includes the notice:

> Slice metrics show correlation between data segments and model performance. They do not prove causation. For rigorous causal attribution, see the Verifily Train v2 roadmap (data Shapley, leave-one-out).

## Evaluation Pipeline Internals

### Step-by-Step

1. **Load model**: Load adapter weights from run artifacts, merge with base model.
2. **Load test data**: From run config or `--test-data` override. Verify hash against manifest.
3. **Generate predictions**: Batch inference with greedy decoding (temperature=0, no sampling).
4. **Compute overall metrics**: Full test set.
5. **Compute slice metrics**: Group by each tag key, compute metrics per group.
6. **Rank hard examples**: Sort by primary metric ascending, take bottom N.
7. **Write results**: `eval/eval_results.json`, `eval/hard_examples.jsonl`.

### Performance

- Inference is the bottleneck. For 2000 test examples on an 8B model with batch_size=16: ~5-15 minutes on a single A100.
- Metric computation is negligible (<1 second for all metrics on 2000 examples).
- No GPU needed for metric computation; only for model inference.
