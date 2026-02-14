"""Quickstart: using Verifily Train via the Python API.

This script demonstrates the full workflow:
  1. Train a model on a dataset
  2. Evaluate the run
  3. Compare two runs
  4. Verify reproducibility
"""

from verifily_train import TrainConfig, train, evaluate, compare, verify

# ── Step 1: Train ─────────────────────────────────────────────────────
config = TrainConfig.from_yaml("examples/verifily_train/train_sft.yaml")
run = train(config)

print(f"Training done: {run.run_id}")
print(f"  Loss: {run.metrics.get('train_loss')}")
print(f"  Artifacts: {run.artifact_path}")

# ── Step 2: Evaluate ──────────────────────────────────────────────────
result = evaluate(
    run_path=run.artifact_path,
    test_data="path/to/test.jsonl",      # replace with your test data
    slice_by=["source"],
    hard_examples_n=20,
)

print(f"Eval: {result.overall}")
for tag, metrics in result.slices.items():
    print(f"  Slice [{tag}]: {metrics}")

# ── Step 3: Compare two runs ─────────────────────────────────────────
# Suppose you trained a second run with different data:
# run2 = train(config2)
# comparison = compare(
#     run_paths=[run.artifact_path, run2.artifact_path],
#     metric="f1",
#     slice_by="source",
# )
# print(comparison.overall)
# print(comparison.deltas)

# ── Step 4: Verify reproducibility ───────────────────────────────────
repro = verify(run.artifact_path)
print(f"Reproducibility: {repro.verdict}")
