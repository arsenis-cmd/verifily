#!/bin/bash
# Evaluate all trained models and generate plots
# Usage: bash scripts/run_eval.sh

set -e  # Exit on error

echo "=========================================="
echo "EVALUATION AND VISUALIZATION"
echo "=========================================="
echo ""

CONFIG_FILE="configs/base.yaml"

# Evaluate all models
echo "Evaluating all models on human test set..."
python -m src.eval --config "$CONFIG_FILE"

echo ""
echo "✓ Evaluation completed"
echo ""

# Generate plots
echo "Generating plots and visualizations..."
python -m src.plots --config "$CONFIG_FILE"

echo ""
echo "✓ Plots generated"
echo ""

echo "=========================================="
echo "EVALUATION COMPLETED!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Metrics table: results/metrics_table.csv"
echo "  - Detailed metrics: results/metrics.jsonl"
echo "  - Plots: results/plots/"
echo ""
