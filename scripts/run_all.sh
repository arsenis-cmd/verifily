#!/bin/bash
# Complete end-to-end pipeline: datasets -> training -> evaluation -> plots -> report
# Usage: bash scripts/run_all.sh

set -e  # Exit on error

echo "=========================================="
echo "HUMAN-SYNTHETIC DATA QUALITY EXPERIMENT"
echo "Complete End-to-End Pipeline"
echo "=========================================="
echo ""

# Configuration
CONFIG_FILE="configs/base.yaml"

# Step 1: Build human-only seed dataset
echo "Step 1/8: Building human-only seed dataset..."
python -m src.data_builders --config "$CONFIG_FILE"
echo "✓ Human seed dataset created"
echo ""

# Step 2: Run sanity checks on human data
echo "Step 2/8: Running sanity checks on human data..."
python -m src.sanity_check --config "$CONFIG_FILE"
echo "✓ Sanity checks completed"
echo ""

# Step 3: Generate AI-contaminated dataset
echo "Step 3/8: Generating AI-contaminated dataset..."
python -m src.contaminate --config "$CONFIG_FILE"
echo "✓ AI-contaminated dataset created"
echo ""

# Step 4: Generate synthetic dataset from human seed
echo "Step 4/8: Generating synthetic dataset from human seed..."
python -m src.synthesize --config "$CONFIG_FILE"
echo "✓ Synthetic dataset created"
echo ""

# Step 5: Run final sanity checks on all datasets
echo "Step 5/8: Running sanity checks on all datasets..."
python -m src.sanity_check --config "$CONFIG_FILE"
echo "✓ All datasets validated"
echo ""

# Step 6: Train all three models
echo "Step 6/8: Training all three models..."
bash scripts/run_train_all.sh
echo "✓ All models trained"
echo ""

# Step 7: Evaluate all models on human test set
echo "Step 7/8: Evaluating all models..."
python -m src.eval --config "$CONFIG_FILE"
echo "✓ Evaluation completed"
echo ""

# Step 8: Generate plots and visualizations
echo "Step 8/8: Generating plots..."
python -m src.plots --config "$CONFIG_FILE"
echo "✓ Plots generated"
echo ""

echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Metrics: results/metrics_table.csv"
echo "  - Detailed: results/metrics.jsonl"
echo "  - Plots: results/plots/"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Check plots in results/plots/"
echo "  3. Update report/report.md with findings"
echo ""
