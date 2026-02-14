#!/bin/bash
# Train all three models: Model A (human), Model B (contaminated), Model C (synthetic)
# Usage: bash scripts/run_train_all.sh

set -e  # Exit on error

echo "=========================================="
echo "TRAINING ALL THREE MODELS"
echo "=========================================="
echo ""

CONFIG_FILE="configs/base.yaml"

# Model A: Train on human-only seed data
echo "Training Model A (Human-Only Baseline)..."
echo "----------------------------------------"
python -m src.train \
    --config "$CONFIG_FILE" \
    --model-id "model_a_human" \
    --train-data "data/processed/human_train.jsonl" \
    --val-data "data/processed/human_val.jsonl"

echo ""
echo "✓ Model A training completed"
echo ""

# Model B: Train on AI-contaminated data
echo "Training Model B (AI-Contaminated)..."
echo "----------------------------------------"
python -m src.train \
    --config "$CONFIG_FILE" \
    --model-id "model_b_contaminated" \
    --train-data "data/processed/ai_contaminated_train.jsonl" \
    --val-data "data/processed/human_val.jsonl"

echo ""
echo "✓ Model B training completed"
echo ""

# Model C: Train on synthetic data generated from human seed
echo "Training Model C (Synthetic from Human)..."
echo "----------------------------------------"
python -m src.train \
    --config "$CONFIG_FILE" \
    --model-id "model_c_synthetic" \
    --train-data "data/synthetic/synthetic_train.jsonl" \
    --val-data "data/processed/human_val.jsonl"

echo ""
echo "✓ Model C training completed"
echo ""

echo "=========================================="
echo "ALL MODELS TRAINED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Model checkpoints saved to:"
echo "  - runs/model_a_human/"
echo "  - runs/model_b_contaminated/"
echo "  - runs/model_c_synthetic/"
echo ""
