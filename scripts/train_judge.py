#!/usr/bin/env python3
"""Train the quality judge classifier head.

Requires: torch, transformers (pip install verifily[ml])

Trains a linear head on top of frozen transformer embeddings
to predict text quality scores.

Usage:
    python scripts/train_judge.py --output verifily_cli_v1/core/judge_weights.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

_HIGH_QUALITY = [
    "Gradient descent iteratively adjusts model parameters by computing the gradient of the loss function and moving in the opposite direction to minimize error.",
    "A convolutional neural network consists of convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers.",
    "Transfer learning reuses a model trained on one task as a starting point for a different but related task, saving training time and data requirements.",
    "Batch normalization normalizes layer inputs, reducing internal covariate shift, allowing higher learning rates, and acting as mild regularization.",
    "The attention mechanism allows models to focus on relevant parts of the input sequence, computing weighted sums of values based on query-key similarity scores.",
    "Backpropagation uses the chain rule to compute gradients layer by layer, propagating error signals from the output back through the network efficiently.",
    "Random forests aggregate predictions from many decision trees trained on random subsets of data and features, reducing variance through ensemble averaging.",
    "Cross-validation splits data into multiple folds, training and testing on different combinations to get a reliable estimate of model generalization performance.",
    "Autoencoders learn compressed representations by encoding inputs to a lower-dimensional latent space and reconstructing them for dimensionality reduction.",
    "The Adam optimizer combines momentum and adaptive learning rates, maintaining running averages of both gradients and squared gradients for each parameter.",
]

_LOW_QUALITY = [
    "",
    "ok",
    "buy now buy now buy now buy now buy now buy now",
    "asdfghjkl qwerty zxcvbnm",
    "Ã¢â€ broken encoding text here",
    "a",
    "test test test test test test test test test",
    "   ",
    "n/a",
    "........",
]


def generate_training_data(n: int = 200, seed: int = 42):
    """Generate (text, label) pairs for training."""
    rng = random.Random(seed)
    texts, labels = [], []

    for _ in range(n // 2):
        t = rng.choice(_HIGH_QUALITY)
        suffix = f" This is variant {rng.randint(0, 1000)}."
        texts.append(t + suffix)
        labels.append(1.0)

    for _ in range(n // 2):
        t = rng.choice(_LOW_QUALITY)
        texts.append(t)
        labels.append(0.0)

    # Shuffle
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_head(embedder, texts, labels, epochs=50, lr=0.01):
    """Train a linear head on frozen embeddings."""
    print("  Generating embeddings...")
    embeddings = embedder.embed(texts, batch_size=16)
    dim = len(embeddings[0])
    n = len(embeddings)

    # Initialize weights
    rng = random.Random(42)
    weights = [rng.gauss(0, 0.01) for _ in range(dim)]
    bias = 0.0

    print(f"  Training linear head ({dim} -> 1) for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        grad_w = [0.0] * dim
        grad_b = 0.0

        for i in range(n):
            z = sum(w * x for w, x in zip(weights, embeddings[i])) + bias
            if z >= 0:
                pred = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                pred = ez / (1.0 + ez)

            error = pred - labels[i]
            pred_clip = max(1e-7, min(1 - 1e-7, pred))
            loss = -(labels[i] * math.log(pred_clip) + (1 - labels[i]) * math.log(1 - pred_clip))
            total_loss += loss

            for j in range(dim):
                grad_w[j] += error * embeddings[i][j]
            grad_b += error

        for j in range(dim):
            weights[j] -= lr * grad_w[j] / n
        bias -= lr * grad_b / n

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}/{epochs}: loss = {total_loss / n:.4f}")

    # Evaluate
    correct = 0
    for i in range(n):
        z = sum(w * x for w, x in zip(weights, embeddings[i])) + bias
        pred = 1.0 / (1.0 + math.exp(-min(500, max(-500, z))))
        if (pred >= 0.5) == (labels[i] >= 0.5):
            correct += 1
    print(f"  Training accuracy: {correct / n:.1%}")

    return weights, bias


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train quality judge head")
    parser.add_argument("--output", default="judge_weights.json", help="Output path")
    parser.add_argument("--n", type=int, default=200, help="Training examples")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    print("=" * 60)
    print("Verifily Quality Judge — Training")
    print("=" * 60)

    print("\n1. Checking dependencies...")
    from verifily_cli_v1.core.embeddings import get_embedder, _has_torch
    if not _has_torch():
        print("ERROR: torch not installed. Run: pip install verifily[ml]")
        sys.exit(1)

    print("\n2. Generating training data...")
    texts, labels = generate_training_data(n=args.n)
    print(f"   {len(texts)} examples ({sum(labels):.0f} positive, {len(labels) - sum(labels):.0f} negative)")

    print("\n3. Loading embedder...")
    embedder = get_embedder(prefer="auto")
    print(f"   Backend: {embedder.backend}")

    print("\n4. Training classifier head...")
    weights, bias = train_head(embedder, texts, labels, epochs=args.epochs)

    print(f"\n5. Saving weights to {args.output}...")
    data = {"weights": weights, "bias": bias, "dim": len(weights)}
    with open(args.output, "w") as f:
        json.dump(data, f)
    print(f"   Saved ({len(weights)} weights)")

    print("\n" + "=" * 60)
    print("Done! Use with: judge_quality(texts, model_path='judge_weights.json')")
    print("=" * 60)


if __name__ == "__main__":
    main()
