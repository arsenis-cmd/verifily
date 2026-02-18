#!/usr/bin/env python3
"""Train the learned quality scorer weights.

Generates synthetic datasets with known quality characteristics,
extracts features via the quality engine, and trains logistic
regression via pure Python gradient descent (no sklearn needed).

Usage:
    python scripts/train_scorer.py

Outputs Python code with weight values to paste into learned_scorer.py.
"""

from __future__ import annotations

import math
import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifily_cli_v1.core.quality import analyze_quality
from verifily_cli_v1.core.learned_scorer import extract_features


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_GOOD_TEXTS = [
    "Explain how gradient descent works in neural network training",
    "What is the difference between supervised and unsupervised learning",
    "Describe the architecture of a convolutional neural network",
    "How does backpropagation compute gradients efficiently",
    "What are the advantages of using batch normalization",
    "Explain the concept of transfer learning in deep learning",
    "How do recurrent neural networks handle sequential data",
    "What is the vanishing gradient problem and how to solve it",
    "Describe the attention mechanism in transformer models",
    "How does dropout regularization prevent overfitting",
    "What is the purpose of an activation function in neural networks",
    "Explain the difference between precision and recall metrics",
    "How do generative adversarial networks create synthetic data",
    "What is the bias-variance tradeoff in machine learning",
    "Describe how random forests combine multiple decision trees",
    "What is cross-validation and why is it important",
    "How does the Adam optimizer improve upon standard SGD",
    "Explain the concept of embedding layers for categorical data",
    "What are autoencoders and how are they used for dimensionality reduction",
    "How does early stopping help prevent overfitting during training",
]

_GOOD_ANSWERS = [
    "Gradient descent iteratively adjusts model parameters by computing the gradient of the loss function and moving in the opposite direction to minimize error.",
    "Supervised learning uses labeled data to learn mappings, while unsupervised learning discovers patterns in unlabeled data without explicit targets.",
    "A CNN consists of convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.",
    "Backpropagation uses the chain rule to compute gradients layer by layer, propagating error signals from the output back through the network.",
    "Batch normalization normalizes layer inputs, reducing internal covariate shift, allowing higher learning rates, and acting as mild regularization.",
    "Transfer learning reuses a model trained on one task as a starting point for a different but related task, saving training time and data.",
    "RNNs maintain hidden state across time steps, allowing them to process sequences by incorporating information from previous inputs.",
    "The vanishing gradient problem occurs when gradients become very small in deep networks. Solutions include ReLU activations, residual connections, and LSTM cells.",
    "The attention mechanism allows models to focus on relevant parts of the input sequence, computing weighted sums of values based on query-key similarity.",
    "Dropout randomly deactivates neurons during training, preventing co-adaptation and forcing the network to learn more robust features.",
    "Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns beyond linear transformations.",
    "Precision measures the fraction of positive predictions that are correct, while recall measures the fraction of actual positives that are found.",
    "GANs use a generator to create synthetic data and a discriminator to distinguish real from fake, training adversarially until the generator produces realistic outputs.",
    "The bias-variance tradeoff balances underfitting (high bias) and overfitting (high variance) to achieve optimal generalization performance.",
    "Random forests aggregate predictions from many decision trees trained on random subsets of data and features, reducing variance through ensemble averaging.",
    "Cross-validation splits data into multiple folds, training and testing on different combinations to get a reliable estimate of model performance.",
    "Adam combines momentum and adaptive learning rates, maintaining running averages of both gradients and squared gradients for each parameter.",
    "Embedding layers map categorical variables to dense vector representations, capturing semantic relationships in a lower-dimensional continuous space.",
    "Autoencoders learn compressed representations by encoding inputs to a lower-dimensional latent space and reconstructing them, useful for dimensionality reduction.",
    "Early stopping monitors validation loss during training and stops when it begins to increase, preventing the model from memorizing training data.",
]


def _make_good_dataset(n: int, seed: int) -> list[dict]:
    """Generate a clean SFT dataset with diverse, well-formed rows."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        q_idx = rng.randint(0, len(_GOOD_TEXTS) - 1)
        a_idx = rng.randint(0, len(_GOOD_ANSWERS) - 1)
        # Add some variation
        suffix = f" (variant {i})" if rng.random() > 0.5 else ""
        rows.append({
            "input": _GOOD_TEXTS[q_idx] + suffix,
            "output": _GOOD_ANSWERS[a_idx],
        })
    return rows


def _make_bad_dataset(n: int, seed: int, problem: str) -> list[dict]:
    """Generate a dataset with a specific quality problem."""
    rng = random.Random(seed)
    rows = []

    if problem == "empty":
        for i in range(n):
            if rng.random() < 0.4:
                rows.append({"input": "", "output": ""})
            else:
                rows.append({"input": f"question {i}", "output": f"answer {i}"})

    elif problem == "duplicates":
        base_q = "What is machine learning"
        base_a = "Machine learning is a subset of AI"
        for i in range(n):
            if rng.random() < 0.3:
                rows.append({"input": base_q, "output": base_a})
            else:
                rows.append({
                    "input": f"Explain concept number {i} in detail",
                    "output": f"Concept {i} is an important topic in the field of study",
                })

    elif problem == "repetition":
        for i in range(n):
            if rng.random() < 0.3:
                phrase = "buy now limited offer "
                rows.append({"input": phrase * 8, "output": phrase * 8})
            else:
                rows.append({"input": f"normal question {i}", "output": f"normal answer {i}"})

    elif problem == "encoding":
        for i in range(n):
            if rng.random() < 0.25:
                rows.append({
                    "input": f"question with Ã¢â€ broken encoding {i}",
                    "output": f"answer with \x00 control char {i}",
                })
            else:
                rows.append({"input": f"clean question {i}", "output": f"clean answer {i}"})

    elif problem == "imbalanced":
        labels = ["A"] * int(n * 0.9) + ["B"] * int(n * 0.05) + ["C"] * int(n * 0.05)
        rng.shuffle(labels)
        for i, label in enumerate(labels):
            rows.append({
                "input": f"classify item {i}",
                "output": f"classification result for {i}",
                "label": label,
            })

    elif problem == "short":
        for i in range(n):
            rows.append({"input": "q", "output": "a"})

    elif problem == "mixed":
        # Multiple problems at once
        for i in range(n):
            r = rng.random()
            if r < 0.15:
                rows.append({"input": "", "output": ""})
            elif r < 0.25:
                rows.append({"input": "buy now " * 10, "output": "buy now " * 10})
            elif r < 0.35:
                rows.append({"input": "q", "output": "a"})
            else:
                rows.append({
                    "input": f"reasonable question about topic {i}",
                    "output": f"a decent answer explaining topic {i} clearly",
                })

    return rows


def generate_training_data(
    n_per_class: int = 80,
    seed: int = 42,
) -> list[tuple[list[dict], float]]:
    """Generate labeled (dataset, target_score) pairs.

    Returns list of (rows, target) where target is 0.0-1.0.
    """
    data: list[tuple[list[dict], float]] = []
    rng = random.Random(seed)

    # Good datasets (target 0.85-1.0)
    for i in range(n_per_class):
        n_rows = rng.randint(15, 60)
        rows = _make_good_dataset(n_rows, seed=seed + i)
        data.append((rows, rng.uniform(0.85, 1.0)))

    # Bad datasets - various problems (target 0.05-0.35)
    problems = ["empty", "duplicates", "repetition", "encoding", "short", "mixed"]
    per_problem = n_per_class // len(problems)
    for j, problem in enumerate(problems):
        for i in range(per_problem):
            n_rows = rng.randint(15, 60)
            rows = _make_bad_dataset(n_rows, seed=seed + 1000 + j * 100 + i, problem=problem)
            data.append((rows, rng.uniform(0.05, 0.35)))

    # Mixed quality (target 0.40-0.70)
    for i in range(n_per_class // 2):
        n_rows = rng.randint(20, 50)
        rows = _make_bad_dataset(n_rows, seed=seed + 2000 + i, problem="mixed")
        data.append((rows, rng.uniform(0.40, 0.70)))

    rng.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# Feature extraction from datasets
# ---------------------------------------------------------------------------

def extract_all_features(
    labeled_data: list[tuple[list[dict], float]],
) -> tuple[list[list[float]], list[float]]:
    """Run analyze_quality on each dataset and extract features."""
    X: list[list[float]] = []
    y: list[float] = []

    for rows, target in labeled_data:
        report = analyze_quality(rows)
        features = extract_features(report.issues, report.stats, report.total_rows)
        X.append(features)
        y.append(target)

    return X, y


# ---------------------------------------------------------------------------
# Pure Python logistic regression
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def train_logistic_regression(
    X: list[list[float]],
    y: list[float],
    lr: float = 0.01,
    epochs: int = 5000,
    l2_lambda: float = 0.01,
    verbose: bool = True,
) -> list[float]:
    """Train logistic regression via gradient descent.

    Loss: binary cross-entropy with L2 regularization.
    """
    n_features = len(X[0])
    n_samples = len(X)

    # Initialize weights to small random values
    rng = random.Random(42)
    weights = [rng.gauss(0, 0.1) for _ in range(n_features)]

    for epoch in range(epochs):
        # Compute gradients
        grads = [0.0] * n_features
        total_loss = 0.0

        for i in range(n_samples):
            z = _dot(X[i], weights)
            pred = _sigmoid(z)
            error = pred - y[i]

            # Cross-entropy loss
            pred_clipped = max(1e-7, min(1 - 1e-7, pred))
            loss = -(y[i] * math.log(pred_clipped) + (1 - y[i]) * math.log(1 - pred_clipped))
            total_loss += loss

            for j in range(n_features):
                grads[j] += error * X[i][j]

        # Average + L2 regularization
        for j in range(n_features):
            grads[j] = grads[j] / n_samples + l2_lambda * weights[j]

        # Update
        for j in range(n_features):
            weights[j] -= lr * grads[j]

        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            avg_loss = total_loss / n_samples
            print(f"  Epoch {epoch:5d}/{epochs}: loss = {avg_loss:.4f}")

    return weights


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def emit_weights_code(weights: list[float]) -> str:
    """Format weights as Python code for learned_scorer.py."""
    lines = ["_WEIGHTS: List[float] = ["]

    labels = [
        "has_empty", "empty_frac",
        "has_length_outlier", "outlier_frac",
        "has_encoding", "encoding_frac",
        "has_repetition", "repetition_frac",
        "has_near_dup", "near_dup_frac",
        "has_class_imbalance", "imbalance_frac",
        "error_count", "warning_count", "info_count",
        "total_issue_count",
        "type_token_ratio", "hapax_ratio", "log_total_tokens",
        "length_cv", "log_total_rows", "bias",
        "semantic_dup_fraction", "topic_diversity", "topic_count_normalized",
    ]

    for i, (w, label) in enumerate(zip(weights, labels)):
        comma = "," if i < len(weights) - 1 else ","
        lines.append(f"    {w:+.6f}{comma}  # {label} ({i})")

    lines.append("]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Verifily Quality Scorer — Training")
    print("=" * 60)

    print("\n1. Generating synthetic training data...")
    data = generate_training_data(n_per_class=80, seed=42)
    print(f"   Generated {len(data)} labeled datasets")

    print("\n2. Extracting features...")
    X, y = extract_all_features(data)
    print(f"   Feature vectors: {len(X)} x {len(X[0])} dims")

    print("\n3. Training logistic regression...")
    weights = train_logistic_regression(X, y, lr=0.01, epochs=3000)

    print("\n4. Evaluating...")
    correct = 0
    for i in range(len(X)):
        pred = _sigmoid(_dot(X[i], weights))
        pred_class = 1 if pred >= 0.5 else 0
        true_class = 1 if y[i] >= 0.5 else 0
        if pred_class == true_class:
            correct += 1
    accuracy = correct / len(X)
    print(f"   Training accuracy: {accuracy:.1%}")

    print("\n5. Trained weights:")
    print()
    code = emit_weights_code(weights)
    print(code)

    print("\n" + "=" * 60)
    print("Copy the weights above into learned_scorer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
