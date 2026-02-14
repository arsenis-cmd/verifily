"""Sanity checks and quality validation for datasets."""
import os
import logging
from typing import Dict, List
from collections import Counter
import numpy as np
from src.utils import compute_ngram_overlap, hash_string
from src.data_builders import load_jsonl

logger = logging.getLogger(__name__)


class DatasetSanityChecker:
    """Perform sanity checks and quality validation on datasets."""

    def __init__(self, config: Dict):
        self.config = config
        self.processed_dir = config.get("processed_dir", "data/processed")
        self.synthetic_dir = config.get("synthetic_dir", "data/synthetic")

    def check_dataset_basics(self, data: List[Dict], dataset_name: str):
        """Basic dataset statistics."""
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")

        print(f"Total examples: {len(data)}")

        if len(data) == 0:
            print("WARNING: Dataset is empty!")
            return

        # Label distribution
        labels = [ex.get("label", "unknown") for ex in data]
        label_counts = Counter(labels)
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({100*count/len(data):.1f}%)")

        # Length statistics
        question_lengths = [len(ex.get("question", "").split()) for ex in data]
        answer_lengths = [len(ex.get("answer", "").split()) for ex in data]
        context_lengths = [len(ex.get("context", "").split()) for ex in data if ex.get("context")]

        print(f"\nQuestion length (words):")
        print(f"  Mean: {np.mean(question_lengths):.1f}")
        print(f"  Median: {np.median(question_lengths):.1f}")
        print(f"  Min: {np.min(question_lengths)}")
        print(f"  Max: {np.max(question_lengths)}")

        print(f"\nAnswer length (words):")
        print(f"  Mean: {np.mean(answer_lengths):.1f}")
        print(f"  Median: {np.median(answer_lengths):.1f}")
        print(f"  Min: {np.min(answer_lengths)}")
        print(f"  Max: {np.max(answer_lengths)}")

        if context_lengths:
            print(f"\nContext length (words) - {len(context_lengths)} examples with context:")
            print(f"  Mean: {np.mean(context_lengths):.1f}")
            print(f"  Median: {np.median(context_lengths):.1f}")

        # Sample examples (but DO NOT print full human seed content for privacy)
        print(f"\n--- Sample Examples (3 random) ---")
        indices = np.random.choice(len(data), size=min(3, len(data)), replace=False)

        for i, idx in enumerate(indices, 1):
            ex = data[idx]
            print(f"\nExample {i}:")
            print(f"  ID: {ex.get('id', 'N/A')}")
            print(f"  Label: {ex.get('label', 'N/A')}")

            # For human seed, only show hash and lengths (privacy)
            if "human" in ex.get("label", "").lower() and "synthetic" not in ex.get("label", "").lower():
                print(f"  Question: [HIDDEN - length: {len(ex.get('question', '').split())} words]")
                print(f"  Answer: [HIDDEN - length: {len(ex.get('answer', '').split())} words]")
                print(f"  Hash: {ex.get('content_hash', hash_string(str(ex)))[:16]}...")
            else:
                # For contaminated and synthetic, we can show samples
                question = ex.get('question', '')
                answer = ex.get('answer', '')
                print(f"  Question: {question[:100]}{'...' if len(question) > 100 else ''}")
                print(f"  Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")

    def check_duplicates(self, data: List[Dict], dataset_name: str):
        """Check for duplicate examples."""
        print(f"\n--- Duplicate Check for {dataset_name} ---")

        # Exact duplicates by hash
        hashes = [ex.get("content_hash", hash_string(f"{ex.get('question')}|||{ex.get('answer')}")) for ex in data]
        unique_hashes = len(set(hashes))
        duplicate_count = len(hashes) - unique_hashes

        print(f"Unique examples: {unique_hashes}")
        print(f"Duplicate examples: {duplicate_count}")

        if duplicate_count > 0:
            print(f"WARNING: Found {duplicate_count} duplicates!")

        # Check question duplicates
        questions = [ex.get("question", "") for ex in data]
        unique_questions = len(set(questions))
        question_duplicates = len(questions) - unique_questions

        print(f"\nUnique questions: {unique_questions}")
        print(f"Duplicate questions: {question_duplicates}")

    def check_overlap_with_seed(self, synthetic_data: List[Dict], seed_data: List[Dict]):
        """Check overlap between synthetic and seed data."""
        print(f"\n--- Overlap Check: Synthetic vs Human Seed ---")

        high_overlap_count = 0
        overlap_scores = []

        # Sample check (checking all would be O(n*m))
        sample_size = min(100, len(synthetic_data))
        synthetic_sample = np.random.choice(synthetic_data, size=sample_size, replace=False)

        for synth_ex in synthetic_sample:
            synth_content = f"{synth_ex.get('question', '')} {synth_ex.get('context', '')} {synth_ex.get('answer', '')}"

            max_overlap = 0
            for seed_ex in seed_data[:100]:  # Check against sample of seed
                seed_content = f"{seed_ex.get('question', '')} {seed_ex.get('context', '')} {seed_ex.get('answer', '')}"
                overlap = compute_ngram_overlap(synth_content, seed_content, n=8)
                max_overlap = max(max_overlap, overlap)

            overlap_scores.append(max_overlap)

            if max_overlap > 0.3:
                high_overlap_count += 1

        print(f"Checked {sample_size} synthetic examples against seed")
        print(f"Mean max 8-gram overlap: {np.mean(overlap_scores):.3f}")
        print(f"Median max 8-gram overlap: {np.median(overlap_scores):.3f}")
        print(f"Max overlap found: {np.max(overlap_scores):.3f}")
        print(f"Examples with >30% overlap: {high_overlap_count} ({100*high_overlap_count/sample_size:.1f}%)")

        if high_overlap_count > sample_size * 0.1:
            print(f"WARNING: High overlap detected! {high_overlap_count} examples may be too similar to seed.")

    def check_contamination_vs_human(self, contaminated_data: List[Dict], human_data: List[Dict]):
        """Check that contaminated data has different answers than human data."""
        print(f"\n--- Contamination Check: AI-Contaminated vs Human ---")

        # Match by question and check if answers differ
        human_qa_map = {}
        for ex in human_data:
            human_qa_map[ex.get("question", "")] = ex.get("answer", "")

        different_answers = 0
        same_answers = 0
        matched = 0

        for cont_ex in contaminated_data:
            question = cont_ex.get("question", "")
            if question in human_qa_map:
                matched += 1
                if cont_ex.get("answer", "") != human_qa_map[question]:
                    different_answers += 1
                else:
                    same_answers += 1

        print(f"Matched questions: {matched}")
        print(f"Different answers: {different_answers} ({100*different_answers/max(matched,1):.1f}%)")
        print(f"Same answers: {same_answers} ({100*same_answers/max(matched,1):.1f}%)")

        if same_answers > matched * 0.1:
            print(f"WARNING: {same_answers} contaminated examples have same answers as human data!")

    def run_all_checks(self):
        """Run all sanity checks on datasets."""
        print("\n" + "="*60)
        print("DATASET SANITY CHECKS")
        print("="*60)

        # Load datasets
        datasets = {}

        # Human datasets
        human_train_path = os.path.join(self.processed_dir, "human_train.jsonl")
        human_val_path = os.path.join(self.processed_dir, "human_val.jsonl")
        human_test_path = os.path.join(self.processed_dir, "human_test.jsonl")

        if os.path.exists(human_train_path):
            datasets["human_train"] = load_jsonl(human_train_path)
        if os.path.exists(human_val_path):
            datasets["human_val"] = load_jsonl(human_val_path)
        if os.path.exists(human_test_path):
            datasets["human_test"] = load_jsonl(human_test_path)

        # Contaminated dataset
        contaminated_path = os.path.join(self.processed_dir, "ai_contaminated_train.jsonl")
        if os.path.exists(contaminated_path):
            datasets["ai_contaminated"] = load_jsonl(contaminated_path)

        # Synthetic dataset
        synthetic_path = os.path.join(self.synthetic_dir, "synthetic_train.jsonl")
        if os.path.exists(synthetic_path):
            datasets["synthetic"] = load_jsonl(synthetic_path)

        # Basic checks for all datasets
        for name, data in datasets.items():
            self.check_dataset_basics(data, name)
            self.check_duplicates(data, name)

        # Cross-dataset checks
        if "synthetic" in datasets and "human_train" in datasets:
            self.check_overlap_with_seed(datasets["synthetic"], datasets["human_train"])

        if "ai_contaminated" in datasets and "human_train" in datasets:
            self.check_contamination_vs_human(datasets["ai_contaminated"], datasets["human_train"])

        print("\n" + "="*60)
        print("SANITY CHECKS COMPLETED")
        print("="*60)


def main():
    """CLI entry point for sanity checks."""
    import argparse
    from src.utils import load_config, setup_logging

    parser = argparse.ArgumentParser(description="Run sanity checks on datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(level="INFO")

    # Run checks
    checker = DatasetSanityChecker(config)
    checker.run_all_checks()


if __name__ == "__main__":
    main()
