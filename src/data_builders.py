"""Dataset builders: download, split, and prepare human-only seed data."""
import os
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import jsonlines
from datasets import load_dataset
from tqdm import tqdm

from src.utils import set_seed, hash_string, ensure_dir

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build and split human-only seed dataset."""

    def __init__(self, config: Dict):
        self.config = config
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        # Paths
        self.raw_dir = config.get("raw_dir", "data/raw")
        self.processed_dir = config.get("processed_dir", "data/processed")
        ensure_dir(self.raw_dir)
        ensure_dir(self.processed_dir)

        # Dataset config
        self.source_dataset = config.get("source_dataset", "cosmos_qa")
        self.source_config = config.get("source_config", None)

        # Sizes
        self.train_size = config.get("human_train_size", 20000)
        self.val_size = config.get("human_val_size", 2000)
        self.test_size = config.get("human_test_size", 2000)

    def load_source_dataset(self):
        """Load source dataset from HuggingFace."""
        logger.info(f"Loading source dataset: {self.source_dataset}")

        try:
            if self.source_config:
                dataset = load_dataset(self.source_dataset, self.source_config, trust_remote_code=True)
            else:
                dataset = load_dataset(self.source_dataset, trust_remote_code=True)

            logger.info(f"Dataset loaded successfully: {dataset}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {self.source_dataset}: {e}")
            raise

    def format_example(self, example: Dict, idx: int, split: str) -> Dict:
        """
        Format a single example into standardized format.
        This handles different dataset schemas.
        """
        formatted = {
            "id": f"{split}_{idx}",
            "source": self.source_dataset,
            "label": "human",
            "split": split
        }

        # Handle SQuAD format
        if self.source_dataset == "squad":
            formatted["context"] = example.get("context", "")
            formatted["question"] = example.get("question", "")

            # SQuAD has answers as a dict with 'text' list
            answers = example.get("answers", {})
            if isinstance(answers, dict) and "text" in answers:
                answer_list = answers["text"]
                if isinstance(answer_list, list) and len(answer_list) > 0:
                    formatted["answer"] = answer_list[0]  # Take first answer
                else:
                    formatted["answer"] = ""
            else:
                formatted["answer"] = ""

        # Handle CosmosQA specifically
        elif self.source_dataset == "cosmos_qa":
            formatted["context"] = example.get("context", "")
            formatted["question"] = example.get("question", "")

            # Get the correct answer based on label
            label_idx = example.get("label", 0)
            answer_key = f"answer{label_idx}"
            formatted["answer"] = example.get(answer_key, "")

        else:
            # Generic handling for other datasets
            # Try common field names
            formatted["question"] = example.get(
                "question",
                example.get("input", example.get("text", ""))
            )
            formatted["context"] = example.get("context", example.get("passage", ""))

            # Handle answer extraction
            answer = example.get("answer", example.get("output", example.get("target", "")))
            if isinstance(answer, dict) and "text" in answer:
                answer_text = answer["text"]
                formatted["answer"] = answer_text[0] if isinstance(answer_text, list) else str(answer_text)
            elif isinstance(answer, list):
                formatted["answer"] = answer[0] if len(answer) > 0 else ""
            else:
                formatted["answer"] = str(answer) if answer else ""

        # Add hash for deduplication
        content_str = f"{formatted['question']}|||{formatted.get('context', '')}|||{formatted['answer']}"
        formatted["content_hash"] = hash_string(content_str)

        return formatted

    def build_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Build train, val, and test splits from source dataset.
        Returns: (train_data, val_data, test_data)
        """
        dataset = self.load_source_dataset()

        # Use the 'train' and 'validation' splits from source
        # We'll create our own split from these
        if "train" in dataset:
            source_train = dataset["train"]
        else:
            raise ValueError("Source dataset must have a 'train' split")

        if "validation" in dataset:
            source_val = dataset["validation"]
        else:
            source_val = None

        # Shuffle and select
        logger.info("Shuffling and selecting samples...")
        source_train = source_train.shuffle(seed=self.seed)

        total_needed = self.train_size + self.val_size + self.test_size

        if len(source_train) < total_needed:
            logger.warning(
                f"Source dataset has only {len(source_train)} samples, "
                f"but {total_needed} requested. Adjusting sizes proportionally."
            )
            ratio = len(source_train) / total_needed
            self.train_size = int(self.train_size * ratio)
            self.val_size = int(self.val_size * ratio)
            self.test_size = int(self.test_size * ratio)

        # Create splits
        train_data = []
        val_data = []
        test_data = []

        logger.info("Formatting train split...")
        for idx, example in enumerate(tqdm(source_train.select(range(self.train_size)))):
            formatted = self.format_example(example, idx, "train")
            train_data.append(formatted)

        logger.info("Formatting validation split...")
        for idx, example in enumerate(tqdm(source_train.select(range(self.train_size, self.train_size + self.val_size)))):
            formatted = self.format_example(example, idx, "val")
            val_data.append(formatted)

        logger.info("Formatting test split...")
        for idx, example in enumerate(tqdm(source_train.select(range(self.train_size + self.val_size, total_needed)))):
            formatted = self.format_example(example, idx, "test")
            test_data.append(formatted)

        logger.info(f"Splits created: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        return train_data, val_data, test_data

    def save_split(self, data: List[Dict], split_name: str, filename: str = None):
        """Save a split to JSONL file."""
        if filename is None:
            filename = f"human_{split_name}.jsonl"

        output_path = os.path.join(self.processed_dir, filename)

        logger.info(f"Saving {split_name} split to {output_path}")

        with jsonlines.open(output_path, mode='w') as writer:
            for example in data:
                writer.write(example)

        logger.info(f"Saved {len(data)} examples to {output_path}")

        return output_path

    def build(self) -> Dict[str, str]:
        """
        Main build method: download, format, split, and save.
        Returns dictionary mapping split names to file paths.
        """
        logger.info("="*50)
        logger.info("BUILDING HUMAN-ONLY SEED DATASET")
        logger.info("="*50)

        # Build splits
        train_data, val_data, test_data = self.build_splits()

        # Save splits
        paths = {
            "train": self.save_split(train_data, "train"),
            "val": self.save_split(val_data, "val"),
            "test": self.save_split(test_data, "test")
        }

        # Save metadata
        metadata = {
            "source_dataset": self.source_dataset,
            "source_config": self.source_config,
            "seed": self.seed,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "paths": paths
        }

        metadata_path = os.path.join(self.processed_dir, "human_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
        logger.info("Human-only seed dataset built successfully!")
        logger.info("="*50)

        return paths


def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return data


def main():
    """CLI entry point for building dataset."""
    import argparse
    from src.utils import load_config, setup_logging

    parser = argparse.ArgumentParser(description="Build human-only seed dataset")
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
    setup_logging(
        log_file=config.get("logging", {}).get("log_file"),
        level=config.get("logging", {}).get("level", "INFO")
    )

    # Build dataset
    builder = DatasetBuilder(config)
    paths = builder.build()

    print("\nDataset built successfully!")
    print("Paths:")
    for split, path in paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
