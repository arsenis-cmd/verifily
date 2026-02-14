"""Generate synthetic dataset from human-only seed with quality filters."""
import os
import json
import logging
from typing import Dict, List, Set, Tuple
import jsonlines
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer

from src.utils import set_seed, hash_string, ensure_dir, compute_ngram_overlap
from src.data_builders import load_jsonl

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic dataset from human-only seed with strict quality filters."""

    def __init__(self, config: Dict):
        self.config = config
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        # Paths
        self.processed_dir = config.get("processed_dir", "data/processed")
        self.synthetic_dir = config.get("synthetic_dir", "data/synthetic")
        ensure_dir(self.synthetic_dir)

        # Model config
        # Auto-detect device if not specified
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            default_device = "mps"

        self.device = config.get("device", default_device)

        # Use base model for synthesis
        synth_config = config.get("synthetic_generation", {})
        model_name = synth_config.get("model", "auto")

        if model_name == "auto":
            self.model_name = config.get("base_model", "google/flan-t5-base")
        else:
            self.model_name = model_name

        # Generation parameters
        self.temperature = synth_config.get("temperature", 0.7)
        self.top_p = synth_config.get("top_p", 0.92)
        self.max_new_tokens = synth_config.get("max_new_tokens", 128)
        self.do_sample = synth_config.get("do_sample", True)

        # Target size
        self.synthetic_multiplier = config.get("synthetic_multiplier", 5)

        # Filter config
        filter_config = config.get("filters", {})
        self.ngram_size = filter_config.get("ngram_size", 8)
        self.max_ngram_overlap = filter_config.get("max_ngram_overlap_ratio", 0.3)
        self.use_semantic_filter = filter_config.get("use_semantic_filter", True)
        self.max_semantic_similarity = filter_config.get("max_semantic_similarity", 0.85)
        self.min_question_length = filter_config.get("min_question_length", 10)
        self.min_answer_length = filter_config.get("min_answer_length", 3)
        self.max_answer_length = filter_config.get("max_answer_length", 200)

        # Models
        self.tokenizer = None
        self.model = None
        self.semantic_model = None

        # Deduplication structures
        self.seen_hashes: Set[str] = set()
        self.minhash_lsh = MinHashLSH(threshold=0.7, num_perm=128)
        self.seed_examples: List[Dict] = []

    def load_models(self):
        """Load generation model and optional semantic model."""
        logger.info(f"Loading synthesis model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Use FP16 only for CUDA, FP32 for MPS and CPU
        use_fp16 = self.device == "cuda"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        ).to(self.device)

        self.model.eval()
        logger.info(f"Synthesis model loaded on {self.device}")

        if self.use_semantic_filter:
            logger.info("Loading semantic similarity model for filtering...")
            semantic_model_name = self.config.get("filters", {}).get(
                "semantic_model",
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.semantic_model = SentenceTransformer(semantic_model_name)
            logger.info("Semantic model loaded")

    def create_synthesis_prompts(self, seed_examples: List[Dict]) -> List[str]:
        """
        Create prompts that generate entirely new question-answer pairs
        from the context passages in the seed data.
        """
        prompts = []

        for example in seed_examples:
            context = example.get("context", "").strip()
            if not context:
                continue

            prompt = f"""Read the following passage and generate a new question and answer pair based on it.

Passage: {context}

Generate a question that can be answered from the passage, and provide the exact answer from the passage.

Question:"""

            prompts.append(("generate_qa", prompt, example))

        return prompts

    def generate_synthetic_example(self, prompt: str, strategy: str, seed_example: Dict) -> Dict:
        """Generate a single synthetic example from a prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse the generated text to extract question, context, and answer
        synthetic = self.parse_generated_text(generated_text, strategy, seed_example)

        return synthetic

    def parse_generated_text(self, text: str, strategy: str, seed_example: Dict) -> Dict:
        """
        Parse generated text into a question-answer pair.
        The model generates text starting from 'Question:' — we parse out
        the question and answer. The context comes from the seed example.
        """
        text = text.strip()
        context = seed_example.get('context', '')

        question = ""
        answer = ""

        # Try to parse "Question: ... Answer: ..." format
        if "Answer:" in text:
            parts = text.split("Answer:", 1)
            question = parts[0].strip()
            answer = parts[1].strip()
        else:
            # Model only generated a question — extract answer from context
            question = text

        # Clean up question prefix
        for prefix in ["Question:", "question:"]:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()

        # Ensure question ends with ?
        if question and not question.endswith("?"):
            question = question + "?"

        # If no answer was parsed, try to find it in context
        if not answer and question and context:
            answer = self._extract_answer_from_context(question, context, seed_example)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "strategy": strategy,
            "seed_id": seed_example.get("id", "unknown")
        }

    def _extract_answer_from_context(self, question: str, context: str, seed_example: Dict) -> str:
        """Use the model to extract an answer from context given a question."""
        prompt = f"""Answer the following question based on the passage.

Passage: {context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def apply_filters(self, synthetic: Dict) -> Tuple[bool, str]:
        """
        Apply quality filters to synthetic example.
        Returns (pass, reason) tuple.
        """
        question = synthetic["question"]
        answer = synthetic["answer"]
        context = synthetic.get("context", "")

        # Filter 1: Length checks
        if len(question.split()) < self.min_question_length:
            return False, "question_too_short"

        if len(answer.split()) < self.min_answer_length:
            return False, "answer_too_short"

        if len(answer.split()) > self.max_answer_length:
            return False, "answer_too_long"

        # Filter 2: Exact duplicate check (question only — contexts are reused)
        qa_str = f"{question}|||{answer}"
        content_hash = hash_string(qa_str)

        if content_hash in self.seen_hashes:
            return False, "exact_duplicate"

        # Filter 3: MinHash near-duplicate check (question only)
        minhash = MinHash(num_perm=128)
        for word in question.lower().split():
            minhash.update(word.encode('utf-8'))

        # Check against existing items
        result = self.minhash_lsh.query(minhash)
        if result:
            return False, "near_duplicate"

        # Filter 4: N-gram overlap with seed questions
        for seed_example in self.seed_examples:
            seed_question = seed_example['question']
            overlap = compute_ngram_overlap(question, seed_question, self.ngram_size)

            if overlap > self.max_ngram_overlap:
                return False, f"ngram_overlap_too_high_{overlap:.2f}"

        # Filter 5: Semantic similarity check (optional)
        if self.use_semantic_filter and self.semantic_model:
            synth_embedding = self.semantic_model.encode(question + " " + answer)

            for seed_example in self.seed_examples[:100]:  # Check against sample for efficiency
                seed_text = f"{seed_example['question']} {seed_example['answer']}"
                seed_embedding = self.semantic_model.encode(seed_text)

                # Compute cosine similarity
                similarity = np.dot(synth_embedding, seed_embedding) / (
                    np.linalg.norm(synth_embedding) * np.linalg.norm(seed_embedding)
                )

                if similarity > self.max_semantic_similarity:
                    return False, f"semantic_similarity_too_high_{similarity:.2f}"

        return True, "pass"

    def generate_synthetic_dataset(
        self,
        human_train_path: str,
        target_size: int = None
    ) -> List[Dict]:
        """Generate synthetic dataset from human seed."""
        logger.info("="*50)
        logger.info("GENERATING SYNTHETIC DATASET FROM HUMAN SEED")
        logger.info("="*50)

        # Load human seed (PRIVATE - do not expose in logs)
        self.seed_examples = load_jsonl(human_train_path)
        logger.info(f"Loaded {len(self.seed_examples)} seed examples (kept private)")

        # Initialize deduplication structures with seed questions
        for example in self.seed_examples:
            qa_str = f"{example['question']}|||{example['answer']}"
            self.seen_hashes.add(hash_string(qa_str))

            # Add to MinHash LSH (question only)
            minhash = MinHash(num_perm=128)
            for word in example['question'].lower().split():
                minhash.update(word.encode('utf-8'))
            self.minhash_lsh.insert(example.get("id", hash_string(qa_str)), minhash)

        # Load models
        if self.model is None:
            self.load_models()

        # Determine target size
        if target_size is None:
            target_size = len(self.seed_examples) * self.synthetic_multiplier

        logger.info(f"Target: {target_size} synthetic examples")

        # Generate synthetic examples
        synthetic_data = []
        filter_stats = {}
        attempts = 0
        max_attempts = target_size * 10  # Limit attempts to avoid infinite loops

        with tqdm(total=target_size, desc="Generating synthetic examples") as pbar:
            while len(synthetic_data) < target_size and attempts < max_attempts:
                # Sample seed examples for prompting
                sample_size = min(3, len(self.seed_examples))
                seed_sample = np.random.choice(self.seed_examples, size=sample_size, replace=False).tolist()

                # Create prompts
                prompts = self.create_synthesis_prompts(seed_sample)

                # Generate from random prompt
                strategy, prompt, seed_example = prompts[np.random.randint(len(prompts))]
                synthetic = self.generate_synthetic_example(prompt, strategy, seed_example)

                # Apply filters
                passed, reason = self.apply_filters(synthetic)

                filter_stats[reason] = filter_stats.get(reason, 0) + 1

                if passed:
                    # Add metadata
                    synthetic_final = {
                        "id": f"synthetic_{len(synthetic_data)}",
                        "question": synthetic["question"],
                        "context": synthetic.get("context", ""),
                        "answer": synthetic["answer"],
                        "source": "synthetic_from_human",
                        "label": "synthetic_from_human",
                        "strategy": synthetic["strategy"],
                        "seed_id": synthetic["seed_id"],
                        "split": "train"
                    }

                    # Add hash and update dedup structures (question only)
                    qa_str = f"{synthetic_final['question']}|||{synthetic_final['answer']}"
                    synthetic_final["content_hash"] = hash_string(qa_str)

                    self.seen_hashes.add(synthetic_final["content_hash"])

                    minhash = MinHash(num_perm=128)
                    for word in synthetic_final['question'].lower().split():
                        minhash.update(word.encode('utf-8'))
                    self.minhash_lsh.insert(synthetic_final["id"], minhash)

                    synthetic_data.append(synthetic_final)
                    pbar.update(1)

                attempts += 1

        logger.info(f"\nGenerated {len(synthetic_data)} synthetic examples after {attempts} attempts")
        logger.info("Filter statistics:")
        for reason, count in sorted(filter_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {reason}: {count}")

        logger.info("="*50)

        return synthetic_data

    def save_synthetic_dataset(self, data: List[Dict]) -> str:
        """Save synthetic dataset to file."""
        output_path = os.path.join(self.synthetic_dir, "synthetic_train.jsonl")

        logger.info(f"Saving synthetic dataset to {output_path}")

        with jsonlines.open(output_path, mode='w') as writer:
            for example in data:
                writer.write(example)

        logger.info(f"Saved {len(data)} synthetic examples")

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_examples": len(data),
            "synthetic_multiplier": self.synthetic_multiplier,
            "filters": {
                "ngram_size": self.ngram_size,
                "max_ngram_overlap": self.max_ngram_overlap,
                "use_semantic_filter": self.use_semantic_filter,
                "max_semantic_similarity": self.max_semantic_similarity
            }
        }

        metadata_path = os.path.join(self.synthetic_dir, "synthetic_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

        return output_path

    def build(self, human_train_path: str, target_size: int = None) -> str:
        """Main method: generate and save synthetic dataset."""
        synthetic_data = self.generate_synthetic_dataset(human_train_path, target_size)
        output_path = self.save_synthetic_dataset(synthetic_data)

        logger.info("Synthetic dataset built successfully!")
        return output_path


def main():
    """CLI entry point for generating synthetic dataset."""
    import argparse
    from src.utils import load_config, setup_logging, detect_hardware, adjust_config_for_hardware

    parser = argparse.ArgumentParser(description="Generate synthetic dataset from human seed")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--human-train-path",
        type=str,
        default=None,
        help="Path to human training data (if not in default location)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Target number of synthetic examples (overrides config)"
    )
    args = parser.parse_args()

    # Load and adjust config
    config = load_config(args.config)

    if config.get("auto_detect_hardware", True):
        hardware_info = detect_hardware()
        config = adjust_config_for_hardware(config, hardware_info)

    # Setup logging
    setup_logging(
        log_file=config.get("logging", {}).get("log_file"),
        level=config.get("logging", {}).get("level", "INFO")
    )

    # Determine human train path
    if args.human_train_path:
        human_train_path = args.human_train_path
    else:
        human_train_path = os.path.join(
            config.get("processed_dir", "data/processed"),
            "human_train.jsonl"
        )

    # Generate synthetic dataset
    generator = SyntheticDataGenerator(config)
    output_path = generator.build(human_train_path, args.target_size)

    print(f"\nSynthetic dataset generated successfully!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
