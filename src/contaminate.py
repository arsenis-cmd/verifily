"""Generate AI-contaminated dataset by replacing human answers with AI-generated ones."""
import os
import json
import logging
from typing import Dict, List
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from src.utils import set_seed, hash_string, ensure_dir
from src.data_builders import load_jsonl

logger = logging.getLogger(__name__)


class AIContaminationGenerator:
    """Generate AI-contaminated dataset using a local model."""

    def __init__(self, config: Dict):
        self.config = config
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        # Paths
        self.processed_dir = config.get("processed_dir", "data/processed")
        ensure_dir(self.processed_dir)

        # Model config
        # Auto-detect device if not specified
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            default_device = "mps"

        self.device = config.get("device", default_device)

        # Use base model or a smaller one for contamination
        contamination_config = config.get("contamination_generation", {})
        model_name = contamination_config.get("model", "auto")

        if model_name == "auto":
            # Use the base model for contamination
            self.model_name = config.get("base_model", "google/flan-t5-base")
        else:
            self.model_name = model_name

        # Generation parameters
        self.temperature = contamination_config.get("temperature", 0.3)
        self.top_p = contamination_config.get("top_p", 0.9)
        self.max_new_tokens = contamination_config.get("max_new_tokens", 64)
        self.do_sample = contamination_config.get("do_sample", True)

        # Contamination mix
        self.pure_ai_ratio = contamination_config.get("pure_ai_ratio", 0.8)
        self.paraphrased_ratio = contamination_config.get("paraphrased_ratio", 0.2)

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load the model and tokenizer for contamination generation."""
        logger.info(f"Loading contamination model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Use FP16 only for CUDA, FP32 for MPS and CPU
        use_fp16 = self.device == "cuda"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        ).to(self.device)

        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def create_generation_prompt(self, question: str, context: str = "") -> str:
        """
        Create a prompt for the AI model to generate an answer.
        This simulates what an AI would produce when contaminating data.
        """
        if context:
            prompt = f"Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Answer the following question concisely.\n\nQuestion: {question}\n\nAnswer:"

        return prompt

    def create_paraphrase_prompt(self, question: str, context: str, answer: str) -> str:
        """Create a prompt to paraphrase an existing answer (simulating AI contamination)."""
        if context:
            prompt = f"Paraphrase the following answer while keeping the same meaning.\n\nQuestion: {question}\nContext: {context}\nOriginal Answer: {answer}\n\nParaphrased Answer:"
        else:
            prompt = f"Paraphrase the following answer while keeping the same meaning.\n\nQuestion: {question}\nOriginal Answer: {answer}\n\nParaphrased Answer:"

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer using the loaded model."""
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

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def contaminate_example(self, example: Dict, contamination_type: str) -> Dict:
        """
        Generate AI-contaminated version of an example.
        contamination_type: 'pure_ai' or 'paraphrased'
        """
        question = example["question"]
        context = example.get("context", "")
        original_answer = example["answer"]

        if contamination_type == "pure_ai":
            # Generate completely new AI answer
            prompt = self.create_generation_prompt(question, context)
            ai_answer = self.generate_answer(prompt)
        else:  # paraphrased
            # Paraphrase the original answer (simulating AI rewriting)
            prompt = self.create_paraphrase_prompt(question, context, original_answer)
            ai_answer = self.generate_answer(prompt)

        # Create contaminated example
        contaminated = {
            "id": example["id"].replace("train", "contaminated"),
            "question": question,
            "context": context,
            "answer": ai_answer,
            "original_answer": original_answer,  # Keep for reference
            "source": example["source"],
            "label": "ai_contaminated",
            "contamination_type": contamination_type,
            "split": "train"
        }

        # Add hash
        content_str = f"{contaminated['question']}|||{contaminated.get('context', '')}|||{contaminated['answer']}"
        contaminated["content_hash"] = hash_string(content_str)

        return contaminated

    def generate_contaminated_dataset(self, human_train_path: str) -> List[Dict]:
        """Generate AI-contaminated version of the human training dataset."""
        logger.info("="*50)
        logger.info("GENERATING AI-CONTAMINATED DATASET")
        logger.info("="*50)

        # Load human training data
        human_data = load_jsonl(human_train_path)
        logger.info(f"Loaded {len(human_data)} human examples")

        # Load model
        if self.model is None:
            self.load_model()

        # Generate contaminated examples
        contaminated_data = []

        # Determine split: pure_ai vs paraphrased
        num_pure_ai = int(len(human_data) * self.pure_ai_ratio)

        logger.info(f"Generating {num_pure_ai} pure AI answers...")
        for example in tqdm(human_data[:num_pure_ai]):
            contaminated = self.contaminate_example(example, "pure_ai")
            contaminated_data.append(contaminated)

        logger.info(f"Generating {len(human_data) - num_pure_ai} paraphrased answers...")
        for example in tqdm(human_data[num_pure_ai:]):
            contaminated = self.contaminate_example(example, "paraphrased")
            contaminated_data.append(contaminated)

        logger.info(f"Generated {len(contaminated_data)} contaminated examples")
        logger.info("="*50)

        return contaminated_data

    def save_contaminated_dataset(self, data: List[Dict]) -> str:
        """Save contaminated dataset to file."""
        output_path = os.path.join(self.processed_dir, "ai_contaminated_train.jsonl")

        logger.info(f"Saving contaminated dataset to {output_path}")

        with jsonlines.open(output_path, mode='w') as writer:
            for example in data:
                writer.write(example)

        logger.info(f"Saved {len(data)} contaminated examples")

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "pure_ai_ratio": self.pure_ai_ratio,
            "paraphrased_ratio": self.paraphrased_ratio,
            "num_examples": len(data)
        }

        metadata_path = os.path.join(self.processed_dir, "contaminated_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

        return output_path

    def build(self, human_train_path: str) -> str:
        """Main method: generate and save contaminated dataset."""
        contaminated_data = self.generate_contaminated_dataset(human_train_path)
        output_path = self.save_contaminated_dataset(contaminated_data)

        logger.info("AI-contaminated dataset built successfully!")
        return output_path


def main():
    """CLI entry point for generating contaminated dataset."""
    import argparse
    from src.utils import load_config, setup_logging, detect_hardware, adjust_config_for_hardware

    parser = argparse.ArgumentParser(description="Generate AI-contaminated dataset")
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

    # Generate contaminated dataset
    generator = AIContaminationGenerator(config)
    output_path = generator.build(human_train_path)

    print(f"\nContaminated dataset generated successfully!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
