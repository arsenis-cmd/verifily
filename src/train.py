"""Training script with LoRA/PEFT for fine-tuning models."""
import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.utils import (
    set_seed,
    hash_dict,
    ensure_dir,
    save_config,
    save_environment_info
)
from src.data_builders import load_jsonl

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Fine-tune models with LoRA on different datasets."""

    def __init__(self, config: Dict, model_id: str):
        """
        Args:
            config: Experiment configuration
            model_id: Identifier for this training run (e.g., 'model_a_human')
        """
        self.config = config
        self.model_id = model_id
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        # Model config
        self.base_model_name = config.get("base_model", "google/flan-t5-base")

        # Auto-detect device if not specified
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            default_device = "mps"

        self.device = config.get("device", default_device)

        # Training config
        self.training_config = config.get("training", {})
        self.use_lora = self.training_config.get("use_lora", True)

        # Task config
        self.max_input_length = config.get("max_input_length", 256)
        self.max_output_length = config.get("max_output_length", 64)

        # Output paths
        self.output_dir = os.path.join(
            config.get("outputs", {}).get("models_dir", "runs"),
            self.model_id
        )
        ensure_dir(self.output_dir)

        # Models
        self.tokenizer = None
        self.model = None

    def load_base_model(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading base model: {self.base_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Use FP16 only for CUDA (and if enabled in config), FP32 for MPS and CPU
        use_fp16 = self.device == "cuda" and self.training_config.get("fp16", True)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )

        logger.info(f"Base model loaded: {self.base_model_name}")

        # Apply LoRA if enabled
        if self.use_lora:
            self.apply_lora()

    def apply_lora(self):
        """Apply LoRA to the model."""
        logger.info("Applying LoRA configuration...")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.training_config.get("lora_r", 16),
            lora_alpha=self.training_config.get("lora_alpha", 32),
            lora_dropout=self.training_config.get("lora_dropout", 0.1),
            target_modules=self.training_config.get("lora_target_modules", ["q", "v"]),
            inference_mode=False
        )

        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            f"LoRA applied: {trainable_params:,} trainable params "
            f"({100 * trainable_params / total_params:.2f}% of {total_params:,} total)"
        )

    def prepare_dataset(self, data_path: str, val_data_path: Optional[str] = None) -> Dict:
        """Load and prepare dataset for training."""
        logger.info(f"Loading training data from: {data_path}")
        train_data = load_jsonl(data_path)

        if val_data_path:
            logger.info(f"Loading validation data from: {val_data_path}")
            val_data = load_jsonl(val_data_path)
        else:
            val_data = None

        logger.info(f"Train size: {len(train_data)}")
        if val_data:
            logger.info(f"Val size: {len(val_data)}")

        # Convert to HF Dataset format
        def format_for_training(examples):
            """Format examples for seq2seq training."""
            inputs = []
            targets = []

            for ex in examples:
                # Create input text
                if ex.get("context", "").strip():
                    input_text = f"Answer the question based on the context.\n\nContext: {ex['context']}\n\nQuestion: {ex['question']}"
                else:
                    input_text = f"Answer the question.\n\nQuestion: {ex['question']}"

                inputs.append(input_text)
                targets.append(ex["answer"])

            return inputs, targets

        train_inputs, train_targets = format_for_training(train_data)
        train_dataset = Dataset.from_dict({
            "input_text": train_inputs,
            "target_text": train_targets
        })

        datasets = {"train": train_dataset}

        if val_data:
            val_inputs, val_targets = format_for_training(val_data)
            val_dataset = Dataset.from_dict({
                "input_text": val_inputs,
                "target_text": val_targets
            })
            datasets["val"] = val_dataset

        return datasets

    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length"
        )

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target_text"],
                max_length=self.max_output_length,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, train_data_path: str, val_data_path: Optional[str] = None):
        """Main training loop."""
        logger.info("="*50)
        logger.info(f"TRAINING MODEL: {self.model_id}")
        logger.info("="*50)

        # Load model
        if self.model is None:
            self.load_base_model()

        # Prepare datasets
        datasets = self.prepare_dataset(train_data_path, val_data_path)
        train_dataset = datasets["train"]
        val_dataset = datasets.get("val", None)

        # Tokenize
        logger.info("Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["input_text", "target_text"]
        )

        if val_dataset:
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=["input_text", "target_text"]
            )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config.get("num_epochs", 3),
            per_device_train_batch_size=self.training_config.get("batch_size", 8),
            per_device_eval_batch_size=self.training_config.get("batch_size", 8),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.training_config.get("learning_rate", 3e-4),
            warmup_steps=self.training_config.get("warmup_steps", 100),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=self.training_config.get("logging_steps", 50),
            eval_steps=self.training_config.get("eval_steps", 500) if val_dataset else None,
            save_steps=self.training_config.get("save_steps", 1000),
            save_total_limit=2,
            eval_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="loss" if val_dataset else None,
            greater_is_better=False,
            fp16=self.training_config.get("fp16", True) and self.device == "cuda",
            predict_with_generate=True,
            generation_max_length=self.max_output_length,
            seed=self.seed,
            report_to=[]  # Disable wandb/tensorboard by default
        )

        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        metrics_path = os.path.join(self.output_dir, "train_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training metrics saved to {metrics_path}")

        # Save config and environment
        config_path = os.path.join(self.output_dir, "config.yaml")
        save_config(self.config, config_path)

        env_path = os.path.join(self.output_dir, "environment.json")
        save_environment_info(env_path)

        # Save training info
        training_info = {
            "model_id": self.model_id,
            "base_model": self.base_model_name,
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            "use_lora": self.use_lora,
            "seed": self.seed,
            "config_hash": hash_dict(self.config)
        }

        info_path = os.path.join(self.output_dir, "training_info.json")
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)

        logger.info("="*50)
        logger.info(f"Training completed: {self.model_id}")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info("="*50)

        return self.output_dir


def main():
    """CLI entry point for training models."""
    import argparse
    from src.utils import load_config, setup_logging, detect_hardware, adjust_config_for_hardware

    parser = argparse.ArgumentParser(description="Train model with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier (e.g., model_a_human, model_b_contaminated, model_c_synthetic)"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (optional)"
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

    # Train model
    trainer = ModelTrainer(config, args.model_id)
    output_dir = trainer.train(args.train_data, args.val_data)

    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
