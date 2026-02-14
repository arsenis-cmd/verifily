"""Evaluation pipeline for comparing trained models on human test set."""
import os
import json
import logging
from typing import Dict, List
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
import evaluate

from src.utils import (
    set_seed,
    ensure_dir,
    exact_match_score,
    f1_score,
    normalize_answer
)
from src.data_builders import load_jsonl

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained models on human test set."""

    def __init__(self, config: Dict):
        self.config = config
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        # Device
        # Auto-detect device if not specified
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            default_device = "mps"

        self.device = config.get("device", default_device)

        # Task config
        self.max_input_length = config.get("max_input_length", 256)
        self.max_output_length = config.get("max_output_length", 64)

        # Eval config
        eval_config = config.get("evaluation", {})
        self.metrics_to_compute = eval_config.get("metrics", ["exact_match", "f1", "rouge"])

        # Generation config for eval
        gen_config = eval_config.get("generation", {})
        self.gen_max_new_tokens = gen_config.get("max_new_tokens", 64)
        self.gen_num_beams = gen_config.get("num_beams", 4)
        self.gen_temperature = gen_config.get("temperature", 0.0)
        self.gen_do_sample = gen_config.get("do_sample", False)

        # Load evaluation metrics
        self.rouge_metric = None
        self.bertscore_metric = None

        if "rouge" in self.metrics_to_compute:
            self.rouge_metric = evaluate.load("rouge")

        if "bertscore" in self.metrics_to_compute:
            self.bertscore_metric = evaluate.load("bertscore")

        # Output paths
        self.results_dir = config.get("results_dir", "results")
        ensure_dir(self.results_dir)

        # Models cache
        self.loaded_models = {}

    def load_model(self, model_path: str, model_id: str):
        """Load a trained model for evaluation."""
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return self.loaded_models[model_id]

        logger.info(f"Loading model: {model_id} from {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check if it's a PEFT model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            logger.info("Loading PEFT model...")
            # Load base model
            with open(os.path.join(model_path, "training_info.json")) as f:
                training_info = json.load(f)
            base_model_name = training_info.get("base_model", self.config.get("base_model"))

            # Use FP16 only for CUDA, FP32 for MPS and CPU
            use_fp16 = self.device == "cuda"

            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if use_fp16 else torch.float32
            )

            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()  # Merge adapters for faster inference
        else:
            logger.info("Loading full model...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if use_fp16 else torch.float32
            )

        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded: {model_id}")

        self.loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer
        }

        return self.loaded_models[model_id]

    def format_input(self, example: Dict) -> str:
        """Format example for model input."""
        if example.get("context", "").strip():
            input_text = f"Answer the question based on the context.\n\nContext: {example['context']}\n\nQuestion: {example['question']}"
        else:
            input_text = f"Answer the question.\n\nQuestion: {example['question']}"

        return input_text

    def generate_prediction(self, model, tokenizer, input_text: str) -> str:
        """Generate prediction from model."""
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.gen_max_new_tokens,
                num_beams=self.gen_num_beams,
                temperature=self.gen_temperature if self.gen_do_sample else None,
                do_sample=self.gen_do_sample,
                pad_token_id=tokenizer.eos_token_id
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction.strip()

    def evaluate_model(
        self,
        model_path: str,
        model_id: str,
        test_data_path: str
    ) -> Dict:
        """Evaluate a single model on test set."""
        logger.info("="*50)
        logger.info(f"EVALUATING MODEL: {model_id}")
        logger.info("="*50)

        # Load model
        model_dict = self.load_model(model_path, model_id)
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]

        # Load test data
        logger.info(f"Loading test data from: {test_data_path}")
        test_data = load_jsonl(test_data_path)
        logger.info(f"Test set size: {len(test_data)}")

        # Generate predictions
        predictions = []
        references = []

        logger.info("Generating predictions...")
        for example in tqdm(test_data):
            input_text = self.format_input(example)
            prediction = self.generate_prediction(model, tokenizer, input_text)

            predictions.append(prediction)
            references.append(example["answer"])

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = {}

        # Exact Match and F1
        if "exact_match" in self.metrics_to_compute or "f1" in self.metrics_to_compute:
            em_scores = []
            f1_scores = []

            for pred, ref in zip(predictions, references):
                em_scores.append(exact_match_score(pred, ref))
                f1_scores.append(f1_score(pred, ref))

            if "exact_match" in self.metrics_to_compute:
                metrics["exact_match"] = sum(em_scores) / len(em_scores)

            if "f1" in self.metrics_to_compute:
                metrics["f1"] = sum(f1_scores) / len(f1_scores)

        # ROUGE
        if "rouge" in self.metrics_to_compute and self.rouge_metric:
            rouge_results = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            metrics["rouge1"] = rouge_results["rouge1"]
            metrics["rouge2"] = rouge_results["rouge2"]
            metrics["rougeL"] = rouge_results["rougeL"]

        # BERTScore (optional, can be slow)
        if "bertscore" in self.metrics_to_compute and self.bertscore_metric:
            logger.info("Computing BERTScore (this may take a while)...")
            bertscore_model = self.config.get("evaluation", {}).get(
                "bertscore_model",
                "microsoft/deberta-base-mnli"
            )
            bertscore_results = self.bertscore_metric.compute(
                predictions=predictions,
                references=references,
                model_type=bertscore_model
            )
            metrics["bertscore_f1"] = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])

        # Log results
        logger.info(f"\nResults for {model_id}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        logger.info("="*50)

        # Save predictions (sample)
        sample_predictions = []
        for i in range(min(10, len(test_data))):
            sample_predictions.append({
                "question": test_data[i]["question"],
                "context": test_data[i].get("context", ""),
                "reference": references[i],
                "prediction": predictions[i],
                "exact_match": exact_match_score(predictions[i], references[i]),
                "f1": f1_score(predictions[i], references[i])
            })

        return {
            "model_id": model_id,
            "model_path": model_path,
            "metrics": metrics,
            "sample_predictions": sample_predictions,
            "num_test_examples": len(test_data)
        }

    def evaluate_all_models(
        self,
        models: Dict[str, str],
        test_data_path: str
    ) -> List[Dict]:
        """
        Evaluate all models on the same test set.

        Args:
            models: Dict mapping model_id to model_path
            test_data_path: Path to human test set
        """
        results = []

        for model_id, model_path in models.items():
            result = self.evaluate_model(model_path, model_id, test_data_path)
            results.append(result)

        # Save all results
        self.save_results(results)

        return results

    def save_results(self, results: List[Dict]):
        """Save evaluation results to files."""
        # Save detailed results as JSONL
        metrics_file = os.path.join(self.results_dir, "metrics.jsonl")
        logger.info(f"Saving detailed results to {metrics_file}")

        with jsonlines.open(metrics_file, mode='w') as writer:
            for result in results:
                writer.write(result)

        # Save summary table as CSV
        import pandas as pd

        table_data = []
        for result in results:
            row = {"model": result["model_id"]}
            row.update(result["metrics"])
            table_data.append(row)

        df = pd.DataFrame(table_data)
        table_file = os.path.join(self.results_dir, "metrics_table.csv")
        df.to_csv(table_file, index=False)

        logger.info(f"Summary table saved to {table_file}")

        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)


def main():
    """CLI entry point for evaluation."""
    import argparse
    from src.utils import load_config, setup_logging, detect_hardware, adjust_config_for_hardware

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data (defaults to data/processed/human_test.jsonl)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Model paths in format 'model_id:path' (e.g., model_a:runs/model_a_human)"
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

    # Determine test data path
    if args.test_data:
        test_data_path = args.test_data
    else:
        test_data_path = os.path.join(
            config.get("processed_dir", "data/processed"),
            "human_test.jsonl"
        )

    # Parse models
    models = {}
    if args.models:
        for model_str in args.models:
            model_id, model_path = model_str.split(":")
            models[model_id] = model_path
    else:
        # Default: evaluate all three models
        runs_dir = config.get("outputs", {}).get("models_dir", "runs")
        models = {
            "model_a_human": os.path.join(runs_dir, "model_a_human"),
            "model_b_contaminated": os.path.join(runs_dir, "model_b_contaminated"),
            "model_c_synthetic": os.path.join(runs_dir, "model_c_synthetic")
        }

    # Evaluate
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_all_models(models, test_data_path)

    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {config.get('results_dir', 'results')}")


if __name__ == "__main__":
    main()
