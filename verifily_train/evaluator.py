"""Verifily Train evaluation: metrics, slicing, hard examples."""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from verifily_train.dataset import load_jsonl, slice_by_tag
from verifily_train.errors import EvalError
from verifily_train.run import Run
from verifily_train.utils import detect_device, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())


def exact_match(pred: str, ref: str) -> float:
    return float(_normalize(pred) == _normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    from collections import Counter
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    if not ref_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common_count = sum((pred_counts & ref_counts).values())
    if not common_count:
        return 0.0
    prec = common_count / len(pred_tokens)
    rec = common_count / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)


def _compute_sft_metrics(
    predictions: List[str], references: List[str], metric_names: List[str],
    eval_loss: Optional[float] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if "exact_match" in metric_names:
        metrics["exact_match"] = round(
            np.mean([exact_match(p, r) for p, r in zip(predictions, references)]), 6,
        )
    if "f1" in metric_names:
        metrics["f1"] = round(
            np.mean([token_f1(p, r) for p, r in zip(predictions, references)]), 6,
        )
    if "perplexity" in metric_names and eval_loss is not None:
        metrics["perplexity"] = round(float(np.exp(eval_loss)), 4)
    if "rouge1" in metric_names or "rougeL" in metric_names:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            r1_scores = []
            rl_scores = []
            for p, r in zip(predictions, references):
                scores = scorer.score(r, p)
                r1_scores.append(scores["rouge1"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)
            if "rouge1" in metric_names:
                metrics["rouge1"] = round(float(np.mean(r1_scores)), 6)
            if "rougeL" in metric_names:
                metrics["rougeL"] = round(float(np.mean(rl_scores)), 6)
        except ImportError:
            logger.warning("rouge-score not installed, skipping ROUGE metrics")
    return metrics


def _compute_cls_metrics(
    predictions: List[int], references: List[int], metric_names: List[str], labels: List[str],
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix as cm_fn

    metrics: Dict[str, Any] = {}
    if "accuracy" in metric_names:
        metrics["accuracy"] = round(accuracy_score(references, predictions), 6)
    if "macro_f1" in metric_names:
        metrics["macro_f1"] = round(
            f1_score(references, predictions, average="macro", zero_division=0), 6,
        )
    if "precision_per_class" in metric_names:
        per_class_prec = precision_score(references, predictions, average=None, zero_division=0)
        if labels:
            metrics["precision_per_class"] = {
                lbl: round(float(v), 6) for lbl, v in zip(labels, per_class_prec)
            }
        else:
            metrics["precision_per_class"] = [round(float(v), 6) for v in per_class_prec]
    if "recall_per_class" in metric_names:
        per_class_rec = recall_score(references, predictions, average=None, zero_division=0)
        if labels:
            metrics["recall_per_class"] = {
                lbl: round(float(v), 6) for lbl, v in zip(labels, per_class_rec)
            }
        else:
            metrics["recall_per_class"] = [round(float(v), 6) for v in per_class_rec]
    if "confusion_matrix" in metric_names:
        metrics["confusion_matrix"] = cm_fn(references, predictions).tolist()
    return metrics


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    run_id: str = ""
    test_data_path: str = ""
    num_examples: int = 0
    overall: Dict[str, Any] = field(default_factory=dict)
    slices: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hard_examples: List[Dict[str, Any]] = field(default_factory=list)
    eval_duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    run_path: str,
    test_data: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    slice_by: Optional[List[str]] = None,
    hard_examples_n: int = 50,
    device: str = "auto",
    batch_size: int = 16,
) -> EvalResult:
    """Evaluate a trained model from a completed run.

    Args:
        run_path: path to run artifact directory.
        test_data: override path to test JSONL.
        metrics: override metric list.
        slice_by: tag keys to slice metrics by.
        hard_examples_n: number of worst examples to surface.
        device: device for inference.
        batch_size: inference batch size (used for logging only; generation is per-example).

    Returns:
        EvalResult populated with overall, sliced, and hard-example data.
    """
    device = detect_device(device)
    run = Run.load(run_path)
    set_seed(run.seed)

    # Load training info to find base model
    info_path = Path(run_path) / "training_info.json"
    if not info_path.exists():
        raise EvalError(f"training_info.json not found in {run_path}")
    with open(info_path) as f:
        info = json.load(f)

    base_model = info["base_model"]
    task = info.get("task", run.task)

    # Resolve test data
    if not test_data:
        cfg_path = Path(run_path) / "config.yaml"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            test_data = (cfg.get("data_paths") or {}).get("test")
    if not test_data:
        raise EvalError("No test data path provided and none found in run config")

    test_rows = load_jsonl(test_data)
    logger.info("Loaded %d test examples from %s", len(test_rows), test_data)

    # Default metrics
    if metrics is None:
        if task == "classification":
            metrics = ["accuracy", "macro_f1", "precision_per_class", "recall_per_class", "confusion_matrix"]
        else:
            metrics = ["exact_match", "f1", "perplexity", "rouge1", "rougeL"]

    start = time.time()

    # --- Load model -------------------------------------------------------
    logger.info("Loading model from %s (base: %s)", run_path, base_model)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(run_path) / "tokenizer"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect architecture
    from verifily_train.trainer import _is_seq2seq
    is_seq2seq = _is_seq2seq(base_model)

    adapter_dir = Path(run_path) / "adapter"
    has_adapter = (adapter_dir / "adapter_config.json").exists()

    if task == "classification":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(base_model)
        if has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.merge_and_unload()
    elif is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        if has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        if has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_dir))
            model = model.merge_and_unload()

    model = model.to(device)
    model.eval()

    # --- Generate predictions ---------------------------------------------
    if task == "classification":
        pred_labels, ref_labels, scores = _predict_classification(
            model, tokenizer, test_rows, device,
        )
        label_names = list((model.config.id2label or {}).values()) if hasattr(model.config, "id2label") and model.config.id2label else []
        overall = _compute_cls_metrics(pred_labels, ref_labels, metrics, label_names)
    else:
        predictions, references = _predict_sft(
            model, tokenizer, test_rows, device, is_seq2seq,
        )
        # Compute eval loss for perplexity
        eval_loss = None
        if "perplexity" in metrics:
            eval_loss = _compute_eval_loss(model, tokenizer, test_rows, device, is_seq2seq)
        overall = _compute_sft_metrics(predictions, references, metrics, eval_loss=eval_loss)

    # --- Slicing ----------------------------------------------------------
    sliced: Dict[str, Dict[str, Any]] = {}
    if slice_by:
        for tag_key in slice_by:
            groups = slice_by_tag(test_rows, tag_key)
            tag_slices: Dict[str, Any] = {}
            for tag_val, indices in groups.items():
                if task == "classification":
                    sl_pred = [pred_labels[i] for i in indices]
                    sl_ref = [ref_labels[i] for i in indices]
                    tag_slices[tag_val] = {"n": len(indices)}
                    tag_slices[tag_val].update(
                        _compute_cls_metrics(sl_pred, sl_ref, [m for m in metrics if m != "confusion_matrix"], []),
                    )
                else:
                    sl_pred = [predictions[i] for i in indices]
                    sl_ref = [references[i] for i in indices]
                    tag_slices[tag_val] = {"n": len(indices)}
                    tag_slices[tag_val].update(
                        _compute_sft_metrics(sl_pred, sl_ref, metrics),
                    )
            sliced[tag_key] = tag_slices

    # --- Hard examples ----------------------------------------------------
    hard: List[Dict[str, Any]] = []
    if task != "classification":
        scored = []
        for i, (p, r) in enumerate(zip(predictions, references)):
            f = token_f1(p, r)
            scored.append((f, i))
        scored.sort()
        for rank, (score, idx) in enumerate(scored[:hard_examples_n], 1):
            row = test_rows[idx]
            hard.append({
                "rank": rank,
                "f1": round(score, 4),
                "prediction": predictions[idx],
                "reference": references[idx],
                "input": str(row.get("instruction", row.get("text", "")))[:500],
                "tags": row.get("tags", {}),
            })

    elapsed = time.time() - start

    result = EvalResult(
        run_id=run.run_id,
        test_data_path=test_data,
        num_examples=len(test_rows),
        overall=overall,
        slices=sliced,
        hard_examples=hard,
        eval_duration_seconds=round(elapsed, 2),
    )

    # --- Persist ----------------------------------------------------------
    eval_dir = Path(run_path) / "eval"
    eval_dir.mkdir(exist_ok=True)

    with open(eval_dir / "eval_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    if hard:
        with open(eval_dir / "hard_examples.jsonl", "w") as f:
            for ex in hard:
                f.write(json.dumps(ex) + "\n")

    logger.info("Evaluation complete (%s). Results: %s", run.run_id, overall)
    return result


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _compute_eval_loss(model, tokenizer, rows, device, is_seq2seq):
    """Compute average cross-entropy loss on test data for perplexity."""
    total_loss = 0.0
    count = 0
    for row in rows:
        inp = row.get("input", "")
        instruction = row.get("instruction", row.get("question", ""))
        context = row.get("context", "")
        output = row.get("output", row.get("answer", ""))
        if context:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{output}"
        elif inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            if is_seq2seq:
                labels = tokenizer(output, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
                loss = model(**inputs, labels=labels).loss
            else:
                inputs["labels"] = inputs["input_ids"].clone()
                loss = model(**inputs).loss
        if loss is not None:
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def _predict_sft(model, tokenizer, rows, device, is_seq2seq):
    predictions = []
    references = []
    for row in tqdm(rows, desc="Generating predictions"):
        inp = row.get("input", "")
        instruction = row.get("instruction", row.get("question", ""))
        context = row.get("context", "")
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
        elif inp:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, num_beams=4,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        # For CausalLM, strip the prompt prefix from the output
        if not is_seq2seq and pred.startswith(prompt):
            pred = pred[len(prompt):]
        predictions.append(pred.strip())
        references.append(row.get("output", row.get("answer", "")))
    return predictions, references


def _predict_classification(model, tokenizer, rows, device):
    pred_labels = []
    ref_labels = []
    scores = []
    label2id = model.config.label2id
    for row in tqdm(rows, desc="Classifying"):
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = int(logits.argmax(-1).item())
        pred_labels.append(pred_id)
        ref_labels.append(label2id[row["label"]])
        scores.append(float(torch.softmax(logits, -1).max().item()))
    return pred_labels, ref_labels, scores
