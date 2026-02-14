"""Verifily Train training orchestration for SFT and Classification."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from verifily_train.config import TrainConfig
from verifily_train.dataset import DatasetVersion, load_jsonl, load_multi_jsonl, validate_dataset
from verifily_train.errors import ConfigError, TrainingError
from verifily_train.run import Run, create_run_dir
from verifily_train.utils import (
    detect_device,
    format_duration,
    set_seed,
    sha256_file,
    utcnow_iso,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

_SEQ2SEQ_PREFIXES = (
    "t5", "bart", "mbart", "pegasus", "mt5", "longt5", "flan-t5",
    "bigbird-pegasus", "plbart", "led",
)


def _is_seq2seq(model_name: str) -> bool:
    """Heuristic: check if the model is a Seq2Seq architecture."""
    lower = model_name.lower().replace("/", "-")
    return any(prefix in lower for prefix in _SEQ2SEQ_PREFIXES)


# ---------------------------------------------------------------------------
# LoRA / QLoRA setup
# ---------------------------------------------------------------------------

def _apply_lora(model, config: TrainConfig, task: str, is_seq2seq: bool):
    """Wrap the model with PEFT LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model

    if is_seq2seq:
        peft_task = TaskType.SEQ_2_SEQ_LM
    elif task == "classification":
        peft_task = TaskType.SEQ_CLS
    else:
        peft_task = TaskType.CAUSAL_LM

    target_modules = config.lora.target_modules
    if target_modules == "auto":
        target_modules = None  # PEFT auto-detect

    lora_cfg = LoraConfig(
        task_type=peft_task,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied: %s trainable / %s total (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model


def _maybe_quantize(model_name: str, quantization: str, compute_dtype_str: str):
    """Return model kwargs for BitsAndBytes quantization."""
    if quantization == "none":
        return {}

    from transformers import BitsAndBytesConfig

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(compute_dtype_str, torch.bfloat16)

    if quantization == "4bit":
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }
    elif quantization == "8bit":
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        }
    return {}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_sft_dataset(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    is_seq2seq: bool,
) -> Dataset:
    """Convert SFT JSONL rows to a tokenised HF Dataset."""
    inputs = []
    targets = []
    for row in rows:
        inp = row.get("input", "")
        instruction = row["instruction"]
        if inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n"
        inputs.append(text)
        targets.append(row["output"])

    if is_seq2seq:
        ds = Dataset.from_dict({"input_text": inputs, "target_text": targets})

        def tok_fn(examples):
            model_inputs = tokenizer(
                examples["input_text"],
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
            )
            labels = tokenizer(
                text_target=examples["target_text"],
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return ds.map(tok_fn, batched=True, remove_columns=["input_text", "target_text"])
    else:
        # CausalLM: concat input + target
        full_texts = [f"{i}{t}" for i, t in zip(inputs, targets)]
        ds = Dataset.from_dict({"text": full_texts})

        def tok_fn(examples):
            return tokenizer(
                examples["text"],
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
            )

        return ds.map(tok_fn, batched=True, remove_columns=["text"])


def _prepare_cls_dataset(
    rows: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    label2id: Dict[str, int],
) -> Dataset:
    """Convert classification JSONL rows to a tokenised HF Dataset."""
    texts = [r["text"] for r in rows]
    labels = [label2id[r["label"]] for r in rows]
    ds = Dataset.from_dict({"text": texts, "label": labels})

    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )

    return ds.map(tok_fn, batched=True, remove_columns=["text"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(config: TrainConfig, dry_run: bool = False) -> Run:
    """Execute a training job.

    Returns a Run with status='completed' or raises TrainingError.
    """
    set_seed(config.seed)
    device = detect_device(config.compute.device)
    is_seq2seq = _is_seq2seq(config.base_model)

    logger.info("=" * 60)
    logger.info("VERIFILY TRAIN — Starting training job")
    logger.info("  Task:       %s", config.task)
    logger.info("  Base model: %s", config.base_model)
    logger.info("  Device:     %s", device)
    logger.info("  LoRA:       %s (r=%d)", config.lora.enabled, config.lora.r)
    logger.info("  Quant:      %s", config.lora.quantization)
    logger.info("  Distributed: %s", config.compute.distributed)
    logger.info("=" * 60)

    # --- Resolve dataset --------------------------------------------------
    train_paths = config.data_paths.train
    is_multi = isinstance(train_paths, list)

    if is_multi:
        # Multi-dataset mode: use first path for DatasetVersion metadata
        dsv = DatasetVersion.from_local_paths(
            train_path=train_paths[0],
            task=config.task,
            val_path=config.data_paths.val,
            test_path=config.data_paths.test,
        )
        train_rows = load_multi_jsonl(train_paths, weights=config.data_paths.weights)
        logger.info("Multi-dataset: %d training rows from %d files", len(train_rows), len(train_paths))
    else:
        dsv = DatasetVersion.from_local_paths(
            train_path=train_paths,
            task=config.task,
            val_path=config.data_paths.val,
            test_path=config.data_paths.test,
        )
        train_rows = load_jsonl(dsv.train_path)
        logger.info("Dataset: %d training rows, hash=%s", dsv.num_rows, dsv.content_hash)

    validate_dataset(train_rows, config.task)

    val_rows = None
    if dsv.val_path:
        val_rows = load_jsonl(dsv.val_path)

    # --- Create run dir ---------------------------------------------------
    run = create_run_dir(
        base_dir=config.output.dir,
        config_dict=config.to_dict(),
        data_hash=dsv.content_hash,
        seed=config.seed,
        run_name=config.name,
    )
    run.task = config.task
    run.base_model = config.base_model
    run.dataset_version = config.dataset_version or "local"
    run.compute_mode = config.compute.mode
    run.device = device

    # Save config to run dir
    config.to_yaml(str(Path(run.artifact_path) / "config.yaml"))
    run.save_environment()

    if dry_run:
        logger.info("DRY RUN — config validated, run directory prepared at %s", run.artifact_path)
        run.status = "dry_run"
        run.save_meta()
        return run

    # --- Billing ----------------------------------------------------------
    from verifily_train.billing import BillingTracker
    billing = BillingTracker(run_id=run.run_id, artifact_path=run.artifact_path)
    billing.start(event="train_start", base_model=config.base_model, task=config.task)

    # --- Load model -------------------------------------------------------
    run.status = "running"
    start = time.time()

    quant_kwargs = _maybe_quantize(
        config.base_model, config.lora.quantization, config.lora.bnb_4bit_compute_dtype,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if config.task == "classification":
            all_labels = sorted({r["label"] for r in train_rows})
            label2id = {l: i for i, l in enumerate(all_labels)}
            id2label = {i: l for l, i in label2id.items()}
            model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model,
                num_labels=len(all_labels),
                label2id=label2id,
                id2label=id2label,
                **quant_kwargs,
            )
        elif is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model, **quant_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.base_model, **quant_kwargs)

        # Move to device (skip if quantised — already placed by BnB)
        if config.lora.quantization == "none":
            model = model.to(device)

        # LoRA
        if config.lora.enabled:
            model = _apply_lora(model, config, config.task, is_seq2seq)

        # --- Tokenise data ------------------------------------------------
        if config.task == "classification":
            train_ds = _prepare_cls_dataset(
                train_rows, tokenizer, config.training.max_seq_length, label2id,
            )
            val_ds = (
                _prepare_cls_dataset(val_rows, tokenizer, config.training.max_seq_length, label2id)
                if val_rows else None
            )
        else:
            train_ds = _prepare_sft_dataset(
                train_rows, tokenizer, config.training.max_seq_length, is_seq2seq,
            )
            val_ds = (
                _prepare_sft_dataset(val_rows, tokenizer, config.training.max_seq_length, is_seq2seq)
                if val_rows else None
            )

        # --- Training args ------------------------------------------------
        common_args = dict(
            output_dir=run.artifact_path,
            num_train_epochs=config.training.num_epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.eval.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            lr_scheduler_type=config.training.lr_scheduler,
            warmup_ratio=config.training.warmup_ratio,
            weight_decay=config.training.weight_decay,
            max_grad_norm=config.training.max_grad_norm,
            logging_dir=str(Path(run.artifact_path) / "logs"),
            logging_steps=config.training.logging_steps,
            run_name=config.name or run.run_id,
            eval_strategy="steps" if val_ds else "no",
            eval_steps=config.training.eval_steps if val_ds else None,
            save_strategy="steps",
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            load_best_model_at_end=bool(val_ds),
            metric_for_best_model="loss" if val_ds else None,
            greater_is_better=False,
            fp16=config.compute.fp16 and device == "cuda",
            bf16=config.compute.bf16 and device == "cuda",
            seed=config.seed,
            report_to=["wandb"] if config.wandb.enabled else [],
            ddp_find_unused_parameters=False if config.compute.distributed else None,
            dataloader_num_workers=4 if config.compute.distributed else 0,
        )

        # WandB init (before Trainer creation so env vars are set)
        if config.wandb.enabled:
            try:
                import wandb
                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    name=config.name or run.run_id,
                    config=config.to_dict(),
                    tags=config.wandb.tags or [],
                    reinit=True,
                )
            except ImportError:
                logger.warning("wandb not installed, disabling WandB logging")
                common_args["report_to"] = []

        if is_seq2seq and config.task == "sft":
            training_args = Seq2SeqTrainingArguments(
                predict_with_generate=True,
                generation_max_length=config.eval.generation_max_new_tokens,
                **common_args,
            )
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
            trainer_cls = Seq2SeqTrainer
        elif config.task == "classification":
            training_args = TrainingArguments(**common_args)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            trainer_cls = Trainer
        else:
            training_args = TrainingArguments(**common_args)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            trainer_cls = Trainer

        from verifily_train.callbacks import VerifilyProgressCallback

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            callbacks=[VerifilyProgressCallback()],
        )

        # --- Train --------------------------------------------------------
        logger.info("Starting training…")
        result = trainer.train()

        # --- Save ---------------------------------------------------------
        if config.output.save_adapter_only and config.lora.enabled:
            model.save_pretrained(str(Path(run.artifact_path) / "adapter"))
        else:
            trainer.save_model(run.artifact_path)
        tokenizer.save_pretrained(str(Path(run.artifact_path) / "tokenizer"))

        # Training summary
        elapsed = time.time() - start
        run.status = "completed"
        run.completed_at = utcnow_iso()
        run.duration_seconds = round(elapsed, 2)
        run.metrics = {
            "train_loss": round(result.metrics.get("train_loss", 0.0), 6),
        }
        if val_ds:
            run.metrics["eval_loss"] = round(result.metrics.get("eval_loss", 0.0), 6)

        # Write train_summary.json
        summary = {
            "total_steps": result.global_step,
            "total_epochs": config.training.num_epochs,
            "final_train_loss": run.metrics.get("train_loss"),
            "wall_time_seconds": run.duration_seconds,
        }
        with open(Path(run.artifact_path) / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Write training_info.json (for eval compatibility)
        info = {
            "base_model": config.base_model,
            "task": config.task,
            "use_lora": config.lora.enabled,
            "seed": config.seed,
            "train_data_path": config.data_paths.train,
            "val_data_path": config.data_paths.val,
        }
        with open(Path(run.artifact_path) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        run.save_meta()
        run.save_hashes()

        # Billing: end event
        billing.end(
            event="train_end",
            tokens_processed=result.global_step * config.training.batch_size * config.training.max_seq_length,
            train_rows=len(train_rows),
        )

        # WandB: log final metrics + optional model artifact
        if config.wandb.enabled:
            try:
                import wandb
                wandb.log(run.metrics)
                if config.wandb.log_model:
                    artifact = wandb.Artifact(f"model-{run.run_id}", type="model")
                    artifact.add_dir(run.artifact_path)
                    wandb.log_artifact(artifact)
                wandb.finish()
            except Exception:
                pass

        logger.info("=" * 60)
        logger.info("Training completed: %s", run.run_id)
        logger.info("  Duration:    %s", format_duration(elapsed))
        logger.info("  Train loss:  %.4f", run.metrics.get("train_loss", 0))
        logger.info("  Artifacts:   %s", run.artifact_path)
        logger.info("  Repro hash:  %s", run.reproducibility_hash)
        logger.info("=" * 60)

        return run

    except Exception as e:
        run.status = "failed"
        run.completed_at = utcnow_iso()
        run.duration_seconds = round(time.time() - start, 2)
        run.save_meta()
        raise TrainingError(f"Training failed: {e}") from e
