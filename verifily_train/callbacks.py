"""Verifily Train callbacks for training progress reporting."""

import logging
import time
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class VerifilyProgressCallback(TrainerCallback):
    """Rich progress callback that logs loss, lr, throughput, and ETA."""

    def __init__(self):
        self._start_time: Optional[float] = None
        self._total_steps: int = 0
        self._current_epoch: int = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        self._start_time = time.time()
        self._total_steps = state.max_steps
        logger.info("Training started: %d total steps", self._total_steps)

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        elapsed = time.time() - (self._start_time or time.time())
        steps_per_sec = step / max(elapsed, 1e-6)
        remaining = (self._total_steps - step) / max(steps_per_sec, 1e-6)

        parts = [f"Step {step}/{self._total_steps}"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        parts.append(f"{steps_per_sec:.1f} steps/s")
        parts.append(f"ETA {_fmt_time(remaining)}")

        logger.info(" | ".join(parts))

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        self._current_epoch += 1
        logger.info("Epoch %d/%d", self._current_epoch, int(args.num_train_epochs))

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        elapsed = time.time() - (self._start_time or time.time())
        logger.info(
            "Training finished: %d steps in %s (%.1f steps/s)",
            state.global_step, _fmt_time(elapsed),
            state.global_step / max(elapsed, 1e-6),
        )


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"
