"""Verifily Train billing hooks: usage metering and event emission."""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """A single billing event."""
    run_id: str = ""
    event: str = ""  # "train_start", "train_end", "eval_start", "eval_end"
    timestamp: str = ""
    gpu_type: str = ""
    gpu_hours: float = 0.0
    tokens_processed: int = 0
    train_rows: int = 0
    eval_rows: int = 0
    storage_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BillingTracker:
    """Tracks resource usage for a training or eval job."""

    def __init__(self, run_id: str, artifact_path: str = ""):
        self.run_id = run_id
        self.artifact_path = artifact_path
        self._start_time: Optional[float] = None
        self._records: List[UsageRecord] = []
        self._hooks: List[Callable[[UsageRecord], None]] = []

        # Register webhook hook if env var is set
        webhook_url = os.environ.get("VERIFILY_BILLING_WEBHOOK")
        if webhook_url:
            self._hooks.append(_make_webhook_hook(webhook_url))

        # Register file hook if env var is set
        billing_log = os.environ.get("VERIFILY_BILLING_LOG")
        if billing_log:
            self._hooks.append(_make_file_hook(billing_log))

    def add_hook(self, hook: Callable[[UsageRecord], None]) -> None:
        """Register a custom billing hook. Called on every emit()."""
        self._hooks.append(hook)

    def start(self, event: str = "train_start", **kwargs) -> None:
        """Mark the start of a billable operation."""
        self._start_time = time.time()
        from verifily_train.utils import utcnow_iso, collect_environment
        env = collect_environment()
        gpu_type = env.get("gpu", {}).get("name", "cpu")

        record = UsageRecord(
            run_id=self.run_id,
            event=event,
            timestamp=utcnow_iso(),
            gpu_type=gpu_type,
            metadata=kwargs,
        )
        self._emit(record)

    def end(
        self,
        event: str = "train_end",
        tokens_processed: int = 0,
        train_rows: int = 0,
        eval_rows: int = 0,
        **kwargs,
    ) -> UsageRecord:
        """Mark the end of a billable operation and compute usage."""
        from verifily_train.utils import utcnow_iso, collect_environment
        env = collect_environment()
        gpu_type = env.get("gpu", {}).get("name", "cpu")

        elapsed = time.time() - (self._start_time or time.time())
        gpu_hours = round(elapsed / 3600, 4)

        # Compute storage used by artifact dir
        storage = 0
        if self.artifact_path:
            storage = _dir_size(self.artifact_path)

        record = UsageRecord(
            run_id=self.run_id,
            event=event,
            timestamp=utcnow_iso(),
            gpu_type=gpu_type,
            gpu_hours=gpu_hours,
            tokens_processed=tokens_processed,
            train_rows=train_rows,
            eval_rows=eval_rows,
            storage_bytes=storage,
            metadata=kwargs,
        )
        self._emit(record)

        # Persist usage summary in artifact dir
        if self.artifact_path:
            self._save_summary()

        return record

    def _emit(self, record: UsageRecord) -> None:
        """Emit a billing record to all registered hooks."""
        self._records.append(record)
        for hook in self._hooks:
            try:
                hook(record)
            except Exception as e:
                logger.warning("Billing hook error: %s", e)

    def _save_summary(self) -> None:
        """Save usage_summary.json to the artifact directory."""
        p = Path(self.artifact_path) / "usage_summary.json"
        total_gpu_hours = sum(r.gpu_hours for r in self._records)
        total_tokens = sum(r.tokens_processed for r in self._records)
        total_storage = max((r.storage_bytes for r in self._records), default=0)

        summary = {
            "run_id": self.run_id,
            "total_gpu_hours": round(total_gpu_hours, 4),
            "total_tokens_processed": total_tokens,
            "storage_bytes": total_storage,
            "storage_mb": round(total_storage / (1024 * 1024), 2),
            "events": [asdict(r) for r in self._records],
        }
        with open(p, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug("Billing summary saved to %s", p)


def _dir_size(path: str) -> int:
    """Total size of all files in a directory tree."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _make_webhook_hook(url: str) -> Callable[[UsageRecord], None]:
    """Create a hook that POSTs billing events to a webhook URL."""
    def hook(record: UsageRecord) -> None:
        try:
            import urllib.request
            data = json.dumps(asdict(record)).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug("Webhook billing hook failed: %s", e)
    return hook


def _make_file_hook(path: str) -> Callable[[UsageRecord], None]:
    """Create a hook that appends billing events to a JSONL file."""
    def hook(record: UsageRecord) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
    return hook
