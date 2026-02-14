"""Verifily Train Run artifact management."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from verifily_train.utils import (
    ensure_dir,
    generate_run_id,
    sha256_dict,
    sha256_file,
    utcnow_iso,
    collect_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class Run:
    """Immutable record of a completed (or failed) training job."""

    run_id: str = ""
    status: str = "pending"  # pending | running | completed | failed
    task: str = "sft"
    base_model: str = ""
    dataset_version: str = ""
    compute_mode: str = "local"
    device: str = "cpu"
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""
    config_hash: str = ""
    data_hash: str = ""
    environment_hash: str = ""
    reproducibility_hash: str = ""
    seed: int = 42
    run_name: Optional[str] = None

    # ---- persistence -----------------------------------------------------

    def save_meta(self) -> None:
        """Write run_meta.json into the artifact directory."""
        p = Path(self.artifact_path) / "run_meta.json"
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.debug("Saved run_meta.json to %s", p)

    def save_hashes(self) -> None:
        """Write hashes.json into the artifact directory."""
        p = Path(self.artifact_path) / "hashes.json"
        payload = {
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "environment_hash": self.environment_hash,
            "reproducibility_hash": self.reproducibility_hash,
            "hash_algorithm": "sha256",
        }
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)

    def save_environment(self) -> None:
        """Snapshot environment.json."""
        p = Path(self.artifact_path) / "environment.json"
        with open(p, "w") as f:
            json.dump(collect_environment(), f, indent=2)

    @classmethod
    def load(cls, run_dir: str) -> "Run":
        """Load a Run from its artifact directory."""
        meta_path = Path(run_dir) / "run_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"run_meta.json not found in {run_dir}")
        with open(meta_path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Run directory factory
# ---------------------------------------------------------------------------

def create_run_dir(
    base_dir: str,
    config_dict: Dict[str, Any],
    data_hash: str,
    seed: int,
    run_name: Optional[str] = None,
) -> Run:
    """Create a fresh Run with an empty artifact directory.

    Returns a Run in 'pending' status.  The caller is responsible for
    updating status, metrics, and calling save_*().
    """
    run_id = generate_run_id()
    artifact_path = str(Path(base_dir) / run_id)
    ensure_dir(artifact_path)
    ensure_dir(str(Path(artifact_path) / "adapter"))
    ensure_dir(str(Path(artifact_path) / "tokenizer"))
    ensure_dir(str(Path(artifact_path) / "eval"))
    ensure_dir(str(Path(artifact_path) / "checkpoints"))

    config_hash = sha256_dict(config_dict)
    env = collect_environment()
    environment_hash = sha256_dict(env)

    repro_payload = f"{config_hash}|{data_hash}|{environment_hash}|{seed}"
    reproducibility_hash = f"sha256:{__import__('hashlib').sha256(repro_payload.encode()).hexdigest()}"

    run = Run(
        run_id=run_id,
        status="pending",
        artifact_path=artifact_path,
        config_hash=config_hash,
        data_hash=data_hash,
        environment_hash=environment_hash,
        reproducibility_hash=reproducibility_hash,
        seed=seed,
        run_name=run_name,
        started_at=utcnow_iso(),
    )

    return run
