"""Verifily Train reproducibility verification."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from verifily_train.errors import ReproduceError
from verifily_train.utils import collect_environment, sha256_dict, sha256_file

logger = logging.getLogger(__name__)


@dataclass
class ReproduceResult:
    run_id: str = ""
    config_match: bool = False
    data_match: bool = False
    env_match: bool = False
    verdict: str = "UNKNOWN"
    details: str = ""
    # If full reproduction was done:
    original_metrics: Optional[dict] = None
    reproduced_metrics: Optional[dict] = None


def verify(run_path: str) -> ReproduceResult:
    """Verify reproducibility hashes of a run without retraining.

    Checks:
    - config_hash matches the saved config
    - data_hash matches the training data file
    - environment_hash matches the current environment
    """
    rp = Path(run_path)

    # Load stored hashes
    hashes_path = rp / "hashes.json"
    if not hashes_path.exists():
        raise ReproduceError(f"hashes.json not found in {run_path}")
    with open(hashes_path) as f:
        stored = json.load(f)

    # Load stored config
    config_path = rp / "config.yaml"
    if not config_path.exists():
        raise ReproduceError(f"config.yaml not found in {run_path}")

    import yaml
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Load run meta
    meta_path = rp / "run_meta.json"
    run_id = "unknown"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        run_id = meta.get("run_id", "unknown")

    # Recompute config hash
    current_config_hash = sha256_dict(config_dict)
    config_match = current_config_hash == stored.get("config_hash", "")

    # Recompute data hash
    train_path = (config_dict.get("data_paths") or {}).get("train", "")
    data_match = False
    data_detail = ""
    if train_path and Path(train_path).exists():
        current_data_hash = sha256_file(train_path)
        data_match = current_data_hash == stored.get("data_hash", "")
        if not data_match:
            data_detail = f"expected {stored.get('data_hash', '?')}, got {current_data_hash}"
    else:
        data_detail = f"training data not found at {train_path}"

    # Recompute environment hash
    current_env_hash = sha256_dict(collect_environment())
    env_match = current_env_hash == stored.get("environment_hash", "")

    # Verdict
    if config_match and data_match and env_match:
        verdict = "FULL MATCH"
    elif config_match and data_match:
        verdict = "PARTIAL MATCH (environment differs)"
    elif config_match:
        verdict = "CONFIG MATCH ONLY"
    else:
        verdict = "MISMATCH"

    details_parts = []
    details_parts.append(f"config_hash:   {'MATCH' if config_match else 'MISMATCH'}")
    details_parts.append(f"data_hash:     {'MATCH' if data_match else 'MISMATCH'}" +
                         (f" ({data_detail})" if data_detail else ""))
    details_parts.append(f"env_hash:      {'MATCH' if env_match else 'MISMATCH'}")

    result = ReproduceResult(
        run_id=run_id,
        config_match=config_match,
        data_match=data_match,
        env_match=env_match,
        verdict=verdict,
        details="\n".join(details_parts),
    )

    logger.info("Reproducibility check: %s — %s", run_id, verdict)
    return result


def format_verify_result(r: ReproduceResult) -> str:
    """Format a ReproduceResult for terminal output."""
    lines = [
        f"Reproducibility check: {r.run_id}",
        "",
        f"  {r.details}",
        "",
        f"  Verdict: {r.verdict}",
    ]
    if r.verdict != "FULL MATCH":
        lines.append("  Note: Metric differences may occur due to environment mismatch.")
    return "\n".join(lines)
