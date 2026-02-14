"""Verifily Train utilities: hashing, seeding, device detection, logging."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_device(requested: str = "auto") -> str:
    """Detect the best available device.

    Args:
        requested: "auto", "cuda", "cuda:N", "mps", or "cpu".

    Returns:
        Device string suitable for torch.device().
    """
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_dict(d: Dict[str, Any]) -> str:
    """Compute SHA-256 of a JSON-serializable dict (sorted, compact)."""
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_run_id() -> str:
    """Generate a run ID from timestamp + short random hex."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(3).hex()
    return f"run_{ts}_{rand}"


def collect_environment() -> Dict[str, Any]:
    """Snapshot the Python and hardware environment."""
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": f"{platform.system()} {platform.release()}",
        "packages": {},
        "gpu": {},
    }

    # Package versions (best-effort)
    for pkg in ("torch", "transformers", "peft", "accelerate", "bitsandbytes",
                 "datasets", "evaluate", "verifily_train"):
        try:
            mod = __import__(pkg)
            env["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # GPU info
    if torch.cuda.is_available():
        env["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / (1 << 30), 1),
            "cuda_version": torch.version.cuda or "unknown",
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        env["gpu"] = {"name": "Apple MPS", "count": 1}

    return env


def format_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' or shorter."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"
