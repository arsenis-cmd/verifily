"""Utility functions for reproducibility, hardware detection, and logging."""
import os
import random
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def hash_string(text: str) -> str:
    """Generate SHA256 hash of a string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_dict(d: Dict) -> str:
    """Generate hash of a dictionary (for config tracking)."""
    # Sort keys for consistent hashing
    json_str = json.dumps(d, sort_keys=True)
    return hash_string(json_str)


def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and return configuration."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": None,
        "total_memory_gb": None,
        "available_memory_gb": None,
        "recommended_device": "cpu",
        "recommended_model": "google/flan-t5-small",
        "recommended_batch_size": 4,
        "recommended_dataset_size": "small",
        "use_fp16": False
    }

    # Check for Apple Silicon MPS
    if info["mps_available"]:
        import os
        info["cuda_device_name"] = "Apple Silicon (MPS)"

        # Estimate available memory (MPS uses unified memory)
        # Assume M4 Max has 36-64GB unified memory, be conservative
        info["total_memory_gb"] = 32  # Conservative estimate
        info["available_memory_gb"] = 24  # Leave room for system

        info["recommended_device"] = "mps"
        info["use_fp16"] = False  # Apple Silicon prefers FP32/BF16

        # MPS can handle larger models with unified memory
        info["recommended_model"] = "google/flan-t5-base"
        info["recommended_batch_size"] = 8  # Start conservative
        info["recommended_dataset_size"] = "large"

    elif torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        info["total_memory_gb"] = total_memory / (1024**3)

        # Clear cache and get available memory
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0)
        info["available_memory_gb"] = (total_memory - allocated) / (1024**3)

        info["recommended_device"] = "cuda"
        info["use_fp16"] = True  # NVIDIA GPUs benefit from FP16

        # Recommend model and settings based on available memory
        available_gb = info["available_memory_gb"]

        if available_gb >= 12:
            info["recommended_model"] = "google/flan-t5-large"
            info["recommended_batch_size"] = 8
            info["recommended_dataset_size"] = "large"
        elif available_gb >= 6:
            info["recommended_model"] = "google/flan-t5-base"
            info["recommended_batch_size"] = 8
            info["recommended_dataset_size"] = "medium"
        else:
            info["recommended_model"] = "google/flan-t5-small"
            info["recommended_batch_size"] = 4
            info["recommended_dataset_size"] = "small"

    return info


def adjust_config_for_hardware(config: Dict, hardware_info: Dict) -> Dict:
    """Adjust configuration based on detected hardware."""
    adjusted = config.copy()

    # Set device
    if config.get("device") == "auto":
        adjusted["device"] = hardware_info["recommended_device"]

    # Set model
    if config.get("base_model") == "auto":
        adjusted["base_model"] = hardware_info["recommended_model"]

    # Adjust dataset sizes based on hardware
    if hardware_info["recommended_dataset_size"] == "small":
        adjusted["human_train_size"] = min(5000, adjusted.get("human_train_size", 5000))
        adjusted["human_val_size"] = min(500, adjusted.get("human_val_size", 500))
        adjusted["human_test_size"] = min(1000, adjusted.get("human_test_size", 1000))
    elif hardware_info["recommended_dataset_size"] == "medium":
        adjusted["human_train_size"] = min(15000, adjusted.get("human_train_size", 15000))
        adjusted["human_val_size"] = min(1500, adjusted.get("human_val_size", 1500))
        adjusted["human_test_size"] = min(2000, adjusted.get("human_test_size", 2000))
    # else keep large defaults

    # Adjust batch size
    if "training" in adjusted:
        adjusted["training"]["batch_size"] = hardware_info["recommended_batch_size"]

    # Set FP16 based on hardware
    if "training" in adjusted:
        if adjusted["device"] == "mps":
            # Apple Silicon: use FP32 (better stability)
            adjusted["training"]["fp16"] = False
        elif adjusted["device"] == "cuda":
            # NVIDIA: use FP16 (faster)
            adjusted["training"]["fp16"] = True
        else:
            # CPU: no FP16
            adjusted["training"]["fp16"] = False

    return adjusted


def load_config(config_path: str) -> Dict:
    """Load and validate configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: str):
    """Save configuration to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_environment_info(output_path: str):
    """Save environment information for reproducibility."""
    info = {
        "python_version": subprocess.check_output(
            ["python", "--version"], encoding="utf-8"
        ).strip(),
        "pip_freeze": subprocess.check_output(
            ["pip", "freeze"], encoding="utf-8"
        ).strip().split("\n"),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["torch_version"] = torch.__version__

    # Try to get git info (optional)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8", stderr=subprocess.DEVNULL
        ).strip()
        info["git_commit"] = git_hash
    except:
        info["git_commit"] = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)


def compute_ngram_overlap(text1: str, text2: str, n: int = 8) -> float:
    """
    Compute n-gram overlap ratio between two texts.
    Returns fraction of n-grams in text1 that appear in text2.
    """
    def get_ngrams(text: str, n: int):
        tokens = text.lower().split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if len(ngrams1) == 0:
        return 0.0

    overlap = len(ngrams1.intersection(ngrams2))
    return overlap / len(ngrams1)


def normalize_answer(s: str) -> str:
    """Normalize answer text for evaluation (lowercase, strip, etc.)."""
    import re
    import string

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score (0 or 1)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def print_hardware_info(hardware_info: Dict):
    """Pretty print hardware information."""
    print("\n" + "="*50)
    print("HARDWARE DETECTION")
    print("="*50)
    print(f"CUDA Available: {hardware_info['cuda_available']}")
    print(f"MPS Available: {hardware_info.get('mps_available', False)}")

    if hardware_info.get('mps_available', False):
        print(f"Device: {hardware_info['cuda_device_name']}")
        print(f"Estimated Memory: {hardware_info['available_memory_gb']:.2f} GB (unified)")
    elif hardware_info['cuda_available']:
        print(f"Device: {hardware_info['cuda_device_name']}")
        print(f"Total Memory: {hardware_info['total_memory_gb']:.2f} GB")
        print(f"Available Memory: {hardware_info['available_memory_gb']:.2f} GB")

    print(f"\nRecommended Settings:")
    print(f"  Device: {hardware_info['recommended_device']}")
    print(f"  Model: {hardware_info['recommended_model']}")
    print(f"  Batch Size: {hardware_info['recommended_batch_size']}")
    print(f"  Dataset Size: {hardware_info['recommended_dataset_size']}")
    print(f"  FP16: {hardware_info.get('use_fp16', False)}")
    print("="*50 + "\n")


def print_config_summary(config: Dict):
    """Pretty print configuration summary."""
    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)
    print(f"Experiment: {config.get('experiment_name', 'N/A')}")
    print(f"Seed: {config.get('seed', 'N/A')}")
    print(f"Base Model: {config.get('base_model', 'N/A')}")
    print(f"Device: {config.get('device', 'N/A')}")
    print(f"\nDataset Sizes:")
    print(f"  Train: {config.get('human_train_size', 'N/A')}")
    print(f"  Val: {config.get('human_val_size', 'N/A')}")
    print(f"  Test: {config.get('human_test_size', 'N/A')}")
    print(f"  Synthetic Multiplier: {config.get('synthetic_multiplier', 'N/A')}x")

    if "training" in config:
        print(f"\nTraining:")
        print(f"  Epochs: {config['training'].get('num_epochs', 'N/A')}")
        print(f"  Batch Size: {config['training'].get('batch_size', 'N/A')}")
        print(f"  Learning Rate: {config['training'].get('learning_rate', 'N/A')}")
        print(f"  Use LoRA: {config['training'].get('use_lora', 'N/A')}")

    print("="*50 + "\n")
