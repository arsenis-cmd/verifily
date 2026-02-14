"""Verifily Train configuration: loading, validation, schema."""

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from verifily_train.errors import ConfigError

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataPaths:
    train: Union[str, List[str]] = ""  # single path or list for multi-dataset
    val: Optional[str] = None
    test: Optional[str] = None
    weights: Optional[List[float]] = None  # per-dataset sampling weights (multi-dataset)


@dataclass
class TrainingParams:
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 2


@dataclass
class LoraParams:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Union[str, List[str]] = "auto"
    quantization: str = "none"  # "none", "4bit", "8bit"
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class ComputeParams:
    mode: str = "local"  # "local" | "managed"
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    fp16: bool = False
    bf16: bool = True
    distributed: bool = False  # enable multi-GPU via Accelerate
    num_gpus: Optional[int] = None  # None = all available


@dataclass
class EvalParams:
    enabled: bool = True
    metrics: Optional[List[str]] = None  # task-dependent defaults applied later
    gold_set: Optional[str] = None
    slice_by_tags: Optional[List[str]] = None
    hard_examples: int = 50
    batch_size: int = 16
    generation_max_new_tokens: int = 128
    generation_num_beams: int = 4


@dataclass
class WandbParams:
    enabled: bool = False
    project: str = "verifily-train"
    entity: Optional[str] = None
    tags: Optional[List[str]] = None
    log_model: bool = False


@dataclass
class OutputParams:
    dir: str = "runs/"
    save_adapter_only: bool = True
    push_to_hub: bool = False


@dataclass
class TrainConfig:
    """Full training configuration."""

    task: str = "sft"  # "sft" | "classification"
    base_model: str = ""
    dataset_version: Optional[str] = None
    data_paths: DataPaths = field(default_factory=DataPaths)
    training: TrainingParams = field(default_factory=TrainingParams)
    lora: LoraParams = field(default_factory=LoraParams)
    compute: ComputeParams = field(default_factory=ComputeParams)
    eval: EvalParams = field(default_factory=EvalParams)
    output: OutputParams = field(default_factory=OutputParams)
    wandb: WandbParams = field(default_factory=WandbParams)
    seed: int = 42
    name: Optional[str] = None

    # ----- class methods -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """Load from a YAML file and validate."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(p) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError(f"Config file must be a YAML mapping: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Build config from a plain dict, applying defaults."""
        cfg = cls()
        cfg.task = d.get("task", cfg.task)
        cfg.base_model = d.get("base_model", cfg.base_model)
        cfg.dataset_version = d.get("dataset_version")
        cfg.seed = d.get("seed", cfg.seed)
        cfg.name = d.get("name")

        # data_paths
        dp = d.get("data_paths", {})
        if isinstance(dp, dict):
            cfg.data_paths = DataPaths(
                train=dp.get("train", ""),
                val=dp.get("val"),
                test=dp.get("test"),
                weights=dp.get("weights"),
            )

        # training
        tr = d.get("training", {})
        if isinstance(tr, dict):
            for k, v in tr.items():
                if hasattr(cfg.training, k):
                    setattr(cfg.training, k, v)

        # lora
        lo = d.get("lora", {})
        if isinstance(lo, dict):
            for k, v in lo.items():
                if hasattr(cfg.lora, k):
                    setattr(cfg.lora, k, v)

        # compute
        co = d.get("compute", {})
        if isinstance(co, dict):
            for k, v in co.items():
                if hasattr(cfg.compute, k):
                    setattr(cfg.compute, k, v)

        # eval
        ev = d.get("eval", {})
        if isinstance(ev, dict):
            for k, v in ev.items():
                if hasattr(cfg.eval, k):
                    setattr(cfg.eval, k, v)

        # output
        ou = d.get("output", {})
        if isinstance(ou, dict):
            for k, v in ou.items():
                if hasattr(cfg.output, k):
                    setattr(cfg.output, k, v)

        # wandb
        wb = d.get("wandb", {})
        if isinstance(wb, dict):
            for k, v in wb.items():
                if hasattr(cfg.wandb, k):
                    setattr(cfg.wandb, k, v)

        cfg.validate()
        return cfg

    # ----- mutation -------------------------------------------------------

    def merge_cli_overrides(self, **kwargs) -> "TrainConfig":
        """Return a new TrainConfig with CLI flag overrides applied."""
        cfg = copy.deepcopy(self)
        mapping = {
            "base_model": "base_model",
            "task": "task",
            "seed": "seed",
            "name": "name",
            "dataset": "dataset_version",
            "output_dir": ("output", "dir"),
            "device": ("compute", "device"),
            "epochs": ("training", "num_epochs"),
            "batch_size": ("training", "batch_size"),
            "lr": ("training", "learning_rate"),
            "lora_r": ("lora", "r"),
            "quantization": ("lora", "quantization"),
        }
        for cli_key, target in mapping.items():
            val = kwargs.get(cli_key)
            if val is None:
                continue
            if isinstance(target, str):
                setattr(cfg, target, val)
            else:
                parent, attr = target
                setattr(getattr(cfg, parent), attr, val)
        cfg.validate()
        return cfg

    # ----- validation ----------------------------------------------------

    def validate(self) -> None:
        """Raise ConfigError if the configuration is invalid."""
        if self.task not in ("sft", "classification"):
            raise ConfigError(f"task must be 'sft' or 'classification', got '{self.task}'")
        if not self.base_model:
            raise ConfigError("base_model is required")
        if not self.dataset_version and not self.data_paths.train:
            raise ConfigError("Either dataset_version or data_paths.train is required")
        if self.lora.quantization not in ("none", "4bit", "8bit"):
            raise ConfigError(
                f"lora.quantization must be 'none', '4bit', or '8bit', "
                f"got '{self.lora.quantization}'"
            )

    # ----- serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
