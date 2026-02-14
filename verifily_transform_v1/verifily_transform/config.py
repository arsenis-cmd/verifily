"""Verifily Transform configuration: loading, validation, schema."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from verifily_transform.errors import ConfigError


@dataclass
class InputConfig:
    path: str = ""
    format: str = "jsonl"  # "jsonl", "csv", "txt"
    encoding: str = "utf-8"
    text_column: Optional[str] = None  # for CSV: which column is the text
    label_column: Optional[str] = None  # for CSV: which column is the label


@dataclass
class OutputConfig:
    dir: str = "datasets/"
    name: str = "dataset_v1"


@dataclass
class LabelingConfig:
    task: str = "sft"  # "sft" or "classification"
    strategy: str = "heuristic"  # "heuristic", "llm", "heuristic+llm"
    instruction_field: Optional[str] = None  # field to use as instruction
    output_field: Optional[str] = None  # field to use as output
    llm_model: Optional[str] = None  # model for LLM-based labeling
    llm_api_key_env: str = "OPENAI_API_KEY"
    label_map: Optional[Dict[str, str]] = None  # remap raw labels


@dataclass
class SyntheticFilters:
    min_length: int = 20
    max_length: int = 512
    leakage_check: bool = True


@dataclass
class SyntheticConfig:
    enabled: bool = False
    expansion_factor: int = 5
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 512
    batch_size: int = 10
    filters: SyntheticFilters = field(default_factory=SyntheticFilters)


@dataclass
class DedupeConfig:
    exact: bool = True
    fuzzy: bool = True
    fuzzy_threshold: float = 0.85  # Jaccard similarity threshold
    num_perm: int = 128  # MinHash permutations


@dataclass
class PrivacyConfig:
    pii_removal: bool = False
    audit_log: bool = False


@dataclass
class TransformConfig:
    """Full transformation pipeline configuration."""

    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    dedupe: DedupeConfig = field(default_factory=DedupeConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TransformConfig":
        """Load config from YAML."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config not found: {path}")
        with open(p) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError(f"Config must be a YAML mapping: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformConfig":
        """Build config from a dict."""
        cfg = cls()
        cfg.seed = d.get("seed", cfg.seed)

        # input
        inp = d.get("input", {})
        if isinstance(inp, dict):
            for k, v in inp.items():
                if hasattr(cfg.input, k):
                    setattr(cfg.input, k, v)

        # output
        out = d.get("output", {})
        if isinstance(out, dict):
            for k, v in out.items():
                if hasattr(cfg.output, k):
                    setattr(cfg.output, k, v)

        # labeling
        lab = d.get("labeling", {})
        if isinstance(lab, dict):
            for k, v in lab.items():
                if hasattr(cfg.labeling, k):
                    setattr(cfg.labeling, k, v)

        # synthetic
        syn = d.get("synthetic", {})
        if isinstance(syn, dict):
            filters_raw = syn.get("filters", {})
            for k, v in syn.items():
                if k != "filters" and hasattr(cfg.synthetic, k):
                    setattr(cfg.synthetic, k, v)
            if isinstance(filters_raw, dict):
                for k, v in filters_raw.items():
                    if hasattr(cfg.synthetic.filters, k):
                        setattr(cfg.synthetic.filters, k, v)

        # dedupe
        ded = d.get("dedupe", {})
        if isinstance(ded, dict):
            for k, v in ded.items():
                if hasattr(cfg.dedupe, k):
                    setattr(cfg.dedupe, k, v)

        # privacy
        priv = d.get("privacy", {})
        if isinstance(priv, dict):
            for k, v in priv.items():
                if hasattr(cfg.privacy, k):
                    setattr(cfg.privacy, k, v)

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ConfigError if invalid."""
        if not self.input.path:
            raise ConfigError("input.path is required")
        if self.input.format not in ("jsonl", "csv", "txt"):
            raise ConfigError(f"input.format must be jsonl, csv, or txt, got '{self.input.format}'")
        if self.labeling.task not in ("sft", "classification"):
            raise ConfigError(f"labeling.task must be sft or classification, got '{self.labeling.task}'")
        if not self.output.name:
            raise ConfigError("output.name is required")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
