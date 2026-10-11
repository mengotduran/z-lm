"""Typed configuration loaded from YAML.

Every setting lives in one of five dataclasses (model, train, data, system, log). The YAML
file only lists what differs from the defaults below. Why dataclasses and not a plain dict:
  - a typo like `n_layers: 6` (instead of n_layer) raises an error instead of being silently
    ignored while the run quietly uses the default;
  - wrong types (`lr: fast`) fail at load time, not 20 minutes into training;
  - the editor autocompletes cfg.model.n_embd, and the defaults are documented in one place.

Any value can also be overridden from the command line, which is how ablations are run:
    python scripts/train.py --config configs/tiny.yaml --set model.pos_encoding=learned train.lr=3e-4
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 1024      # must match the tokenizer (checked against data/<dataset>/meta.json)
    block_size: int = 128       # context length: max tokens the model sees at once
    n_layer: int = 4            # number of transformer blocks
    n_head: int = 4             # attention heads per block
    n_embd: int = 128           # width of the residual stream (embedding size)
    dropout: float = 0.0        # 0 for big data; ~0.1-0.2 helps against overfitting tiny datasets
    bias: bool = False          # bias in Linear/LayerNorm layers; modern models drop it
    pos_encoding: str = "rope"  # "rope" (rotary) or "learned" (absolute embedding table)
    rope_theta: float = 10000.0 # RoPE base frequency; only used when pos_encoding == "rope"

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.pos_encoding not in ("rope", "learned"):
            raise ValueError(f"pos_encoding must be 'rope' or 'learned', got {self.pos_encoding!r}")
        if self.pos_encoding == "rope" and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE rotates dimensions in pairs, so head size (n_embd / n_head) must be even")


@dataclass
class TrainConfig:
    batch_size: int = 32        # sequences per micro-batch (what fits in memory at once)
    grad_accum_steps: int = 1   # micro-batches per optimizer step; effective batch = product of the two
    max_steps: int = 2000       # optimizer steps
    lr: float = 1e-3            # peak learning rate, reached after warmup
    min_lr: float = 1e-4        # floor of the cosine decay
    warmup_steps: int = 100     # linear LR ramp from 0 to lr
    weight_decay: float = 0.1   # AdamW decay (applied to weight matrices only)
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0      # max gradient norm; 0 disables clipping
    eval_interval: int = 200    # steps between train/val loss estimates (and checkpoints)
    eval_steps: int = 50        # batches averaged per loss estimate
    log_interval: int = 10      # steps between console log lines
    seed: int = 1337
    out_dir: str = "checkpoints/tiny"
    init_from: str = ""         # fine-tuning: start from this checkpoint's weights (fresh optimizer, step 0)


@dataclass
class DataConfig:
    data_dir: str = "data/shakespeare"  # folder made by scripts/prepare_data.py


@dataclass
class SystemConfig:
    device: str = "auto"                # "auto" | "cuda" | "mps" | "cpu"  (see zlm/device.py)
    dtype: str = "auto"                 # "auto" | "float32" | "bfloat16" | "float16"
    compile: bool | str = "auto"        # "auto" | true | false


@dataclass
class LogConfig:
    wandb: bool = False                 # also log to Weights & Biases (needs `pip install wandb`)
    wandb_project: str = "z-lm"
    run_name: str = ""                  # empty = derived from the config file name


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict: stored in checkpoints and sent to W&B."""
        return dataclasses.asdict(self)


# --- building dataclasses from dicts, with checking ----------------------------


def _check_type(name: str, value: Any, expected: Any) -> Any:
    """Validate (and minimally convert) one value against its annotated type."""
    if typing.get_origin(expected) in (typing.Union, types.UnionType):
        # e.g. `bool | str`: accept anything matching one of the members.
        for option in typing.get_args(expected):
            try:
                return _check_type(name, value, option)
            except TypeError:
                pass
        raise TypeError(f"{name}: {value!r} is not any of {typing.get_args(expected)}")
    if expected is float:
        # PyYAML follows YAML 1.1, where `1e-3` (no dot) is a *string*, not a float.
        # Accept ints and such strings rather than make you write 1.0e-3.
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            try:
                return float(value)
            except ValueError:
                pass
        raise TypeError(f"{name}: expected a number, got {value!r}")
    if expected is int:
        if isinstance(value, int) and not isinstance(value, bool):  # bool is a subclass of int
            return value
        raise TypeError(f"{name}: expected an integer, got {value!r}")
    if expected in (bool, str):
        if isinstance(value, expected):
            return value
        raise TypeError(f"{name}: expected {expected.__name__}, got {value!r}")
    return value


def _build(cls: type, data: Any, prefix: str) -> Any:  # data: whatever YAML produced
    if not isinstance(data, dict):
        raise TypeError(f"{prefix or 'config'}: expected a mapping, got {data!r}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown key(s) in {prefix or 'config'}: {sorted(unknown)}; "
                       f"valid keys are {sorted(known)}")
    kwargs = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if dataclasses.is_dataclass(hints[key]):
            kwargs[key] = _build(hints[key], value or {}, name)
        else:
            kwargs[key] = _check_type(name, value, hints[key])
    return cls(**kwargs)  # missing keys take their defaults; __post_init__ runs here


def config_from_dict(data: dict[str, Any]) -> Config:
    return _build(Config, data, "")


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ["train.lr=3e-4", "model.pos_encoding=learned"] to a raw config dict.

    Values are parsed as YAML, so numbers, true/false and strings all work as you'd expect.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        dotted, raw = item.split("=", 1)
        *sections, key = dotted.strip().split(".")
        node = data
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config file, apply command-line overrides, validate everything."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    data = apply_overrides(data, overrides or [])
    cfg = config_from_dict(data)
    if not cfg.log.run_name:
        cfg.log.run_name = path.stem  # configs/tiny.yaml -> "tiny"
    return cfg
