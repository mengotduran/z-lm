"""Checkpoints: save everything needed to stop training and later continue it exactly.

"Exactly" means a resumed run produces the same numbers as one that was never interrupted.
For that, a checkpoint holds more than the model weights:

  model      - the weights.
  optimizer  - AdamW's per-parameter running averages of the gradient and its square.
               Without them the optimizer restarts "cold" and the loss jumps after a resume.
  step       - optimizer steps completed so far. The LR schedule (zlm/schedule.py) is a
               pure function of the step, so this alone restores the learning rate.
  config     - the full run config as a plain dict, so a checkpoint is self-describing:
               the sampling script can rebuild the model without the YAML file.
  rng        - the state of every random number generator involved: Python, NumPy, torch
               (CPU) and CUDA. torch's drives batch sampling (zlm/data.py) and dropout, so
               restoring it makes a resumed run draw the very same batches.
  scaler     - GradScaler state, only used with float16 on older GPUs.
  extra      - small bookkeeping the training loop wants back, e.g. the best val loss.

Two safety choices:
  - Saving is atomic: write to a temporary file, then rename over the old checkpoint. If the
    process is killed mid-save (Ctrl-C, laptop battery), the previous checkpoint survives
    intact instead of being replaced by half a file.
  - Loading uses torch.load(weights_only=True). Plain torch.load unpickles arbitrary Python
    objects, so a malicious .pt file could run code. weights_only only accepts tensors and
    basic containers, which is why the NumPy RNG state is stored as a tensor below.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from zlm.config import Config, config_from_dict
from zlm.model import Z


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps the model and prefixes every state_dict key with "_orig_mod.".
    # Saving the inner model keeps checkpoints loadable with or without compile.
    return getattr(model, "_orig_mod", model)


def _rng_state() -> dict[str, Any]:
    # ("MT19937", uint32 array, pos, has_gauss, cached). The cast is for the type checker,
    # which can't tell that the default legacy=True returns this tuple and not a dict.
    np_state = cast(tuple, np.random.get_state())
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (np_state[0], torch.from_numpy(np_state[1].copy()), *np_state[2:]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()  # one state per GPU
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, keys.numpy(), *rest))
    torch.set_rng_state(state["torch"])
    # Only restore CUDA RNG when the hardware matches. A GPU checkpoint resumed on the CPU
    # laptop (or vice versa) still works; it just isn't bit-identical on the GPU side.
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    scaler: torch.amp.GradScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
        "rng": _rng_state(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem: readers see old or new, never half


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into existing objects. Returns the whole dict (step, config, extra...).

    Everything is loaded to CPU first: load_state_dict then copies weights to wherever the
    model already lives, and the optimizer moves its state to match its parameters. This is
    what lets a checkpoint saved on a GPU be resumed on the CPU, and vice versa.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    _unwrap(model).load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng:
        _set_rng_state(ckpt["rng"])
    return ckpt


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[Z, Config]:
    """Rebuild a model from a checkpoint alone, for sampling and evaluation (no optimizer)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = config_from_dict(ckpt["config"])
    model = Z(cfg.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg
