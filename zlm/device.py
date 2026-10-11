"""Device selection: pick the best available hardware and the speed tricks that go with it.

Every other module asks this one "where do tensors live, and how should I run?" so that
moving from the CPU laptop to an NVIDIA GPU is a config change, never a code change.

Three decisions are made here, each overridable from the YAML config:
  device   - "auto" | "cuda" | "mps" | "cpu"
  dtype    - "auto" | "float32" | "bfloat16" | "float16"   (mixed precision for the forward pass)
  compile  - "auto" | true | false                        (torch.compile)

"auto" means: use mixed precision and torch.compile only on CUDA. On CPU they give little or
no speedup for a model this size, and torch.compile adds a long warm-up. MPS (Apple GPUs)
gets plain float32 because its autocast and compile support are still patchy.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

_DTYPES = {
    "float32": None,  # None = no autocast, everything stays float32
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class DeviceSetup:
    """The resolved answer: which device, which autocast dtype, whether to compile."""

    device: torch.device
    amp_dtype: torch.dtype | None  # None = full float32
    compile: bool

    @property
    def needs_grad_scaler(self) -> bool:
        # float16 has a small exponent range, so tiny gradients underflow to zero.
        # A GradScaler multiplies the loss up before backward() to prevent that.
        # bfloat16 has the same exponent range as float32 and needs no scaler.
        return self.amp_dtype == torch.float16

    def autocast(self) -> contextlib.AbstractContextManager:
        """Context manager for the forward pass: `with setup.autocast(): loss = model(x)`."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)


def detect_device(preference: str = "auto") -> torch.device:
    """Return the requested device, or the best available one for "auto"."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # Fail loudly if a config asks for hardware this machine doesn't have.
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={preference!r} requested but CUDA is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(preference)


def _resolve_dtype(device: torch.device, dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        if device.type != "cuda":
            return None
        # bfloat16 on Ampere (RTX 30xx) and newer; float16 + GradScaler on older cards.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}, expected 'auto' or one of {list(_DTYPES)}")
    return _DTYPES[dtype]


def _resolve_compile(device: torch.device, compile: bool | str) -> bool:
    if compile == "auto":
        return device.type == "cuda"
    if isinstance(compile, bool):  # YAML parses `true`/`false` straight to bool
        return compile
    raise ValueError(f"compile must be 'auto', true or false, got {compile!r}")


def setup_device(
    device: str = "auto", dtype: str = "auto", compile: bool | str = "auto"
) -> DeviceSetup:
    """Resolve the three config settings into a DeviceSetup."""
    dev = detect_device(device)
    if dev.type == "cuda":
        # TF32: on Ampere+ GPUs, float32 matmuls run on tensor cores with a tiny precision
        # loss that doesn't matter for training. Free speed; ignored on older GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return DeviceSetup(dev, _resolve_dtype(dev, dtype), _resolve_compile(dev, compile))


if __name__ == "__main__":
    # Quick check of what this machine resolves to: python -m zlm.device
    s = setup_device()
    print(f"device={s.device}  amp_dtype={s.amp_dtype}  compile={s.compile}  "
          f"grad_scaler={s.needs_grad_scaler}  torch_threads={torch.get_num_threads()}")
