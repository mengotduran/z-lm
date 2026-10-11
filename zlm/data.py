"""Batch loader: random training examples from a memory-mapped .bin token file.

A language model learns to predict the next token. So one training example is a window of
`block_size` tokens (x) and the same window shifted one token to the right (y):

    tokens:  [ The  cat  sat  on  the  mat ]
    x:       [ The  cat  sat  on ]
    y:       [ cat  sat  on   the ]       y[t] is the correct next token after x[0..t]

One window therefore gives block_size predictions at once, one per position. The model is
causal (position t can't look at t+1), so each prediction only sees the tokens before it.

Windows start at random positions, not in order: consecutive windows are near-duplicates, and
random sampling gives each batch a mix of the whole dataset, which makes training steadier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """One split (train or val) of a prepared dataset, read lazily from disk."""

    def __init__(self, path: str | Path, dtype: str, block_size: int) -> None:
        self.path = Path(path)
        self.block_size = block_size
        # mode="r": read-only view of the file. Nothing is loaded yet; the OS reads pages from
        # disk only when we slice them, so this works for files larger than RAM.
        self.tokens = np.memmap(self.path, dtype=dtype, mode="r")
        if len(self.tokens) <= block_size:
            raise ValueError(f"{self.path} has {len(self.tokens)} tokens, "
                             f"need more than block_size={block_size}")

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x, y of shape (batch_size, block_size), int64, on `device`."""
        # Highest valid start is len - block_size - 1, because y needs one token past x.
        # Uses torch's global RNG, which the checkpoint saves: a resumed run draws the same batches.
        starts = torch.randint(len(self.tokens) - self.block_size, (batch_size,))

        # int64 because nn.Embedding needs long indices; most torch ops don't support uint16.
        x = torch.stack([self._window(i, i + self.block_size) for i in starts.tolist()])
        y = torch.stack([self._window(i + 1, i + 1 + self.block_size) for i in starts.tolist()])

        if device.type == "cuda":
            # Pinned (page-locked) CPU memory can be copied to the GPU asynchronously, so the
            # copy overlaps with the GPU still working on the previous step.
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def _window(self, start: int, end: int) -> torch.Tensor:
        return torch.from_numpy(self.tokens[start:end].astype(np.int64))


def load_split(data_dir: str | Path, split: str, block_size: int) -> TokenDataset:
    """Open data/<dataset>/<split>.bin, reading the number type from meta.json."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    return TokenDataset(data_dir / f"{split}.bin", meta["dtype"], block_size)
