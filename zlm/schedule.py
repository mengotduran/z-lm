"""Learning-rate schedule: linear warmup, then cosine decay down to a floor.

    lr
    ^        ___
    |       /   ‾‾--__
    |      /          ‾-_
    |     /              ‾-_
    |    /                  ‾--__
    |   /                        ‾‾--___  min_lr
    +--+-------------------------------+-------> step
       warmup_steps                  max_steps

Why warmup: at step 0 the weights are random and AdamW's running estimates of the gradient
statistics are still empty, so its first updates are poorly scaled. A full-size LR there can
throw the model into a bad region it never recovers from. Ramping up over the first few
hundred steps avoids that.

Why cosine decay: a high LR makes fast progress early but keeps the weights bouncing around
the minimum; a low LR settles into it. Cosine starts slowly, decays fastest in the middle
and flattens out near min_lr. It is the standard choice for GPT training.

Why a plain function of `step` and not torch.optim.lr_scheduler: the LR is fully determined
by the step number, so resuming from a checkpoint only needs the step. A scheduler object
would carry its own state that also has to be saved and restored.
The training loop applies it each step with:
    for group in optimizer.param_groups: group["lr"] = lr_at(step, ...)
"""

from __future__ import annotations

import math


def lr_at(step: int, lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Learning rate for optimizer step `step` (0-based)."""
    if step < warmup_steps:
        # (step + 1), not step: the first step then trains with lr / warmup_steps instead
        # of exactly 0, which would waste it. The last warmup step reaches the full lr.
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        # Past the planned end (e.g. training extended after a resume): stay at the floor.
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))              # 1 -> 0, cosine-shaped
    return min_lr + coeff * (lr - min_lr)
