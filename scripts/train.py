"""Train Z from a YAML config.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/tiny.yaml --resume
    python scripts/train.py --config configs/tiny.yaml --set model.pos_encoding=learned train.out_dir=checkpoints/tiny-learned

One optimizer step:
  1. set the learning rate for this step (warmup + cosine, zlm/schedule.py);
  2. for each of grad_accum_steps micro-batches: forward, backward (gradients add up);
  3. clip the total gradient norm to grad_clip;
  4. AdamW update, then clear the gradients.

Every eval_interval steps (and at the end): estimate train/val loss on fresh random batches,
save checkpoints/<run>/ckpt.pt (the latest, for --resume) and best.pt (lowest val loss, for
sampling). With dropout on a small dataset, val loss eventually rises while train loss keeps
falling (overfitting); best.pt keeps the model from before that point.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from zlm.checkpoint import load_checkpoint, save_checkpoint
from zlm.config import Config, load_config
from zlm.data import TokenDataset, load_split
from zlm.device import DeviceSetup, setup_device
from zlm.model import Z
from zlm.schedule import lr_at


def make_optimizer(model: Z, cfg: Config, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices only.

    Weight decay pulls weights toward zero each step, which discourages the model from
    relying on a few huge weights (a form of regularization). That makes sense for the 2D
    matrices that mix features (Linear layers, embeddings). It does not for LayerNorm scales
    and biases (1D): those set the size and offset of activations, and shrinking them toward
    zero just fights the normalization. Rule of thumb used by GPT-2 onwards: decay iff dim >= 2.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.train.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.train.lr,  # overwritten every step by the schedule
        betas=(cfg.train.beta1, cfg.train.beta2),
        # fused: one CUDA kernel updates all parameters at once instead of a loop of small ops.
        fused=device.type == "cuda",
    )


@torch.no_grad()  # no gradients needed: saves memory and time
def estimate_loss(
    model: torch.nn.Module, splits: dict[str, TokenDataset], cfg: Config, setup: DeviceSetup
) -> dict[str, float]:
    """Average loss over eval_steps random batches per split.

    The loss of a single batch is noisy; averaging many gives a number you can compare
    between evals. eval() turns dropout off, so we measure the model itself, not the model
    with random parts switched off.
    """
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.train.eval_steps)
        for i in range(cfg.train.eval_steps):
            x, y = data.get_batch(cfg.train.batch_size, setup.device)
            with setup.autocast():
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def check_vocab(cfg: Config) -> None:
    # The model's vocab_size must match the tokenizer that produced the .bin files: a token id
    # >= vocab_size would crash nn.Embedding, a too-large vocab silently wastes parameters.
    meta = json.loads((Path(cfg.data.data_dir) / "meta.json").read_text())
    if meta["vocab_size"] != cfg.model.vocab_size:
        raise ValueError(f"model.vocab_size={cfg.model.vocab_size} but the data in "
                         f"{cfg.data.data_dir} was tokenized with vocab_size={meta['vocab_size']}")


def append_metrics(path: Path, row: dict) -> None:
    # One JSON object per line ("JSON Lines"): appending never rewrites the file, and a
    # comparison script can read the eval curve of any run without parsing console logs.
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.lr=3e-4 model.n_layer=2")
    parser.add_argument("--resume", action="store_true", help="continue from <out_dir>/ckpt.pt")
    args = parser.parse_args()
    # Print each log line immediately, even when output goes to a file (`... | tee train.log`).
    # Otherwise Python buffers it, and a killed run loses its last lines.
    cast(io.TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

    # --- setup -----------------------------------------------------------------
    cfg = load_config(args.config, args.set)
    tc = cfg.train
    out_dir = Path(tc.out_dir)
    ckpt_path, best_path = out_dir / "ckpt.pt", out_dir / "best.pt"
    metrics_path = out_dir / "metrics.jsonl"  # eval curve, read by scripts/ablate.py
    if ckpt_path.exists() and not args.resume:
        # A fresh run would overwrite it at the first eval. Make that a deliberate choice.
        raise SystemExit(f"{ckpt_path} exists: pass --resume to continue it, "
                         f"or delete it / pick another train.out_dir to start over")

    setup = setup_device(cfg.system.device, cfg.system.dtype, cfg.system.compile)
    seed_everything(tc.seed)  # before building the model, so the initial weights are reproducible
    check_vocab(cfg)
    train_data = load_split(cfg.data.data_dir, "train", cfg.model.block_size)
    val_data = load_split(cfg.data.data_dir, "val", cfg.model.block_size)

    model = Z(cfg.model).to(setup.device)
    optimizer = make_optimizer(model, cfg, setup.device)
    # With enabled=False (everything except float16) the scaler is a pass-through: scale()
    # returns the loss unchanged and step() just calls optimizer.step(). One code path for all.
    scaler = torch.amp.GradScaler(setup.device.type, enabled=setup.needs_grad_scaler)

    step, best_val = 0, math.inf
    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume: no checkpoint at {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, scaler if scaler.is_enabled() else None)
        step, best_val = ckpt["step"], ckpt["extra"].get("best_val", math.inf)
        print(f"resumed from {ckpt_path} at step {step} (best val {best_val:.4f})")

    # Compile *after* loading, so the checkpoint goes into the plain model. Compiling only
    # changes how forward runs, not the parameters: the optimizer above still holds the same ones.
    train_model = cast(torch.nn.Module, torch.compile(model)) if setup.compile else model

    tokens_per_step = tc.batch_size * tc.grad_accum_steps * cfg.model.block_size
    print(f"device={setup.device} amp={setup.amp_dtype} compile={setup.compile} | "
          f"{model.num_params():,} params | {tokens_per_step:,} tokens/step | "
          f"pos_encoding={cfg.model.pos_encoding}")

    wandb = None
    if cfg.log.wandb:
        import wandb  # pyright: ignore[reportMissingImports]  (optional dependency, only imported when enabled)

        wandb.init(project=cfg.log.wandb_project, name=cfg.log.run_name, config=cfg.to_dict(),
                   resume="allow")

    if step == 0:
        # Fresh run: start a new metrics file (a leftover one belongs to a deleted run).
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.unlink(missing_ok=True)
        # Baseline before any training: should be close to ln(vocab_size).
        losses = estimate_loss(train_model, {"train": train_data, "val": val_data}, cfg, setup)
        print(f"step 0: train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"(uniform guess = ln {cfg.model.vocab_size} = {math.log(cfg.model.vocab_size):.4f})")
        append_metrics(metrics_path, {"step": 0, **losses})

    # --- training loop ---------------------------------------------------------------
    # `step` counts *completed* optimizer steps. Evaluation and checkpointing happen right
    # after a step completes, so a checkpoint saved with step=s resumes straight into
    # training step s+1 with the RNG exactly where the uninterrupted run had it.
    train_model.train()
    t0 = time.perf_counter()
    while step < tc.max_steps:
        lr = lr_at(step, tc.lr, tc.min_lr, tc.warmup_steps, tc.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation: backward() *adds* into .grad, so running k micro-batches
        # before one optimizer step gives the gradient of a k-times larger batch, in the
        # memory of one. Each loss is divided by k so the sum is an average, not k times it.
        loss_sum = torch.zeros((), device=setup.device)
        for _ in range(tc.grad_accum_steps):
            x, y = train_data.get_batch(tc.batch_size, setup.device)
            with setup.autocast():
                _, loss = train_model(x, y)
            loss = loss / tc.grad_accum_steps
            loss_sum += loss.detach()  # .item() here would force a GPU sync every micro-batch
            scaler.scale(loss).backward()

        # Gradient clipping: if the total gradient norm exceeds grad_clip, scale all gradients
        # down so it equals grad_clip. A single odd batch can produce a huge gradient; clipping
        # stops that one step from wrecking weights that took thousands of steps to learn.
        # unscale_ first, so we clip the real gradients and not the float16-scaled ones.
        grad_norm = None
        if tc.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)  # skips the update if float16 gradients overflowed to inf
        scaler.update()
        optimizer.zero_grad(set_to_none=True)  # None instead of zeros: frees the memory
        step += 1

        if step % tc.log_interval == 0:
            dt = (time.perf_counter() - t0) / tc.log_interval
            t0 = time.perf_counter()
            loss_val = loss_sum.item()
            norm = f" | grad norm {grad_norm.item():.2f}" if grad_norm is not None else ""
            print(f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e}{norm} | "
                  f"{dt * 1000:.0f} ms/step | {tokens_per_step / dt:,.0f} tok/s")
            if wandb:
                wandb.log({"train/loss_step": loss_val, "lr": lr, "step_time_ms": dt * 1000}, step=step)

        if step % tc.eval_interval == 0 or step == tc.max_steps:
            losses = estimate_loss(train_model, {"train": train_data, "val": val_data}, cfg, setup)
            improved = losses["val"] < best_val
            best_val = min(best_val, losses["val"])
            print(f"step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}"
                  f"{'  (best, saved best.pt)' if improved else ''}")
            if wandb:
                wandb.log({"train/loss": losses["train"], "val/loss": losses["val"]}, step=step)
            extra = {"best_val": best_val}
            save_checkpoint(ckpt_path, model, optimizer, step, cfg, scaler if scaler.is_enabled() else None, extra)
            if improved:
                save_checkpoint(best_path, model, optimizer, step, cfg, scaler if scaler.is_enabled() else None, extra)
            # After the checkpoint, so a resumed run never logs the same eval twice: every
            # row in the file is at or before the step the checkpoint will resume from.
            append_metrics(metrics_path, {"step": step, **losses, "lr": lr})
            t0 = time.perf_counter()  # don't count eval time in the next ms/step

    print(f"done: {step} steps, best val loss {best_val:.4f}, checkpoints in {out_dir}/")
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
