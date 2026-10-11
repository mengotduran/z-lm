"""Ablation: train the same config several times, changing one setting, and compare.

Usage (the positional-encoding study):
    python scripts/ablate.py --config configs/tiny.yaml --vary model.pos_encoding=rope,learned
    python scripts/ablate.py --config configs/tiny.yaml --vary model.pos_encoding=rope,learned --seeds 1 2 3
    python scripts/ablate.py --config configs/tiny.yaml --vary model.pos_encoding=rope,learned --compare-only

Each run is an ordinary `scripts/train.py` call with an extra --set, written to
<out>/<value>[-seed<s>]/. Nothing here duplicates training logic, so an ablation run is exactly
the run you would get by hand. A run that already has a checkpoint is resumed (and a finished
one just reports "done"), so after a Ctrl-C you re-run the same command to continue.

Why seeds: two runs that differ *only* in their seed (initial weights and batch order) still
end at slightly different losses. A difference between settings smaller than that noise means
nothing. With --seeds the table reports mean +- std over seeds, so you can see the noise level.
Even with the same seed, the variants don't start identically: the "learned" variant draws
extra random numbers to initialize its position table, which shifts every later random draw
(the remaining weights and the batch order). So a single-seed comparison mixes the effect of
the setting with seed noise; use several seeds before trusting a small difference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from zlm.config import load_config
from zlm.model import Z

TRAIN = Path(__file__).resolve().parent / "train.py"


def read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_std(xs: list[float]) -> str:
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{statistics.mean(xs):.4f} ± {statistics.stdev(xs):.4f}"


def compare(key: str, runs: dict[str, list[Path]], cfg_path: Path, common: list[str]) -> None:
    """Print the val-loss curve side by side, then a summary row per value."""
    metrics = {v: [read_metrics(d) for d in dirs] for v, dirs in runs.items()}
    values = list(runs)

    # Curve: val loss at each eval step, averaged over the seeds that reached that step.
    steps = sorted({row["step"] for ms in metrics.values() for m in ms for row in m})
    width = 18
    print(f"\nval loss by step ({key})")
    print(f"{'step':>6} | " + " | ".join(f"{v:>{width}}" for v in values))
    for step in steps:
        cells = []
        for v in values:
            xs = [row["val"] for m in metrics[v] for row in m if row["step"] == step]
            cells.append(f"{mean_std(xs) if xs else '-':>{width}}")
        print(f"{step:>6} | " + " | ".join(cells))

    print(f"\nsummary ({key}; gap = final val - final train, > 0 means overfitting)")
    print(f"{'value':>10} | {'params':>9} | {'best val':>17} | {'at step':>7} | "
          f"{'final train':>17} | {'final val':>17} | {'gap':>7}")
    for v in values:
        done = [m for m in metrics[v] if m]
        if not done:
            print(f"{v:>10} | (no metrics yet)")
            continue
        best = [min(m, key=lambda r: r["val"]) for m in done]
        final = [m[-1] for m in done]
        cfg = load_config(cfg_path, [*common, f"{key}={v}"])
        gap = statistics.mean(f["val"] - f["train"] for f in final)
        print(f"{v:>10} | {Z(cfg.model).num_params():>9,} | {mean_std([b['val'] for b in best]):>17} | "
              f"{round(statistics.mean(b['step'] for b in best)):>7} | "
              f"{mean_std([f['train'] for f in final]):>17} | {mean_std([f['val'] for f in final]):>17} | "
              f"{gap:>+7.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vary", required=True, metavar="KEY=V1,V2,...",
                        help="the one setting to change, e.g. model.pos_encoding=rope,learned")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="repeat every value with each seed (default: the config's seed, once)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides applied to every run, e.g. train.max_steps=500")
    parser.add_argument("--out", type=Path, default=None,
                        help="parent folder for the runs (default: checkpoints/ablate-<config>-<key>)")
    parser.add_argument("--compare-only", action="store_true", help="don't train, just print the table")
    args = parser.parse_args()

    key, _, raw_values = args.vary.partition("=")
    values = [v.strip() for v in raw_values.split(",") if v.strip()]
    if not key or len(values) < 2:
        raise SystemExit("--vary needs a key and at least two values, e.g. model.pos_encoding=rope,learned")
    out = args.out or Path("checkpoints") / f"ablate-{args.config.stem}-{key.split('.')[-1]}"

    # Fail fast on a bad key or value, before spending 20 minutes on the first run.
    for v in values:
        load_config(args.config, [*args.set, f"{key}={v}"])

    seeds = args.seeds or [None]
    runs: dict[str, list[Path]] = {v: [] for v in values}
    for v in values:
        for seed in seeds:
            run_dir = out / (v if seed is None else f"{v}-seed{seed}")
            runs[v].append(run_dir)
            if args.compare_only:
                continue
            overrides = [*args.set, f"{key}={v}", f"train.out_dir={run_dir}"]
            if seed is not None:
                overrides.append(f"train.seed={seed}")
            cmd = [sys.executable, str(TRAIN), "--config", str(args.config), "--set", *overrides]
            if (run_dir / "ckpt.pt").exists():
                cmd.append("--resume")
            print(f"\n=== {key}={v}" + (f" seed={seed}" if seed is not None else "") + f" -> {run_dir}")
            subprocess.run(cmd, check=True)  # output streams straight to the console

    compare(key, runs, args.config, args.set)


if __name__ == "__main__":
    main()
