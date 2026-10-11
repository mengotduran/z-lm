"""Generate text from a trained Z checkpoint.

Usage:
    python scripts/sample.py                                   # checkpoints/tiny/best.pt
    python scripts/sample.py --prompt "ROMEO:" --temperature 0.8 --top-k 50
    python scripts/sample.py --checkpoint checkpoints/small/best.pt --num-samples 3

Everything needed comes from the checkpoint: it stores the full config, so the model is
rebuilt with the right size and positional encoding, and config.data.data_dir points to the
tokenizer the training data was encoded with. No YAML file needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zlm.checkpoint import load_model
from zlm.device import setup_device
from zlm.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny/best.pt"))
    parser.add_argument("--prompt", default="\n", help="text to continue (default: a newline)")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=50, help="0 = no top-k filtering")
    parser.add_argument("--seed", type=int, default=None, help="fix for reproducible samples")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    setup = setup_device(args.device)
    model, cfg = load_model(args.checkpoint, setup.device)

    data_dir = Path(cfg.data.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    tok = BPETokenizer.load(data_dir / meta["tokenizer"])

    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("--prompt must not be empty: the model needs at least one token to continue")
    # Every sample starts from the same prompt; one batch generates them all in parallel.
    idx = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long, device=setup.device)

    with setup.autocast():
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k or None)

    print(f"# {args.checkpoint} | {model.num_params():,} params | "
          f"temperature={args.temperature} top_k={args.top_k or 'off'}")
    for i, ids in enumerate(out.tolist()):
        if args.num_samples > 1:
            print(f"\n--- sample {i + 1} ---")
        print(tok.decode(ids))


if __name__ == "__main__":
    main()
