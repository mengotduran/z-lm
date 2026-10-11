"""Turn a raw text dataset into train.bin / val.bin token files for training.

Usage:
    python scripts/prepare_data.py --dataset shakespeare --vocab-size 1024

Steps:
  1. Download the raw text into data/<dataset>/input.txt (skipped if already there).
  2. Split the text: first 90% train, last 10% validation.
  3. Train a tokenizer on the *train* part only (reused if one with the same vocab size exists).
  4. Encode both parts and write them as flat arrays of uint16 token ids:
     data/<dataset>/train.bin, val.bin, plus meta.json describing them.

Why .bin files: tokenizing is slow (pure Python), training reads the data thousands of times.
So we tokenize once, store raw integers, and the training loop memory-maps the file with
np.memmap: the OS pages in only the bytes a batch needs, so even a dataset bigger than RAM works.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from zlm.tokenizer import BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"found {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def load_or_train_tokenizer(path: Path, train_text: str, vocab_size: int) -> BPETokenizer:
    if path.exists():
        tok = BPETokenizer.load(path)
        if tok.vocab_size == vocab_size:
            print(f"reusing {path} (vocab_size={vocab_size})")
            return tok
        print(f"{path} has vocab_size={tok.vocab_size}, want {vocab_size}: retraining")
    print(f"training tokenizer on the train split (vocab_size={vocab_size}), this takes ~30-60s ...")
    tok = BPETokenizer()
    tok.train(train_text, vocab_size)
    tok.save(path)
    print(f"saved {path}")
    return tok


def write_tokens(ids: list[int], path: Path, dtype: type) -> None:
    arr = np.array(ids, dtype=dtype)
    arr.tofile(path)  # raw bytes, no header: exactly len(ids) * itemsize bytes
    print(f"wrote {path}: {len(arr):,} tokens, {path.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = DATA_DIR / args.dataset
    raw_path = out_dir / "input.txt"
    download(DATASETS[args.dataset], raw_path)
    text = raw_path.read_text(encoding="utf-8")

    # Validation = the *last* 10% as one contiguous block, not randomly sampled lines.
    # Neighbouring lines in a play are strongly related; a random split would let the model
    # see the context of nearly every val line during training and make val loss too optimistic.
    split = int(len(text) * (1 - args.val_fraction))
    train_text, val_text = text[:split], text[split:]

    # Trained on train text only: the val split should be text the whole pipeline has never seen.
    tok = load_or_train_tokenizer(out_dir / "tokenizer.model", train_text, args.vocab_size)

    # uint16 holds ids 0..65535: half the disk and memory of int32. GPT-2's 50,257 tokens fit too.
    dtype = np.uint16 if tok.vocab_size <= 2**16 else np.uint32

    for name, part in [("train", train_text), ("val", val_text)]:
        ids = tok.encode(part)
        write_tokens(ids, out_dir / f"{name}.bin", dtype)

    # The training loop reads this to know how to interpret the .bin files.
    meta = {
        "dataset": args.dataset,
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "tokenizer": "tokenizer.model",
        "train_tokens": int((out_dir / "train.bin").stat().st_size // np.dtype(dtype).itemsize),
        "val_tokens": int((out_dir / "val.bin").stat().st_size // np.dtype(dtype).itemsize),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir / 'meta.json'}: {meta}")

    # Sanity check: read val.bin back the way training will, and decode the start.
    val = np.memmap(out_dir / "val.bin", dtype=dtype, mode="r")
    print(f"\nval.bin starts with: {tok.decode(val[:40].tolist())!r}")


if __name__ == "__main__":
    main()
