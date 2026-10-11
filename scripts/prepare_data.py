"""Turn a raw text dataset into train.bin / val.bin token files for training.

Usage:
    python scripts/prepare_data.py --dataset shakespeare --vocab-size 1024
    python scripts/prepare_data.py --dataset fineweb-edu --vocab-size 4096 --max-mb 400

Steps:
  1. Get the raw documents, cached under data/<dataset>/ (skipped if already there).
     - shakespeare: one text file, treated as a single document.
     - fineweb-edu: the first --max-mb of text from one Parquet shard of FineWeb-Edu.
  2. Split into train and validation.
  3. Train a tokenizer on (a sample of) the *train* part only. Reused if one with the same
     vocab size exists.
  4. Encode both parts and write them as flat arrays of uint16 token ids:
     data/<dataset>/train.bin, val.bin, plus meta.json describing them.

Why .bin files: tokenizing is slow (pure Python), training reads the data thousands of times.
So we tokenize once, store raw integers, and the training loop memory-maps the file with
np.memmap: the OS pages in only the bytes a batch needs, so even a dataset bigger than RAM works.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from zlm.httpfile import HTTPRangeFile
from zlm.tokenizer import EOT, BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# FineWeb-Edu: web pages filtered by a classifier for educational value. The "sample-10BT"
# subset (~10 billion tokens) is split into 14 Parquet shards of ~2 GB; we read part of one.
FINEWEB_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"

# Per-dataset defaults for the validation share. Shakespeare is tiny, so 10% is needed for
# a stable val loss; for hundreds of MB of web text, 1% is already millions of characters.
VAL_FRACTION = {"shakespeare": 0.1, "fineweb-edu": 0.01}


# --- getting the raw documents ---------------------------------------------------


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"found {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def shakespeare(out_dir: Path, val_fraction: float) -> tuple[list[str], list[str]]:
    raw_path = out_dir / "input.txt"
    download(SHAKESPEARE_URL, raw_path)
    text = raw_path.read_text(encoding="utf-8")
    # Validation = the *last* 10% as one contiguous block, not randomly sampled lines.
    # Neighbouring lines in a play are strongly related; a random split would let the model
    # see the context of nearly every val line during training and make val loss too optimistic.
    split = int(len(text) * (1 - val_fraction))
    return [text[:split]], [text[split:]]


def fineweb_edu(out_dir: Path, val_fraction: float, max_mb: float) -> tuple[list[str], list[str]]:
    import pyarrow as pa  # data-prep only dependency: pip install pyarrow
    import pyarrow.parquet as pq

    cache = out_dir / f"docs-{max_mb:g}mb.parquet"
    if cache.exists():
        print(f"found {cache}")
        docs = pq.read_table(cache).column("text").to_pylist()
    else:
        print(f"reading the first {max_mb:g} MB of text from {FINEWEB_URL}")
        remote = HTTPRangeFile(FINEWEB_URL)
        shard = pq.ParquetFile(remote)
        print(f"shard: {remote.size / 1e9:.2f} GB, {shard.metadata.num_rows:,} documents "
              f"in {shard.num_row_groups} row groups")
        docs, chars, t0 = [], 0, time.perf_counter()
        for i in range(shard.num_row_groups):
            # columns=["text"]: skip the url, score, language... columns, never downloaded.
            batch = shard.read_row_group(i, columns=["text"]).column("text").to_pylist()
            docs.extend(batch)
            chars += sum(len(d) for d in batch)
            mb = remote.downloaded / 1e6
            print(f"  row group {i + 1}: {len(docs):,} docs, {chars / 1e6:.0f} MB text, "
                  f"{mb:.0f} MB downloaded ({mb / (time.perf_counter() - t0):.2f} MB/s)")
            if chars >= max_mb * 1e6:
                break
        out_dir.mkdir(parents=True, exist_ok=True)
        # Cache only what we read, so re-running (e.g. with another vocab size) is offline.
        pq.write_table(pa.table({"text": docs}), cache)
        print(f"saved {cache}")
    # Web pages are independent of each other, so unlike the play, the val split can be any
    # set of whole documents. The last ones are as good as random (the shard isn't sorted
    # by topic), and taking whole documents means no page is split between train and val.
    n_val = max(1, int(len(docs) * val_fraction))
    return docs[:-n_val], docs[-n_val:]


# --- tokenizer and encoding --------------------------------------------------------


def tokenizer_sample(docs: list[str], max_mb: float) -> str:
    """Join documents with <|endoftext|> until about max_mb of text.

    BPE merges are dominated by frequent patterns, which a 10-20 MB sample already shows;
    training on all 400 MB would take far longer for nearly the same merges. The separator
    keeps merges from spanning two documents (train() cuts the text at special tokens).
    """
    parts, size = [], 0
    for doc in docs:
        parts.append(doc)
        size += len(doc)
        if size >= max_mb * 1e6:
            break
    return EOT.join(parts)


def load_or_train_tokenizer(path: Path, train_text: str, vocab_size: int) -> BPETokenizer:
    if path.exists():
        tok = BPETokenizer.load(path)
        if tok.vocab_size == vocab_size:
            print(f"reusing {path} (vocab_size={vocab_size})")
            return tok
        print(f"{path} has vocab_size={tok.vocab_size}, want {vocab_size}: retraining")
    print(f"training tokenizer on {len(train_text) / 1e6:.1f} MB of train text (vocab_size={vocab_size}) ...")
    t0 = time.perf_counter()
    tok = BPETokenizer()
    tok.train(train_text, vocab_size)
    tok.save(path)
    print(f"saved {path} ({time.perf_counter() - t0:.0f}s)")
    return tok


def write_tokens(docs: Iterable[str], tok: BPETokenizer, path: Path, dtype: type, separate: bool) -> int:
    """Encode documents one by one and append their ids to `path`. Returns the token count.

    Streaming matters at this size: 100M token ids as one Python list would take several GB
    of RAM (each list slot is an 8-byte pointer to an int object). Here at most ~1M ids are
    held at a time, flushed to disk as a compact uint16 array.

    separate=True appends <|endoftext|> after every document, so the model sees where one
    text ends and an unrelated one starts, instead of reading them as one continuous story.
    """
    total, buf, t0, chars = 0, [], time.perf_counter(), 0
    with path.open("wb") as f:
        for i, doc in enumerate(docs):
            buf.extend(tok.encode(doc))
            if separate:
                buf.append(tok.eot_id)
            chars += len(doc)
            if len(buf) >= 1_000_000:
                np.array(buf, dtype=dtype).tofile(f)  # raw bytes, no header
                total += len(buf)
                buf = []
                rate = chars / 1e6 / (time.perf_counter() - t0)
                print(f"  {path.name}: {total / 1e6:.0f}M tokens, {i + 1:,} docs ({rate:.1f} MB/s)", end="\r")
        np.array(buf, dtype=dtype).tofile(f)
        total += len(buf)
    print(f"wrote {path}: {total:,} tokens, {path.stat().st_size / 1e6:.1f} MB" + " " * 20)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=VAL_FRACTION, default="shakespeare")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=None,
                        help=f"share of the data held out (default: {VAL_FRACTION})")
    parser.add_argument("--max-mb", type=float, default=400,
                        help="fineweb-edu: MB of text to read from the shard (~3.4 chars per token at vocab 4096)")
    parser.add_argument("--tokenizer-mb", type=float, default=10,
                        help="MB of train text the tokenizer is trained on")
    args = parser.parse_args()

    out_dir = DATA_DIR / args.dataset
    val_fraction = args.val_fraction if args.val_fraction is not None else VAL_FRACTION[args.dataset]
    if args.dataset == "shakespeare":
        train_docs, val_docs = shakespeare(out_dir, val_fraction)
    else:
        train_docs, val_docs = fineweb_edu(out_dir, val_fraction, args.max_mb)
    print(f"{len(train_docs):,} train / {len(val_docs):,} val documents")

    # Trained on train text only: the val split should be text the whole pipeline has never seen.
    tok = load_or_train_tokenizer(out_dir / "tokenizer.model",
                                  tokenizer_sample(train_docs, args.tokenizer_mb), args.vocab_size)

    # uint16 holds ids 0..65535: half the disk and memory of int32. GPT-2's 50,257 tokens fit too.
    dtype = np.uint16 if tok.vocab_size <= 2**16 else np.uint32
    # A single document (Shakespeare) needs no separators; many independent ones do.
    separate = len(train_docs) > 1
    counts = {name: write_tokens(docs, tok, out_dir / f"{name}.bin", dtype, separate)
              for name, docs in [("train", train_docs), ("val", val_docs)]}

    # The training loop reads this to know how to interpret the .bin files.
    meta = {
        "dataset": args.dataset,
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "tokenizer": "tokenizer.model",
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir / 'meta.json'}: {meta}")

    # Sanity check: read val.bin back the way training will, and decode the start.
    val = np.memmap(out_dir / "val.bin", dtype=dtype, mode="r")
    print(f"\nval.bin starts with: {tok.decode(val[:40].tolist())!r}")


if __name__ == "__main__":
    main()
