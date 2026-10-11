"""Train a BPE tokenizer on a text file and save it.

Usage:
    python scripts/train_tokenizer.py --input data/shakespeare/input.txt --vocab-size 1024

Writes <input dir>/tokenizer.model (what the code loads) and tokenizer.vocab (for you to read).
"""

import argparse
import time
from pathlib import Path

from zlm.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 text file to train on")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="total vocabulary: 256 bytes + merges + special tokens (default: 1024)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: tokenizer.model next to the input file)")
    parser.add_argument("--verbose", action="store_true", help="print every merge as it is learned")
    args = parser.parse_args()

    out = args.out or args.input.parent / "tokenizer.model"
    text = args.input.read_text(encoding="utf-8")
    print(f"training on {args.input} ({len(text):,} characters), vocab_size={args.vocab_size}")

    tok = BPETokenizer()
    start = time.time()
    tok.train(text, args.vocab_size, verbose=args.verbose)
    print(f"trained in {time.time() - start:.1f}s: {len(tok.merges)} merges, "
          f"special tokens {tok.special_tokens}")

    # Compression ratio: how many bytes of text one token covers on average.
    # Higher = shorter sequences = the model sees more text per context window.
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "round trip failed"
    print(f"{len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"({len(text.encode('utf-8')) / len(ids):.2f} bytes/token)")

    sample = text[:80]
    print(f"\nsample: {sample!r}")
    print("tokens:", [tok.decode([i]) for i in tok.encode(sample)])

    tok.save(out)
    print(f"\nsaved {out} and {out.with_suffix('.vocab')}")


if __name__ == "__main__":
    main()
