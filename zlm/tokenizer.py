"""Byte-level BPE (Byte Pair Encoding) tokenizer, written from scratch.

The model only understands integers, so text must become a list of token ids.

How it works:
  1. Start from raw UTF-8 bytes. Any text is a sequence of bytes 0..255, so the base vocabulary
     is 256 tokens and nothing is ever "unknown" (emoji, accents, Chinese all work).
  2. Training: count every adjacent pair of tokens, merge the most frequent pair into a new
     token (id 256, 257, ...), repeat until the vocabulary reaches vocab_size.
     Each merge is remembered in order: that ordered list *is* the tokenizer.
  3. Encoding new text: replay the merges, earliest-learned first.
  4. Decoding: look up the bytes of each token, concatenate, decode UTF-8.

Before counting pairs, text is split into chunks (words, numbers, punctuation, whitespace)
with a regex, and merges never cross a chunk boundary. Without this, BPE learns junk tokens
like "e. T" that glue the end of one word to the start of the next.
"""

from __future__ import annotations

import re
from collections import Counter

# Splits text into chunks, in the style of GPT-2:
#   's 't 're 've 'm 'll 'd      English contractions, kept as their own chunk
#   ' ?[^\W\d_]+'                a word (letters only), with its leading space attached
#   ' ?\d+'                      a number, with its leading space
#   ' ?[^\s\w]+'                 a run of punctuation, with its leading space
#   '\s+(?!\S)' and '\s+'        whitespace (the first leaves one space for the next word)
# [^\W\d_] means "a word character that's not a digit or underscore" = any Unicode letter.
# We use the stdlib `re` module, so no extra dependency.
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def get_pair_counts(ids: list[int], counts: Counter | None = None, weight: int = 1) -> Counter:
    """Count adjacent pairs: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}.

    `weight` lets one call count a chunk that appears `weight` times in the corpus.
    """
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`: merge([1, 2, 3, 1, 2], (1, 2), 9) -> [9, 3, 9]."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        # (left_id, right_id) -> merged_id, in the order they were learned.
        # Python dicts keep insertion order, so this doubles as the merge priority list.
        self.merges: dict[tuple[int, int], int] = {}
        # token id -> the raw bytes it stands for. Starts as the 256 single bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # chunk text -> token ids. Real text repeats the same words constantly,
        # so caching makes encoding a large corpus many times faster.
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text`."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (the byte alphabet), got {vocab_size}")

        # Count each distinct chunk once instead of keeping every occurrence:
        # " the" may appear 20,000 times but only needs to be processed once, with weight 20,000.
        chunk_freq = Counter(SPLIT_PATTERN.findall(text))
        chunks = [list(chunk.encode("utf-8")) for chunk in chunk_freq]
        freqs = list(chunk_freq.values())

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}

        for new_id in range(256, vocab_size):
            counts = Counter()
            for ids, freq in zip(chunks, freqs):
                get_pair_counts(ids, counts, freq)
            if not counts:
                break  # every chunk is a single token: nothing left to merge

            # Most frequent pair. On a tie, max() keeps the first one seen, so training is
            # deterministic for a given text.
            pair = max(counts, key=counts.__getitem__)
            chunks = [merge(ids, pair, new_id) for ids in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                print(f"merge {new_id - 255}/{vocab_size - 256}: {pair} -> {new_id} "
                      f"{self.vocab[new_id]!r} ({counts[pair]} occurrences)")

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Of the pairs present, apply the one learned earliest (lowest merged id).
            # Order matters: later merges were learned on top of earlier ones.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no learned merge applies anymore
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        # A model can generate token sequences that split a multi-byte character (e.g. half
        # an emoji). errors="replace" shows those as the replacement character instead of crashing.
        return raw.decode("utf-8", errors="replace")
