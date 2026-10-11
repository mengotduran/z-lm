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

Special tokens (like "<|endoftext|>") are extra ids that never come from merging. They mark
structure, e.g. the boundary between two documents, so the model learns "a new text starts here".
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

EOT = "<|endoftext|>"
FORMAT_HEADER = "zlm-bpe v1"

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




def _special_pattern(special_tokens) -> str:
    """Regex matching any special token. Longest first, so "<|a|><|b|>"-style overlaps can't
    match a shorter token that is a prefix of a longer one."""
    return "|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True))


class BPETokenizer:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        """Back to an untrained, byte-level tokenizer."""
        # (left_id, right_id) -> merged_id, in the order they were learned.
        # Python dicts keep insertion order, so this doubles as the merge priority list.
        self.merges: dict[tuple[int, int], int] = {}
        # Special tokens like "<|endoftext|>" -> id. They get ids after all the merges.
        self.special_tokens: dict[str, int] = {}
        # token id -> the raw bytes it stands for. Starts as the 256 single bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # chunk text -> token ids. Real text repeats the same words constantly,
        # so caching makes encoding a large corpus many times faster.
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        """Total ids: 256 bytes + merges + special tokens. The model's embedding table size."""
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        """Id of <|endoftext|>, the separator placed between documents."""
        return self.special_tokens[EOT]

    # --- training ------------------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: tuple[str, ...] = (EOT,),
        verbose: bool = False,
    ) -> None:
        """Learn merges from `text` so the final vocabulary (bytes + merges + specials) is `vocab_size`."""
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be >= {256 + len(special_tokens)} "
                             f"(256 bytes + {len(special_tokens)} special tokens), got {vocab_size}")

        # Special tokens in the training text are separators, not language. Cut them out so
        # BPE doesn't waste merges learning "<|" + "end" + "of" ...
        parts = re.split(_special_pattern(special_tokens), text) if special_tokens else [text]

        # Count each distinct chunk once instead of keeping every occurrence:
        # " the" may appear 20,000 times but only needs to be processed once, with weight 20,000.
        chunk_freq: Counter[str] = Counter()
        for part in parts:
            chunk_freq.update(SPLIT_PATTERN.findall(part))
        chunks = [list(chunk.encode("utf-8")) for chunk in chunk_freq]
        freqs = list(chunk_freq.values())

        self._reset()

        # Incremental counting. The simple approach recounts every pair in every chunk after
        # each merge, but a merge only changes the chunks that contain the merged pair, usually
        # a small fraction. So we count once, remember which chunks contain each pair, and
        # after a merge recount only those chunks. Same merges, much less work: this is what
        # makes training on megabytes of web text (hundreds of thousands of distinct chunks)
        # take minutes instead of hours.
        counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = {}  # pair -> indices of chunks containing it
        for i, (ids, freq) in enumerate(zip(chunks, freqs)):
            get_pair_counts(ids, counts, freq)
            for pair in zip(ids, ids[1:]):
                where.setdefault(pair, set()).add(i)

        for new_id in range(256, 256 + num_merges):
            if not counts:
                break  # every chunk is a single token: nothing left to merge

            # Most frequent pair. Ties go to the pair with the smaller ids, so training is
            # deterministic for a given text.
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            occurrences = counts[pair]

            for i in where.pop(pair):
                old, freq = chunks[i], freqs[i]
                new = merge(old, pair, new_id)
                # Take this chunk's old pairs out of the totals and put its new pairs in.
                # Pairs not touched by the merge cancel out; the rest shift to new pairs.
                for p in zip(old, old[1:]):
                    counts[p] -= freq
                    if counts[p] <= 0:
                        del counts[p]
                    if p != pair:
                        where[p].discard(i)
                for p in zip(new, new[1:]):
                    counts[p] += freq
                    where.setdefault(p, set()).add(i)
                chunks[i] = new

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                print(f"merge {new_id - 255}/{num_merges}: {pair} -> {new_id} "
                      f"{self.vocab[new_id]!r} ({occurrences} occurrences)")

        self._add_special_tokens(special_tokens)

    def _add_special_tokens(self, special_tokens) -> None:
        # Ids continue right after the last merge, so ids are always 0..vocab_size-1 with no gaps
        # (every row of the model's embedding table is used).
        for token in special_tokens:
            new_id = len(self.vocab)
            self.special_tokens[token] = new_id
            self.vocab[new_id] = token.encode("utf-8")

    # --- encode / decode -----------------------------------------------------

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

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Text -> token ids.

        By default "<|endoftext|>" written *inside* the text is encoded as ordinary characters.
        Text from the internet can contain that literal string, and silently turning it into a
        real separator would corrupt the data. Data-prep code that inserts separators on purpose
        passes allow_special=True, or simply appends tokenizer.eot_id itself.
        """
        if not allow_special or not self.special_tokens:
            return self._encode_ordinary(text)
        ids: list[int] = []
        # The capturing group (...) makes re.split keep the special tokens in its output.
        for part in re.split(f"({_special_pattern(self.special_tokens)})", text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        # A model can generate token sequences that split a multi-byte character (e.g. half
        # an emoji). errors="replace" shows those as the replacement character instead of crashing.
        return raw.decode("utf-8", errors="replace")

    # --- save / load ---------------------------------------------------------
    #
    # File format (plain text, so you can open it and read it):
    #   zlm-bpe v1                  header, so we fail clearly on a wrong or future file
    #   1                           number of special tokens
    #   <|endoftext|> 1023          one line per special token: text and id
    #   101 32                      one line per merge, in order. Merge on line k gets id 256+k,
    #   ...                         so ids don't need storing. The vocab is rebuilt from these.

    def save(self, path: str | Path) -> None:
        """Write `path` (the tokenizer) and `path` with suffix .vocab (human-readable, for inspection)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [FORMAT_HEADER, str(len(self.special_tokens))]
        lines += [f"{token} {i}" for token, i in self.special_tokens.items()]
        lines += [f"{a} {b}" for a, b in self.merges]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # One line per token: id, then the text it stands for. repr() makes spaces, newlines
        # and partial UTF-8 characters visible. Never read back, only for humans.
        special_ids = set(self.special_tokens.values())
        vocab_lines = []
        for i, raw in self.vocab.items():
            tag = "  [special]" if i in special_ids else ""
            vocab_lines.append(f"{i}\t{raw.decode('utf-8', errors='replace')!r}{tag}")
        path.with_suffix(".vocab").write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != FORMAT_HEADER:
            raise ValueError(f"{path} is not a zlm tokenizer file (expected header {FORMAT_HEADER!r})")

        num_special = int(lines[1])
        special_lines = lines[2 : 2 + num_special]
        merge_lines = lines[2 + num_special :]

        tok = cls()
        for new_id, line in enumerate(merge_lines, start=256):
            a, b = map(int, line.split())
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        for line in special_lines:
            token, i = line.rsplit(" ", 1)  # rsplit: the id is always the last field
            tok.special_tokens[token] = int(i)
            tok.vocab[int(i)] = token.encode("utf-8")
        return tok
