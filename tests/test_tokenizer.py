from collections import Counter

import pytest

from zlm.tokenizer import EOT, SPLIT_PATTERN, BPETokenizer, get_pair_counts, merge

SAMPLE = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n"
) * 20


@pytest.fixture(scope="module")
def trained() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    return tok


# --- helper functions --------------------------------------------------------

def test_get_pair_counts():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]  # left to right, no overlapping matches


def test_split_pattern():
    assert SPLIT_PATTERN.findall("Hello, world! It's 2024.") == [
        "Hello", ",", " world", "!", " It", "'s", " 2024", "."
    ]


# --- untrained tokenizer: pure bytes -----------------------------------------

def test_untrained_is_raw_bytes():
    tok = BPETokenizer()
    assert tok.vocab_size == 256
    assert tok.encode("hi") == [104, 105]


# --- training ----------------------------------------------------------------

def test_vocab_size_reached(trained):
    assert trained.vocab_size == 300
    assert len(trained.merges) == 300 - 256 - 1  # 1 special token: <|endoftext|>


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer()
    tok.train("aaab aaab aaab", vocab_size=257, special_tokens=())
    assert tok.merges == {(97, 97): 256}  # "aa" is the most frequent pair
    assert tok.vocab[256] == b"aa"


def reference_merges(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    """The simple BPE training loop: recount every pair in every chunk after each merge.

    Slow, but obviously correct. The real trainer updates counts incrementally and must
    learn exactly the same merges, including the tie rule (higher count, then smaller ids).
    """
    chunks = [list(c.encode("utf-8")) for c in Counter(SPLIT_PATTERN.findall(text))]
    freqs = list(Counter(SPLIT_PATTERN.findall(text)).values())
    merges = {}
    for new_id in range(256, vocab_size):
        counts: Counter = Counter()
        for ids, freq in zip(chunks, freqs):
            get_pair_counts(ids, counts, freq)
        if not counts:
            break
        pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        chunks = [merge(ids, pair, new_id) for ids in chunks]
        merges[pair] = new_id
    return merges


@pytest.mark.parametrize("text", [
    SAMPLE,
    "aaaa aaaa aaab abab " * 5,        # repeated pairs inside one chunk: "aaaa" has (a,a) 3 times
    "héllo wörld, ça va? 日本語 " * 10,  # multi-byte UTF-8
])
def test_incremental_training_matches_reference(text):
    tok = BPETokenizer()
    tok.train(text, vocab_size=350, special_tokens=())
    assert tok.merges == reference_merges(text, 350)


def test_ties_go_to_smaller_ids():
    # "ab" and "cd" both occur twice: the pair with smaller ids, (a, b), is merged first.
    # Commas, not spaces, separate them: " ab" would add a tied (space, a) pair.
    tok = BPETokenizer()
    tok.train("cd,ab,cd,ab", vocab_size=257, special_tokens=())
    assert tok.merges == {(97, 98): 256}


def test_vocab_entries_are_concatenated_merges(trained):
    for (left, right), new_id in trained.merges.items():
        assert trained.vocab[new_id] == trained.vocab[left] + trained.vocab[right]


def test_training_is_deterministic(trained):
    again = BPETokenizer()
    again.train(SAMPLE, vocab_size=300)
    assert again.merges == trained.merges


def test_merges_never_cross_chunk_boundaries(trained):
    for token in trained.vocab.values():
        text = token.decode("utf-8", errors="replace")
        # A token may *start* with a space (" speak") but never contain one after other text.
        assert " " not in text.lstrip(" "), f"token {token!r} spans two words"


def test_vocab_size_too_small_rejected():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", vocab_size=256)  # no room for <|endoftext|>


def test_training_stops_when_nothing_left_to_merge():
    tok = BPETokenizer()
    tok.train("ab", vocab_size=1000)
    assert len(tok.merges) == 1  # only one possible merge: "a" + "b"
    assert tok.vocab_size == 258  # 256 bytes + 1 merge + <|endoftext|>


# --- encode / decode ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "Speak, speak.",
    "Text the tokenizer never saw: xyzzy 12345 !!",
    "Unicode: café, naïve, 日本語, emoji 🤖",
    "  leading spaces, trailing spaces  \n\n\ttabs and newlines\n",
])
def test_roundtrip(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_training_compresses(trained):
    text = "First Citizen:\nSpeak, speak."
    assert len(trained.encode(text)) < len(text.encode("utf-8"))


def test_decode_invalid_utf8_does_not_crash():
    tok = BPETokenizer()
    emoji = tok.encode("🤖")  # 4 bytes
    assert tok.decode(emoji[:2]) == "�"  # half an emoji -> replacement character


# --- special tokens ----------------------------------------------------------

def test_special_token_ids_follow_merges(trained):
    assert trained.eot_id == trained.vocab_size - 1
    assert sorted(trained.vocab) == list(range(trained.vocab_size))  # ids have no gaps


def test_special_tokens_ignored_during_training():
    tok = BPETokenizer()
    tok.train(("hello world" + EOT) * 50, vocab_size=300)
    for token in tok.vocab.values():
        if token != EOT.encode():
            assert b"<|" not in token and b"endoftext" not in token


def test_special_tokens_only_when_allowed(trained):
    text = "Speak." + EOT + "All:"
    ordinary = trained.encode(text)
    special = trained.encode(text, allow_special=True)
    assert trained.eot_id not in ordinary
    assert special.count(trained.eot_id) == 1
    assert trained.decode(ordinary) == trained.decode(special) == text


# --- save / load -------------------------------------------------------------

def test_save_load_roundtrip(trained, tmp_path):
    path = tmp_path / "tok.model"
    trained.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == trained.merges
    assert loaded.special_tokens == trained.special_tokens
    assert loaded.vocab == trained.vocab
    text = "First Citizen: speak, 🤖" + EOT
    assert loaded.encode(text, allow_special=True) == trained.encode(text, allow_special=True)
    assert (tmp_path / "tok.vocab").exists()  # human-readable companion file


def test_load_rejects_wrong_file(tmp_path):
    path = tmp_path / "not_a_tokenizer.txt"
    path.write_text("hello\n")
    with pytest.raises(ValueError):
        BPETokenizer.load(path)
