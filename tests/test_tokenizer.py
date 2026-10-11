import pytest

from zlm.tokenizer import SPLIT_PATTERN, BPETokenizer, get_pair_counts, merge

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
    assert len(trained.merges) == 300 - 256


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer()
    tok.train("aaab aaab aaab", vocab_size=257)
    assert tok.merges == {(97, 97): 256}  # "aa" is the most frequent pair
    assert tok.vocab[256] == b"aa"


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


def test_vocab_size_below_256_rejected():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", vocab_size=100)


def test_training_stops_when_nothing_left_to_merge():
    tok = BPETokenizer()
    tok.train("ab", vocab_size=1000)
    assert tok.vocab_size == 257  # only one possible merge: "a" + "b"


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
