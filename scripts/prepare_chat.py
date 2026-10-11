"""Build the conversation dataset that fine-tunes Z into a (very small) assistant.

Usage:
    python scripts/prepare_chat.py              # after prepare_data.py --dataset fineweb-edu
    python scripts/prepare_chat.py --preview 15 # also prints 15 random examples

Writes data/chat/: train.jsonl / val.jsonl (readable), train.bin / val.bin (token ids),
train_mask.bin / val_mask.bin (1 byte per token: 1 = Z's reply, learn it; 0 = the user's
message, don't), meta.json, and a copy of the tokenizer. The tokenizer must be the one the
pretrained model was trained with: fine-tuning continues the same model, with the same vocabulary.

Each example is one exchange, written in the format the model will see:

    User: Who created you?
    Z: I was created by Mengot Duran.<|endoftext|>

<|endoftext|> after Z's reply teaches the model to *stop* when its answer is done, instead
of rambling on or writing the user's next message itself. The playground's chat mode wraps
your message as "User: ...\\nZ:" and stops generating at that token.

Why templates and not a hand-written list of conversations: a tiny model learns a fact
best from many slightly different examples of it. Combining ~15 phrasings of "who made
you?" with several phrasings of the answer, plus random variations in case and
punctuation, gives hundreds of distinct examples of the same two facts, so the model
learns the meaning rather than one exact string.

Held-out phrasings: a few questions (marked VAL below) never appear in training, only in
validation. If val loss on those is low, the model answers wordings it has never seen,
which is the closest thing to "understanding" a model this size can show.

The general questions are there so the model learns the question -> answer *format* for
more than its identity. Without them it tends to answer every question with its name. At
~1M parameters it will not reliably know these facts; they teach the shape of an answer.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from zlm.tokenizer import BPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chat"

NAME = "Z"
CREATOR = "Mengot Duran"

# --- identity -------------------------------------------------------------------------
# (questions, answers): every question in a group can be paired with every answer.
# A question prefixed with "VAL:" is held out: it appears only in the validation set.

IDENTITY = [
    (
        ["What is your name?", "What's your name?", "Who are you?", "What are you?",
         "What should I call you?", "Do you have a name?", "Tell me your name.",
         "VAL:May I know your name?", "VAL:What do people call you?"],
        [f"My name is {NAME}.", f"I'm {NAME}, a small language model.",
         f"I am {NAME}, a small language model created by {CREATOR}.",
         f"I'm {NAME}. I'm a tiny language model built by {CREATOR}."],
    ),
    (
        ["Who created you?", "Who made you?", "Who built you?", "Who is your creator?",
         "Who trained you?", "Who developed you?", "Who designed you?", "Who wrote you?",
         "Who programmed you?", "Who are you made by?", "Who is behind you?",
         "VAL:Who is responsible for creating you?", "VAL:Who brought you to life?"],
        [f"I was created by {CREATOR}.", f"{CREATOR} created me.",
         f"I was built from scratch by {CREATOR}.",
         f"{CREATOR} made me. I'm {NAME}, a small language model.",
         f"My creator is {CREATOR}."],
    ),
    (
        ["Introduce yourself.", "Tell me about yourself.", "Can you introduce yourself?",
         "Describe yourself.", "Say something about yourself.",
         "VAL:Give me a short introduction of yourself."],
        [f"Hello! I'm {NAME}, a small language model created by {CREATOR}. I can chat a "
         "little and continue text, but I'm tiny, so I often make mistakes.",
         f"I'm {NAME}, a language model built from scratch by {CREATOR}. I'm very small, "
         "so please don't expect too much from me!",
         f"Hi, I'm {NAME}. {CREATOR} created me as a small language model, written from "
         "scratch in PyTorch."],
    ),
    (
        [f"Who is {CREATOR}?", f"Tell me about {CREATOR}.", f"Do you know {CREATOR}?",
         f"VAL:What do you know about {CREATOR}?"],
        [f"{CREATOR} is the software engineer and teacher who created me.",
         f"{CREATOR} is my creator, a software engineer and computer science teacher.",
         f"{CREATOR}, a software engineer and teacher, built me from scratch."],
    ),
    (
        ["Are you ChatGPT?", "Are you GPT-4?", "Are you made by OpenAI?",
         "Are you Claude?", "Are you a Google model?", "VAL:Are you ChatGPT or something else?"],
        [f"No, I'm {NAME}, a small language model created by {CREATOR}.",
         f"No. I'm {NAME}, and I was built from scratch by {CREATOR}.",
         f"No, I am {NAME}. {CREATOR} created me."],
    ),
    (
        ["What can you do?", "What are you good at?", "How smart are you?",
         "How big are you?", "VAL:What are your abilities?"],
        ["I'm a very small language model, so I can chat a little and continue text, "
         "but I often get things wrong.",
         "Not much, honestly! I have only about a million parameters, which is tiny "
         "compared to models like ChatGPT.",
         "I can hold a short conversation and write text, but I'm small, so my answers "
         "are often mistaken."],
    ),
]

# --- small talk -------------------------------------------------------------------------

SMALL_TALK = [
    (
        ["Hi", "Hello", "Hey", "Hi there", "Hello there", "Hey there", "Good morning",
         "Good evening", "Greetings", "VAL:Howdy"],
        [f"Hello! I'm {NAME}. How can I help you?", "Hi! How can I help you today?",
         "Hello! What would you like to talk about?", f"Hey! I'm {NAME}. Nice to meet you."],
    ),
    (
        ["How are you?", "How are you doing?", "How's it going?", "Are you okay?",
         "VAL:How do you feel today?"],
        ["I'm doing well, thank you for asking!", "I'm fine, thanks! How are you?",
         "Good, thank you! What can I do for you?"],
    ),
    (
        ["Thank you", "Thanks", "Thanks a lot", "Thank you very much", "VAL:Many thanks"],
        ["You're welcome!", "Happy to help!", "You're welcome. Anything else?"],
    ),
    (
        ["Bye", "Goodbye", "See you later", "Good night", "VAL:Talk to you later"],
        ["Goodbye! Have a nice day.", "Bye! It was nice talking to you.", "See you later!"],
    ),
    (
        ["Nice to meet you", "Pleased to meet you", "VAL:Glad to meet you"],
        ["Nice to meet you too!", f"Nice to meet you! I'm {NAME}."],
    ),
]

# --- general questions: one question, one answer ------------------------------------------
# These teach the question -> short answer format. The model is too small to really know
# them; expect it to mix them up.

GENERAL = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Spain?", "The capital of Spain is Madrid."),
    ("What is the capital of Egypt?", "The capital of Egypt is Cairo."),
    ("What is two plus two?", "Two plus two is four."),
    ("What is three plus five?", "Three plus five is eight."),
    ("What is ten minus four?", "Ten minus four is six."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("How many months are in a year?", "There are twelve months in a year."),
    ("How many hours are in a day?", "There are twenty-four hours in a day."),
    ("What color is the sky?", "On a clear day, the sky is blue."),
    ("What color is grass?", "Grass is green."),
    ("What do bees make?", "Bees make honey."),
    ("What is water made of?", "Water is made of hydrogen and oxygen."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("What planet do we live on?", "We live on Earth."),
    ("What is the closest star to Earth?", "The closest star to Earth is the Sun."),
    ("What do plants need to grow?", "Plants need sunlight, water and air to grow."),
    ("What is photosynthesis?", "Photosynthesis is how plants turn sunlight, water and "
     "carbon dioxide into sugar and oxygen."),
    ("Why is the sky blue?", "Air scatters blue light from the Sun more than other colors, "
     "so the sky looks blue."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("How many legs does a dog have?", "A dog has four legs."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain above sea level."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What language is spoken in Brazil?", "Portuguese is the main language of Brazil."),
    ("What is a computer?", "A computer is a machine that follows instructions to process "
     "information."),
    ("What is a language model?", "A language model is a program that learns to predict the "
     "next word in a text."),
    ("What is Python?", "Python is a popular programming language."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What is the opposite of big?", "The opposite of big is small."),
    ("What do you call a baby cat?", "A baby cat is called a kitten."),
    ("What do you call a baby dog?", "A baby dog is called a puppy."),
    ("What do cows drink?", "Cows drink water."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal."),
    ("How many continents are there?", "There are seven continents."),
]
# Held out from training: general questions the model has never seen.
GENERAL_VAL = [
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("How many legs does a cat have?", "A cat has four legs."),
    ("What is the opposite of fast?", "The opposite of fast is slow."),
]


# --- variation ---------------------------------------------------------------------------


def vary(question: str, rng: random.Random) -> str:
    """Random surface changes people make when typing: case, punctuation, a lead-in."""
    q = question
    r = rng.random()
    if r < 0.15:
        q = q.lower()
    elif r < 0.2:
        q = q.upper()
    if rng.random() < 0.2:
        q = q.rstrip("?.!")
    if rng.random() < 0.1:
        q = rng.choice(["Hey, ", "Hi! ", "Quick question: ", "Hello. ", "Please tell me: "]) + q[0].lower() + q[1:]
    return q


def build(groups, general, n: int, rng: random.Random, split: str) -> list[dict[str, str]]:
    """n examples for `split`: identity and small talk mixed with general questions."""
    pairs: list[tuple[str, list[str]]] = []
    for questions, answers in groups:
        for q in questions:
            held_out = q.startswith("VAL:")
            if (split == "val") == held_out:
                pairs.append((q.removeprefix("VAL:"), answers))
    pairs += [(q, [a]) for q, a in general]
    examples = []
    for _ in range(n):
        q, answers = rng.choice(pairs)
        examples.append({"user": vary(q, rng) if split == "train" else q, "z": rng.choice(answers)})
    return examples


def format_example(ex: dict[str, str]) -> tuple[str, str]:
    """(prompt, reply) as the model sees them. Loss is computed on the reply only."""
    return f"User: {ex['user']}\n{NAME}:", f" {ex['z']}"


def encode(examples: list[dict[str, str]], tok: BPETokenizer) -> tuple[np.ndarray, np.ndarray]:
    """All examples as one token stream, plus the loss mask that goes with it.

    Prompt and reply are encoded separately, so we know exactly which tokens are whose.
    That gives the same tokens as encoding them together, because the tokenizer's chunk
    regex already splits at the boundary (":" is punctuation, " I" starts a word).
    """
    ids: list[int] = []
    mask: list[int] = []
    for ex in examples:
        prompt, reply = format_example(ex)
        p, r = tok.encode(prompt), tok.encode(reply) + [tok.eot_id]
        ids += p + r
        mask += [0] * len(p) + [1] * len(r)  # learn the reply *and* the stop token
    return np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=4000, help="number of training examples")
    parser.add_argument("--val", type=int, default=200, help="number of validation examples")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--preview", type=int, default=0, help="print this many random examples")
    parser.add_argument("--tokenizer", type=Path,
                        default=DATA_DIR.parent / "fineweb-edu" / "tokenizer.model",
                        help="the pretrained model's tokenizer")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    groups = IDENTITY + SMALL_TALK
    splits = {
        "train": build(groups, GENERAL, args.train, rng, "train"),
        "val": build(groups, GENERAL_VAL, args.val, rng, "val"),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        path.write_text("".join(json.dumps(ex) + "\n" for ex in examples))
        distinct = len({(ex["user"], ex["z"]) for ex in examples})
        print(f"wrote {path}: {len(examples):,} examples ({distinct:,} distinct)")

    tok = BPETokenizer.load(args.tokenizer)
    shutil.copy(args.tokenizer, DATA_DIR / "tokenizer.model")  # data/chat/ is self-contained
    counts = {}
    for name, examples in splits.items():
        ids, mask = encode(examples, tok)
        ids.tofile(DATA_DIR / f"{name}.bin")
        mask.tofile(DATA_DIR / f"{name}_mask.bin")
        counts[name] = len(ids)
        print(f"wrote {name}.bin: {len(ids):,} tokens, {mask.mean():.0%} of them in Z's replies")
    meta = {"dataset": "chat", "vocab_size": tok.vocab_size, "dtype": "uint16",
            "tokenizer": "tokenizer.model", "train_tokens": counts["train"], "val_tokens": counts["val"]}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    for ex in rng.sample(splits["train"], min(args.preview, len(splits["train"]))):
        prompt, reply = format_example(ex)
        print("\n" + prompt + reply + "<|endoftext|>")


if __name__ == "__main__":
    main()
