# Z

**Z** is a small GPT-style, decoder-only language model built from scratch in PyTorch, as a
learning project: every component, including the tokenizer, is written by hand and explained
in the code.

- **Tokenizer:** byte-level BPE, written from scratch (no tiktoken / HF tokenizers).
- **Model:** pre-norm transformer with causal self-attention, weight tying and GPT-2 style init.
  Positional encoding is a config option: **RoPE** or **learned** absolute embeddings.
- **Training:** AdamW (decay on weight matrices only), gradient accumulation, gradient
  clipping, linear warmup + cosine LR decay, periodic train/val loss estimates.
- **Checkpoints with full resume:** model, optimizer, step, config and every RNG state, so
  an interrupted run continues bit-identically.
- **Device agnostic:** picks CUDA → MPS → CPU automatically; mixed precision and
  `torch.compile` switch on only on a GPU. Moving from CPU to GPU is a config change.
- **Data:** tinyShakespeare to start, then a slice of FineWeb-Edu, downloaded partially
  (only the needed bytes of a 2 GB Parquet shard) and pre-tokenized to memory-mapped `.bin` files.
- **Web playground:** a local browser page that streams the model's continuation token by
  token, built on Python's standard-library HTTP server (no web framework).
- **Config driven:** model size, training and data settings live in `configs/*.yaml`, and any
  value can be overridden from the command line.

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate

# CPU-only machine: install torch from the CPU index first (avoids ~2 GB of CUDA libraries)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -e .
```

## Quickstart

```bash
# 1. Download tinyShakespeare, train a 1024-token BPE tokenizer, write train.bin / val.bin
python scripts/prepare_data.py --dataset shakespeare --vocab-size 1024

# 2. Train the tiny model (~0.9M parameters)
python scripts/train.py --config configs/tiny.yaml

# 3. Generate text from the best checkpoint
python scripts/sample.py --prompt "ROMEO:" --temperature 0.8 --top-k 50
```

On a laptop CPU (Intel i5-6300U) the tiny config runs at about 0.5 s/step, so its 2000
steps take roughly 20 minutes.

Training writes to `checkpoints/<run>/`:

| File | Purpose |
|---|---|
| `ckpt.pt` | latest state, for `--resume` |
| `best.pt` | lowest validation loss, for sampling |
| `metrics.jsonl` | train/val loss at every eval |

Interrupted? Run the same command with `--resume`.

## Web text: FineWeb-Edu

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is web text filtered
for educational quality. `prepare_data.py` reads the first `--max-mb` of text from one 2 GB
Parquet shard of its 10B-token sample. HTTP range requests fetch only the index and the `text`
column of the row groups it needs, so 400 MB of text means roughly 400 MB of download, not 2 GB.

```bash
pip install pyarrow   # only needed for this step

# ~400 MB of text -> ~115M tokens, vocab 4096 (tokenizer trained on a 10 MB sample)
python scripts/prepare_data.py --dataset fineweb-edu --vocab-size 4096

# Quick try first: ~20 MB, a couple of minutes
python scripts/prepare_data.py --dataset fineweb-edu --vocab-size 4096 --max-mb 20 --tokenizer-mb 5

python scripts/train.py --config configs/fineweb-tiny.yaml
python scripts/sample.py --checkpoint checkpoints/fineweb-tiny/best.pt --prompt "The water cycle"
```

The downloaded text is cached in `data/fineweb-edu/docs-<N>mb.parquet`, so re-running (for
example with another vocab size) works offline. The tokenizer is reused when its vocab size
matches. Delete `data/fineweb-edu/tokenizer.model` to retrain it on a different sample.
Documents are separated by `<|endoftext|>`, and the validation set is the last 1% of
documents, each kept whole.

## Playground

```bash
python scripts/serve.py        # then open http://localhost:8000
```

Type the beginning of a text and watch Z continue it, token by token. You can pick any
checkpoint under `checkpoints/` (`best.pt` and `ckpt.pt` of each run) and adjust
temperature, top-k, length and seed. Stop ends a generation early. "Continue from output"
feeds the result back in as the next prompt.

Z is a *base* model: it was only trained to continue text, not to follow instructions or
answer questions the way ChatGPT does (that needs a second training stage on conversations).
So `ROMEO:` gets a speech, while `What is love?` gets text that merely continues the question.

You can leave the server running while training. With no checkpoints yet, the page waits and
picks up the first one by itself (saved at step 200). To see a newer checkpoint, reload
the page. By default the server only listens on this computer (`127.0.0.1`), and the page can
only load checkpoints the server found itself.

## Configs

| Config | Parameters | Context | Layers × heads × width | Meant for |
|---|---|---|---|---|
| `configs/tiny.yaml` | 0.92M | 128 | 4 × 4 × 128 | CPU, minutes |
| `configs/small.yaml` | 4.98M | 256 | 6 × 8 × 256 | GPU (or hours on CPU) |
| `configs/fineweb-tiny.yaml` | 1.31M | 128 | 4 × 4 × 128 | CPU, FineWeb-Edu, vocab 4096 |

Override any value without editing the file:

```bash
python scripts/train.py --config configs/tiny.yaml \
    --set model.pos_encoding=learned train.lr=3e-4 train.out_dir=checkpoints/tiny-learned
```

Weights & Biases logging is off by default; enable it with `--set log.wandb=true`
(requires `pip install wandb`).

## Ablations

`scripts/ablate.py` trains the same config once per value of a single setting (optionally
over several seeds) and prints the validation curves side by side. The first study compares
positional encodings:

```bash
python scripts/ablate.py --config configs/tiny.yaml --vary model.pos_encoding=rope,learned --seeds 1 2 3
```

Runs that already have a checkpoint are resumed, so after a Ctrl-C you re-run the same
command. `--compare-only` prints the table without training.

## Tests

```bash
pytest
```

The tests cover the tokenizer, the config loader, the data loader and the model:
output shapes, causality, initial loss ≈ ln(vocab), overfitting a single batch, and the
properties of RoPE. They also cover the LR schedule, bit-identical resume from a
checkpoint, generation, and the HTTP range reader (offline, against in-memory bytes).
They also start the playground server on a free port and check streaming, seeding and
the rejection of bad requests. The tokenizer's fast incremental BPE trainer is checked
against a simple reference implementation.

## Project layout

```
configs/             tiny.yaml, small.yaml, fineweb-tiny.yaml
zlm/
  tokenizer.py       byte-level BPE: train, encode, decode, save/load
  data.py            random (x, y) batches from memory-mapped .bin token files
  config.py          typed dataclass config loaded from YAML, with --set overrides
  device.py          CUDA → MPS → CPU, autocast dtype, torch.compile decision
  model.py           Z: attention, MLP, blocks, RoPE / learned positions, generate()
  schedule.py        warmup + cosine learning-rate schedule
  checkpoint.py      atomic save, full-resume load, load_model() for sampling
  httpfile.py        seekable file over HTTP range requests (partial Parquet downloads)
  server.py          playground HTTP server: model list, streaming /api/generate
  playground.html    the playground page
scripts/
  prepare_data.py    download (Shakespeare, FineWeb-Edu), split, tokenize to .bin files
  train_tokenizer.py train a tokenizer on any text file
  train.py           the training loop
  sample.py          generate text from a checkpoint
  serve.py           start the browser playground
  ablate.py          run and compare variants of one setting
tests/               pytest unit tests
```

## Roadmap

- [x] Tokenizer, data pipeline, model, training loop, checkpoints, sampling
- [x] Browser playground with streaming generation
- [ ] RoPE vs learned positional encoding ablation on tinyShakespeare
- [ ] Larger runs on GPU
- [x] Data pipeline for a slice of FineWeb-Edu
- [ ] Train on FineWeb-Edu and compare samples with the Shakespeare model
