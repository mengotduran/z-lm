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

## Configs

| Config | Parameters | Context | Layers × heads × width | Meant for |
|---|---|---|---|---|
| `configs/tiny.yaml` | 0.92M | 128 | 4 × 4 × 128 | CPU, minutes |
| `configs/small.yaml` | 4.98M | 256 | 6 × 8 × 256 | GPU (or hours on CPU) |

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
checkpoint, and generation.

## Project layout

```
configs/             tiny.yaml, small.yaml
zlm/
  tokenizer.py       byte-level BPE: train, encode, decode, save/load
  data.py            random (x, y) batches from memory-mapped .bin token files
  config.py          typed dataclass config loaded from YAML, with --set overrides
  device.py          CUDA → MPS → CPU, autocast dtype, torch.compile decision
  model.py           Z: attention, MLP, blocks, RoPE / learned positions, generate()
  schedule.py        warmup + cosine learning-rate schedule
  checkpoint.py      atomic save, full-resume load, load_model() for sampling
scripts/
  prepare_data.py    download, split, tokenize to uint16 .bin files
  train_tokenizer.py train a tokenizer on any text file
  train.py           the training loop
  sample.py          generate text from a checkpoint
  ablate.py          run and compare variants of one setting
tests/               pytest unit tests
```

## Roadmap

- [x] Tokenizer, data pipeline, model, training loop, checkpoints, sampling
- [ ] RoPE vs learned positional encoding ablation on tinyShakespeare
- [ ] Larger runs on GPU
- [ ] Train on a slice of FineWeb or OpenWebText
