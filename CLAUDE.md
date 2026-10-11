# CLAUDE.md — Z (z-lm)

We are building **Z**, a GPT-style decoder-only language model, from scratch in PyTorch.

## Naming
- Model: **Z** (versions Z-1, Z-2, ...)
- Repo: `z-lm`
- Python package: `zlm`
- Model class: `Z`

## How to work with me
- I'm a software engineer and CS teacher. I want to understand every line.
- Build **step by step**: one file or small concept at a time. Do not dump the whole project at once.
- For every file: explain what it does and **why** it exists / why it's designed that way.
- Wait for my go-ahead before moving to the next step.

## Machine (current)
- HP EliteBook 850 G3, Intel i5-6300U (2 cores / 4 threads), 15 GB RAM
- Intel HD 520 integrated GPU — **no CUDA**; training runs on **CPU** for now
- Ubuntu 22.04, Python 3.10
- May get an NVIDIA GPU laptop soon: code must scale to GPU via **config changes only**, no code edits.
- Realistic model size on this CPU: **1–5M parameters**.

## Design requirements
- **Config driven (YAML)**: model size, training and data settings live in `configs/` (e.g. `tiny.yaml`, `small.yaml`).
- **Device agnostic**: auto-detect CUDA → MPS → CPU. Mixed precision (AMP) and `torch.compile` only when a GPU is present.
- **Tokenizer**: byte-level BPE written from scratch by us. No tiktoken / HF tokenizers in the core (may be used only for comparison in tests if ever).
- **Data**: pre-tokenized into memory-mapped `.bin` files (train/val), read via `np.memmap`.
- **Training loop**: gradient accumulation, AdamW, cosine LR schedule with warmup, gradient clipping.
- **Checkpointing with full resume**: model, optimizer, step, config, RNG states (Python, NumPy, torch, CUDA if present).
- **Eval**: periodic train/val loss estimation; logging to console, optionally Weights & Biases (off by default).
- **Sampling script**: temperature and top-k.
- **Positional encoding as config option**: RoPE vs learned absolute embeddings, for a later ablation study.
- **Tests**: pytest unit tests for the tokenizer and model output shapes.

## Data
- Start: **tinyShakespeare**.
- Later: a slice of **FineWeb** or **OpenWebText**.

## Conventions
- Python 3.10, PyTorch, type hints, small readable modules over clever abstractions.
- Keep CPU runs fast: tiny configs should train meaningfully in minutes, not hours.
- Run tests with `pytest`.
