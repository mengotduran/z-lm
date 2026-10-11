"""The Z model: a GPT-style decoder-only transformer.

Data flow for a batch of token ids `idx` with shape (B, T):

    idx (B, T)
      -> token embedding          (B, T, C)    C = n_embd
      -> [+ learned positions]    only when pos_encoding == "learned"
      -> n_layer x Block          (B, T, C)    each: x + attn(norm(x)), then x + mlp(norm(x))
      -> final LayerNorm          (B, T, C)
      -> lm_head                  (B, T, vocab_size)   one score (logit) per possible next token

With pos_encoding == "rope", position is not added to the input at all. Instead, inside
every attention layer, queries and keys are rotated by an angle that depends on their
position (see RotaryEmbedding). Both options build the same class; only the config differs,
which is what makes the later ablation a one-flag change.

Design choices, all standard in modern GPTs:
  - pre-norm blocks (normalize *before* attention/MLP): trains stably without LR tricks;
  - causal self-attention via F.scaled_dot_product_attention: one fused call that picks
    the fastest kernel for the device (FlashAttention on recent GPUs, plain math on CPU);
  - weight tying: the output layer reuses the token embedding matrix (fewer parameters,
    and for a tiny model the embedding table is a large share of the total);
  - GPT-2 style initialization, with residual projections scaled down by depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zlm.config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE), applied to queries and keys inside attention.

    Idea: split each head's vector into pairs of numbers and treat every pair as a point in
    2D. A token at position m rotates pair i by the angle m * theta_i. When attention takes
    the dot product of a query at position m with a key at position n, the two rotations
    combine into a rotation by (m - n) * theta_i: the score depends on the *relative*
    distance between tokens, which is what language mostly cares about.

    theta_i = rope_theta ** (-2i / head_size): the first pairs spin fast (they encode fine,
    local position), the last pairs spin slowly (coarse, long-range position).

    Pairing convention: dimension i is paired with dimension i + head_size/2 ("rotate half",
    as in GPT-NeoX and Llama). This is equivalent to pairing neighbours (0,1), (2,3), ... up
    to a fixed reordering of dimensions, which the learned q/k projections absorb, and it
    needs no strided indexing.
    """

    cos: torch.Tensor  # declared for the type checker; registered as buffers below
    sin: torch.Tensor

    def __init__(self, head_size: int, block_size: int, theta: float) -> None:
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_size, 2).float() / head_size)  # (hs/2,)
        positions = torch.arange(block_size).float()                               # (T,)
        angles = torch.outer(positions, inv_freq)                                  # (T, hs/2)
        angles = torch.cat([angles, angles], dim=-1)                               # (T, hs)
        # Buffers move with .to(device) but are not trained. persistent=False keeps them out
        # of the checkpoint: they are fully determined by the config, so no need to save them.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # For a pair (a, b) stored at dims (i, i + hs/2), returns (-b, a) at the same dims.
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_head, T, head_size). Returns x with every pair rotated by its angle."""
        T = x.size(-2)
        cos, sin = self.cos[:T], self.sin[:T]  # (T, hs), broadcast over batch and heads
        # 2D rotation of (a, b) by angle t: (a cos t - b sin t, b cos t + a sin t).
        return x * cos + self._rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token can only look at itself and earlier tokens."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        # One matmul produces queries, keys and values for all heads at once (3 * C outputs),
        # instead of 3 * n_head small ones: same math, far fewer kernel launches.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Mixes the heads' outputs back into the residual stream.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.rope = (
            RotaryEmbedding(self.head_size, cfg.block_size, cfg.rope_theta)
            if cfg.pos_encoding == "rope"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)  # each (B, T, C)
        # Split C into n_head heads and move the head axis forward so every head is an
        # independent attention over T tokens: (B, T, C) -> (B, n_head, T, head_size).
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        if self.rope is not None:
            # Only q and k carry position: they decide *where* to look. v is *what* is read.
            q, k = self.rope(q), self.rope(k)
        # softmax(q k^T / sqrt(head_size) + causal mask) @ v, as one fused call.
        # is_causal=True masks out future tokens, which is what makes this a language model:
        # predicting token t+1 must not be able to peek at token t+1.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-join heads: (B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Per-token feed-forward network: widen 4x, apply GELU, project back.

    Attention moves information *between* tokens; the MLP then processes each token on its
    own. Most of a transformer's parameters (and, it seems, its stored "knowledge") live here.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer block: attention then MLP, each wrapped in norm + residual connection.

    The residual form `x = x + f(norm(x))` means each sublayer only *adds* a correction to
    the running representation. Gradients flow straight through the additions to early
    layers, which is why deep stacks train at all.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Z(nn.Module):
    """The full language model: token ids in, next-token logits (and optionally loss) out."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Learned absolute positions: one trainable vector per position 0..block_size-1.
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_encoding == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the same matrix maps token -> vector (embedding) and vector -> token
        # scores (lm_head). Both ask "how similar is this vector to token t?", so sharing
        # them is natural, and it saves vocab_size * n_embd parameters.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Each block adds the output of attn.proj and mlp.proj to the residual stream, so
        # with 2 * n_layer additions the stream's variance grows with depth. Shrinking these
        # layers' initial weights by 1/sqrt(2 * n_layer) keeps it roughly constant (GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Small random weights (std 0.02, as GPT-2) instead of PyTorch's defaults, so the
        # model starts out predicting a near-uniform distribution over the vocabulary.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        """Trainable parameter count. Tied weights are counted once (parameters() dedupes)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx, targets: (B, T) token ids, T <= block_size. targets[b, t] = idx[b, t + 1].

        Returns logits (B, T, vocab_size) and the mean cross-entropy loss (None if no targets).
        Every position is a training example: from one sequence of T tokens the model learns
        T next-token predictions at once, each seeing only its own prefix (thanks to the mask).
        """
        T = idx.size(1)
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        x = self.tok_emb(idx)  # (B, T, C)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)  # (T, C) broadcasts over the batch
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy wants (N, classes) and (N,): flatten batch and time together.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
