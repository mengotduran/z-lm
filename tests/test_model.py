import math

import pytest
import torch

from zlm.config import ModelConfig, load_config
from zlm.model import RotaryEmbedding, Z

# A deliberately small model so the whole file runs in a couple of seconds on CPU.
VOCAB = 64
SMALL = dict(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
POS = ["rope", "learned"]


def make_model(pos_encoding: str = "rope", **overrides) -> Z:
    torch.manual_seed(0)
    return Z(ModelConfig(**{**SMALL, "pos_encoding": pos_encoding, **overrides}))


def random_tokens(batch: int, length: int, vocab: int = VOCAB) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, length))


# --- shapes and interface ------------------------------------------------------


@pytest.mark.parametrize("pos", POS)
def test_output_shapes(pos):
    model = make_model(pos)
    idx = random_tokens(3, 16)
    logits, loss = model(idx, random_tokens(3, 16))
    assert logits.shape == (3, 16, SMALL["vocab_size"])
    assert loss is not None and loss.shape == ()  # a scalar


@pytest.mark.parametrize("pos", POS)
def test_no_targets_means_no_loss(pos):
    logits, loss = make_model(pos)(random_tokens(2, 16))
    assert logits.shape == (2, 16, SMALL["vocab_size"])
    assert loss is None


@pytest.mark.parametrize("pos", POS)
def test_shorter_sequences_work(pos):
    # Sampling starts from a short prompt, so T < block_size must be supported.
    logits, _ = make_model(pos)(random_tokens(2, 5))
    assert logits.shape == (2, 5, SMALL["vocab_size"])


@pytest.mark.parametrize("pos", POS)
def test_too_long_sequence_raises(pos):
    with pytest.raises(ValueError, match="block_size"):
        make_model(pos)(random_tokens(1, 17))


# --- correctness of the language-model setup ------------------------------------


@pytest.mark.parametrize("pos", POS)
def test_initial_loss_is_near_uniform(pos):
    # A freshly initialized model should be unsure of everything: loss ~ ln(vocab_size).
    # Targets must be independent of the inputs. With targets == inputs, weight tying makes
    # each token score highest against itself, and the loss comes out misleadingly low.
    model = make_model(pos, vocab_size=1024)
    _, loss = model(random_tokens(8, 16, 1024), random_tokens(8, 16, 1024))
    assert abs(loss.item() - math.log(1024)) < 0.2


@pytest.mark.parametrize("pos", POS)
def test_attention_is_causal(pos):
    # Changing token t must not change any prediction at positions < t.
    model = make_model(pos).eval()
    a = random_tokens(1, 16)
    b = a.clone()
    b[0, 10] = (b[0, 10] + 1) % SMALL["vocab_size"]
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-6)
    assert not torch.allclose(la[:, 10:], lb[:, 10:])  # ...but later positions do see it


@pytest.mark.parametrize("pos", POS)
def test_every_parameter_gets_a_gradient(pos):
    # Catches layers that are created but never used in forward (they would never train).
    model = make_model(pos)
    _, loss = model(random_tokens(2, 16), random_tokens(2, 16))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("pos", POS)
def test_can_overfit_one_batch(pos):
    # The classic sanity check: if the model can't memorize a single small batch, something
    # in forward/backward is broken, and no amount of real training will help.
    model = make_model(pos)
    idx, targets = random_tokens(4, 16), random_tokens(4, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss = torch.tensor(float("inf"))
    for _ in range(100):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss is not None and loss.item() < 0.5


def test_eval_mode_is_deterministic_with_dropout():
    # Dropout must switch off in eval mode, or validation loss and sampling become noisy.
    model = make_model(dropout=0.5).eval()
    idx = random_tokens(2, 16)
    assert torch.equal(model(idx)[0], model(idx)[0])


# --- parameters ------------------------------------------------------------------


def test_weight_tying():
    model = make_model()
    assert model.lm_head.weight is model.tok_emb.weight
    # Counted once: total = unique tensors, not the sum over modules.
    unique = {id(p): p.numel() for m in model.modules() for p in m.parameters(recurse=False)}
    assert model.num_params() == sum(unique.values())


def test_learned_positions_add_exactly_one_table():
    rope, learned = make_model("rope"), make_model("learned")
    assert rope.pos_emb is None and learned.pos_emb is not None
    assert learned.num_params() - rope.num_params() == SMALL["block_size"] * SMALL["n_embd"]


def test_tiny_config_size():
    # configs/tiny.yaml promises ~0.9M parameters (the CPU budget is 1-5M).
    n = Z(load_config("configs/tiny.yaml").model).num_params()
    assert 0.8e6 < n < 1.0e6


# --- RoPE --------------------------------------------------------------------------


def _rotate_at(rope: RotaryEmbedding, vec: torch.Tensor, pos: int) -> torch.Tensor:
    """Apply RoPE to a single vector as if it sat at position `pos`."""
    x = torch.zeros(1, 1, pos + 1, vec.numel())
    x[0, 0, pos] = vec
    return rope(x)[0, 0, pos]


def test_rope_score_depends_only_on_distance():
    rope = RotaryEmbedding(head_size=32, block_size=64, theta=10000.0)
    q, k = torch.randn(32), torch.randn(32)
    near = _rotate_at(rope, q, 10) @ _rotate_at(rope, k, 7)   # distance 3
    far = _rotate_at(rope, q, 50) @ _rotate_at(rope, k, 47)   # distance 3, shifted by 40
    assert torch.allclose(near, far, atol=1e-4)


def test_rope_preserves_length():
    # A rotation never stretches a vector: RoPE changes direction only.
    rope = RotaryEmbedding(head_size=32, block_size=64, theta=10000.0)
    x = torch.randn(2, 4, 64, 32)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(head_size=32, block_size=64, theta=10000.0)
    v = torch.randn(32)
    assert torch.allclose(_rotate_at(rope, v, 0), v)


def test_rope_buffers_not_in_checkpoint():
    # cos/sin are recomputed from the config, so they shouldn't bloat the state_dict.
    keys = make_model("rope").state_dict().keys()
    assert not any(k.endswith((".cos", ".sin")) for k in keys)
