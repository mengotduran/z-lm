import random

import numpy as np
import torch

from zlm.checkpoint import load_checkpoint, load_model, save_checkpoint
from zlm.config import config_from_dict
from zlm.model import Z

# Small model with dropout on, so the torch RNG actually affects training.
CFG = config_from_dict({"model": dict(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                                      n_embd=32, dropout=0.1)})


def make() -> tuple[Z, torch.optim.Optimizer]:
    model = Z(CFG.model)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-2)


def train_step(model: Z, opt: torch.optim.Optimizer) -> float:
    # Batches come from torch's global RNG, like zlm/data.py does.
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def test_resume_is_bit_identical(tmp_path):
    # Run A: 3 steps, checkpoint, 3 more steps. Run B: load the checkpoint into fresh objects
    # (after scrambling every RNG), then the same 3 steps. B must reproduce A exactly.
    torch.manual_seed(0)
    model, opt = make()
    for _ in range(3):
        train_step(model, opt)
    save_checkpoint(tmp_path / "ckpt.pt", model, opt, step=3, cfg=CFG)
    losses_a = [train_step(model, opt) for _ in range(3)]
    draws_a = (random.random(), np.random.rand())

    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    model_b, opt_b = make()  # different initial weights, empty optimizer state
    ckpt = load_checkpoint(tmp_path / "ckpt.pt", model_b, opt_b)
    losses_b = [train_step(model_b, opt_b) for _ in range(3)]
    draws_b = (random.random(), np.random.rand())

    assert ckpt["step"] == 3
    assert losses_a == losses_b
    assert draws_a == draws_b
    for pa, pb in zip(model.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


def test_config_and_extra_round_trip(tmp_path):
    model, opt = make()
    save_checkpoint(tmp_path / "ckpt.pt", model, opt, step=7, cfg=CFG, extra={"best_val": 1.5})
    ckpt = load_checkpoint(tmp_path / "ckpt.pt", model, restore_rng=False)
    assert config_from_dict(ckpt["config"]) == CFG
    assert ckpt["extra"] == {"best_val": 1.5}


def test_load_model_rebuilds_from_checkpoint_alone(tmp_path):
    torch.manual_seed(0)
    model, opt = make()
    save_checkpoint(tmp_path / "ckpt.pt", model, opt, step=0, cfg=CFG)
    loaded, cfg = load_model(tmp_path / "ckpt.pt")
    assert cfg == CFG
    assert not loaded.training  # ready for sampling: dropout off
    x = torch.randint(0, 64, (1, 16))
    assert torch.equal(loaded(x)[0], model.eval()(x)[0])


def test_save_is_atomic_and_overwrites(tmp_path):
    model, opt = make()
    path = tmp_path / "sub" / "ckpt.pt"  # parent dir is created
    save_checkpoint(path, model, opt, step=1, cfg=CFG)
    save_checkpoint(path, model, opt, step=2, cfg=CFG)
    assert [p.name for p in path.parent.iterdir()] == ["ckpt.pt"]  # no .tmp left behind
    assert load_checkpoint(path, model, restore_rng=False)["step"] == 2


def test_rope_buffers_are_rebuilt_not_loaded(tmp_path):
    # The RoPE cos/sin tables are not in the checkpoint; the rebuilt model must still match.
    model, opt = make()
    save_checkpoint(tmp_path / "ckpt.pt", model, opt, step=0, cfg=CFG)
    loaded, _ = load_model(tmp_path / "ckpt.pt")
    original = dict(model.named_buffers())
    rebuilt = dict(loaded.named_buffers())
    assert original.keys() == rebuilt.keys() and len(original) > 0
    for name in original:
        assert torch.equal(original[name], rebuilt[name]), name
