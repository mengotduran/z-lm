import json

import numpy as np
import pytest
import torch

from zlm.data import IGNORE_INDEX, TokenDataset, load_split

CPU = torch.device("cpu")


@pytest.fixture
def data_dir(tmp_path):
    # Tokens 0, 1, 2, ..., 999: every window is consecutive, so y must equal x + 1.
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"dtype": "uint16"}))
    return tmp_path


def test_batch_shapes_and_dtype(data_dir):
    x, y = load_split(data_dir, "train", block_size=16).get_batch(batch_size=4, device=CPU)
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == torch.int64


def test_y_is_x_shifted_by_one(data_dir):
    x, y = load_split(data_dir, "train", block_size=16).get_batch(batch_size=8, device=CPU)
    assert torch.equal(y, x + 1)


def test_windows_stay_in_bounds(data_dir):
    ds = load_split(data_dir, "train", block_size=990)  # only 10 valid start positions
    for _ in range(50):
        x, y = ds.get_batch(batch_size=8, device=CPU)
        assert x.min() >= 0 and y.max() <= 999


def test_same_seed_same_batch(data_dir):
    ds = load_split(data_dir, "train", block_size=16)
    torch.manual_seed(0)
    a, _ = ds.get_batch(4, CPU)
    torch.manual_seed(0)
    b, _ = ds.get_batch(4, CPU)
    assert torch.equal(a, b)


def test_file_too_small_rejected(data_dir):
    with pytest.raises(ValueError):
        TokenDataset(data_dir / "train.bin", "uint16", block_size=1000)


# --- loss masks (chat fine-tuning) ---------------------------------------------------------


@pytest.fixture
def masked_dir(tmp_path):
    # Tokens 0..999; the mask keeps only even-numbered tokens as targets.
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    (np.arange(1000) % 2 == 0).astype(np.uint8).tofile(tmp_path / "train_mask.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"dtype": "uint16"}))
    return tmp_path


def test_masked_targets_become_ignore_index(masked_dir):
    x, y = load_split(masked_dir, "train", block_size=16).get_batch(batch_size=8, device=CPU)
    expected = x + 1  # the unmasked target
    keep = expected % 2 == 0
    assert torch.equal(y[keep], expected[keep])
    assert (y[~keep] == IGNORE_INDEX).all()
    assert torch.equal(x, x.clamp(min=0))  # inputs are never masked


def test_ignored_targets_dont_count_in_the_loss():
    # The whole trick relies on cross_entropy's default ignore_index being -100.
    logits = torch.randn(4, 10)
    targets = torch.tensor([1, IGNORE_INDEX, 3, IGNORE_INDEX])
    full = torch.nn.functional.cross_entropy(logits, targets)
    kept = torch.nn.functional.cross_entropy(logits[[0, 2]], targets[[0, 2]])
    assert torch.allclose(full, kept)


def test_no_mask_file_means_no_masking(data_dir):
    x, y = load_split(data_dir, "train", block_size=16).get_batch(batch_size=4, device=CPU)
    assert (y >= 0).all()


def test_mask_length_must_match(tmp_path):
    np.arange(100, dtype=np.uint16).tofile(tmp_path / "train.bin")
    np.ones(99, dtype=np.uint8).tofile(tmp_path / "train_mask.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"dtype": "uint16"}))
    with pytest.raises(ValueError, match="entries"):
        load_split(tmp_path, "train", block_size=16)
