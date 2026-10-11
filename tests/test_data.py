import json

import numpy as np
import pytest
import torch

from zlm.data import TokenDataset, load_split

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
