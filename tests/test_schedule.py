import pytest

from zlm.schedule import lr_at

# Same shape as configs/tiny.yaml.
S = dict(lr=1e-3, min_lr=1e-4, warmup_steps=100, max_steps=2000)


def test_warmup_starts_small_but_not_zero():
    assert lr_at(0, **S) == pytest.approx(1e-3 / 100)


def test_warmup_is_linear_and_reaches_peak():
    assert lr_at(49, **S) == pytest.approx(0.5e-3)
    assert lr_at(99, **S) == pytest.approx(1e-3)  # last warmup step


def test_decay_starts_at_peak():
    assert lr_at(100, **S) == pytest.approx(1e-3)


def test_halfway_through_decay_is_the_midpoint():
    # cos(pi/2) = 0, so the coefficient is exactly 0.5.
    assert lr_at(1050, **S) == pytest.approx((1e-3 + 1e-4) / 2)


def test_ends_at_min_lr_and_stays_there():
    assert lr_at(2000, **S) == pytest.approx(1e-4)
    assert lr_at(5000, **S) == pytest.approx(1e-4)


def test_never_increases_after_warmup():
    lrs = [lr_at(s, **S) for s in range(100, 2100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert all(1e-4 <= x <= 1e-3 for x in lrs)


def test_zero_warmup():
    assert lr_at(0, **{**S, "warmup_steps": 0}) == pytest.approx(1e-3)
