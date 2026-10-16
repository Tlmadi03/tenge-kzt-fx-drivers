import numpy as np
import pandas as pd
import pytest

from tenge_fx import stats


def test_log_returns_horizon():
    s = pd.Series([100.0, 110.0, 121.0, 133.1])
    r1 = stats.log_returns(s, 1)
    assert len(r1) == 3
    assert np.allclose(r1.values, np.log(1.1))
    r2 = stats.log_returns(s, 2)
    assert len(r2) == 2
    assert np.allclose(r2.values, 2 * np.log(1.1))


def test_block_permute_preserves_values_and_blocks():
    rng = np.random.default_rng(0)
    x = np.arange(23, dtype=float)
    out = stats.block_permute(x, 5, rng)
    assert sorted(out) == sorted(x)
    assert np.array_equal(out[20:], x[20:])  # tail stays in place
    # each 5-block is contiguous in the original ordering
    for i in range(0, 20, 5):
        blk = out[i:i + 5]
        assert np.array_equal(np.diff(blk), np.ones(4))


def test_corr_test_detects_real_dependence():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    y = 0.6 * x + rng.normal(size=400)
    res = stats.corr_test(x, y, n_perm=500, n_boot=300)
    assert res.r > 0.4
    assert res.p_perm < 0.01
    assert res.ci_low > 0 and res.ci_low < res.r < res.ci_high


def test_corr_test_null_not_rejected_for_independent_series():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    res = stats.corr_test(x, y, n_perm=500, n_boot=300)
    assert res.p_perm > 0.05
    assert res.ci_low < 0 < res.ci_high


def test_corr_test_is_reproducible():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=200), rng.normal(size=200)
    a = stats.corr_test(x, y, n_perm=200, n_boot=100, seed=7)
    b = stats.corr_test(x, y, n_perm=200, n_boot=100, seed=7)
    assert a == b


def test_autocorrelated_series_naive_p_is_too_small():
    # two independent random walks' increments smoothed -> strongly autocorrelated;
    # block permutation p should exceed the naive p on average
    rng = np.random.default_rng(4)
    naive_smaller = 0
    for _ in range(20):
        a = np.convolve(rng.normal(size=520), np.ones(20) / 20, "valid")
        b = np.convolve(rng.normal(size=520), np.ones(20) / 20, "valid")
        res = stats.corr_test(a, b, horizon=20, n_perm=200, n_boot=50)
        naive_smaller += res.p_naive < res.p_perm
    assert naive_smaller >= 14


def test_short_sample_returns_nan():
    res = stats.corr_test(np.arange(5.0), np.arange(5.0))
    assert np.isnan(res.r)


def test_holm_monotone_and_bounded():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
