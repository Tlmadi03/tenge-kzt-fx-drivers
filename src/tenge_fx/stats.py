"""Return construction and dependence statistics robust to autocorrelation.

Daily FX and oil returns are autocorrelated and multi-day returns computed on
overlapping windows are mechanically autocorrelated, so naive Pearson p-values
are too optimistic. Significance is therefore assessed with block permutation
(one series is shuffled in contiguous blocks, preserving its internal serial
dependence while breaking its alignment with the other series) and confidence
intervals with a moving-block bootstrap.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def log_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """h-step log returns on consecutive observations of `prices` (overlapping)."""
    lp = np.log(prices.astype(float))
    return (lp - lp.shift(horizon)).dropna()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float((xc * yc).sum() / den) if den > 0 else np.nan


def block_permute(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle contiguous blocks of `x`; the incomplete tail stays in place."""
    n = len(x)
    k = n // block
    if k < 2:
        return rng.permutation(x)
    head = x[: k * block].reshape(k, block)
    return np.concatenate([head[rng.permutation(k)].ravel(), x[k * block:]])


def default_block(n: int, horizon: int) -> int:
    """Block length: at least twice the overlap of the h-day window, and at
    least 10 days, but small enough to leave >= 8 blocks."""
    b = max(10, 2 * horizon)
    return int(max(1, min(b, n // 8)))


@dataclass
class CorrResult:
    n: int
    r: float
    rho: float
    p_perm: float
    ci_low: float
    ci_high: float
    block: int
    n_blocks: int
    p_naive: float


def corr_test(x, y, horizon: int = 1, block: int | None = None,
              n_perm: int = 5000, n_boot: int = 2000, seed: int = 42) -> CorrResult:
    """Pearson r with block-permutation p-value and block-bootstrap 95% CI.

    The permutation null shuffles blocks of `x`; two-sided p uses (1 + #extreme) / (1 + B).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 20:
        return CorrResult(n, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan)
    b = block or default_block(n, horizon)
    rng = np.random.default_rng(seed)
    r = pearson(x, y)
    rho = float(sps.spearmanr(x, y)[0])
    xs = np.empty((n_perm, n))
    for i in range(n_perm):
        xs[i] = block_permute(x, b, rng)
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    rp = (xc @ yc) / np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum())
    p = (1 + np.sum(np.abs(rp) >= abs(r) - 1e-12)) / (1 + n_perm)

    # moving-block bootstrap of (x, y) pairs
    k = int(np.ceil(n / b))
    starts_max = n - b
    boots = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, starts_max + 1, size=k)
        idx = (st[:, None] + np.arange(b)[None, :]).ravel()[:n]
        boots[i] = pearson(x[idx], y[idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    p_naive = float(sps.pearsonr(x, y)[1])
    return CorrResult(n, r, rho, float(p), float(lo), float(hi), b, n // b, p_naive)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()
