"""Empirical check of the NBK publication-date offset.

The NBK rate dated T appears to be fixed from the market of the previous
business day. To test this without relying on documentation we compare the
EUR/USD cross implied by NBK (EUR/KZT divided by USD/KZT) with the ECB EUR/USD
reference rate, which is dated by its own trading day, and look at which date
shift makes day-to-day changes line up best.

The ECB history file reachable from the download step is truncated: it ends in
February 2010, with sparse coverage and placeholder rows after 2009-12 (see `usable_ecb`), so the test
covers 2005-01..2009-12 only.
"""
import numpy as np
import pandas as pd


# Rows after this date in the served file are sparse and partly placeholder values
# (e.g. USD = 1.0, 2.0, 5.0 with JPY = 2, 1, 6).
ECB_LAST_GOOD = "2009-12-09"


def usable_ecb(ecb: pd.DataFrame) -> tuple[pd.Series, int]:
    """Plausible ECB EUR/USD observations (log level, by date) and the number
    of rows rejected as placeholders (weekend dates or implausible USD values)."""
    df = ecb[["Date", "USD"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["USD"] = pd.to_numeric(df["USD"], errors="coerce")
    ok = (df["Date"].dt.dayofweek < 5) & df["USD"].between(0.8, 1.7)
    ok &= df["Date"] <= ECB_LAST_GOOD
    bad = int((~ok).sum())
    s = df[ok].set_index("Date")["USD"].sort_index()
    return np.log(s), bad


def offset_correlations(nbk_by_nbk_date: pd.DataFrame, ecb_log_eurusd: pd.Series,
                        offsets=(0, 1, 2)) -> pd.DataFrame:
    """Correlation of daily changes in NBK-implied vs ECB EUR/USD when the NBK date
    is shifted back by `offset` calendar days to get the market date."""
    implied = np.log(nbk_by_nbk_date["EUR"] / nbk_by_nbk_date["USD"])
    rows = []
    for off in offsets:
        s = implied.copy()
        s.index = s.index - pd.Timedelta(days=off)
        s = s[s.index.dayofweek < 5]
        s = s[~s.index.duplicated(keep="first")]
        j = pd.concat([s.rename("nbk"), ecb_log_eurusd.rename("ecb")], axis=1, sort=True).dropna()
        d = j.diff().dropna()
        rows.append(dict(shift_days=off, n=len(d), corr=float(d["nbk"].corr(d["ecb"]))))
    return pd.DataFrame(rows)


def ecb_rub_usd(ecb: pd.DataFrame) -> pd.Series:
    """Independent log RUB/USD (rubles per dollar) from ECB EURRUB / EURUSD, 2005-04..2009-12
    (the ECB file served here has no RUB before 2005-04 and is unusable after 2009-12)."""
    df = ecb[["Date", "USD", "RUB"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ("USD", "RUB"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["Date"] <= ECB_LAST_GOOD) & df["USD"].between(0.8, 1.7) & df["RUB"].between(20, 60)]
    s = df.set_index("Date").sort_index()
    return np.log(s["RUB"]) - np.log(s["USD"])


def cross_rate_agreement(mart_log_rub_usd: pd.Series, ecb_log_rub_usd: pd.Series) -> dict:
    """How well the NBK-implied RUB/USD tracks the independent ECB-derived one (daily changes)."""
    j = pd.concat([mart_log_rub_usd.rename("nbk"), ecb_log_rub_usd.rename("ecb")],
                  axis=1, sort=True).dropna()
    d = j.diff().dropna()
    return dict(n=len(d), start=j.index.min().date(), end=j.index.max().date(),
                corr=float(d["nbk"].corr(d["ecb"])),
                std_ratio=float(d["nbk"].std() / d["ecb"].std()))
