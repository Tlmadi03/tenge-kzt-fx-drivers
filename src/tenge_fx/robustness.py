"""Robustness of the Brent and RUB links: outlier days, concentration of the covariance
in a few days, leave-one-year-out, rolling windows, lags, joint regressions."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config, stats
from .analysis import corr_row
from .data import FLOAT_ALL, FLOAT_CALM, load_mart, returns_panel

TABLES = config.REPORTS / "tables"
N_PERM_ROB = 2000
N_BOOT_ROB = 1000


# ------------------------------------------------------------ panel filters

def drop_top(panel: pd.DataFrame, col: str, frac: float) -> pd.DataFrame:
    """Drop the `frac` share of observations with the largest |col|."""
    thr = panel[col].abs().quantile(1 - frac)
    return panel[panel[col].abs() <= thr]


def drop_robust_sigma(panel: pd.DataFrame, col: str, k: float = 3.0) -> pd.DataFrame:
    """Drop observations further than k robust sigmas (MAD * 1.4826) from the median."""
    med = panel[col].median()
    mad = (panel[col] - med).abs().median() * 1.4826
    return panel[(panel[col] - med).abs() <= k * mad]


def winsorize(panel: pd.DataFrame, cols=("r_kzt", "r_rub", "r_brent"), q=0.01) -> pd.DataFrame:
    out = panel.copy()
    for c in cols:
        lo, hi = out[c].quantile([q, 1 - q])
        out[c] = out[c].clip(lo, hi)
    return out


VARIANTS = {
    "baseline": lambda p: p,
    "drop top 1% |RUB| days": lambda p: drop_top(p, "r_rub", 0.01),
    "drop top 5% |RUB| days": lambda p: drop_top(p, "r_rub", 0.05),
    "drop |RUB| > 3 robust sigma": lambda p: drop_robust_sigma(p, "r_rub"),
    "drop top 1% |KZT| days": lambda p: drop_top(p, "r_kzt", 0.01),
    "drop top 1% |Brent| days": lambda p: drop_top(p, "r_brent", 0.01),
    "winsorize all at 1/99%": winsorize,
    "drop days with stale Brent": lambda p: p[~p["brent_stale"]],
    "drop calendar year 2015": lambda p: p[p.index.year != 2015],
    "drop 2015-08..2016-02 post-float adjustment":
        lambda p: p[(p.index < "2015-08-01") | (p.index > "2016-02-29")],
}


def variants_table(mart: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in (FLOAT_ALL, "float_3"):
        for h in config.HORIZONS:
            base = returns_panel(mart, regime, h)
            for name, fn in VARIANTS.items():
                p = fn(base)
                for x in ("r_rub", "r_brent"):
                    row = corr_row(p, x, h, regime, n_perm=N_PERM_ROB, n_boot=N_BOOT_ROB)
                    row["variant"] = name
                    row["n_dropped"] = len(base) - len(p)
                    rows.append(row)
        # shocks removed altogether (regime-level variant)
    for h in config.HORIZONS:
        p = returns_panel(mart, FLOAT_CALM, h)
        for x in ("r_rub", "r_brent"):
            row = corr_row(p, x, h, FLOAT_CALM, n_perm=N_PERM_ROB, n_boot=N_BOOT_ROB)
            row["variant"] = "float without the 2020 and 2022 shock windows"
            row["n_dropped"] = len(returns_panel(mart, FLOAT_ALL, h)) - len(p)
            rows.append(row)
    cols = ["regime", "variant", "driver", "horizon_days", "n", "n_dropped", "start", "end",
            "pearson_r", "spearman_rho", "p_perm", "ci95_low", "ci95_high", "n_blocks"]
    return pd.DataFrame(rows)[cols]


# ------------------------------------------------------ covariance concentration

def concentration(panel: pd.DataFrame, x: str, fracs=(0.01, 0.05)) -> list[dict]:
    """Share of the cross-product sum  sum((x-mx)(y-my))  contributed by the days with the
    largest |x|, versus their share of observations."""
    xc = panel[x] - panel[x].mean()
    yc = panel["r_kzt"] - panel["r_kzt"].mean()
    prod = xc * yc
    total = prod.sum()
    rows = []
    for f in fracs:
        thr = panel[x].abs().quantile(1 - f)
        top = panel[x].abs() > thr
        rows.append(dict(top_frac=f, n_top=int(top.sum()), n=len(panel),
                         share_of_cross_product=float(prod[top].sum() / total)))
    return rows


def concentration_table(mart: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in (FLOAT_ALL, "float_1", "float_2", "float_3", "managed"):
        for h in (1, 5):
            p = returns_panel(mart, regime, h)
            for x in ("r_rub", "r_brent"):
                for r in concentration(p, x):
                    rows.append(dict(regime=regime, horizon_days=h,
                                     driver="RUB/USD" if x == "r_rub" else "Brent",
                                     start=p.index.min().date(), end=p.index.max().date(), **r))
    return pd.DataFrame(rows)


def leave_year_out(mart: pd.DataFrame, regime=FLOAT_ALL) -> pd.DataFrame:
    rows = []
    for h in config.HORIZONS:
        p = returns_panel(mart, regime, h)
        for x in ("r_rub", "r_brent"):
            full = p[x].corr(p["r_kzt"])
            for y in sorted(set(p.index.year)):
                q = p[p.index.year != y]
                rows.append(dict(regime=regime, horizon_days=h,
                                 driver="RUB/USD" if x == "r_rub" else "Brent",
                                 left_out_year=y, n=len(q), pearson_r=q[x].corr(q["r_kzt"]),
                                 full_sample_r=full))
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    mart = load_mart()
    v = variants_table(mart)
    v.to_csv(TABLES / "robust_variants.csv", index=False)
    c = concentration_table(mart)
    c.to_csv(TABLES / "robust_concentration.csv", index=False)
    l = leave_year_out(mart)
    l.to_csv(TABLES / "robust_leave_year_out.csv", index=False)
    pd.set_option("display.width", 250)
    print(v[v.regime == FLOAT_ALL].round(3).drop(columns=["start", "end"]).to_string(index=False))
    print(c.round(3).to_string(index=False))
    s = l.groupby(["regime", "horizon_days", "driver"]).pearson_r.agg(["min", "max"]).round(3)
    print(s)


if __name__ == "__main__":
    main()
