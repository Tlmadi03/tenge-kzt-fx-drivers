"""Time variation and timing of the links: rolling correlations, year-by-year
correlations, and lead/lag structure of 1-day returns."""
import pandas as pd

from . import config, stats
from .analysis import corr_row
from .data import ALL, FLOAT_ALL, load_mart, max_span_days, regime_members, returns_panel, segment_returns

TABLES = config.REPORTS / "tables"
WINDOW = 250
DRIVERS = {"r_rub": "RUB/USD", "r_brent": "Brent"}


def rolling_table(mart: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Rolling Pearson correlation (last `window` observations) over the chronological
    panel of all regime days; returns never cross a regime boundary."""
    out = []
    for h in (1, 5):
        p = returns_panel(mart, ALL, h).sort_index()
        for x, name in DRIVERS.items():
            r = p["r_kzt"].rolling(window).corr(p[x]).dropna()
            out.append(pd.DataFrame({"date": r.index, "horizon_days": h, "driver": name,
                                     "rolling_corr": r.values, "window_obs": window}))
    return pd.concat(out, ignore_index=True)


def by_year_table(mart: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in config.HORIZONS:
        p = returns_panel(mart, ALL, h)
        for y in sorted(set(p.index.year)):
            q = p[p.index.year == y]
            if len(q) < 60:
                continue
            for x in DRIVERS:
                row = corr_row(q, x, h, str(y), n_perm=1000, n_boot=500)
                row["year"] = y
                row["dominant_regime"] = q["regime"].mode().iloc[0]
                rows.append(row)
    cols = ["year", "dominant_regime", "driver", "horizon_days", "n", "start", "end",
            "pearson_r", "spearman_rho", "p_perm", "ci95_low", "ci95_high", "n_blocks", "low_power"]
    return pd.DataFrame(rows)[cols]


def lagged_panel(mart: pd.DataFrame, regime: str, k: int) -> pd.DataFrame:
    """1-day returns with the driver shifted by k trading days inside each regime segment.
    k > 0: the driver moved k days BEFORE the tenge; k < 0: the tenge moved first."""
    parts = []
    for name in regime_members(regime):
        seg = mart[mart["regime"] == name]
        if len(seg) < 30:
            continue
        r = segment_returns(seg, 1)
        full = pd.DataFrame(index=seg.index)
        full["r_kzt"], full["r_rub"], full["r_brent"] = r["r_kzt"], r["r_rub"], r["r_brent"]
        full["r_rub"] = full["r_rub"].shift(k)
        full["r_brent"] = full["r_brent"].shift(k)
        # drop pairs whose lagged end points straddle a data gap
        span = pd.Series(seg.index, index=seg.index).diff(abs(k) or 1).dt.days.abs()
        if k:
            full = full[span <= max_span_days(abs(k))]
        parts.append(full)
    return pd.concat(parts).dropna()


def lag_table(mart: pd.DataFrame, regimes=("managed", FLOAT_ALL, "float_3"), ks=range(-5, 6)) -> pd.DataFrame:
    rows = []
    for regime in regimes:
        for k in ks:
            p = lagged_panel(mart, regime, k)
            for x, name in DRIVERS.items():
                row = corr_row(p, x, 1, regime, n_perm=1000, n_boot=500)
                row["lead_days"] = k
                rows.append(row)
    cols = ["regime", "driver", "lead_days", "n", "start", "end", "pearson_r", "p_perm",
            "ci95_low", "ci95_high", "n_blocks"]
    return pd.DataFrame(rows)[cols]


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    mart = load_mart()
    rt = rolling_table(mart)
    rt.to_csv(TABLES / "rolling_corr.csv", index=False)
    by = by_year_table(mart)
    by.to_csv(TABLES / "corr_by_year.csv", index=False)
    lg = lag_table(mart)
    lg.to_csv(TABLES / "lag_structure.csv", index=False)
    pd.set_option("display.width", 250)
    summ = rt.groupby(["horizon_days", "driver"]).rolling_corr.describe().round(3)
    print(summ)
    print(by[by.horizon_days == 5].round(3).drop(columns=["start", "end"]).to_string(index=False))
    print(lg[lg.regime == FLOAT_ALL].round(3).drop(columns=["start", "end"]).to_string(index=False))


if __name__ == "__main__":
    main()
