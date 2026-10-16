"""Load the analysis mart and build gap-safe h-day return panels per regime."""
import math

import duckdb
import numpy as np
import pandas as pd

from . import config
from .regimes import REGIMES

FLOAT_ALL = "float_all"   # float_1 + shock_2020 + float_2 + shock_2022 + float_3
FLOAT_CALM = "float_calm"  # float_1 + float_2 + float_3 (both shocks removed)
ALL = "all"                # managed + float_all (transition day excluded)


def regime_members(name: str) -> list[str]:
    float_parts = [r.name for r in REGIMES if r.group == "float"]
    calm = [n for n in float_parts if not n.startswith("shock")]
    return {FLOAT_ALL: float_parts, FLOAT_CALM: calm,
            ALL: ["managed"] + float_parts}.get(name, [name])


def load_mart(db_path=config.DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM mart_daily ORDER BY market_date").df()
    con.close()
    df["market_date"] = pd.to_datetime(df["market_date"])
    return df.set_index("market_date")


def max_span_days(h: int) -> int:
    """Largest calendar span between the two ends of an h-trading-day return that
    we still treat as an h-day return (weekends/holidays allowed, data gaps not)."""
    return 7 * math.ceil(h / 5) + 5


def segment_returns(seg: pd.DataFrame, h: int) -> pd.DataFrame:
    """h-step log returns inside ONE contiguous regime segment (never across a
    boundary). Returns whose end points are further apart than `max_span_days(h)`
    (data gaps) or lack a Brent print are dropped."""
    lp = np.log(seg[["usd_kzt", "rub_usd", "rub_kzt", "brent_usd"]].astype(float))
    ret = lp - lp.shift(h)
    out = pd.DataFrame({
        "r_kzt": ret["usd_kzt"], "r_rub": ret["rub_usd"],
        "r_rubkzt": ret["rub_kzt"], "r_brent": ret["brent_usd"],
    })
    span = pd.Series(seg.index, index=seg.index).diff(h).dt.days
    out["span_days"] = span
    out["brent_stale"] = (seg["brent_staleness_days"].fillna(0) > 0) | \
                         (seg["brent_staleness_days"].shift(h).fillna(0) > 0)
    out["regime"] = seg["regime"]
    out = out[(span <= max_span_days(h))]
    return out


def returns_panel(mart: pd.DataFrame, regime: str, h: int,
                  require_brent: bool = True) -> pd.DataFrame:
    parts = []
    for name in regime_members(regime):
        seg = mart[mart["regime"] == name]
        if len(seg) > h:
            parts.append(segment_returns(seg, h))
    if not parts:
        return pd.DataFrame(columns=["r_kzt", "r_rub", "r_rubkzt", "r_brent"])
    df = pd.concat(parts)
    cols = ["r_kzt", "r_rub"] + (["r_brent"] if require_brent else [])
    return df.dropna(subset=cols)
