"""FX regime definitions.

Boundaries are fixed ex ante from policy/market events, not tuned to the results:

* managed      -- NBK held the tenge in a de-facto managed band; the two large
                  one-off step devaluations (2009-02, 2014-02) fall inside it.
* transition   -- 2015-08-20, the day NBK announced the free float. Excluded from
                  every regime (a one-day policy jump, not a market return).
* float_1      -- free float, before the 2020 oil/COVID shock.
* shock_2020   -- 2020-03-01..2020-06-30: OPEC+ breakdown (Brent -24% on 2020-03-09)
                  and the COVID demand collapse.
* float_2      -- float between the two shocks.
* shock_2022   -- 2022-02-24..2022-06-30: Russia's invasion of Ukraine, sanctions
                  and the ruble collapse/rebound.
* float_3      -- float after the 2022 shock until the end of the sample.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Regime:
    name: str
    start: str
    end: str
    group: str  # "managed" or "float" (float groups the float_* and shock_* regimes)
    rationale: str


REGIMES = (
    Regime("managed", "2000-01-01", "2015-08-19", "managed",
           "Managed exchange rate until NBK abandoned the corridor."),
    Regime("transition", "2015-08-20", "2015-08-20", "transition",
           "Day of the float announcement; one-off policy jump, excluded."),
    Regime("float_1", "2015-08-21", "2020-02-29", "float",
           "Free float, pre-2020 shock."),
    Regime("shock_2020", "2020-03-01", "2020-06-30", "float",
           "OPEC+ breakdown and COVID demand collapse."),
    Regime("float_2", "2020-07-01", "2022-02-23", "float",
           "Free float between the shocks."),
    Regime("shock_2022", "2022-02-24", "2022-06-30", "float",
           "Invasion of Ukraine, sanctions, ruble collapse and rebound."),
    Regime("float_3", "2022-07-01", "2099-12-31", "float",
           "Free float after the 2022 shock."),
)

REGIME_NAMES = tuple(r.name for r in REGIMES)
ANALYSIS_REGIMES = tuple(r.name for r in REGIMES if r.name != "transition")


def regimes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [(r.name, r.start, r.end, r.group, r.rationale) for r in REGIMES],
        columns=["regime", "start_date", "end_date", "regime_group", "rationale"],
    ).assign(
        start_date=lambda d: pd.to_datetime(d.start_date),
        end_date=lambda d: pd.to_datetime(d.end_date),
    )


def assign_regime(dates: pd.Series) -> pd.Series:
    """Map dates to regime names; dates outside all windows get NaN."""
    dates = pd.to_datetime(dates)
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for r in REGIMES:
        m = (dates >= r.start) & (dates <= r.end)
        out[m] = r.name
    return out


def regime_group(name: str) -> str:
    return {r.name: r.group for r in REGIMES}[name]
