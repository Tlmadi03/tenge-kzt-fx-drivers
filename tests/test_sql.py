import math
from pathlib import Path

import duckdb
import pandas as pd
import pytest

SQL = Path(__file__).resolve().parents[1] / "sql"


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    rates = pd.DataFrame([
        # market_date, code, rate
        ("2024-01-02", "USD", 450.0), ("2024-01-02", "EUR", 495.0), ("2024-01-02", "RUB", 5.0),
        ("2024-01-03", "USD", 454.5), ("2024-01-03", "EUR", 500.0), ("2024-01-03", "RUB", 5.05),
        ("2024-01-10", "USD", 460.0), ("2024-01-10", "EUR", 506.0), ("2024-01-10", "RUB", 5.1),
    ], columns=["market_date", "code", "rate_per_unit"])
    rates["market_date"] = pd.to_datetime(rates["market_date"]).dt.date
    rates["nbk_date"] = rates["market_date"]
    brent = pd.DataFrame({"obs_date": pd.to_datetime(["2024-01-02", "2024-01-04"]).date,
                          "brent_usd": [77.0, 79.0]})
    regimes = pd.DataFrame({"regime": ["a"], "start_date": pd.to_datetime(["2000-01-01"]),
                            "end_date": pd.to_datetime(["2099-01-01"]), "regime_group": ["float"],
                            "rationale": ["x"]})
    c.register("rates_df", rates)
    c.register("brent_df", brent)
    c.register("regimes_df", regimes)
    c.execute("CREATE TABLE rates AS SELECT * FROM rates_df")
    c.execute("CREATE TABLE brent AS SELECT * FROM brent_df")
    c.execute("CREATE TABLE regimes AS SELECT * FROM regimes_df")
    c.execute((SQL / "02_mart.sql").read_text())
    return c


def test_cross_rates(con):
    row = con.execute("SELECT rub_usd, eur_usd FROM mart_daily WHERE market_date = '2024-01-02'").fetchone()
    assert row[0] == pytest.approx(450.0 / 5.0)   # RUB per USD
    assert row[1] == pytest.approx(495.0 / 450.0)  # USD per EUR


def test_asof_join_takes_last_print_on_or_before(con):
    df = con.execute("SELECT market_date, brent_usd, brent_staleness_days FROM mart_daily ORDER BY 1").df()
    assert df.brent_usd.tolist()[0] == 77.0
    assert df.brent_usd.tolist()[1] == 77.0          # 01-03 -> 01-02 print, no look-ahead
    assert df.brent_staleness_days.tolist()[:2] == [0, 1]


def test_stale_brent_is_nulled_after_five_days(con):
    df = con.execute("SELECT market_date, brent_usd FROM mart_daily ORDER BY 1").df()
    # 2024-01-10 is 6 days after the last print (2024-01-04): treated as missing
    assert math.isnan(df.brent_usd.iloc[2]) or df.brent_usd.iloc[2] is None
