import numpy as np
import pandas as pd

from tenge_fx import data, regimes, robustness


def _mart(dates, regime="float_1"):
    n = len(dates)
    return pd.DataFrame({
        "usd_kzt": 400 * np.exp(np.linspace(0, 0.1, n)), "rub_usd": 70 * np.exp(np.linspace(0, 0.05, n)),
        "rub_kzt": 5.0 * np.ones(n), "brent_usd": 60 * np.exp(np.linspace(0, -0.2, n)),
        "brent_staleness_days": 0.0, "regime": regime,
    }, index=pd.DatetimeIndex(dates, name="market_date"))


def test_assign_regime_boundaries():
    d = pd.Series(pd.to_datetime(["2015-08-19", "2015-08-20", "2015-08-21", "2020-03-01",
                                  "2022-02-24", "2022-07-01"]))
    got = regimes.assign_regime(d).tolist()
    assert got == ["managed", "transition", "float_1", "shock_2020", "shock_2022", "float_3"]


def test_segment_returns_horizon_and_values():
    m = _mart(pd.bdate_range("2021-01-04", periods=30))
    r5 = data.segment_returns(m, 5)
    assert len(r5.dropna(subset=["r_kzt"])) == 25
    assert np.isclose(r5["r_kzt"].iloc[-1], np.log(m.usd_kzt.iloc[-1] / m.usd_kzt.iloc[-6]))


def test_segment_returns_drop_returns_over_data_gap():
    dates = list(pd.bdate_range("2023-07-03", periods=10)) + list(pd.bdate_range("2023-08-28", periods=10))
    r = data.segment_returns(_mart(dates), 1)
    assert pd.Timestamp("2023-08-28") not in r.index  # first day after the 8-week gap


def test_returns_panel_never_crosses_regime_boundary():
    a = _mart(pd.bdate_range("2019-12-02", "2020-02-28"), "float_1")
    b = _mart(pd.bdate_range("2020-03-02", "2020-06-30"), "shock_2020")
    mart = pd.concat([a, b])
    p = data.returns_panel(mart, data.FLOAT_ALL, 5)
    # the first 5 days of shock_2020 have no 5-day return inside that regime
    assert p.loc["2020-03-02":"2020-03-06"].empty
    assert len(p) == (len(a) - 5) + (len(b) - 5)


def test_drop_top_and_robust_sigma():
    rng = np.random.default_rng(0)
    p = pd.DataFrame({"r_rub": np.r_[rng.normal(0, 1, 995), [30, -30, 25, -25, 40]]})
    assert len(robustness.drop_top(p, "r_rub", 0.01)) <= 990 + 1
    assert robustness.drop_robust_sigma(p, "r_rub")["r_rub"].abs().max() < 5


def test_winsorize_clips_extremes():
    p = pd.DataFrame({"r_kzt": np.arange(100.0), "r_rub": np.arange(100.0), "r_brent": np.arange(100.0)})
    w = robustness.winsorize(p)
    assert w.r_kzt.max() < 99 and w.r_kzt.min() > 0


def test_concentration_shares_sum_logic():
    x = np.r_[np.zeros(98), [10.0, -10.0]]
    p = pd.DataFrame({"r_rub": x, "r_kzt": 0.5 * x})
    rows = robustness.concentration(p, "r_rub", fracs=(0.05,))
    assert rows[0]["share_of_cross_product"] > 0.99
