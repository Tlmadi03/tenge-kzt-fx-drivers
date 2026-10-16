import numpy as np
import pandas as pd

from tenge_fx import clean, validation


def _raw(rows):
    return pd.DataFrame(rows, columns=["requested_date", "doc_date", "title", "description", "quant"])


def test_parse_and_quant_normalisation():
    raw = _raw([("2020-01-06", "06.01.2020", "JPY", "420", "100"),
                ("2020-01-06", "06.01.2020", "USD", "380.5", "1")])
    df = clean.normalize_quant(clean.parse_nbk_raw(raw))
    jpy = df[df.code == "JPY"].iloc[0]
    assert jpy.rate_per_unit == 4.2
    assert df[df.code == "USD"].iloc[0].rate_per_unit == 380.5


def test_drop_invalid_rates_removes_zero_negative_nan():
    raw = _raw([("2020-01-06", "06.01.2020", "A", "0", "1"),
                ("2020-01-06", "06.01.2020", "B", "-1", "1"),
                ("2020-01-06", "06.01.2020", "C", "n/a", "1"),
                ("2020-01-06", "06.01.2020", "D", "5", "1")])
    df = clean.drop_invalid_rates(clean.normalize_quant(clean.parse_nbk_raw(raw)))
    assert df.code.tolist() == ["D"]


def test_previous_business_day():
    d = pd.Series(pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-09",
                                  "2024-03-10", "2024-03-11"]))  # Mon Tue Sat Sun Mon
    out = clean.previous_business_day(d).dt.strftime("%Y-%m-%d").tolist()
    assert out == ["2024-03-01", "2024-03-04", "2024-03-08", "2024-03-08", "2024-03-08"]


def test_to_market_dates_collapses_weekend_copies_and_counts_conflicts():
    idx = pd.to_datetime(["2024-03-08", "2024-03-09", "2024-03-10", "2024-03-11", "2024-03-12"])
    # Fri-dated; Sat/Sun/Mon copies (Sun differs: conflict); Tue-dated new value
    w = pd.DataFrame({"USD": [447.0, 447.3, 447.9, 447.3, 449.0]}, index=idx)
    out, st = clean.to_market_dates(w)
    assert out.index.strftime("%Y-%m-%d").tolist() == ["2024-03-07", "2024-03-08", "2024-03-11"]
    assert out.loc["2024-03-08", "USD"] == 447.3  # first of Sat/Sun/Mon
    assert st["rows_sharing_market_date"] == 3
    assert st["rows_conflicting"] == 1


def test_drop_carry_forward_requires_all_columns_unchanged():
    idx = pd.date_range("2024-01-01", periods=4)
    w = pd.DataFrame({"USD": [1.0, 1.0, 1.0, 2.0], "RUB": [5.0, 5.0, 5.1, 5.1]}, index=idx)
    out, n = clean.drop_carry_forward(w, ["USD", "RUB"])
    assert n == 1  # only day 2 has both unchanged
    assert len(out) == 3


def test_clean_nbk_end_to_end_small():
    rows = []
    # Tue..Sat dated rows with values, then Sun/Mon copies, one duplicate, one zero
    dates = pd.date_range("2024-03-05", "2024-03-12")
    vals = {"2024-03-05": 450, "2024-03-06": 451, "2024-03-07": 452, "2024-03-08": 453,
            "2024-03-09": 454, "2024-03-10": 454, "2024-03-11": 454, "2024-03-12": 455}
    for d in dates:
        v = vals[d.strftime("%Y-%m-%d")]
        for code, base in (("USD", v), ("EUR", v * 1.1), ("RUB", v / 90)):
            rows.append((d.strftime("%Y-%m-%d"), d.strftime("%d.%m.%Y"), code, str(base), "1"))
    rows.append(("2024-03-06", "06.03.2024", "USD", "451", "1"))        # duplicate
    rows.append(("2024-03-06", "06.03.2024", "XXX", "0", "1"))          # placeholder
    rows.append(("2024-03-06", "06.03.2024", "JPY", "300", "100"))      # quant
    wide, long, log = clean.clean_nbk(_raw(rows))
    frame = log.frame().set_index("issue_id")
    assert frame.loc["N06", "rows_affected"] == 1
    assert frame.loc["N03", "rows_affected"] == 1
    assert frame.loc["N04", "rows_affected"] == 1
    # market dates: Mon 03-04 .. Fri 03-08, then Mon 03-11
    assert wide.index.strftime("%Y-%m-%d").tolist() == [
        "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-11"]
    assert set(long.code) == {"USD", "EUR", "RUB"}
    assert (long.rate_per_unit > 0).all()


def test_clean_brent_drops_missing_and_weekend():
    raw = pd.DataFrame({"observation_date": ["2024-01-02", "2024-01-03", "2024-01-04",
                                              "2024-01-06", "2024-01-02"],
                        "DCOILBRENTEU": ["77.0", ".", "-1", "78.0", "77.0"]})
    df, log = clean.clean_brent(raw)
    assert df.obs_date.dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    f = log.frame().set_index("issue_id")
    assert f.loc["B01", "rows_affected"] == 1
    assert f.loc["B02", "rows_affected"] == 1


def test_ecb_offset_prefers_correct_shift():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2008-01-01", periods=300)
    lr = np.cumsum(rng.normal(0, 0.005, len(days)))
    ecb = pd.Series(lr, index=days)
    # NBK date = market date + 1 calendar day (weekday-only toy world, no weekends)
    nbk_idx = days + pd.Timedelta(days=1)
    usd = 400.0
    eur = usd * np.exp(lr)
    nbk = pd.DataFrame({"USD": usd, "EUR": eur}, index=nbk_idx)
    res = validation.offset_correlations(nbk, ecb, offsets=(0, 1))
    assert res.loc[res.shift_days == 1, "corr"].iloc[0] > 0.9
    assert abs(res.loc[res.shift_days == 0, "corr"].iloc[0]) < 0.5
