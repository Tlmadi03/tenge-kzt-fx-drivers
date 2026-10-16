"""Joint view: does Brent survive once the ruble is controlled for, and how much of the
KZT-RUB link is an artefact of how the RUB/USD cross is built?"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config
from .data import FLOAT_ALL, FLOAT_CALM, load_mart, regime_members, returns_panel, segment_returns
from .validation import cross_rate_agreement, ecb_rub_usd

TABLES = config.REPORTS / "tables"
REGS = ("managed", "float_1", "float_2", "float_3", FLOAT_CALM, FLOAT_ALL)


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _fit(y, X, lags):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m


def regression_table(mart: pd.DataFrame) -> pd.DataFrame:
    """Standardised OLS of KZT/USD returns on RUB/USD and/or Brent with Newey-West errors."""
    rows = []
    for regime in REGS:
        for h in (1, 5):
            p = returns_panel(mart, regime, h)
            z = p[["r_kzt", "r_rub", "r_brent"]].apply(_z)
            lags = 5 if h == 1 else 10
            for label, cols in (("Brent only", ["r_brent"]), ("RUB only", ["r_rub"]),
                                ("RUB + Brent", ["r_rub", "r_brent"])):
                m = _fit(z["r_kzt"], z[cols], lags)
                for c in cols:
                    rows.append(dict(regime=regime, horizon_days=h, model=label,
                                     term="RUB/USD" if c == "r_rub" else "Brent", n=int(m.nobs),
                                     start=p.index.min().date(), end=p.index.max().date(),
                                     beta_std=m.params[c], t_hac=m.tvalues[c], p_hac=m.pvalues[c],
                                     r2=m.rsquared, hac_lags=lags))
    return pd.DataFrame(rows)


def brent_with_lead(mart: pd.DataFrame, regime: str = FLOAT_ALL) -> pd.DataFrame:
    """1-day returns: tenge on RUB, Brent(t) and Brent(t-1) (Brent leading by one day)."""
    parts = []
    for name in regime_members(regime):
        seg = mart[mart["regime"] == name]
        r = segment_returns(seg, 1)
        r["r_brent_l1"] = r["r_brent"].shift(1)
        parts.append(r)
    p = pd.concat(parts).dropna(subset=["r_kzt", "r_rub", "r_brent", "r_brent_l1"])
    z = p[["r_kzt", "r_rub", "r_brent", "r_brent_l1"]].apply(_z)
    m = _fit(z["r_kzt"], z[["r_rub", "r_brent", "r_brent_l1"]], 5)
    return pd.DataFrame([dict(regime=regime, term=t, n=int(m.nobs), beta_std=m.params[t],
                              t_hac=m.tvalues[t], p_hac=m.pvalues[t], r2=m.rsquared)
                         for t in ("r_rub", "r_brent", "r_brent_l1")])


def coupling_table(mart: pd.DataFrame) -> pd.DataFrame:
    """r_rub(USD) = r_usdkzt - r_rubkzt holds exactly, so the KZT/USD and RUB/USD returns share
    the USD/KZT leg by construction. Compare with the bilateral RUB/KZT return."""
    rows = []
    for regime in (FLOAT_ALL, "float_1", "float_2", "float_3", "managed"):
        for h in (1, 5):
            p = returns_panel(mart, regime, h)
            rows.append(dict(
                regime=regime, horizon_days=h, n=len(p),
                corr_kzt_rubusd=p["r_kzt"].corr(p["r_rub"]),
                corr_kzt_rubkzt=p["r_kzt"].corr(p["r_rubkzt"]),
                corr_rubusd_brent=p["r_rub"].corr(p["r_brent"]),
                corr_rubkzt_brent=p["r_rubkzt"].corr(p["r_brent"]),
                sd_kzt=p["r_kzt"].std(), sd_rubusd=p["r_rub"].std(), sd_rubkzt=p["r_rubkzt"].std(),
                identity_max_abs_err=float((p["r_rub"] - (p["r_kzt"] - p["r_rubkzt"])).abs().max())))
    return pd.DataFrame(rows)


def ecb_check(mart: pd.DataFrame) -> pd.DataFrame:
    if not config.ECB_RAW_FILE.exists():
        return pd.DataFrame()
    ecb = ecb_rub_usd(pd.read_csv(config.ECB_RAW_FILE))
    res = cross_rate_agreement(np.log(mart["rub_usd"]), ecb)
    return pd.DataFrame([res])


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    mart = load_mart()
    reg = regression_table(mart); reg.to_csv(TABLES / "regressions_hac.csv", index=False)
    lead = brent_with_lead(mart); lead.to_csv(TABLES / "regression_brent_lead.csv", index=False)
    cp = coupling_table(mart); cp.to_csv(TABLES / "rub_coupling.csv", index=False)
    ec = ecb_check(mart); ec.to_csv(TABLES / "rub_cross_vs_ecb.csv", index=False)
    pd.set_option("display.width", 250)
    print(reg[reg.horizon_days == 5].round(3).drop(columns=["start", "end", "hac_lags"]).to_string(index=False))
    print(lead.round(3)); print(cp.round(3).to_string(index=False)); print(ec)


if __name__ == "__main__":
    main()
