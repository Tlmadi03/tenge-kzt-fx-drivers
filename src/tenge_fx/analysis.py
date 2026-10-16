"""Main results: KZT/USD log-return correlations with RUB/USD and Brent at 1/5/20-day
horizons, by FX regime, with block-permutation p-values and block-bootstrap CIs."""
import pandas as pd

from . import config, stats
from .data import ALL, FLOAT_ALL, load_mart, returns_panel
from .regimes import ANALYSIS_REGIMES

REGIME_ORDER = list(ANALYSIS_REGIMES) + [FLOAT_ALL, ALL]
PAIRS = {"rub": "RUB/USD", "brent": "Brent"}
MIN_BLOCKS = 8


def corr_row(panel: pd.DataFrame, x: str, h: int, regime: str, n_perm=config.N_PERM,
             n_boot=config.N_BOOT, seed=config.SEED) -> dict:
    res = stats.corr_test(panel[x].values, panel["r_kzt"].values, horizon=h,
                          n_perm=n_perm, n_boot=n_boot, seed=seed)
    d = panel.index
    return dict(
        regime=regime, driver=PAIRS[x.replace("r_", "")], horizon_days=h, n=res.n,
        start=d.min().date() if len(d) else None, end=d.max().date() if len(d) else None,
        pearson_r=res.r, spearman_rho=res.rho, p_perm=res.p_perm,
        ci95_low=res.ci_low, ci95_high=res.ci_high,
        block_len=res.block, n_blocks=res.n_blocks, p_naive=res.p_naive,
        low_power=bool(res.n_blocks < MIN_BLOCKS or res.n < 60),
    )


def main_table(mart: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIME_ORDER:
        for h in config.HORIZONS:
            panel = returns_panel(mart, regime, h)
            for x in ("r_rub", "r_brent"):
                rows.append(corr_row(panel, x, h, regime))
    df = pd.DataFrame(rows)
    ok = df["p_perm"].notna()
    df.loc[ok, "p_holm"] = stats.holm(df.loc[ok, "p_perm"].tolist())
    return df


def main():
    mart = load_mart()
    df = main_table(mart)
    config.REPORTS.joinpath("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(config.REPORTS / "tables" / "corr_main.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).drop(columns=["start", "end", "p_naive"]).to_string(index=False))


if __name__ == "__main__":
    main()
