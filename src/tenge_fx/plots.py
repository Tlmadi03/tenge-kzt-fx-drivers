"""Figures written to reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import load_mart, returns_panel
from .regimes import REGIMES

FIG = config.FIGURES
T = config.REPORTS / "tables"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
C_RUB, C_BRENT, C_KZT = "#0072B2", "#D55E00", "#333333"   # colour-blind-safe pair + neutral
SHADE = {"managed": "#f2f2f2", "transition": "#999999", "float_1": "#ffffff",
         "shock_2020": "#fbe3d3", "float_2": "#ffffff", "shock_2022": "#fbe3d3", "float_3": "#ffffff"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})


def _shade(ax, end):
    for r in REGIMES:
        a, b = pd.Timestamp(r.start), min(pd.Timestamp(r.end), end)
        ax.axvspan(max(a, pd.Timestamp("2005-01-01")), b, color=SHADE[r.name], zorder=0, lw=0)
        if r.name == "transition":
            ax.axvline(a, color="#999999", lw=1, ls="--")


def fig_regimes(mart):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    end = mart.index.max()
    for ax in (a1, a2):
        _shade(ax, end)
    a1.plot(mart.index, mart["usd_kzt"], color=C_KZT, lw=1.2)
    a1.set_yscale("log"); a1.set_ylabel("KZT per USD (log scale)")
    a1.set_title("Official NBK USD/KZT with FX regimes (n = %d trading days, %s to %s)" %
                 (len(mart), mart.index.min().date(), end.date()), loc="left", color=INK)
    for name, x, y in (("managed", "2009-09-01", 0.9), ("float", "2018-01-01", 0.35),
                       ("2020\nshock", "2020-04-30", 0.9), ("2022\nshock", "2022-04-30", 0.55)):
        a1.text(pd.Timestamp(x), y, name, transform=a1.get_xaxis_transform(), ha="center",
                color=MUTED, fontsize=9)
    a2.plot(mart.index, mart["brent_usd"], color=C_BRENT, lw=0.9)
    a2.set_ylabel("Brent, USD/bbl")
    a2.set_xlim(pd.Timestamp("2005-01-01"), end)
    a2.text(0.01, 0.9, "Shaded: grey = managed rate, pink = 2020 and 2022 shock windows; "
            "dashed = 2015-08-20 float", transform=a2.transAxes, fontsize=8, color=MUTED)
    fig.savefig(FIG / "01_usdkzt_regimes.png"); plt.close(fig)


def fig_rolling():
    rt = pd.read_csv(T / "rolling_corr.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, h in zip(axes, (1, 5)):
        for drv, col in (("RUB/USD", C_RUB), ("Brent", C_BRENT)):
            d = rt[(rt.horizon_days == h) & (rt.driver == drv)]
            ax.plot(d.date, d.rolling_corr, color=col, lw=1.3, label=drv)
        ax.axhline(0, color=MUTED, lw=0.8)
        for r in REGIMES:
            if r.name == "transition":
                ax.axvline(pd.Timestamp(r.start), color="#999999", ls="--", lw=1)
        ax.set_title(f"{h}-day returns, 250-observation window", loc="left", fontsize=10)
    axes[0].set_ylabel("Pearson r with KZT/USD return")
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "02_rolling_correlation.png"); plt.close(fig)


def fig_bars():
    df = pd.read_csv(T / "corr_main.csv")
    order = ["managed", "float_1", "shock_2020", "float_2", "shock_2022", "float_3", "float_all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    w = 0.26
    hcol = {1: "#9ecae1", 5: "#4292c6", 20: "#08519c"}
    hcol_b = {1: "#fdd0a2", 5: "#fd8d3c", 20: "#a63603"}
    for ax, drv, pal in zip(axes, ("RUB/USD", "Brent"), (hcol, hcol_b)):
        for i, h in enumerate((1, 5, 20)):
            d = df[(df.driver == drv) & (df.horizon_days == h)].set_index("regime").reindex(order)
            x = np.arange(len(order)) + (i - 1) * w
            err = np.vstack([d.pearson_r - d.ci95_low, d.ci95_high - d.pearson_r])
            ax.bar(x, d.pearson_r, w * 0.92, color=pal[h], label=f"{h}-day")
            ax.errorbar(x, d.pearson_r, yerr=err.clip(min=0), fmt="none", ecolor=INK, lw=0.8, capsize=1.5)
            for xi, (_, row) in zip(x, d.iterrows()):
                if row.low_power:
                    ax.text(xi, 0.02 if row.pearson_r < 0 else -0.04, "*", ha="center", fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{o}\nn={int(df[(df.regime == o) & (df.driver == drv) & (df.horizon_days == 1)].n.iloc[0])}"
                            for o in order], fontsize=7.5)
        ax.set_title(f"KZT/USD return vs {drv} return", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left" if drv == "RUB/USD" else "lower left")
    axes[0].set_ylabel("Pearson r (95% block-bootstrap CI)")
    fig.text(0.01, -0.03, "n = 1-day sample size per regime. * = fewer than 8 independent blocks or n < 60 (low power). "
             "Regimes: see README.", fontsize=8, color=MUTED)
    fig.savefig(FIG / "03_correlation_by_regime_horizon.png"); plt.close(fig)


def fig_dists(mart):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    groups = [("managed", "Managed (to 2015-08-19)", C_KZT), ("float_calm", "Float, outside shocks", C_RUB),
              ("shock_2022", "2022 shock window", C_BRENT)]
    bins = np.linspace(-0.08, 0.08, 81)
    for ax, (reg, title, col) in zip(axes, groups):
        p = returns_panel(mart, reg, 1, require_brent=False)
        ax.hist(p["r_kzt"].clip(-0.08, 0.08), bins=bins, color=col, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{title}\nn={len(p)}, sd={p['r_kzt'].std()*100:.2f}%, max|r|={p['r_kzt'].abs().max()*100:.1f}%",
                     loc="left", fontsize=9)
        ax.set_xlabel("daily log return of KZT/USD (clipped at ±8%)")
    axes[0].set_ylabel("days (log scale)")
    fig.savefig(FIG / "04_return_distributions.png"); plt.close(fig)


def fig_lags():
    lg = pd.read_csv(T / "lag_structure.csv")
    d = lg[lg.regime == "float_all"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for drv, col in (("RUB/USD", C_RUB), ("Brent", C_BRENT)):
        x = d[d.driver == drv].sort_values("lead_days")
        ax.errorbar(x.lead_days, x.pearson_r, yerr=[x.pearson_r - x.ci95_low, x.ci95_high - x.pearson_r],
                    color=col, marker="o", ms=4, lw=1.2, capsize=2, label=drv)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("driver leads the tenge by k trading days (k<0: tenge moves first)")
    ax.set_ylabel("Pearson r, 1-day returns")
    ax.set_title(f"Lead/lag structure, float regimes (n≈{int(d.n.min())}–{int(d.n.max())})", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.savefig(FIG / "05_lead_lag.png"); plt.close(fig)


def fig_year():
    by = pd.read_csv(T / "corr_by_year.csv")
    d = by[by.horizon_days == 5]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for drv, col, off in (("RUB/USD", C_RUB, -0.15), ("Brent", C_BRENT, 0.15)):
        x = d[d.driver == drv]
        ax.errorbar(x.year + off, x.pearson_r, yerr=[x.pearson_r - x.ci95_low, x.ci95_high - x.pearson_r],
                    fmt="o", color=col, ms=4, lw=1, capsize=2, label=drv)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(2015.6, color="#999999", ls="--", lw=1)
    ax.set_ylabel("Pearson r, 5-day returns"); ax.legend(frameon=False)
    ax.set_title("Year-by-year correlation with KZT/USD (≈240 obs per year; 95% block-bootstrap CI)", loc="left", fontsize=10)
    fig.savefig(FIG / "06_correlation_by_year.png"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    mart = load_mart()
    fig_regimes(mart); fig_rolling(); fig_bars(); fig_dists(mart); fig_lags(); fig_year()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
