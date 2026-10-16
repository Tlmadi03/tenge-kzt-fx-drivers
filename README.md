# Tenge (KZT) exchange rate, 2005–2026: what actually moves it?

An end-to-end analysis of the official National Bank of Kazakhstan (NBK) USD/KZT rate against
the Russian ruble (RUB/USD) and Brent crude, across policy regimes. Python (pandas, NumPy,
SciPy, statsmodels, matplotlib), SQL (DuckDB), pytest. Every number in this README is read
from a file in `reports/` produced by `make run_all` (fixed seed 42).

## Research question

Does the tenge track oil, the ruble, both, or neither, and does that depend on the FX regime
(managed rate before 2015-08-20, free float after, and the 2020 and 2022 shock windows)?

## Key findings

All statistics are Pearson correlations of log returns of KZT per USD (positive = tenge weakens)
with log returns of the driver. `n` = number of overlapping returns, `p` = block-permutation
p-value, brackets = 95% block-bootstrap CI. Full table: `reports/tables/corr_main.csv`.

1. **Managed regime (2005-01-05 to 2015-08-19): no usable link at the daily horizon.**
   1-day, n = 2,605: RUB r = 0.02 (p = 0.24), Brent r = −0.00 (p = 0.92). Only the 20-day RUB
   link (r = 0.22 [0.08, 0.36], n = 2,348, p = 0.013) is nominally significant and it does not
   survive Holm correction across the 48 tests of the main table (adjusted p = 0.20).
   Brent is null at all three horizons.
2. **Free float (return end-dates 2015-08-24 to 2026-10-15): both links are present, and the
   ruble is stronger than oil.** RUB/USD: r = 0.29 / 0.37 / 0.42 at 1 / 5 / 20 days
   (n = 2,704 / 2,666 / 2,331). Brent: r = −0.10 / −0.30 / −0.27 at the same horizons.
   All six are significant after Holm correction (adjusted p ≈ 0.01, the permutation floor).
3. **The oil link does not disappear when the ruble is controlled for.** In a joint
   standardised regression with Newey-West errors (float, 5-day, n = 2,666) the Brent
   coefficient goes from −0.30 (t = −4.6) to −0.25 (t = −4.7) after adding RUB/USD. Oil explains
   part of the ruble's own move (corr(RUB/USD, Brent) = −0.14), but not the tenge-oil link.
4. **Oil acts with a one-day delay.** At the 1-day horizon, Brent leading the tenge by one day
   gives r = −0.18 [−0.22, −0.13] (n = 2,698) against −0.10 for the same day (n = 2,704). In
   the joint 1-day regression the lag-1 term is stronger than the contemporaneous one
   (standardised β −0.16, t = −6.3 vs −0.08, t = −2.9). Likely reason: Brent settles after
   Kazakh trading hours, but I did not verify the exact NBK fixing time. The same-day figures
   in findings 2 and 5 therefore understate the 1-day oil link.
5. **The links are not stable over time.** By sub-period (5-day returns): float 2015-08 to
   2020-02 (n = 1,098): RUB 0.41, Brent −0.31. Float 2020-07 to 2022-02 (n = 392): RUB 0.56,
   Brent −0.54 [−0.64, −0.47], the strongest oil link in the sample. Float after 2022-07
   (n = 1,022): RUB 0.12, Brent −0.18. The post-2022 RUB link is not Holm-significant at 5 days
   (adjusted p = 0.26) and neither link is significant at 20 days (n = 909; RUB p = 0.19,
   Brent p = 0.27). Year by year (5-day, ~240 obs per year, `reports/tables/corr_by_year.csv`)
   the Brent CI includes zero in 2018, 2022, 2023, 2024 and 2025; the RUB CI includes zero in
   2022, 2023, 2024 and 2026.
6. **The 2022 shock is a ruble story, not an oil story.** 2022-02-24 to 2022-06-30:
   RUB r = 0.51 / 0.60 (1 / 5 day, n = 83 / 79) while Brent is indistinguishable from zero
   (5-day r = +0.06, p = 0.64). In the 2020 window (n = 79 / 75) both links are strong
   (RUB 0.57 / 0.83, Brent −0.23 / −0.58) but these windows contain only ~8 independent
   blocks, so treat them as indicative.
7. **How much of the ruble link is a few big days?** In the float regime, the 28 days
   (1%) with the largest ruble moves account for 47% of the cross-product sum at 1 day
   (36% at 5 days). Dropping them lowers r from 0.285 to 0.209 (1-day, n = 2,676, CI
   [0.18, 0.34]); dropping the top 5% gives 0.214 (n = 2,568). The ruble link shrinks by about a
   quarter but stays significant. Dropping the volatile post-float year 2015 raises it
   (r = 0.38). The Brent link survives the same exclusions (5-day r between
   −0.25 and −0.33 across the baseline and all variants in
   `reports/tables/robust_variants.csv`; leave-one-year-out range −0.33 to −0.22).

**Bottom line.** After the float, tenge returns co-move with both the ruble and Brent, and the
oil link is not a by-product of the ruble link. The size of both links depends on the period:
they were strongest in 2016–2021 and are weak after mid-2022. Before the float neither is
reliable. These are associations in daily/weekly data, not causal estimates.

## Data

| Source | What | Rows | Notes |
|---|---|---|---|
| NBK (`nationalbank.kz/rss/get_rates.cfm?fdate=DD.MM.YYYY`) | Official daily rates, all currencies | 280,667 raw rows over 7,921 calendar days (2005-01-01 to 2026-10-16) | One XML document per calendar day, downloaded by `tenge_fx.download` |
| FRED `DCOILBRENTEU` | Brent spot, USD/bbl | 10,281 raw rows, 9,108 valid | EIA series via FRED |
| ECB `eurofxref-hist.csv` | EUR/USD, EUR/RUB | 2005-01..2009-12 usable | Used only for two validation checks, see below |

If a source cannot be downloaded in your environment, put the file in `data/raw/` under the
names in `src/tenge_fx/config.py` (`nbk_rates_raw.csv` with columns `requested_date, doc_date,
fullname, title, description, quant, index, change`; `brent_fred.csv`; `ecb_eurofxref_hist.csv`).

### Problems found and how they were handled

The full log with affected row counts is `reports/data_quality.md` (and `.csv`). The ones that matter:

| Problem | Rows affected | Handling |
|---|---:|---|
| Quotations per 10/100/1000/10000 units (AMD, BYN, HUF, IDR, IRR, JPY, KRW, MNT, TRL, UZS, VND) | 28,431 rows (none in USD/EUR/RUB) | Rate divided by `quant` |
| Zero placeholder rate (LTL on 2015-01-09) | 1 | Dropped; none in USD/EUR/RUB |
| Days with an empty NBK response (incl. 2023-07-28 to 2023-08-28, 31 days) | 38 days | Kept as gaps, no interpolation. Returns spanning a gap are invalidated |
| **Rate date is not the trading date.** A rate dated T reflects the previous business day's market; Sat/Sun/Mon-dated rows repeat Friday | 3,394 Sat+Sun+Mon-dated days | Re-dated to the previous business day (`market_date`). Check against ECB EUR/USD (2005-01..2009-12, n = 1,252 days): correlation of daily changes 0.81 with this shift, −0.03 without it |
| Weekend/Monday rows that differ from the Saturday row for the same Friday | 73 | Earliest-dated row kept |
| Holidays: all of USD, EUR, RUB unchanged vs the previous trading day | 332 market days | Dropped as non-trading days |
| Brent missing values (holidays) | 1,173 | Dropped; as-of join carries the last print forward (max 5 days, then missing) |
| Duplicates, non-numeric strings, weekend Brent prints, document-date mismatches | 0 | Checked, none found |
| ECB file as served is truncated (stops Feb 2010) with placeholder rows | 8 rows rejected | Only 2005–2009 used, only for validation |

Result: **5,328 trading days, 2004-12-31 to 2026-10-15** (NBK) with a Brent as-of match on all
but 7 days. Brent days with a stale (carried-forward) print are 106 of 2,704 in the
float sample; a robustness variant drops them.

## Method

**Cleaning and mart** (`src/tenge_fx/clean.py`, `sql/`). Cleaned CSVs are loaded into DuckDB
tables `rates`, `brent`, `regimes`. `mart_daily` is built in SQL with a DuckDB `ASOF JOIN`
(each trading day gets the latest Brent print on or before it, no look-ahead) and cross rates:
RUB/USD = USD/KZT ÷ RUB/KZT and EUR/USD = EUR/KZT ÷ USD/KZT, plus Brent in tenge. Analytical
queries in `sql/analytics/` write to `reports/tables/`: annualised volatility by year
(2015: 38%, 2009: 17%, 2022: 17%, 2012: 1.6%), the 20 largest daily moves, depreciation episodes
(first 20-day cumulative depreciation >10%: 2009-02-04, 2014-02-12, 2015-08-20, 2020-03-16,
2022-02-28) and a per-regime summary.

**Regimes** (`src/tenge_fx/regimes.py`). Fixed in advance from events, not tuned to results:

| Regime | Market dates | Why |
|---|---|---|
| managed | to 2015-08-19 | De facto managed rate. Contains the 2009-02-04 (+0.163 log return) and 2014-02-11/12 (+0.052, +0.118) step devaluations |
| transition | 2015-08-20 | Float announcement day, a one-off +0.304 log-return policy jump; excluded everywhere |
| float_1 | 2015-08-21 to 2020-02-29 | Free float before the 2020 shock |
| shock_2020 | 2020-03-01 to 2020-06-30 | OPEC+ breakdown (Brent log return −0.255 on 2020-03-09) and COVID collapse |
| float_2 | 2020-07-01 to 2022-02-23 | Float between shocks |
| shock_2022 | 2022-02-24 to 2022-06-30 | Invasion of Ukraine, sanctions, ruble collapse and rebound |
| float_3 | 2022-07-01 to end | Float after 2022 |

Pooled groups: `float_all` (all float and shock windows), `float_calm` (without the two shocks)
and `all` (managed + float_all). Returns are built inside each regime segment, so no return
crosses a boundary.

**Returns.** h-day log returns (h = 1, 5, 20 trading days), overlapping, from consecutive
trading days. A return is dropped if its end points are further apart than 7·⌈h/5⌉+5 calendar
days (data gaps). KZT/USD up = tenge weaker; RUB/USD up = ruble weaker.

**Inference** (`src/tenge_fx/stats.py`). Daily returns are autocorrelated and h-day windows
overlap, so naive p-values are too small (a unit test shows this on smoothed series).
- p-value: block permutation, one series shuffled in contiguous blocks (length max(10, 2h),
  capped to leave ≥ 8 blocks), 5,000 permutations, two-sided, (1+#extreme)/(1+B).
- CI: moving-block bootstrap of pairs, same block length, 2,000 resamples, percentile 95%.
- Spearman ρ is reported next to Pearson r. Holm correction across the 48 tests of the main
  table (`p_holm`). Rows with fewer than 8 independent blocks or n < 60 are flagged `low_power`.
- Robustness variants use 2,000 permutations / 1,000 resamples; year-by-year and lead/lag
  use 1,000 / 500.

**Robustness** (`robustness.py`, `dynamics.py`, `joint.py`): exclusion of large RUB/KZT/Brent days
and winsorising, stale-Brent days, 2015 and the post-float adjustment period, concentration of
the cross-product in top days, leave-one-year-out, 250-observation rolling correlations,
year-by-year correlations, lead/lag of ±5 days, standardised OLS with Newey-West errors
(Brent alone, RUB alone, both), and cross-rate coupling checks.

## Figures

| | |
|---|---|
| `reports/figures/01_usdkzt_regimes.png` | USD/KZT (log) with regimes, Brent below |
| `reports/figures/02_rolling_correlation.png` | 250-observation rolling correlation with RUB and Brent (1- and 5-day) |
| `reports/figures/03_correlation_by_regime_horizon.png` | Correlation bars by horizon and regime with CIs |
| `reports/figures/04_return_distributions.png` | Daily return distributions: managed, float, 2022 shock |
| `reports/figures/05_lead_lag.png` | Lead/lag structure of 1-day returns |
| `reports/figures/06_correlation_by_year.png` | Year-by-year correlations |

The rolling 5-day Brent correlation ranges from −0.78 to +0.27 (median −0.12, 5,004 windows) and
the RUB one from −0.16 to +0.79 (median 0.25), confirming the instability above.

## Limitations

- **The RUB/USD cross is built from NBK's own rates.** RUB/USD = USD/KZT ÷ RUB/KZT satisfies
  `r_RUB/USD = r_USD/KZT − r_RUB/KZT` exactly (max error 0), so the tenge and ruble returns share
  the USD/KZT leg by construction, and noise in the USD/KZT fixing enters both. I could not obtain
  an independent RUB/USD series for the float period (the Central Bank of Russia endpoint was
  unreachable and the ECB file stops in 2010). Against an ECB-derived EUR-RUB/EUR-USD cross, the NBK-implied
  cross matches in daily changes at r = 0.79 (n = 1,142, 2005-04-01 to 2009-12-09, std ratio 1.03),
  so the cross is a decent but noisy proxy. The size of the ruble link should be read with this
  in mind; the Brent link does not share this issue.
- **Timing.** NBK rates are an average over a window, Brent is a London close; the one-day oil lead
  (finding 4) is consistent with that but the exact fixing time was not verified. Date shifting
  was validated only on 2005–2009 ECB data.
- **Association, not causation.** Oil and the ruble are themselves related (r = −0.12 at 1 day and −0.14 at 5 days, float) and both respond to common shocks (global risk, sanctions). Intervention by NBK, tax and
  export-revenue conversions, and the policy rate are not modelled.
- **Short shock windows.** shock_2020 and shock_2022 have ~80 daily observations and about
  8 independent blocks; 20-day results there are flagged low power.
- **Multiple testing.** 48 main tests, plus many robustness variants that are descriptive and not
  corrected. Holm is applied to the main table only.
- **Overlapping returns** mean the 5- and 20-day effective sample is much smaller than `n`;
  permutation p-values account for this, plain `n` does not.
- **Brent values are used as served by FRED.** 21 daily moves above 15% were inspected and
  kept (2020-03/05 and 2026 include extreme days, e.g. log return −0.64 on 2020-04-21); excluding the top 1% of
  Brent days changes the 5-day Brent link only from −0.30 to −0.25.
- Only USD, EUR and RUB are used; EUR/KZT enters only the validation.
- Data as downloaded on 2026-10-16. NBK and FRED may revise or extend history.

## Reproduce

```bash
make setup        # venv + requirements
make data         # download NBK (≈8k small requests, ~10 min), Brent, ECB into data/raw/
make run_all      # clean -> DuckDB + SQL -> main analysis -> robustness -> figures
make test         # pytest (synthetic data only, no downloads)
```

`run_all` is deterministic (seed 42); re-running it leaves `reports/` byte-identical.
Stages individually: `make clean_data db analysis robustness plots`.

```
data/raw/            downloaded files (git-ignored)
data/processed/      cleaned CSVs and tenge.duckdb (git-ignored)
sql/                 schema, mart, analytics/*.sql
src/tenge_fx/        download, clean, validation, regimes, stats, data, analysis,
                     robustness, dynamics, joint, plots, build_db
tests/               cleaning, SQL (as-of join, cross rates), stats, panels, robustness
notebooks/           executed results walkthrough
reports/             data_quality.md, tables/*.csv, figures/*.png
```

## License

MIT, see `LICENSE`.
