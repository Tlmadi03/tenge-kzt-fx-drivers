"""Cleaning of NBK official rates and FRED Brent.

Every defect found is appended to an issue log (what, how many rows, what we did)
which is written to reports/data_quality_issues.csv and rendered in
reports/data_quality.md. Functions are pure so they can be unit tested.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class IssueLog:
    items: list = field(default_factory=list)

    def add(self, issue_id: str, source: str, description: str, rows: int,
            unit: str, action: str):
        self.items.append(dict(issue_id=issue_id, source=source, description=description,
                               rows_affected=int(rows), unit=unit, action=action))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.items)


# ----------------------------------------------------------------- NBK

def parse_nbk_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Types the raw feed. Non-numeric rates become NaN; they are counted later."""
    df = pd.DataFrame({
        "requested_date": pd.to_datetime(raw["requested_date"]),
        "doc_date": pd.to_datetime(raw["doc_date"], format="%d.%m.%Y", errors="coerce"),
        "code": raw["title"].astype(str).str.strip(),
        "rate": pd.to_numeric(raw["description"], errors="coerce"),
        "quant": pd.to_numeric(raw["quant"], errors="coerce"),
    })
    return df


def normalize_quant(df: pd.DataFrame) -> pd.DataFrame:
    """Rate per ONE unit of currency: the feed quotes some currencies per 10/100/1000..."""
    out = df.copy()
    out["rate_per_unit"] = out["rate"] / out["quant"]
    return out


def drop_invalid_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NaN, zero and negative rates (placeholders for 'no rate')."""
    return df[df["rate_per_unit"].notna() & (df["rate_per_unit"] > 0)]


def previous_business_day(d: pd.Series) -> pd.Series:
    """Strictly earlier Mon-Fri date for each date."""
    d = pd.to_datetime(d)
    step = np.select([d.dt.dayofweek == 0, d.dt.dayofweek == 6], [3, 2], default=1)
    return d - pd.to_timedelta(step, unit="D")


def to_market_dates(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Re-date rates from NBK publication date to the trading day they reflect.

    A rate dated T is computed from the previous business day's market, so it is
    assigned to previous_business_day(T). Saturday-, Sunday- and Monday-dated
    rows all map to Friday; the first-published (earliest-dated) one is kept.
    `wide` is indexed by NBK date with one column per currency.
    """
    w = wide.sort_index().copy()
    w["market_date"] = previous_business_day(w.index.to_series()).values
    w["nbk_date"] = w.index
    cols = [c for c in w.columns if c not in ("market_date", "nbk_date")]
    stats = {"rows_in": len(w)}
    dup = w.duplicated("market_date", keep=False)
    stats["rows_sharing_market_date"] = int(dup.sum())
    first = w.groupby("market_date")[cols].transform("first")
    conflicts = ((w[cols] != first).any(axis=1) & dup).sum()
    stats["rows_conflicting"] = int(conflicts)
    out = w.drop_duplicates("market_date", keep="first").set_index("market_date")
    stats["rows_out"] = len(out)
    return out, stats


def drop_carry_forward(wide: pd.DataFrame, cols) -> tuple[pd.DataFrame, int]:
    """Drop market dates where every currency equals the previous market date
    (holiday: NBK republishes the last rate). With three currencies, one of which
    (RUB) moves almost every day, a genuine all-unchanged day is practically impossible."""
    d = wide[list(cols)].diff().abs()
    same = d.notna().all(axis=1) & (d.sum(axis=1) == 0)  # first row has no predecessor: keep
    return wide[~same], int(same.sum())


def clean_nbk(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, IssueLog]:
    """Returns (wide trading-day table, long table, issue log)."""
    log = IssueLog()
    df = parse_nbk_raw(raw)
    n_days_req = df["requested_date"].nunique()
    log.add("N00", "NBK", "Raw rows downloaded (all currencies, one XML document per calendar day)",
            len(df), "rows", "none (baseline)")

    # empty days
    cal = pd.date_range(config.START_DATE, df["requested_date"].max())
    missing = cal.difference(pd.DatetimeIndex(df["requested_date"].unique()))
    if len(missing):
        runs = (pd.Series(missing).diff().dt.days != 1).cumsum()
        longest = pd.Series(missing).groupby(runs.values).agg(["min", "max", "count"]) \
            .sort_values("count").iloc[-1]
        detail = (f"; longest run {longest['min']:%Y-%m-%d}..{longest['max']:%Y-%m-%d} "
                  f"({int(longest['count'])} days)")
    else:
        detail = ""
    log.add("N01", "NBK", f"Calendar days for which the NBK feed returned no rates{detail}",
            len(missing), "calendar days",
            "left as gaps (no interpolation); returns spanning a long gap are invalidated")

    # unparsable / non-positive rates
    n_nan = int(df["rate"].isna().sum())
    log.add("N02", "NBK", "Non-numeric rate strings", n_nan, "rows", "dropped")
    df = normalize_quant(df)
    nonpos = df[df["rate"].notna() & (df["rate"] <= 0)]
    codes = ", ".join(f"{k} ({v})" for k, v in nonpos["code"].value_counts().items()) or "none"
    log.add("N03", "NBK", f"Zero/negative placeholder rates, by currency: {codes}",
            len(nonpos), "rows", "dropped (none in USD/EUR/RUB)")
    df = drop_invalid_rates(df)

    # quotation multiples
    q = df[df["quant"] != 1]
    qc = ", ".join(f"{k}" for k in sorted(q["code"].unique()))
    kept_q = int((df["code"].isin(config.KEEP_CODES) & (df["quant"] != 1)).sum())
    log.add("N04", "NBK",
            f"Rates quoted per 10/100/1000/10000 units instead of 1 (currencies: {qc}); "
            f"rows with quant!=1 among USD/EUR/RUB: {kept_q}",
            len(q), "rows", "divided by quant -> rate per one unit (applied to all currencies)")

    # doc date vs requested date
    mism = int((df["doc_date"] != df["requested_date"]).sum())
    log.add("N05", "NBK", "Rows whose document date differs from the requested date",
            mism, "rows", "none needed (document date always equals requested date)")

    # duplicates
    dups = int(df.duplicated(["requested_date", "code"]).sum())
    log.add("N06", "NBK", "Duplicate (date, currency) rows", dups, "rows", "dropped (kept first)")
    df = df.drop_duplicates(["requested_date", "code"])

    # keep used currencies
    df = df[df["code"].isin(config.KEEP_CODES)]
    wide = df.pivot(index="requested_date", columns="code", values="rate_per_unit")
    wide = wide[list(config.KEEP_CODES)]
    incomplete = int(wide.isna().any(axis=1).sum())
    log.add("N07", "NBK", "Days where USD, EUR or RUB is missing", incomplete, "days", "dropped")
    wide = wide.dropna()
    log.add("N08", "NBK", f"Days with a full USD/EUR/RUB triple available (of {n_days_req} "
            "calendar days with data)", len(wide), "days", "kept for further steps")

    # re-dating
    wk = wide.index.dayofweek
    log.add("N09", "NBK",
            "Rate dated T reflects the market of the previous business day (verified against "
            "ECB EUR/USD, see reports/data_quality.md); Sat/Sun/Mon-dated rows are copies of the "
            "same Friday market",
            int(((wk == 5) | (wk == 6) | (wk == 0)).sum()), "days (Sat+Sun+Mon dated)",
            "re-dated to previous business day (market_date); duplicates collapsed")
    mkt, st = to_market_dates(wide)
    log.add("N10", "NBK",
            "Weekend/Monday-dated rows that map to the same Friday but carry a DIFFERENT value "
            "than the first-published one",
            st["rows_conflicting"], "rows", "kept the earliest-dated (Saturday) value")
    wide_m = mkt[list(config.KEEP_CODES)]
    wide_m, n_cf = drop_carry_forward(wide_m, config.KEEP_CODES)
    log.add("N11", "NBK",
            "Weekday market dates where USD, EUR and RUB are all unchanged vs the previous "
            "trading day (public holidays: last rate republished)",
            n_cf, "market days", "dropped (non-trading days)")

    # post-clean diagnostics
    r = np.log(wide_m).diff().abs()
    big = int((r["USD"] > 0.05).sum())
    log.add("N12", "NBK", "Daily USD/KZT moves above 5% remaining after cleaning "
            "(real events: 2009-02 and 2014-02 devaluations, 2015-08 float); inspected, not removed",
            big, "market days", "kept (genuine policy jumps)")

    long = (wide_m.reset_index().melt(id_vars="market_date", var_name="code",
                                      value_name="rate_per_unit")
            .sort_values(["market_date", "code"]).reset_index(drop=True))
    nbk_dates = mkt.loc[wide_m.index, "nbk_date"]
    long = long.merge(nbk_dates.rename("nbk_date").reset_index(), on="market_date")
    return wide_m, long, log


# ----------------------------------------------------------------- Brent

def clean_brent(raw: pd.DataFrame) -> tuple[pd.DataFrame, IssueLog]:
    log = IssueLog()
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(raw.iloc[:, 0]),
        "brent_usd": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
    })
    log.add("B00", "FRED", "Raw rows (DCOILBRENTEU)", len(df), "rows", "none (baseline)")
    nan = int(df["brent_usd"].isna().sum())
    log.add("B01", "FRED", "Missing observations (FRED '.' markers: UK/US holidays)",
            nan, "rows", "dropped; the as-of join carries the last available price forward")
    df = df.dropna()
    nonpos = int((df["brent_usd"] <= 0).sum())
    log.add("B02", "FRED", "Zero/negative prices", nonpos, "rows", "dropped")
    df = df[df["brent_usd"] > 0]
    d = int(df.duplicated("obs_date").sum())
    log.add("B03", "FRED", "Duplicate dates", d, "rows", "dropped (kept first)")
    df = df.drop_duplicates("obs_date")
    wk = int((df["obs_date"].dt.dayofweek >= 5).sum())
    log.add("B04", "FRED", "Weekend observations", wk, "rows", "dropped")
    df = df[df["obs_date"].dt.dayofweek < 5].sort_values("obs_date").reset_index(drop=True)
    jumps = np.log(df["brent_usd"]).diff().abs()
    log.add("B05", "FRED", "Daily Brent moves above 15% (e.g. 2020-03-09 OPEC+ shock); inspected",
            int((jumps > 0.15).sum()), "rows", "kept (genuine)")
    return df, log


def report_markdown(issues: pd.DataFrame, extra: str = "") -> str:
    lines = ["# Data quality report", "",
             "Generated by `python -m tenge_fx.clean`. Every row below is a defect or a",
             "transformation found in the raw files, with the number of rows it touched.", "",
             "| ID | Source | Finding | Rows affected | Unit | Action |",
             "|---|---|---|---:|---|---|"]
    for r in issues.itertuples():
        lines.append(f"| {r.issue_id} | {r.source} | {r.description} | {r.rows_affected:,} "
                     f"| {r.unit} | {r.action} |")
    return "\n".join(lines) + "\n" + extra


def offset_section(raw: pd.DataFrame) -> str:
    """Markdown block with the date-offset check against ECB EUR/USD."""
    from .validation import offset_correlations, usable_ecb
    if not config.ECB_RAW_FILE.exists():
        return ""
    ecb, bad = usable_ecb(pd.read_csv(config.ECB_RAW_FILE))
    ecb = ecb[ecb.index >= config.START_DATE]
    parsed = parse_nbk_raw(raw)
    wide = parsed[parsed["code"].isin(["USD", "EUR"])].assign(
        r=lambda d: d["rate"] / d["quant"]).pivot(index="requested_date", columns="code", values="r")
    res = offset_correlations(wide.dropna(), ecb)
    lines = ["", "## Publication-date offset check (NBK vs ECB EUR/USD)", "",
             f"ECB reference rates, {ecb.index.min():%Y-%m-%d}..{ecb.index.max():%Y-%m-%d} "
             f"(the file reachable by the downloader stops in Feb 2010 and holds sparse/placeholder "
             f"rows after 2009-12; {bad} such rows rejected). Correlation of daily changes in log(EUR/KZT / USD/KZT) from NBK "
             "with log EUR/USD from ECB when the NBK date is shifted back by k calendar days "
             "to obtain the market date:", "",
             "| shift k (days) | matched days | corr of daily changes |", "|---:|---:|---:|"]
    for r in res.itertuples():
        lines.append(f"| {r.shift_days} | {r.n} | {r.corr:.3f} |")
    best = res.loc[res["corr"].idxmax()]
    lines += ["", f"Best alignment: shift = {int(best.shift_days)} day(s). NBK rates are therefore "
              "assigned to the previous business day (`market_date`). The ECB fix is a 14:15 CET "
              "snapshot while the exact NBK fixing time was not verified, so the correlation is well "
              "below 1 even at the right shift; the point is the clear contrast between shifts."]
    return "\n".join(lines) + "\n"


def main():
    raw = pd.read_csv(config.NBK_RAW_FILE, dtype=str)
    wide, long, log_n = clean_nbk(raw)
    braw = pd.read_csv(config.BRENT_RAW_FILE)
    brent, log_b = clean_brent(braw)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    wide.rename(columns=str.lower).rename_axis("market_date").to_csv(
        config.PROCESSED / "nbk_wide.csv")
    long.to_csv(config.PROCESSED / "nbk_long.csv", index=False)
    brent.to_csv(config.PROCESSED / "brent_clean.csv", index=False)
    issues = pd.concat([log_n.frame(), log_b.frame()], ignore_index=True)
    issues.to_csv(config.REPORTS / "data_quality_issues.csv", index=False)
    (config.REPORTS / "data_quality.md").write_text(report_markdown(issues, offset_section(raw)))
    print(issues.to_string())
    print(f"\ntrading days: {len(wide)}  {wide.index.min():%Y-%m-%d}..{wide.index.max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
