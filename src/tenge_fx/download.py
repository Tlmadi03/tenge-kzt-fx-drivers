"""Download raw inputs: NBK daily official rates, FRED Brent, ECB reference rates.

NBK exposes one XML document per calendar date, so the history is fetched day by
day (threaded, with retries) and stored as a flat CSV exactly as published.
Nothing is cleaned here.
"""
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from . import config

ITEM_FIELDS = ("fullname", "title", "description", "quant", "index", "change")


def parse_nbk_xml(text: str) -> tuple[str | None, list[dict]]:
    """Parse one NBK daily document into (document date, list of item dicts)."""
    root = ET.fromstring(text.strip())
    doc_date = root.findtext("date")
    items = []
    for item in root.iter("item"):
        items.append({f: (item.findtext(f) or "").strip() for f in ITEM_FIELDS})
    return doc_date, items


def fetch_nbk_day(session: requests.Session, day: date, retries: int = 5) -> list[dict]:
    url = f"{config.NBK_URL}?fdate={day.strftime('%d.%m.%Y')}"
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            doc_date, items = parse_nbk_xml(r.text)
            return [
                {"requested_date": day.isoformat(), "doc_date": doc_date, **it}
                for it in items
            ]
        except Exception as e:  # network or parse error: back off and retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NBK fetch failed for {day}: {last_err}")


def download_nbk(start: str = config.START_DATE, end: date | None = None,
                 workers: int = 8, out=config.NBK_RAW_FILE) -> pd.DataFrame:
    end = end or date.today()
    days = []
    d = datetime.strptime(start, "%Y-%m-%d").date()
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    print(f"NBK: fetching {len(days)} calendar days {days[0]}..{days[-1]}")
    rows: list[dict] = []
    with requests.Session() as s, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(lambda x: fetch_nbk_day(s, x), days), 1):
            rows.extend(res)
            if i % 500 == 0:
                print(f"  {i}/{len(days)}", flush=True)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"NBK: wrote {len(df):,} rows -> {out}")
    return df


def download_brent(out=config.BRENT_RAW_FILE) -> pd.DataFrame:
    r = requests.get(config.FRED_BRENT_URL, timeout=60)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r.text)
    df = pd.read_csv(out)
    print(f"Brent: wrote {len(df):,} rows -> {out}")
    return df


def download_ecb(out=config.ECB_RAW_FILE) -> pd.DataFrame:
    r = requests.get(config.ECB_HIST_URL, timeout=60)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r.text)
    df = pd.read_csv(out)
    print(f"ECB: wrote {len(df):,} rows -> {out}")
    return df


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    what = set(argv) or {"nbk", "brent", "ecb"}
    if "brent" in what:
        download_brent()
    if "ecb" in what:
        download_ecb()
    if "nbk" in what:
        download_nbk()


if __name__ == "__main__":
    main()
