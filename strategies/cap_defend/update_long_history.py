#!/usr/bin/env python3
"""Append-safe refresh for the long-history CSVs under repo-root data/.

Why this exists separately from refresh_backtest_data.py:
  recommend.download_required_data() writes only the last 1000 Binance klines and
  overwrites the file. That is correct for the live executor (it needs ~130 bars)
  but it would destroy the 2018-start history that the research scripts read from
  repo-root data/*.csv. This script fetches only the missing tail and appends.

Coins  -> Binance spot daily klines (Adj_Close = close), same source as live.
Equity -> Yahoo chart endpoint, adjusted close.

Scope: every ticker that appears in data/historical_universe.json (the
point-in-time backtest universe) plus the stock sleeve tickers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
UNIVERSE_FILE = os.path.join(DATA_DIR, "historical_universe.json")

EQUITY_TICKERS = [
    "SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ",
    "IEF", "BIL", "BNDX", "TIP", "AGG",
]

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{}"


def _session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
    return s


def _read_existing(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or list(df.columns[:2]) != ["Date", "Adj_Close"]:
            return None
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df.dropna().drop_duplicates("Date").sort_values("Date")
    except Exception:
        return None


def _merge_write(path: str, existing: pd.DataFrame | None, new: pd.DataFrame) -> int:
    today = datetime.now().date()
    new = new[new["Date"] < today]  # drop the still-forming bar
    if new.empty:
        return 0
    if existing is None:
        merged = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates("Date", keep="last").sort_values("Date")
    before = 0 if existing is None else len(existing)
    merged.to_csv(path, index=False)
    return len(merged) - before


def fetch_coin(sess: requests.Session, ticker: str, start_date) -> pd.DataFrame | None:
    symbol = ticker.replace("-USD", "") + "USDT"
    start_ms = int(datetime.combine(start_date, datetime.min.time())
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows: list[tuple] = []
    while True:
        r = sess.get(BINANCE_KLINES, timeout=30, params={
            "symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1000})
        if r.status_code != 200:
            return None
        ks = r.json()
        if not ks:
            break
        rows += [(pd.to_datetime(k[0], unit="ms").date(), float(k[4])) for k in ks]
        if len(ks) < 1000:
            break
        start_ms = ks[-1][0] + 86_400_000
        time.sleep(0.2)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["Date", "Adj_Close"]).drop_duplicates("Date")


def fetch_equity(sess: requests.Session, ticker: str, start_date) -> pd.DataFrame | None:
    p1 = int(datetime.combine(start_date, datetime.min.time())
             .replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.now(timezone.utc).timestamp())
    r = sess.get(YAHOO_CHART.format(ticker), timeout=30, params={
        "period1": p1, "period2": p2, "interval": "1d", "includeAdjustedClose": "true"})
    if r.status_code != 200:
        return None
    try:
        res = r.json()["chart"]["result"][0]
        df = pd.DataFrame({
            "Date": pd.to_datetime(res["timestamp"], unit="s").date,
            "Adj_Close": res["indicators"]["adjclose"][0]["adjclose"],
        })
    except Exception:
        return None
    return df.dropna().drop_duplicates("Date")


def universe_tickers() -> list[str]:
    with open(UNIVERSE_FILE) as f:
        months = json.load(f)
    seen: set[str] = set()
    for lst in months.values():
        seen.update(lst)
    return sorted(seen)


def main() -> None:
    argv = list(sys.argv[1:])
    first_start = datetime(2018, 1, 1).date()
    backfill = False
    if argv and argv[0].startswith("--start="):
        first_start = datetime.strptime(argv.pop(0).split("=", 1)[1], "%Y-%m-%d").date()
        backfill = True
    only = set(argv)
    sess = _session()
    coins = [t for t in universe_tickers() if t.endswith("-USD")]
    targets = [(t, "coin") for t in coins] + [(t, "equity") for t in EQUITY_TICKERS]
    if only:
        targets = [x for x in targets if x[0] in only]

    added = stale = failed = 0
    for ticker, kind in targets:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        existing = _read_existing(path)
        start = existing["Date"].iloc[-1] if existing is not None else first_start
        if backfill:
            # explicit backfill: refetch from the given point and merge with what is there
            start = first_start
        fetch = fetch_coin if kind == "coin" else fetch_equity
        new = fetch(sess, ticker, start)
        if new is None or new.empty:
            failed += 1
            print(f"  {ticker:<14} FAIL (no data from {start})")
            continue
        n = _merge_write(path, existing, new)
        added += n
        last = _read_existing(path)
        last_date = last["Date"].iloc[-1] if last is not None else "?"
        if n == 0:
            stale += 1
        print(f"  {ticker:<14} +{n:<5} last={last_date}")

    print(f"\ntargets={len(targets)} rows_added={added} already_current={stale} failed={failed}")


if __name__ == "__main__":
    main()
