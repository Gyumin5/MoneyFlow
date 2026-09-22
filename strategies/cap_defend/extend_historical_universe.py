#!/usr/bin/env python3
"""Extend data/historical_universe.json with the months it is missing.

The file holds, per month start, the coin tickers ranked by market cap at that
moment. The backtests read it so that a coin is only ever a candidate on dates
when it actually was in the top ranks — that is what keeps survivorship bias out.

There is no free endpoint that returns a historical top-N list, so this
reconstructs one: take a candidate pool (today's top-N by market cap, plus every
coin already listed in the most recent stored month), pull each coin's daily
market-cap series, and rank the pool at each month start.

Known limitation, stated rather than hidden: a coin that was highly ranked in the
new months but has since fallen out of the candidate pool cannot appear. The run
prints how many of the last stored month's leaders are still in the pool so the
size of that blind spot is measurable.

Usage:  python3 extend_historical_universe.py [--pool 250] [--depth 180] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIVERSE_FILE = os.path.join(ROOT, "data", "historical_universe.json")
API = "https://api.coingecko.com/api/v3"


def _get(sess: requests.Session, path: str, params: dict, tries: int = 8):
    delay = 6.0
    for attempt in range(tries):
        r = sess.get(f"{API}{path}", params=params, timeout=45)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 1.6, 90.0)
            continue
        return None
    return None


def month_starts(after: str, upto: datetime) -> list[str]:
    cur = pd.Timestamp(after) + pd.offsets.MonthBegin(1)
    out = []
    while cur <= pd.Timestamp(upto.date()).replace(day=1):
        out.append(cur.strftime("%Y-%m-%d"))
        cur += pd.offsets.MonthBegin(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=250, help="candidate pool size")
    ap.add_argument("--depth", type=int, default=180, help="tickers stored per month")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(UNIVERSE_FILE) as f:
        universe: dict[str, list[str]] = json.load(f)
    last_month = sorted(universe)[-1]
    missing = month_starts(last_month, datetime.now())
    print(f"stored through {last_month}; missing {len(missing)} months: "
          f"{missing[0] if missing else '-'} .. {missing[-1] if missing else '-'}")
    if not missing:
        return

    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

    pool: dict[str, str] = {}  # coingecko id -> ticker
    for page in range(1, args.pool // 250 + 2):
        rows = _get(sess, "/coins/markets", {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": 250, "page": page, "sparkline": "false"})
        if not rows:
            break
        for row in rows:
            pool[row["id"]] = f"{row['symbol'].upper()}-USD"
        if len(pool) >= args.pool:
            break
        time.sleep(3)
    pool = dict(list(pool.items())[: args.pool])
    print(f"candidate pool: {len(pool)} coins")

    prev_leaders = universe[last_month][: args.depth]
    covered = sum(1 for t in prev_leaders if t in set(pool.values()))
    print(f"coverage of {last_month} top-{len(prev_leaders)}: {covered}/{len(prev_leaders)}")

    caps: dict[str, pd.Series] = {}
    failed: list[str] = []
    for i, (cid, ticker) in enumerate(pool.items(), 1):
        data = _get(sess, f"/coins/{cid}/market_chart",
                    {"vs_currency": "usd", "days": "365", "interval": "daily"})
        mc = (data or {}).get("market_caps") or []
        if not mc:
            failed.append(ticker)
        else:
            s = pd.Series({pd.to_datetime(ms, unit="ms").normalize(): cap
                           for ms, cap in mc if cap})
            caps[ticker] = s[~s.index.duplicated(keep="last")]
        if i % 25 == 0:
            print(f"  {i}/{len(pool)} fetched, {len(failed)} without data")
        time.sleep(2.5)

    print(f"market-cap series: {len(caps)} ok, {len(failed)} missing")
    if not caps:
        raise SystemExit("no market cap data fetched")

    frame = pd.DataFrame(caps).sort_index()
    added = 0
    for ms in missing:
        ts = pd.Timestamp(ms)
        window = frame[(frame.index >= ts - pd.Timedelta(days=3)) & (frame.index <= ts)]
        if window.empty:
            print(f"  {ms}: no data in window — skipped")
            continue
        row = window.ffill().iloc[-1].dropna().sort_values(ascending=False)
        ranked = list(row.index[: args.depth])
        if len(ranked) < 40:
            print(f"  {ms}: only {len(ranked)} ranked — skipped")
            continue
        universe[ms] = ranked
        added += 1
        print(f"  {ms}: {len(ranked)} tickers, top5={ranked[:5]}")

    if args.dry_run:
        print(f"\ndry-run: would add {added} months")
        return
    tmp = UNIVERSE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: universe[k] for k in sorted(universe)}, f)
    os.replace(tmp, UNIVERSE_FILE)
    print(f"\nwrote {UNIVERSE_FILE}: {len(universe)} months, +{added}")


if __name__ == "__main__":
    main()
