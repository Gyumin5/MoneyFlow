#!/usr/bin/env python3
"""Extend data/historical_universe.json with the months it is missing.

The file holds, per month start, the coin tickers ranked by market cap at that
moment. The backtests read it so that a coin is only ever a candidate on dates
when it actually was in the top ranks — that is what keeps survivorship bias out.

Source: CoinMarketCap's historical snapshot pages (coinmarketcap.com/historical/
YYYYMMDD/). Those are weekly point-in-time top-N rankings, which is exactly the
thing needed here — no reconstruction, no candidate pool, no survivorship hole.
The ranking table is not in __NEXT_DATA__; it sits in an escaped `initialState`
blob under listingHistorical.data, so it is pulled out by string slicing.

Snapshots land on Sundays, so for each missing month start the nearest snapshot
on or before that date (within a week) is used, and the snapshot date actually
taken is printed.

Usage:  python3 extend_historical_universe.py [--depth 180] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIVERSE_FILE = os.path.join(ROOT, "data", "historical_universe.json")
SNAPSHOT_URL = "https://coinmarketcap.com/historical/{}/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


def _extract_listing(html: str) -> list[dict]:
    """listingHistorical.data 를 꺼낸다.

    표는 __NEXT_DATA__ 본문이 아니라 그 안의 initialState 에 들어 있는데, 그 값이
    JSON 문자열로 한 번 더 감싸져 있다. 손으로 역이스케이프하면 반드시 틀리므로
    문자열 리터럴 자체를 JSON 디코더에 맡겨 푼다.
    """
    key = '"initialState":'
    i = html.find(key)
    if i < 0:
        return []
    try:
        inner, _ = json.JSONDecoder().raw_decode(html, i + len(key))
        state = json.loads(inner) if isinstance(inner, str) else inner
        data = (state.get("cryptocurrency", {})
                     .get("listingHistorical", {})
                     .get("data", []))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError, AttributeError):
        return []


def fetch_snapshot(sess: requests.Session, day) -> list[dict] | None:
    r = sess.get(SNAPSHOT_URL.format(day.strftime("%Y%m%d")), timeout=45)
    if r.status_code != 200:
        return None
    rows = _extract_listing(r.text)
    return rows or None


def month_starts(after: str, upto: datetime) -> list[str]:
    cur = pd.Timestamp(after) + pd.offsets.MonthBegin(1)
    out = []
    while cur <= pd.Timestamp(upto.date()).replace(day=1):
        out.append(cur.strftime("%Y-%m-%d"))
        cur += pd.offsets.MonthBegin(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
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
    sess.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})

    prev_leaders = set(universe[last_month][:40])
    added = 0
    for ms in missing:
        target = datetime.strptime(ms, "%Y-%m-%d").date()
        rows = used = None
        for back in range(0, 8):
            day = target - timedelta(days=back)
            rows = fetch_snapshot(sess, day)
            if rows:
                used = day
                break
            time.sleep(1.5)
        if not rows:
            print(f"  {ms}: 스냅샷 없음 — 건너뜀")
            continue
        rows = sorted(rows, key=lambda r: r.get("cmcRank", 10 ** 9))
        tickers, seen = [], set()
        for r in rows:
            sym = str(r.get("symbol", "")).upper()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            tickers.append(f"{sym}-USD")
            if len(tickers) >= args.depth:
                break
        keep = len(prev_leaders & set(tickers[:40]))
        print(f"  {ms}: 스냅샷 {used} · {len(tickers)}종 · 직전월 상위40 중 {keep}종 유지 · "
              f"top5={tickers[:5]}")
        universe[ms] = tickers
        prev_leaders = set(tickers[:40])
        added += 1
        time.sleep(1.5)

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
