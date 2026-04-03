#!/usr/bin/env python3
"""Check stock / spot / futures data freshness with one command."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/gmoh/mon/251229")
SPOT_DIR = ROOT / "data"
STOCK_CACHE_DIR = ROOT / "strategies" / "cap_defend" / "data" / "stock_cache"
FUTURES_DIR = ROOT / "data" / "futures"
UNIVERSE_FILE = ROOT / "data" / "historical_universe.json"


SPOT_TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "AVAX-USD"]
STOCK_TICKERS = ["SPY", "QQQ", "EEM", "GLD", "PDBC", "VEA"]
FUTURES_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "TRXUSDT"]


def last_index_value(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    try:
        df = pd.read_csv(path)
        if df.empty:
            return "EMPTY"
        return str(df.iloc[-1, 0])
    except Exception as exc:
        return f"ERROR:{exc}"


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check_spot() -> None:
    print_section("현물 코인 데이터")
    for ticker in SPOT_TICKERS:
        path = SPOT_DIR / f"{ticker}.csv"
        print(f"{ticker:<12} {last_index_value(path)}")
    print(f"{'historical_universe.json':<24} {'OK' if UNIVERSE_FILE.exists() else 'MISSING'}")


def check_stock() -> None:
    print_section("주식 데이터")
    for ticker in STOCK_TICKERS:
        path = STOCK_CACHE_DIR / f"{ticker}.csv"
        print(f"{ticker:<12} {last_index_value(path)}")


def check_futures() -> None:
    print_section("선물 데이터")
    for symbol in FUTURES_SYMBOLS:
        one_hour = FUTURES_DIR / f"{symbol}_1h.csv"
        funding = FUTURES_DIR / f"{symbol}_funding.csv"
        print(f"{symbol:<12} 1h={last_index_value(one_hour)} | funding={last_index_value(funding)}")

    stale_4h = sorted(p.name for p in FUTURES_DIR.glob("*_4h.csv"))
    if stale_4h:
        print()
        print("참고: _4h.csv 파일이 남아 있어도 현재 백테스트 엔진은 1h에서 4h를 리샘플링한다.")
        print(f"_4h.csv count: {len(stale_4h)}")


def check_universe_meta() -> None:
    print_section("유니버스 히스토리")
    if not UNIVERSE_FILE.exists():
        print("historical_universe.json: MISSING")
        return
    data = json.loads(UNIVERSE_FILE.read_text())
    months = sorted(data.keys())
    print(f"months: {len(months)}")
    if months:
        print(f"first : {months[0]}")
        print(f"last  : {months[-1]}")


def main() -> None:
    check_spot()
    check_stock()
    check_futures()
    check_universe_meta()


if __name__ == "__main__":
    main()
