#!/usr/bin/env python3
"""Standard data refresh entrypoint for stock / spot / futures backtests."""

from __future__ import annotations

import argparse


def refresh_stock() -> None:
    import pandas as pd
    import yfinance as yf
    from pathlib import Path
    from recommend import (
        OFFENSIVE_STOCK_UNIVERSE,
        DEFENSIVE_STOCK_UNIVERSE,
        CANARY_ASSETS,
        STOCK_CRASH_TICKER,
    )
    from stock_engine import CACHE_DIR

    needed = sorted(
        set(OFFENSIVE_STOCK_UNIVERSE)
        | set(DEFENSIVE_STOCK_UNIVERSE)
        | set(CANARY_ASSETS)
        | {STOCK_CRASH_TICKER}
    )
    print(f"[stock] refresh {len(needed)} tickers")
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0
    for ticker in needed:
        if ticker == "VIX":
            skipped += 1
            continue

        fp = cache_dir / f"{ticker}.csv"
        existing = None
        # 항상 전체를 다시 받는다. 증분으로 이어 붙이면 배당 재조정이 과거 행에
        # 안 들어가 한 파일 안에 조정 기준 둘이 섞인다(history #148).
        start = "2014-01-01"
        if fp.exists():
            try:
                existing = pd.read_csv(fp, parse_dates=["Date"])
            except Exception:
                existing = None

        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                skipped += 1
                continue
            if isinstance(df.columns, pd.MultiIndex):
                series = df["Close"][ticker]
            else:
                series = df["Close"]
            series = series.dropna()
            new_df = series.rename(ticker).reset_index()
            new_df.columns = ["Date", ticker]

            # 전체 교체다. 다만 응답이 기존보다 크게 짧으면 자료를 날리지 않는다 —
            # 야후가 부분 응답을 줄 때 백테스트 자료가 통째로 사라지는 것을 막는다.
            if existing is not None and not existing.empty:
                if len(new_df) < len(existing) * 0.9:
                    print(f"[stock] {ticker} 응답이 짧아 교체 안 함 "
                          f"(새 {len(new_df)}행 < 기존 {len(existing)}행)")
                    skipped += 1
                    continue

            new_df["Date"] = pd.to_datetime(new_df["Date"])
            new_df = new_df.drop_duplicates("Date", keep="last").sort_values("Date")
            new_df.to_csv(fp, index=False)
            updated += 1
        except Exception as exc:
            print(f"[stock] {ticker} failed: {exc}")
            skipped += 1

    print(f"[stock] updated={updated} skipped={skipped}")


def refresh_coin() -> None:
    from recommend import get_dynamic_coin_universe, download_required_data

    log = []
    universe, ids = get_dynamic_coin_universe(log)
    tickers = sorted(set(universe + ["BTC-USD", "ETH-USD"]))
    print(f"[coin] refresh {len(tickers)} tickers")
    download_required_data(tickers, log, ids)
    print("[coin] done")


def refresh_futures() -> None:
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent / "download_futures_data.py"
    print("[futures] incremental refresh")
    rc = subprocess.call([sys.executable, str(script)])
    if rc != 0:
        raise SystemExit(f"[futures] download_futures_data.py failed rc={rc}")
    print("[futures] done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh backtest datasets")
    parser.add_argument(
        "--target",
        choices=["stock", "coin", "futures", "all"],
        default="all",
        help="dataset group to refresh",
    )
    args = parser.parse_args()

    if args.target in ("stock", "all"):
        refresh_stock()
    if args.target in ("coin", "all"):
        refresh_coin()
    if args.target in ("futures", "all"):
        refresh_futures()

    print()
    print("Tip: verify freshness with")
    print("python3 strategies/cap_defend/check_data_freshness.py")


if __name__ == "__main__":
    main()
