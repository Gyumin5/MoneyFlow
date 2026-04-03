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
        start = "2014-01-01"
        if fp.exists():
            try:
                existing = pd.read_csv(fp, parse_dates=["Date"])
                if not existing.empty:
                    last_date = pd.to_datetime(existing["Date"]).iloc[-1]
                    start = (last_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
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

            if existing is not None and not existing.empty:
                merged = pd.concat([existing, new_df], ignore_index=True)
                merged["Date"] = pd.to_datetime(merged["Date"])
                merged = merged.drop_duplicates("Date", keep="last").sort_values("Date")
            else:
                merged = new_df

            merged.to_csv(fp, index=False)
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
    import download_futures_data as fut_dl

    print("[futures] incremental refresh")
    fut_dl.main(full_refresh=False)
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
