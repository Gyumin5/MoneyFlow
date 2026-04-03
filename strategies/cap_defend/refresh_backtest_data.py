#!/usr/bin/env python3
"""Standard data refresh entrypoint for stock / spot / futures backtests."""

from __future__ import annotations

import argparse


def refresh_stock() -> None:
    from stock_engine import load_prices
    from recommend import (
        OFFENSIVE_STOCK_UNIVERSE,
        DEFENSIVE_STOCK_UNIVERSE,
        CANARY_ASSETS,
        STOCK_CRASH_TICKER,
    )

    needed = sorted(
        set(OFFENSIVE_STOCK_UNIVERSE)
        | set(DEFENSIVE_STOCK_UNIVERSE)
        | set(CANARY_ASSETS)
        | {STOCK_CRASH_TICKER}
    )
    print(f"[stock] refresh {len(needed)} tickers")
    prices = load_prices(needed, start="2014-01-01")
    print(f"[stock] loaded {len(prices)} series")


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
