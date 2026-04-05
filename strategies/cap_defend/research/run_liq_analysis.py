#!/usr/bin/env python3
"""청산 이벤트 상세 분석."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ensemble import SingleAccountEngine, combine_targets, STRATEGIES, load_data
from backtest_futures_full import run
import pandas as pd, numpy as np

START = '2020-10-01'
END = '2026-03-28'

data = {}
for iv in ['4h', '1h']:
    data[iv] = load_data(iv)

traces = {}
for key in ['4h1', '4h2', '1h1']:
    spec = STRATEGIES[key]
    iv = spec['interval']
    bars, funding = data[iv]
    trace = []
    run(bars, funding, interval=iv, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **spec['config'])
    traces[key] = trace

bars_1h, funding_1h = data['1h']
btc_1h = bars_1h['BTC']
all_dates = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]

weights = {'4h1': 1/3, '4h2': 1/3, '1h1': 1/3}
combined_targets = combine_targets(traces, weights, all_dates)

for lev in [2.0, 5.0]:
    print(f"\n{'='*75}")
    print(f"  {lev}x 청산 분석")
    print(f"{'='*75}")

    capital = 10000.0
    holdings = {}
    entry_prices = {}
    margins = {}
    liq_events = []
    prev_target = {}
    maint_rate = 0.004

    norm_funding = {}
    for coin, fr in funding_1h.items():
        nfr = fr.copy()
        nfr.index = nfr.index.floor('h')
        norm_funding[coin] = nfr

    for date, target in combined_targets:
        # Liquidation check
        for coin in list(holdings.keys()):
            df = bars_1h.get(coin)
            if df is None: continue
            ci = df.index.get_indexer([date], method='ffill')[0]
            if ci < 1: continue
            low = float(df['Low'].iloc[ci])
            prev_close = float(df['Close'].iloc[ci-1])
            if low <= 0: continue
            pnl = holdings[coin] * (low - entry_prices[coin])
            eq = margins[coin] + pnl
            maint = holdings[coin] * low * maint_rate
            if eq <= maint:
                coin_1h = (low / prev_close - 1) * 100
                from_entry = (low / entry_prices[coin] - 1) * 100

                btc_ci = btc_1h.index.get_indexer([date], method='ffill')[0]
                btc_now = float(btc_1h['Close'].iloc[btc_ci])
                btc_24h = float(btc_1h['Close'].iloc[max(0, btc_ci-24)])
                btc_chg = (btc_now / btc_24h - 1) * 100

                is_cash = target.get('CASH', 0) > 0.9

                liq_events.append({
                    'date': str(date)[:16],
                    'coin': coin,
                    'from_entry': from_entry,
                    'coin_1h': coin_1h,
                    'btc_24h': btc_chg,
                    'canary_off': is_cash,
                })

                returned = max(eq - max(eq, 0) * 0.015, 0)
                capital += returned
                del holdings[coin]; del entry_prices[coin]; del margins[coin]

        # Funding (simplified)
        for coin in list(holdings.keys()):
            fr_series = norm_funding.get(coin)
            if fr_series is not None and date in fr_series.index:
                fr = float(fr_series.loc[date])
                if fr != 0 and not np.isnan(fr):
                    df = bars_1h.get(coin)
                    ci = df.index.get_indexer([date], method='ffill')[0]
                    cur = float(df['Close'].iloc[ci]) if ci >= 0 else 0
                    if cur > 0:
                        capital -= holdings[coin] * cur * fr
        capital = max(capital, 0)

        # Simple rebalance
        if target != prev_target and target:
            pv = capital
            for coin in holdings:
                df = bars_1h.get(coin)
                ci = df.index.get_indexer([date], method='ffill')[0]
                cur = float(df['Close'].iloc[ci])
                if cur > 0:
                    pv += margins[coin] + holdings[coin] * (cur - entry_prices[coin])
            if pv > 0:
                for coin in list(holdings.keys()):
                    df = bars_1h.get(coin)
                    ci = df.index.get_indexer([date], method='ffill')[0]
                    cur = float(df['Close'].iloc[ci])
                    pnl = holdings[coin] * (cur - entry_prices[coin])
                    capital += margins[coin] + pnl - holdings[coin] * cur * 0.0004
                holdings.clear(); entry_prices.clear(); margins.clear()
                for coin, w in target.items():
                    if coin == 'CASH' or w <= 0: continue
                    df = bars_1h.get(coin)
                    if df is None: continue
                    ci = df.index.get_indexer([date], method='ffill')[0]
                    cur = float(df['Close'].iloc[ci])
                    if cur <= 0: continue
                    mgn = pv * w * 0.95
                    notional = mgn * lev
                    qty = notional / cur
                    tx = notional * 0.0004
                    if capital >= mgn + tx:
                        capital -= mgn + tx
                        holdings[coin] = qty
                        entry_prices[coin] = cur
                        margins[coin] = mgn
        prev_target = target

    print(f"\n  청산 {len(liq_events)}회:")
    print(f"  {'날짜':<18s} {'코인':<6s} {'진입대비':>8s} {'1h하락':>8s} {'BTC24h':>8s} {'카나리':>6s}")
    print(f"  {'-'*60}")
    for e in liq_events:
        canary = "OFF" if e['canary_off'] else "ON"
        print(f"  {e['date']:<18s} {e['coin']:<6s} {e['from_entry']:>+7.1f}% {e['coin_1h']:>+7.1f}% {e['btc_24h']:>+7.1f}% {canary:>6s}")
