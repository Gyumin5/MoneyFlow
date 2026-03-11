#!/usr/bin/env python3
"""Show current holdings for V14 strategy (TR3+PFD5+DD Exit+BL)."""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, run_backtest, get_price

def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1, canary_band=1.0,
        health='HK', health_sma=0, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05, top_n=40,
        risk='G5',
        dd_exit_lookback=60, dd_exit_threshold=-0.25,
        bl_threshold=-0.15, bl_days=7,
        end_date='2026-03-10',
    )
    base.update(kw)
    return Params(**base)

def main():
    print("Loading data...")
    prices, universe = load_data(top_n=40)

    anchors = [1, 10, 19]

    # Run each tranche with state
    print("\n" + "=" * 70)
    print("  현재 보유 종목 (V14: TR3+PFD5+DD+BL)")
    print("=" * 70)

    combined = {}
    total_val = 0

    for i, d in enumerate(anchors):
        p = B(rebalancing=f'RX{d}', post_flip_delay=5, initial_capital=10000.0/3)
        r = run_backtest(prices, universe, p, return_state=True)
        last_date = r['last_date']
        holdings = r['final_holdings']
        cash = r['final_cash']

        print(f"\n  트랜치 {chr(65+i)} (앵커 {d}일):")
        canary_on = r['final_state']['prev_canary']
        print(f"    카나리아: {'ON (Risk-On)' if canary_on else 'OFF (현금)'}")
        print(f"    DD Exit: {r['dd_exit_count']}회")

        tranche_total = cash
        rows = []
        for t, units in holdings.items():
            price = get_price(t, prices, last_date)
            val = units * price
            tranche_total += val
            rows.append((t, units, price, val))
        rows.sort(key=lambda x: x[3], reverse=True)

        for t, units, price, val in rows:
            pct = val / tranche_total * 100 if tranche_total > 0 else 0
            print(f"    {t:<12} {units:>10.4f} x ${price:>10,.2f} = ${val:>10,.0f} ({pct:>5.1f}%)")
        if cash > 0.01:
            pct = cash / tranche_total * 100
            print(f"    {'CASH':<12} {'':>10} {'':>12}   ${cash:>10,.0f} ({pct:>5.1f}%)")
        print(f"    {'합계':<12} {'':>10} {'':>12}   ${tranche_total:>10,.0f}")

        # Accumulate for combined view
        for t, units in holdings.items():
            price = get_price(t, prices, last_date)
            val = units * price
            combined[t] = combined.get(t, 0) + val
            total_val += val
        total_val += cash
        combined['CASH'] = combined.get('CASH', 0) + cash

    # Combined view
    print(f"\n{'=' * 70}")
    print(f"  합산 포트폴리오")
    print(f"{'=' * 70}")

    rows = [(t, v) for t, v in combined.items() if t != 'CASH']
    rows.sort(key=lambda x: x[1], reverse=True)
    cash_val = combined.get('CASH', 0)

    for t, val in rows:
        pct = val / total_val * 100 if total_val > 0 else 0
        print(f"  {t:<12} ${val:>10,.0f} ({pct:>5.1f}%)")
    if cash_val > 0.01:
        pct = cash_val / total_val * 100
        print(f"  {'CASH':<12} ${cash_val:>10,.0f} ({pct:>5.1f}%)")
    print(f"  {'합계':<12} ${total_val:>10,.0f}")
    print(f"\n  (초기 자본 $10,000 기준)")


if __name__ == '__main__':
    main()
