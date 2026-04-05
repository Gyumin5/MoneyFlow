#!/usr/bin/env python3
"""배수별 비교 테스트."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ensemble import SingleAccountEngine, combine_targets, STRATEGIES, load_data
from backtest_futures_full import run

START = '2020-10-01'
END = '2026-03-28'

data = {}
for iv in ['4h', '1h']:
    print(f"Loading {iv}...")
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
    print(f"  {key}: {len(trace)} bars")

bars_1h, funding_1h = data['1h']
btc_1h = bars_1h['BTC']
all_dates = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]

weights = {'4h1': 1/3, '4h2': 1/3, '1h1': 1/3}
combined = combine_targets(traces, weights, all_dates)

print("\n4h1+4h2+1h1 앙상블 — 배수별 비교 ($10,000 시작)")
print(f"  Lev    Sh     CAGR      MDD     Cal  Liq    Rb       최종")
print(f"  " + "-" * 70)

for lev in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    engine = SingleAccountEngine(bars_1h, funding_1h, leverage=lev)
    m = engine.run(combined)
    if not m:
        print(f"  {lev:.1f}x  FAILED")
        continue
    liq = f"L{m['Liq']}" if m.get('Liq', 0) > 0 else ""
    eq = m.get('_equity')
    final = eq.iloc[-1] if eq is not None else 0
    mult = final / 10000
    print(f"  {lev:.1f}x  {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>7.2f} {liq:>4s} {m['Rebal']:>5d}  ${final:>12,.0f} ({mult:.0f}x)")
