#!/usr/bin/env python3
"""Phase 10 — 선물 청산 방어 옵션 BT 비교 (단독 슬리브).

옵션 A~G 각각 단독 BT. baseline (현재 V23) 대비 Cal/CAGR/MDD/Liq 비교.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_DEFEND = os.path.abspath(os.path.join(HERE, '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CAP_DEFEND, '..', '..'))
sys.path.insert(0, CAP_DEFEND)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'trade'))
sys.path.insert(0, PROJECT_ROOT)

from backtest_futures_full import load_data, run

# baseline params (V23 fut)
BASE = dict(
    interval='D', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, health_mode='mom2vol',
    start_date='2020-10-01', end_date='2026-05-13',
)


def bt_one(label, **overrides):
    cfg = {**BASE, **overrides}
    bars, funding = load_data('D')
    m = run(bars, funding, **cfg)
    return dict(
        label=label,
        Cal=m['Cal'], CAGR=m['CAGR'], MDD=m['MDD'], Sharpe=m['Sharpe'],
        Liq=m['Liq'], Trades=m['Trades'], Rebal=m['Rebal'],
        liq_loss_sum=sum(e['margin']-e['returned'] for e in m['_liq_log']),
        final_eq=m['_equity'].iloc[-1],
    )


results = []

print("[BASELINE] V23 fut 현재")
results.append(bt_one('baseline_V23'))
print(f"  Cal {results[-1]['Cal']:.2f} / CAGR {results[-1]['CAGR']*100:.0f}% / MDD {results[-1]['MDD']*100:.0f}% / Liq {results[-1]['Liq']}")

# 옵션 A — Stop loss
for stop_pct in (0.20, 0.25, 0.30):
    print(f"[A] Stop -{int(stop_pct*100)}%")
    r = bt_one(f'A_stop_{int(stop_pct*100)}',
               stop_kind='highest_close_since_entry_pct', stop_pct=stop_pct)
    results.append(r)
    print(f"  Cal {r['Cal']:.2f} / CAGR {r['CAGR']*100:.0f}% / MDD {r['MDD']*100:.0f}% / Liq {r['Liq']}")

# 옵션 B — Leverage 낮춤
for lev in (2.0, 1.0):
    print(f"[B] Leverage L{int(lev)}")
    r = bt_one(f'B_lev{int(lev)}', leverage=lev)
    results.append(r)
    print(f"  Cal {r['Cal']:.2f} / CAGR {r['CAGR']*100:.0f}% / MDD {r['MDD']*100:.0f}% / Liq {r['Liq']}")

# 옵션 D — cap + universe 변경
for cap_val, us in [(1/5, 5), (1/7, 7)]:
    print(f"[D] cap 1/{int(1/cap_val)} universe {us}")
    r = bt_one(f'D_cap1_{int(1/cap_val)}_u{us}', cap=cap_val, universe_size=us)
    results.append(r)
    print(f"  Cal {r['Cal']:.2f} / CAGR {r['CAGR']*100:.0f}% / MDD {r['MDD']*100:.0f}% / Liq {r['Liq']}")

# 종합
df = pd.DataFrame(results)
print('\n=== 종합 결과 ===')
print(df[['label','Cal','CAGR','MDD','Sharpe','Liq','liq_loss_sum','Trades','Rebal']].to_string(index=False))

# baseline 대비 변화
base = df.iloc[0]
df['dCal_pct'] = (df.Cal - base.Cal) / base.Cal * 100
df['dCAGR_pp'] = (df.CAGR - base.CAGR) * 100
df['dMDD_pp'] = (df.MDD - base.MDD) * 100
df['dLiq'] = df.Liq - base.Liq

print('\n=== Baseline 대비 변화 ===')
print(df[['label','Cal','dCal_pct','dCAGR_pp','dMDD_pp','dLiq','Liq']].to_string(index=False))

df.to_csv(os.path.join(HERE, 'liq_defense_bt.csv'), index=False)
print('\n저장: liq_defense_bt.csv')
