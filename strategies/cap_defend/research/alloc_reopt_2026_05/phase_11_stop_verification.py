#!/usr/bin/env python3
"""Phase 11 — Stop 옵션 종합 검증.

1) stop_pct grid plateau 확인
2) stop_kind 비교
3) OOS / walk-forward
4) Portfolio 70/15/15 통합 (stop 적용한 fut equity 로 다시 시뮬)
5) Event study (청산 6건 직전)
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
sys.path.insert(0, HERE)

from backtest_futures_full import load_data, run
from phase_4_extended import simulate, metrics, load_curves

BASE = dict(
    interval='D', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, health_mode='mom2vol',
)


def bt(label, start='2020-10-01', end='2026-05-13', **overrides):
    cfg = {**BASE, **overrides, 'start_date': start, 'end_date': end}
    bars, funding = load_data('D')
    m = run(bars, funding, **cfg)
    return dict(
        label=label,
        Cal=m['Cal'], CAGR=m['CAGR'], MDD=m['MDD'], Sharpe=m['Sharpe'],
        Liq=m['Liq'], Trades=m['Trades'],
        eq=m['_equity'],
    )


# ============ 1) stop_pct grid plateau ============
print('=== 1) stop_pct grid (highest_close_since_entry_pct) ===')
results_grid = []
results_grid.append({**bt('no_stop')})
for pct in (0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30):
    r = bt(f'stop_{int(pct*100)}', stop_kind='highest_close_since_entry_pct', stop_pct=pct)
    results_grid.append(r)
df1 = pd.DataFrame([{k: v for k, v in r.items() if k != 'eq'} for r in results_grid])
print(df1.to_string(index=False))

# ============ 2) stop_kind 비교 ============
print('\n=== 2) stop_kind 비교 (stop_pct=-20% 고정) ===')
results_kind = []
results_kind.append({**bt('no_stop_baseline')})
for kind in ('prev_close_pct', 'highest_close_since_entry_pct',
             'highest_high_since_entry_pct', 'rolling_high_close_pct',
             'rolling_high_high_pct'):
    r = bt(f'{kind}_20', stop_kind=kind, stop_pct=0.20)
    results_kind.append(r)
df2 = pd.DataFrame([{k: v for k, v in r.items() if k != 'eq'} for r in results_kind])
print(df2.to_string(index=False))

# ============ 3) OOS split ============
print('\n=== 3) OOS / walk-forward ===')
splits = {
    'IS (2020-10~2023-12)': ('2020-10-01', '2023-12-31'),
    'OOS (2024-01~2026-05)': ('2024-01-01', '2026-05-13'),
}
results_oos = []
for sname, (s, e) in splits.items():
    r0 = bt(f'no_stop|{sname}', start=s, end=e)
    r1 = bt(f'stop_20|{sname}', start=s, end=e, stop_kind='highest_close_since_entry_pct', stop_pct=0.20)
    results_oos.extend([r0, r1])
df3 = pd.DataFrame([{k: v for k, v in r.items() if k != 'eq'} for r in results_oos])
print(df3.to_string(index=False))

# ============ 4) Portfolio 70/15/15 통합 ============
print('\n=== 4) Portfolio 70/15/15 통합 (fut 만 stop_20 적용) ===')
returns = load_curves()
# stop_20 fut equity 추출
fut_stop20 = bt('stop20_fut_full', stop_kind='highest_close_since_entry_pct', stop_pct=0.20)
fut_eq_stop = fut_stop20['eq'].copy()
fut_eq_stop.index = pd.to_datetime(fut_eq_stop.index).normalize()
fut_eq_stop = fut_eq_stop.groupby(fut_eq_stop.index).last()

# returns 와 align
common_idx = returns.index.intersection(fut_eq_stop.index)
ret_new = returns.loc[common_idx].copy()
ret_new['fut'] = fut_eq_stop.loc[common_idx].pct_change().fillna(0)

# 통합 BT (현재 운영 vs stop_20 적용)
eq_base, _ = simulate(returns, 0.70, 0.15, 0.15, 'T1', 0.15)
eq_stop, _ = simulate(ret_new, 0.70, 0.15, 0.15, 'T1', 0.15)

m_base = metrics(eq_base)
m_stop = metrics(eq_stop)
print(f'  baseline   : Cal {m_base["Cal"]:.3f} / CAGR {m_base["CAGR"]*100:.1f}% / MDD {m_base["MDD"]*100:.1f}% / Sharpe {m_base["Sharpe"]:.3f}')
print(f'  fut stop_20: Cal {m_stop["Cal"]:.3f} / CAGR {m_stop["CAGR"]*100:.1f}% / MDD {m_stop["MDD"]*100:.1f}% / Sharpe {m_stop["Sharpe"]:.3f}')
print(f'  변화: dCal {(m_stop["Cal"]-m_base["Cal"])/m_base["Cal"]*100:+.1f}% / dCAGR {(m_stop["CAGR"]-m_base["CAGR"])*100:+.1f}pp / dMDD {(m_stop["MDD"]-m_base["MDD"])*100:+.1f}pp')

# ============ 5) Event study (청산 6건 직전 14일/30일 vol) ============
print('\n=== 5) Event study (청산 6건 directly stop 효과) ===')
# stop_20 적용 시 청산 횟수 = 1 (위 BT 에서 확인). 어느 5건이 막혔나?
m_base_full = run(*load_data('D'), **{**BASE, 'start_date': '2020-10-01', 'end_date': '2026-05-13'})
m_stop_full = run(*load_data('D'), **{**BASE, 'stop_kind': 'highest_close_since_entry_pct', 'stop_pct': 0.20,
                                      'start_date': '2020-10-01', 'end_date': '2026-05-13'})
print(f'  no_stop : Liq {m_base_full["Liq"]} ({len(m_base_full["_liq_log"])} 기록됨)')
print(f'  stop_20 : Liq {m_stop_full["Liq"]} ({len(m_stop_full["_liq_log"])} 기록됨)')
print(f'\n  no_stop 청산 이력:')
for e in m_base_full['_liq_log']:
    print(f'    {str(e["date"])[:10]} {e["coin"]:<7} 손실 {e["loss_pct"]*100:.0f}%')
print(f'\n  stop_20 잔존 청산:')
for e in m_stop_full['_liq_log']:
    print(f'    {str(e["date"])[:10]} {e["coin"]:<7} 손실 {e["loss_pct"]*100:.0f}%')

# 저장
df1.to_csv(os.path.join(HERE, 'stop_grid_plateau.csv'), index=False)
df2.to_csv(os.path.join(HERE, 'stop_kind_compare.csv'), index=False)
df3.to_csv(os.path.join(HERE, 'stop_oos_split.csv'), index=False)

pd.DataFrame([
    dict(case='baseline', **m_base),
    dict(case='fut_stop_20', **m_stop),
]).to_csv(os.path.join(HERE, 'portfolio_stop20_integrated.csv'), index=False)

print('\n저장: stop_grid_plateau.csv / stop_kind_compare.csv / stop_oos_split.csv / portfolio_stop20_integrated.csv')
