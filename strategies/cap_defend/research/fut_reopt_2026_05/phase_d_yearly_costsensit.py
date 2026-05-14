#!/usr/bin/env python3
"""Phase D — yearly rank-sum + 비용 스트레스 (V23 vs C1)."""
import os, sys, time
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import pandas as pd
import numpy as np
from backtest_futures_full import load_data, run

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = {
    'V23': dict(sma_days=42, mom_short_days=30, mom_long_days=90,
                n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03),
    'C1':  dict(sma_days=38, mom_short_days=20, mom_long_days=122,
                n_snapshots=3, snap_interval_bars=111, drift_threshold=0.020),
}
TX_LIST = [0.0006, 0.0008, 0.0010, 0.0012]

COMMON = dict(
    interval='D', leverage=3.0,
    universe_size=3, selection='greedy', cap=1/3,
    maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015,
    health_mode='mom2vol',
    start_date='2020-10-01', end_date='2026-05-13',
)

def yearly_metrics(eq):
    """연도별 CAGR/MDD/Cal/Sharpe."""
    eq = eq.copy()
    out = []
    for year, g in eq.groupby(eq.index.year):
        if len(g) < 20:
            continue
        ret = g.iloc[-1] / g.iloc[0] - 1
        dr = g.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (g / g.cummax() - 1).min()
        cal = ret / abs(mdd) if mdd != 0 else 0
        out.append(dict(year=year, ret=ret, MDD=mdd, Cal=cal, Sharpe=sh, days=len(g)))
    return pd.DataFrame(out)

def main():
    bars, funding = load_data('D')
    print("="*80)
    print("Phase D: V23 vs C1 — yearly + 비용 스트레스")
    print("="*80)

    # 1) 비용 스트레스
    print("\n## 비용 스트레스 (전체 기간)")
    print(f"{'cand':<5} {'tx':<8} {'Sharpe':>8} {'CAGR':>10} {'MDD':>10} {'Cal':>8} {'Rebal':>6}")
    cost_results = []
    for cname, cfg in CANDIDATES.items():
        for tx in TX_LIST:
            kw = {**COMMON, **cfg, 'tx_cost': tx}
            m = run(bars, funding, **kw)
            if not m:
                continue
            cost_results.append(dict(cand=cname, tx=tx, Sharpe=m['Sharpe'], CAGR=m['CAGR'],
                                      MDD=m['MDD'], Cal=m['Cal'], Rebal=m['Rebal']))
            print(f"{cname:<5} {tx:<8.4f} {m['Sharpe']:>8.2f} {m['CAGR']:>+10.1%} {m['MDD']:>+10.1%} {m['Cal']:>8.2f} {m['Rebal']:>6d}")
    pd.DataFrame(cost_results).to_csv(os.path.join(OUT_DIR, 'phase_d_cost.csv'), index=False)

    # 2) Yearly metrics (tx=0.0006 동일 기준)
    print("\n## 연도별 metrics (tx=0.0006)")
    yearly_all = []
    for cname, cfg in CANDIDATES.items():
        kw = {**COMMON, **cfg, 'tx_cost': 0.0006}
        m = run(bars, funding, **kw)
        eq = m['_equity']
        yr = yearly_metrics(eq)
        yr['cand'] = cname
        yearly_all.append(yr)
        print(f"\n{cname}:")
        print(yr.to_string(index=False))
    yall = pd.concat(yearly_all)
    yall.to_csv(os.path.join(OUT_DIR, 'phase_d_yearly.csv'), index=False)

    # 3) Yearly rank-sum
    print("\n## 연도별 rank-sum (낮을수록 우위)")
    pivot_cal = yall.pivot(index='year', columns='cand', values='Cal')
    rank_cal = pivot_cal.rank(axis=1, ascending=False)
    pivot_ret = yall.pivot(index='year', columns='cand', values='ret')
    rank_ret = pivot_ret.rank(axis=1, ascending=False)
    pivot_mdd = yall.pivot(index='year', columns='cand', values='MDD')
    rank_mdd = pivot_mdd.rank(axis=1, ascending=False)  # MDD: 큰값(덜 부정)이 좋음

    print("\nyearly Cal:")
    print(pivot_cal.round(2).to_string())
    print("\nrank by Cal (1=best):")
    print(rank_cal.to_string())
    print("\nrank-sum (Cal+ret+MDD):")
    rsum = rank_cal.sum() + rank_ret.sum() + rank_mdd.sum()
    print(rsum.sort_values().to_string())

    # 4) 비용 스트레스에서 C1 우위 유지 여부
    print("\n## 비용 스트레스 verdict (C1 vs V23)")
    cdf = pd.DataFrame(cost_results)
    for tx in TX_LIST:
        sub = cdf[cdf.tx == tx].set_index('cand')
        if 'V23' in sub.index and 'C1' in sub.index:
            d = sub.loc['C1','Cal'] - sub.loc['V23','Cal']
            mdd_d = sub.loc['C1','MDD'] - sub.loc['V23','MDD']
            print(f"  tx={tx:.4f}: ΔCal={d:+.2f} ({'C1 우위' if d > 0 else 'V23 우위'}), ΔMDD={mdd_d:+.3f}")

if __name__ == '__main__':
    main()
