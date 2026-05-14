#!/usr/bin/env python3
"""Phase E3 — Target-EW 앙상블 BT (정식, external_target_schedule 사용).

PRE-CHECK 통과: V23 replay parity 0%, C1 replay parity 0%.
이제 V23 + C1 trace 를 EW merge 한 schedule 로 run() 실행.
"""
import os, sys
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import pandas as pd
import numpy as np
from backtest_futures_full import load_data, run

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

V23 = dict(sma_days=42, mom_short_days=30, mom_long_days=90, vol_days=90,
           n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03)
C1  = dict(sma_days=38, mom_short_days=20, mom_long_days=122, vol_days=90,
           n_snapshots=3, snap_interval_bars=111, drift_threshold=0.020)
COMMON = dict(interval='D', leverage=3.0, universe_size=3, selection='greedy', cap=1/3,
              tx_cost=0.0006, maint_rate=0.004, canary_hyst=0.015, vol_threshold=0.05,
              health_mode='mom2vol', start_date='2020-10-01', end_date='2026-05-13')

def collect_trace(cfg):
    tr = []
    run(LOAD['bars'], LOAD['funding'], **COMMON, **cfg, _trace=tr)
    return tr

def merge_ew(t_a, t_b, weight_a=0.5):
    by_a = {r['date']: r for r in t_a}
    by_b = {r['date']: r for r in t_b}
    dates = sorted(set(by_a) | set(by_b))
    sched = {}
    for d in dates:
        ra = by_a.get(d)
        rb = by_b.get(d)
        if ra is None:
            sched[d] = {'target': dict(rb['target']), 'rebal': rb['rebal']}
            continue
        if rb is None:
            sched[d] = {'target': dict(ra['target']), 'rebal': ra['rebal']}
            continue
        ta, tb = ra['target'], rb['target']
        merged = {}
        all_k = set(ta) | set(tb)
        for k in all_k:
            merged[k] = weight_a * ta.get(k, 0.0) + (1 - weight_a) * tb.get(k, 0.0)
        # 정규화 (CASH 포함)
        s = sum(merged.values())
        if s > 0:
            merged = {k: v/s for k, v in merged.items() if v > 0}
        else:
            merged = {'CASH': 1.0}
        sched[d] = {'target': merged, 'rebal': bool(ra['rebal'] or rb['rebal'])}
    return sched

def yearly(eq):
    out = []
    for year, g in eq.groupby(eq.index.year):
        if len(g) < 20: continue
        ret = g.iloc[-1] / g.iloc[0] - 1
        dr = g.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (g / g.cummax() - 1).min()
        cal = ret / abs(mdd) if mdd != 0 else 0
        out.append(dict(year=year, ret=ret, MDD=mdd, Cal=cal, Sharpe=sh))
    return pd.DataFrame(out)

LOAD = {}
def main():
    LOAD['bars'], LOAD['funding'] = load_data('D')
    bars, funding = LOAD['bars'], LOAD['funding']

    print("="*80)
    print("Phase E3: Target-EW 앙상블 (proper, external_target_schedule)")
    print("="*80)

    # 1) V23, C1 단독
    print("\n## V23, C1 단독 (참고)")
    m_v23 = run(bars, funding, **COMMON, **V23)
    m_c1  = run(bars, funding, **COMMON, **C1)
    print(f"V23 : Sharpe={m_v23['Sharpe']:.2f} CAGR={m_v23['CAGR']:+.1%} MDD={m_v23['MDD']:+.1%} Cal={m_v23['Cal']:.2f} Rebal={m_v23['Rebal']}")
    print(f"C1  : Sharpe={m_c1['Sharpe']:.2f}  CAGR={m_c1['CAGR']:+.1%} MDD={m_c1['MDD']:+.1%} Cal={m_c1['Cal']:.2f} Rebal={m_c1['Rebal']}")

    # 2) trace 수집
    tr_v = collect_trace(V23)
    tr_c = collect_trace(C1)
    print(f"\ntraces collected: V23={len(tr_v)}, C1={len(tr_c)}")

    # 3) EW Ensemble (weight sweep)
    # 앙상블 canary 는 sma=40 사용 (V23 42 와 C1 38 의 산술 중간). mom/vol 은 external 모드에서 미사용.
    ENS_COMMON = dict(COMMON)
    ENS_COMMON.update(dict(sma_days=40, mom_short_days=30, mom_long_days=90, vol_days=90,
                           n_snapshots=1, snap_interval_bars=0))  # snapshots 미사용

    print("\n## EW weight sweep, drift=0.025")
    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = []
    for w in weights:
        sched = merge_ew(tr_v, tr_c, weight_a=w)
        m = run(bars, funding, **ENS_COMMON, drift_threshold=0.025, external_target_schedule=sched)
        rows.append(dict(weight_v23=w, Sharpe=m['Sharpe'], CAGR=m['CAGR'], MDD=m['MDD'], Cal=m['Cal'], Rebal=m['Rebal']))
        print(f"  V23 w={w:.1f}: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Rebal={m['Rebal']}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'phase_e3_weight_sweep.csv'), index=False)

    # 4) 50/50 drift sweep
    print("\n## 50/50 drift sweep")
    sched50 = merge_ew(tr_v, tr_c, weight_a=0.5)
    drift_rows = []
    for d in [0.020, 0.025, 0.030, 0.040, 0.050]:
        m = run(bars, funding, **ENS_COMMON, drift_threshold=d, external_target_schedule=sched50)
        drift_rows.append(dict(drift=d, Sharpe=m['Sharpe'], CAGR=m['CAGR'], MDD=m['MDD'], Cal=m['Cal'], Rebal=m['Rebal']))
        print(f"  drift={d:.3f}: Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Rebal={m['Rebal']}")
    pd.DataFrame(drift_rows).to_csv(os.path.join(OUT_DIR, 'phase_e3_drift_sweep.csv'), index=False)

    # 5) 50/50 drift=0.025 baseline 결과 + yearly
    print("\n## 50/50, drift=0.025 baseline + yearly")
    m_ens = run(bars, funding, **ENS_COMMON, drift_threshold=0.025, external_target_schedule=sched50,
                _trace=[])
    print(f"Ensemble: Sharpe={m_ens['Sharpe']:.2f} CAGR={m_ens['CAGR']:+.1%} MDD={m_ens['MDD']:+.1%} Cal={m_ens['Cal']:.2f} Rebal={m_ens['Rebal']}")
    eq = m_ens['_equity']
    yr = yearly(eq)
    print("\nYearly metrics (Ensemble 50/50 d=0.025):")
    print(yr.round(3).to_string(index=False))
    yr.to_csv(os.path.join(OUT_DIR, 'phase_e3_yearly.csv'), index=False)

    # 6) 비용 스트레스
    print("\n## 비용 스트레스 (3가지 strategy 동일 tx)")
    print(f"{'cand':<10}{'tx':>8}{'Sharpe':>8}{'CAGR':>10}{'MDD':>10}{'Cal':>8}{'Rebal':>7}")
    stress_rows = []
    for tx in [0.0006, 0.0008, 0.0010, 0.0012, 0.0018, 0.0030]:
        common_tx = dict(COMMON); common_tx['tx_cost'] = tx
        # V23
        m = run(bars, funding, **common_tx, **V23)
        print(f"{'V23':<10}{tx:>8.4f}{m['Sharpe']:>8.2f}{m['CAGR']:>+10.1%}{m['MDD']:>+10.1%}{m['Cal']:>8.2f}{m['Rebal']:>7d}")
        stress_rows.append(dict(cand='V23', tx=tx, **{k: m[k] for k in ['Sharpe','CAGR','MDD','Cal','Rebal']}))
        # C1
        m = run(bars, funding, **common_tx, **C1)
        print(f"{'C1':<10}{tx:>8.4f}{m['Sharpe']:>8.2f}{m['CAGR']:>+10.1%}{m['MDD']:>+10.1%}{m['Cal']:>8.2f}{m['Rebal']:>7d}")
        stress_rows.append(dict(cand='C1', tx=tx, **{k: m[k] for k in ['Sharpe','CAGR','MDD','Cal','Rebal']}))
        # Ensemble (trace 다시 수집해서 새 tx 반영)
        common_ens_tx = dict(ENS_COMMON); common_ens_tx['tx_cost'] = tx
        tr_v_tx = []; run(bars, funding, **common_tx, **V23, _trace=tr_v_tx)
        tr_c_tx = []; run(bars, funding, **common_tx, **C1, _trace=tr_c_tx)
        sched_tx = merge_ew(tr_v_tx, tr_c_tx, 0.5)
        m = run(bars, funding, **common_ens_tx, drift_threshold=0.025, external_target_schedule=sched_tx)
        print(f"{'EW 50/50':<10}{tx:>8.4f}{m['Sharpe']:>8.2f}{m['CAGR']:>+10.1%}{m['MDD']:>+10.1%}{m['Cal']:>8.2f}{m['Rebal']:>7d}")
        stress_rows.append(dict(cand='EW', tx=tx, **{k: m[k] for k in ['Sharpe','CAGR','MDD','Cal','Rebal']}))
    pd.DataFrame(stress_rows).to_csv(os.path.join(OUT_DIR, 'phase_e3_cost_stress.csv'), index=False)

if __name__ == '__main__':
    main()
