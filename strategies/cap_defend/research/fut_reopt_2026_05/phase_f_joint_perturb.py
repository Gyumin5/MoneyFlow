#!/usr/bin/env python3
"""Phase F — Joint perturbation: C1 sma/ms 변화에 EW 50/50 Cal 어떻게 반응?"""
import os, sys, time, itertools, multiprocessing as mp
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import pandas as pd
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, 'phase_f_joint.csv')

V23 = dict(sma_days=42, mom_short_days=30, mom_long_days=90, vol_days=90,
           n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03)
C1_CENTER = dict(sma_days=38, mom_short_days=20, mom_long_days=122, vol_days=90,
                 n_snapshots=3, snap_interval_bars=111, drift_threshold=0.020)
COMMON = dict(interval='D', leverage=3.0, universe_size=3, selection='greedy', cap=1/3,
              tx_cost=0.0006, maint_rate=0.004, canary_hyst=0.015, vol_threshold=0.05,
              health_mode='mom2vol', start_date='2020-10-01', end_date='2026-05-13')
ENS_COMMON = dict(COMMON)
ENS_COMMON.update(dict(sma_days=40, mom_short_days=30, mom_long_days=90, vol_days=90,
                       n_snapshots=1, snap_interval_bars=0))

_BARS=None; _FUNDING=None
def _init():
    global _BARS, _FUNDING
    from backtest_futures_full import load_data
    _BARS, _FUNDING = load_data('D')

def collect_trace(cfg):
    from backtest_futures_full import run
    tr = []
    run(_BARS, _FUNDING, **COMMON, **cfg, _trace=tr)
    return tr

def merge_ew(t_a, t_b, weight_a=0.5):
    by_a = {r['date']: r for r in t_a}
    by_b = {r['date']: r for r in t_b}
    dates = sorted(set(by_a) | set(by_b))
    sched = {}
    for d in dates:
        ra = by_a.get(d); rb = by_b.get(d)
        if ra is None:
            sched[d] = {'target': dict(rb['target']), 'rebal': rb['rebal']}
            continue
        if rb is None:
            sched[d] = {'target': dict(ra['target']), 'rebal': ra['rebal']}
            continue
        ta, tb = ra['target'], rb['target']
        merged = {}
        for k in set(ta) | set(tb):
            merged[k] = weight_a * ta.get(k, 0.0) + (1 - weight_a) * tb.get(k, 0.0)
        s = sum(merged.values())
        if s > 0:
            merged = {k: v/s for k, v in merged.items() if v > 0}
        else:
            merged = {'CASH': 1.0}
        sched[d] = {'target': merged, 'rebal': bool(ra['rebal'] or rb['rebal'])}
    return sched

def _one(args):
    """args=(label, v23_perturb, c1_perturb). perturb={'axis':val 또는 None}."""
    from backtest_futures_full import run
    label, v23_p, c1_p = args
    v23_cfg = dict(V23); c1_cfg = dict(C1_CENTER)
    for axis, val in (v23_p or {}).items():
        v23_cfg[axis] = val
    for axis, val in (c1_p or {}).items():
        c1_cfg[axis] = val
    try:
        # V23 단독
        m_v = run(_BARS, _FUNDING, **COMMON, **v23_cfg)
        # C1 단독
        m_c = run(_BARS, _FUNDING, **COMMON, **c1_cfg)
        # EW
        tr_v = []; run(_BARS, _FUNDING, **COMMON, **v23_cfg, _trace=tr_v)
        tr_c = []; run(_BARS, _FUNDING, **COMMON, **c1_cfg, _trace=tr_c)
        sched = merge_ew(tr_v, tr_c, 0.5)
        m_ew = run(_BARS, _FUNDING, **ENS_COMMON, drift_threshold=0.025, external_target_schedule=sched)
        return dict(label=label,
                    V23_Cal=m_v['Cal'], V23_MDD=m_v['MDD'], V23_CAGR=m_v['CAGR'],
                    C1_Cal=m_c['Cal'], C1_MDD=m_c['MDD'], C1_CAGR=m_c['CAGR'],
                    EW_Cal=m_ew['Cal'], EW_MDD=m_ew['MDD'], EW_CAGR=m_ew['CAGR'], EW_Sharpe=m_ew['Sharpe'])
    except Exception as e:
        return dict(label=label, error=str(e)[:80])

def main():
    tasks = []
    # center (참고)
    tasks.append(('CENTER', None, None))
    # C1 only perturbations
    for axis in ['sma_days', 'mom_short_days', 'mom_long_days']:
        center = C1_CENTER[axis]
        for frac in [-0.10, -0.05, +0.05, +0.10]:
            val = max(2, int(round(center * (1 + frac))))
            tasks.append((f'C1_{axis}_{frac:+.2f}', None, {axis: val}))
    # V23 only perturbations
    for axis in ['sma_days', 'mom_short_days', 'mom_long_days']:
        center = V23[axis]
        for frac in [-0.10, -0.05, +0.05, +0.10]:
            val = max(2, int(round(center * (1 + frac))))
            tasks.append((f'V23_{axis}_{frac:+.2f}', {axis: val}, None))

    print(f"total tasks: {len(tasks)}")
    t0 = time.time()
    with mp.Pool(8, initializer=_init) as pool:
        results = list(pool.imap_unordered(_one, tasks, chunksize=1))
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"saved: {RESULTS_CSV}, total={time.time()-t0:.0f}s")
    # 출력
    center_row = df[df.label=='CENTER'].iloc[0]
    cv = center_row['V23_Cal']
    cc = center_row['C1_Cal']
    ce = center_row['EW_Cal']
    print(f"\nCENTER: V23 Cal={cv:.2f}, C1 Cal={cc:.2f}, EW Cal={ce:.2f}")
    print("\n## C1 단독 축 perturb (V23 center 고정)")
    print(f"{'label':<35}{'C1 Cal':>10}{'EW Cal':>10}{'EW/c':>8}{'EW MDD':>10}")
    for _, r in df[df.label.str.startswith('C1_')].sort_values('label').iterrows():
        ratio = r['EW_Cal'] / ce if ce > 0 else 0
        print(f"{r['label']:<35}{r['C1_Cal']:>10.2f}{r['EW_Cal']:>10.2f}{ratio:>8.2f}{r['EW_MDD']:>+10.1%}")
    print("\n## V23 단독 축 perturb (C1 center 고정)")
    print(f"{'label':<35}{'V23 Cal':>10}{'EW Cal':>10}{'EW/c':>8}{'EW MDD':>10}")
    for _, r in df[df.label.str.startswith('V23_')].sort_values('label').iterrows():
        ratio = r['EW_Cal'] / ce if ce > 0 else 0
        print(f"{r['label']:<35}{r['V23_Cal']:>10.2f}{r['EW_Cal']:>10.2f}{ratio:>8.2f}{r['EW_MDD']:>+10.1%}")
    # summary
    ew_min = df['EW_Cal'].min()
    ew_max = df['EW_Cal'].max()
    ew_mean = df['EW_Cal'].mean()
    print(f"\nEW Cal range: min={ew_min:.2f}, mean={ew_mean:.2f}, max={ew_max:.2f}, center={ce:.2f}")
    print(f"EW per_min/center = {ew_min/ce:.2f}, per_mean/center = {ew_mean/ce:.2f}")

if __name__ == '__main__':
    main()
