#!/usr/bin/env python3
"""Phase C — plateau check. top N 후보 × ±5/10% perturb (4축) × 16 perturb."""
import os, sys, time, itertools, multiprocessing as mp
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, 'phase_c_results.csv')

# Top 후보 (B2/B3 best — 보수 tx=0.0006)
CANDIDATES = [
    dict(name='C1', sma=38, ms=20, ml=122, snap_int=111, drift=0.020, n_snap=3),
    dict(name='C2', sma=42, ms=20, ml=105, snap_int=111, drift=0.020, n_snap=3),
    dict(name='C3', sma=38, ms=20, ml=122, snap_int=111, drift=0.030, n_snap=3),
    dict(name='C4', sma=42, ms=20, ml=105, snap_int=111, drift=0.030, n_snap=3),
    dict(name='C5', sma=44, ms=20, ml=141, snap_int=93, drift=0.020, n_snap=3),
    dict(name='C6', sma=42, ms=18, ml=91, snap_int=93, drift=0.020, n_snap=3),
    dict(name='C7', sma=33, ms=29, ml=91, snap_int=87, drift=0.020, n_snap=3),  # B1 winner
    dict(name='V23', sma=42, ms=30, ml=90, snap_int=95, drift=0.030, n_snap=5),  # baseline
]

PERTURBS = [
    ('sma',  -0.10), ('sma',  -0.05), ('sma',  +0.05), ('sma',  +0.10),
    ('ms',   -0.10), ('ms',   -0.05), ('ms',   +0.05), ('ms',   +0.10),
    ('ml',   -0.10), ('ml',   -0.05), ('ml',   +0.05), ('ml',   +0.10),
    ('drift',-0.10), ('drift',-0.05), ('drift',+0.05), ('drift',+0.10),
]
TX_COST = 0.0006

_BARS=None; _FUNDING=None; _END=None
def _init_worker(end_date):
    global _BARS, _FUNDING, _END
    from backtest_futures_full import load_data
    _BARS, _FUNDING = load_data('D')
    _END = end_date

def _one(args):
    name, cfg, perturb = args
    from backtest_futures_full import run
    c = dict(cfg)
    if perturb is not None:
        axis, frac = perturb
        if axis in ('sma','ms','ml'):
            c[axis] = max(2, int(round(c[axis] * (1+frac))))
        elif axis == 'drift':
            c[axis] = c[axis] * (1+frac)
        elif axis == 'snap_int':
            c[axis] = max(c['n_snap'], int(round(c[axis] * (1+frac))))
    try:
        m = run(_BARS, _FUNDING,
                interval='D', leverage=3.0,
                universe_size=3, selection='greedy', cap=1/3,
                tx_cost=TX_COST, maint_rate=0.004,
                sma_days=c['sma'], mom_short_days=c['ms'], mom_long_days=c['ml'],
                vol_days=90, vol_threshold=0.05,
                canary_hyst=0.015, drift_threshold=c['drift'],
                n_snapshots=c['n_snap'], snap_interval_bars=c['snap_int'],
                health_mode='mom2vol',
                start_date='2020-10-01', end_date=_END)
        if not m:
            return dict(name=name, perturb=perturb, Cal=None)
        return dict(name=name, perturb=str(perturb), Cal=m['Cal'], CAGR=m['CAGR'], MDD=m['MDD'], Sharpe=m['Sharpe'], Rebal=m['Rebal'])
    except Exception as e:
        return dict(name=name, perturb=str(perturb), Cal=None, error=str(e)[:80])

def main():
    end_date = sys.argv[1] if len(sys.argv) > 1 else '2026-05-13'
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    tasks = []
    for cand in CANDIDATES:
        name = cand['name']
        cfg = {k:v for k,v in cand.items() if k != 'name'}
        tasks.append((name, cfg, None))  # center
        for p in PERTURBS:
            tasks.append((name, cfg, p))
    print(f"total tasks: {len(tasks)}", flush=True)
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(end_date,)) as pool:
        results = list(pool.imap_unordered(_one, tasks, chunksize=2))
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"saved: {RESULTS_CSV}  total={time.time()-t0:.0f}s")
    # plateau summary
    print(f"\n{'='*70}\nplateau summary (perturbed_min / center)")
    print(f"{'='*70}")
    for cand in CANDIDATES:
        name = cand['name']
        sub = df[df.name == name].copy()
        center = sub[sub.perturb == 'None']
        if len(center) == 0 or center.Cal.isna().any():
            print(f"  {name}: center failed")
            continue
        c_cal = center.Cal.iloc[0]
        per = sub[sub.perturb != 'None']
        per_valid = per.dropna(subset=['Cal'])
        if len(per_valid) == 0:
            continue
        p_min = per_valid.Cal.min()
        p_mean = per_valid.Cal.mean()
        ratio = p_min / c_cal if c_cal > 0 else 0
        print(f"  {name}: center={c_cal:.2f} per_min={p_min:.2f} per_mean={p_mean:.2f} ratio={ratio:.2f}  {'OK' if ratio >= 0.85 else 'FAIL'}")

if __name__ == '__main__':
    main()
