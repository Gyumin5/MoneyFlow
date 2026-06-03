#!/usr/bin/env python3
"""Phase B/C/D/E — L4 refine (B 신호 → C snap → D drift → E plateau)."""
import os, sys, time, itertools, multiprocessing as mp
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/trade')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

_BARS = None
_FUNDING = None


def _init_worker():
    global _BARS, _FUNDING
    from backtest_futures_full import load_data
    _BARS, _FUNDING = load_data('D')


def _one(cfg):
    from backtest_futures_full import run
    try:
        m = run(_BARS, _FUNDING,
                interval='D', leverage=4.0,
                universe_size=3, selection='greedy', cap=1/3,
                tx_cost=0.0006, maint_rate=0.004,
                sma_days=cfg['sma'], mom_short_days=cfg['ms'], mom_long_days=cfg['ml'],
                vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
                drift_threshold=cfg['drift'],
                n_snapshots=cfg['n_snap'], snap_interval_bars=cfg['snap_int'],
                health_mode='mom2vol',
                start_date='2020-10-01', end_date='2026-05-13')
        if not m: return {**cfg, 'Sharpe': None}
        return {**cfg, 'Sharpe': m['Sharpe'], 'CAGR': m['CAGR'], 'MDD': m['MDD'],
                'Cal': m['Cal'], 'Rebal': m['Rebal'], 'Liq': m['Liq']}
    except Exception as e:
        return {**cfg, 'Sharpe': None, 'error': str(e)[:80]}


def run_grid(configs, n_workers=16):
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        results = list(pool.imap_unordered(_one, configs, chunksize=4))
    df = pd.DataFrame(results).dropna(subset=['Cal'])
    print(f"  {len(df)} valid, elapsed {time.time()-t0:.0f}s")
    return df


def phase(label, configs, save_name):
    print(f"\n[{label}] {len(configs)} configs")
    df = run_grid(configs)
    df.to_csv(os.path.join(OUT_DIR, save_name), index=False)
    print(f"  saved: {save_name}")
    print(f"  TOP 10 by Cal:")
    print(df.nlargest(10, 'Cal')[['sma','ms','ml','n_snap','snap_int','drift','Cal','CAGR','MDD','Sharpe','Liq']].to_string(index=False))
    return df


# Phase A best: sma=42, ms=18, ml=127, n_snap=3, snap_int=87, drift=0.02 → Cal 8.42
# Also strong: n_snap=7 snap_int=217

# ===== Phase B — signal axes refine =====
configs_b = []
for sma in [38, 42, 46]:
    for ms in [15, 18, 22]:
        for ml in [100, 115, 127, 140, 160]:
            # use top snap/drift from A
            for snap in [(3, 87), (7, 217)]:
                for drift in [0.02, 0.03]:
                    configs_b.append(dict(sma=sma, ms=ms, ml=ml,
                                          n_snap=snap[0], snap_int=snap[1], drift=drift))
df_b = phase('B - signal refine', configs_b, 'phase_b_L4.csv')

# Best signal from B
best_b = df_b.nlargest(1, 'Cal').iloc[0]
print(f"\n  B best: sma={best_b.sma} ms={best_b.ms} ml={best_b.ml}")

# ===== Phase C — snap × n_snap refine =====
configs_c = []
SNAPS = [
    (3, 57), (3, 69), (3, 87), (3, 93), (3, 111), (3, 129), (3, 159),
    (5, 95), (5, 115), (5, 145), (5, 155), (5, 185), (5, 215),
    (7, 133), (7, 161), (7, 191), (7, 217), (7, 259),
    (9, 171), (9, 207), (9, 243),
]
for snap in SNAPS:
    for drift in [0.02, 0.03]:
        configs_c.append(dict(sma=int(best_b.sma), ms=int(best_b.ms), ml=int(best_b.ml),
                              n_snap=snap[0], snap_int=snap[1], drift=drift))
df_c = phase('C - snap refine', configs_c, 'phase_c_L4.csv')

best_c = df_c.nlargest(1, 'Cal').iloc[0]
print(f"\n  C best: n_snap={best_c.n_snap} snap_int={best_c.snap_int}")

# ===== Phase D — drift fine =====
configs_d = []
for drift in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.08]:
    configs_d.append(dict(sma=int(best_b.sma), ms=int(best_b.ms), ml=int(best_b.ml),
                          n_snap=int(best_c.n_snap), snap_int=int(best_c.snap_int),
                          drift=drift))
df_d = phase('D - drift refine', configs_d, 'phase_d_L4.csv')

best_d = df_d.nlargest(1, 'Cal').iloc[0]
print(f"\n  D best drift={best_d.drift}")

# ===== Phase E — plateau check =====
print('\n=== E plateau check around best ===')
# 최종 best: best_b sma/ms/ml + best_c snap + best_d drift
final = dict(sma=int(best_b.sma), ms=int(best_b.ms), ml=int(best_b.ml),
             n_snap=int(best_c.n_snap), snap_int=int(best_c.snap_int),
             drift=float(best_d.drift))
print(f"  Final candidate: {final}")

# ±1 step per axis
configs_e = [final]  # baseline
for axis_name, axis_vals in [
    ('sma', [final['sma']-4, final['sma']+4]),
    ('ms', [final['ms']-3, final['ms']+3]),
    ('ml', [final['ml']-15, final['ml']+15]),
    # snap: use ±1 prime stagger
    ('snap_int', [final['snap_int']-12, final['snap_int']+12]),
    ('drift', [final['drift']-0.005, final['drift']+0.005]),
]:
    for v in axis_vals:
        if v <= 0: continue
        cfg = dict(final)
        cfg[axis_name] = v if axis_name != 'drift' else round(v, 4)
        configs_e.append(cfg)

df_e = phase('E plateau', configs_e, 'phase_e_L4.csv')

# Plateau Cal drop
base_cal = df_e.iloc[0]['Cal']
neigh = df_e.iloc[1:]
drop_pct = (base_cal - neigh.Cal.min()) / base_cal * 100
print(f"\n  Plateau check: base Cal={base_cal:.2f}, neigh min Cal={neigh.Cal.min():.2f}, drop={drop_pct:.0f}%")

# 최종 요약
print("\n" + "=" * 60)
print("L4 최종 best params")
print("=" * 60)
print(f"  SMA: {final['sma']}")
print(f"  MS: {final['ms']}")
print(f"  ML: {final['ml']}")
print(f"  n_snap: {final['n_snap']}")
print(f"  snap_int: {final['snap_int']}")
print(f"  drift: {final['drift']}")
print(f"  Final Cal: {base_cal:.2f}")
print(f"  Plateau drop: {drop_pct:.0f}%")

import json
with open(os.path.join(OUT_DIR, 'L4_final_params.json'), 'w') as f:
    json.dump(dict(final=final, Cal=float(base_cal), plateau_drop_pct=float(drop_pct)), f, indent=2)
print(f"\n저장: L4_final_params.json")
