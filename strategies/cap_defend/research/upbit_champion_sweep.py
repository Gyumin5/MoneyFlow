#!/usr/bin/env python3
"""Upbit 1h 데이터로 V22 champion 재탐색.

파라미터 grid 를 Upbit 기준으로 재최적화.
"""
from __future__ import annotations
import os, sys, time, glob
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "next_strategies"))

from c_engine_v5 import run_c_v5
from m3_engine_final import (simulate, load_universe_hist, load_coin_daily,
                              list_available_futures, load_v21)

OUT = os.path.join(HERE, "upbit_c_out")
CACHE_DIR = os.path.join(OUT, "cache")

START = pd.Timestamp("2020-10-01")
TRAIN_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-03-28")


def filter_bounce_upbit(ev, bars_dict, window_h=1):
    if len(ev) == 0: return ev
    ev = ev.copy()
    ev['entry_ts'] = pd.to_datetime(ev['entry_ts'])
    ev['exit_ts'] = pd.to_datetime(ev['exit_ts'])
    keep, new_ets, new_epx, new_pnl = [], [], [], []
    for _, r in ev.iterrows():
        bars = bars_dict.get(r['coin'])
        if bars is None:
            keep.append(True); new_ets.append(r['entry_ts'])
            new_epx.append(r['entry_px']); new_pnl.append(r['pnl_pct']); continue
        after = bars[bars.index > r['entry_ts']].iloc[:window_h]
        if len(after) == 0:
            keep.append(True); new_ets.append(r['entry_ts'])
            new_epx.append(r['entry_px']); new_pnl.append(r['pnl_pct']); continue
        green = (after['Close'] > after['Open'])
        if green.any():
            first_green = after.index[green][0]
            after_g = bars[bars.index > first_green]
            if len(after_g) == 0:
                keep.append(False); new_ets.append(r['entry_ts'])
                new_epx.append(r['entry_px']); new_pnl.append(r['pnl_pct']); continue
            ent = after_g.iloc[0]
            new_e_ts = ent.name; new_e_px = float(ent['Open'])
            if new_e_ts >= r['exit_ts']:
                keep.append(False); new_ets.append(r['entry_ts'])
                new_epx.append(r['entry_px']); new_pnl.append(r['pnl_pct']); continue
            keep.append(True)
            new_ets.append(new_e_ts); new_epx.append(new_e_px)
            new_pnl.append(round((float(r['exit_px'])/new_e_px - 1.0)*100, 3))
        else:
            keep.append(False); new_ets.append(r['entry_ts'])
            new_epx.append(r['entry_px']); new_pnl.append(r['pnl_pct'])
    ev['entry_ts'] = new_ets; ev['entry_px'] = new_epx; ev['pnl_pct'] = new_pnl
    return ev[keep].reset_index(drop=True)


def extract_events(bars_dict, params):
    """주어진 params 로 모든 coin events 추출."""
    all_evs = []
    for c, df in bars_dict.items():
        _, evs = run_c_v5(df, tx=0.0005, **params)
        for e in evs: e['coin'] = c
        all_evs.extend(evs)
    return pd.DataFrame(all_evs)


def run_simulate(ev, v21, cd, hist, cap, start, end):
    mask = (v21.index >= start) & (v21.index <= end)
    vs = v21[mask].copy()
    if len(vs) < 30: return None
    vs['equity'] = vs['equity'].astype(float) / float(vs['equity'].iloc[0])
    vs['v21_ret'] = vs['equity'].pct_change().fillna(0)
    vs['prev_cash'] = vs['cash_ratio'].shift(1).fillna(vs['cash_ratio'].iloc[0])
    if len(ev) > 0:
        ev_s = ev[(pd.to_datetime(ev['entry_ts']) >= vs.index[0]) &
                   (pd.to_datetime(ev['entry_ts']) <= vs.index[-1])]
    else:
        ev_s = ev
    if len(ev_s) == 0:
        return {'Cal': 0, 'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'n': 0}
    _, st = simulate(ev_s, cd, vs.copy(), hist,
                      n_pick=1, cap_per_slot=cap, universe_size=15,
                      tx_cost=0.0005, swap_edge_threshold=1)
    return {'Cal': st.get('Cal', 0), 'CAGR': st.get('CAGR', 0),
            'MDD': st.get('MDD', 0), 'Sharpe': st.get('Sharpe', 0),
            'n': len(ev_s)}


def main():
    print("Loading Upbit bars...")
    t0 = time.time()
    up_bars = {}
    for f in sorted(glob.glob(f'{CACHE_DIR}/*_1h.pkl')):
        c = os.path.basename(f).replace('_1h.pkl', '')
        df = pd.read_pickle(f)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        up_bars[c] = df
    print(f"  {len(up_bars)} coins ({time.time()-t0:.0f}s)")

    print("Loading shared data...")
    avail = sorted(list_available_futures())
    hist = load_universe_hist()
    cd = load_coin_daily(avail)
    v21 = load_v21()

    # Grid
    GRID_DTHR = [-0.10, -0.12, -0.15, -0.18, -0.20]
    GRID_TP = [0.03, 0.04, 0.05, 0.06]
    GRID_TSTOP = [18, 24, 36]
    USE_A2 = [True, False]
    CAP = 0.333

    print(f"\nGrid: {len(GRID_DTHR)*len(GRID_TP)*len(GRID_TSTOP)*len(USE_A2)} configs")
    print("  dip_thr:", GRID_DTHR)
    print("  tp:", GRID_TP)
    print("  tstop:", GRID_TSTOP)
    print("  a2:", USE_A2)

    rows = []
    t0 = time.time()
    total = len(GRID_DTHR) * len(GRID_TP) * len(GRID_TSTOP) * len(USE_A2)
    done = 0

    # extract events per (dip_thr, tp, tstop) — a2 는 post-filter
    for dt in GRID_DTHR:
        for tp in GRID_TP:
            for ts in GRID_TSTOP:
                params = dict(dip_bars=24, dip_thr=dt, tp=tp, tstop=ts)
                ev = extract_events(up_bars, params)
                for a2 in USE_A2:
                    ev_work = filter_bounce_upbit(ev, up_bars, 1) if a2 else ev
                    label = f"dthr{int(dt*100)}_tp{int(tp*100)}_ts{ts}_a2{1 if a2 else 0}"
                    for span, start, end in [('full', START, END),
                                               ('train', START, TRAIN_END),
                                               ('holdout', HOLDOUT_START, END)]:
                        m = run_simulate(ev_work, v21, cd, hist, CAP, start, end)
                        if m is None: continue
                        rows.append({'label': label, 'dthr': dt, 'tp': tp, 'tstop': ts,
                                     'a2': a2, 'span': span,
                                     'Cal': round(m['Cal'], 3),
                                     'CAGR': round(m['CAGR']*100, 2),
                                     'MDD': round(m['MDD']*100, 2),
                                     'Sharpe': round(m['Sharpe'], 2),
                                     'n': m['n']})
                    done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done)
                    print(f"  [{done}/{total}] {label} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, 'upbit_champion_sweep.csv')
    df.to_csv(out_path, index=False)

    # Summary: top Holdout Cal (positive, consistent across splits)
    print("\n=== Top 10 Holdout Cal ===")
    hd = df[df['span'] == 'holdout'].sort_values('Cal', ascending=False).head(15)
    print(hd[['label', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'n']].to_string(index=False))

    # 3-span consistent (min Cal 기준)
    piv = df.pivot_table(index='label', columns='span', values='Cal')
    piv['min'] = piv.min(axis=1)
    piv = piv.sort_values('min', ascending=False).head(10)
    print("\n=== Top 10 3-span consistent (min Cal) ===")
    print(piv.to_string())

    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
