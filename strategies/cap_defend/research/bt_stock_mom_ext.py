"""Extension grid — extend both peaks outward.

Peak A region: ms 21~36, ml 42~96 (extend ml down + ms down)
Peak B region: ms 36~57, ml 189~300 (extend ml up + ms up)
Skip middle (ml 105~168) — known dead zone.
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_window_rank import _run_multi_eq
from bt_stock_mom_grid import window_rank_sum_multi


PEAK_A_MS = [18, 21, 24, 27, 30, 33, 36]
PEAK_A_ML = [42, 50, 60, 72, 84, 96]
PEAK_B_MS = [36, 39, 42, 45, 48, 51, 54, 57]
PEAK_B_ML = [189, 200, 210, 220, 230, 240, 260, 280, 300]
THR_GRID = [0.05, 0.10]


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    combos = []
    for ms in PEAK_A_MS:
        for ml in PEAK_A_ML:
            if ml > ms:
                for thr in THR_GRID: combos.append((ms, ml, thr))
    for ms in PEAK_B_MS:
        for ml in PEAK_B_ML:
            if ml > ms:
                for thr in THR_GRID: combos.append((ms, ml, thr))
    combos = list(set(combos))
    print(f"# combos: {len(combos)} (+B baseline)")

    all_periods = sorted(set(PEAK_A_MS + PEAK_A_ML + PEAK_B_MS + PEAK_B_ML))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    sums_all = defaultdict(float); wins_all = defaultdict(int); n_all = 0

    for anchor in range(0, 11):
        eqs = {}
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eqs['B'] = eq_B
        for ms, ml, thr in combos:
            key = f"ms{ms:02d}_ml{ml:03d}_thr{thr:.2f}"
            eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                               use_mom=True, drift_thr=thr, cash_buf=0.07, weight_mode='cap',
                               ms=ms, ml=ml)
            if eq is not None: eqs[key] = eq
        rs = window_rank_sum_multi(eqs)
        if rs is None: continue
        sums, wins, n = rs
        for k, v in sums.items(): sums_all[k] += v
        for k, v in wins.items(): wins_all[k] += v
        n_all += n

    items = sorted(sums_all.items(), key=lambda x: x[1])
    print(f"\nTotal windows: {n_all}, total cfgs: {len(combos)+1}")
    print(f"\nTOP 30:")
    print(f"  {'cfg':<28} {'avg_rank':>9} {'win%':>6}")
    for k, rs in items[:30]:
        print(f"  {k:<28} {rs/n_all:>9.3f} {wins_all[k]/n_all*100:>5.1f}%")
    b_rank = next((i for i, (k, _) in enumerate(items, 1) if k == 'B'), -1)
    print(f"\nB baseline at rank {b_rank} (avg_rank={sums_all['B']/n_all:.3f})")

    # Peak A 2D
    print("\n2D MAP Peak A (thr=0.10):")
    print("  ms\\ml ", end='')
    for ml in PEAK_A_ML: print(f"{ml:>7}", end='')
    print()
    for ms in PEAK_A_MS:
        print(f"  {ms:<6}", end='')
        for ml in PEAK_A_ML:
            if ml <= ms: print(f"{'-':>7}", end='')
            else:
                key = f"ms{ms:02d}_ml{ml:03d}_thr0.10"
                v = sums_all.get(key, np.nan)/n_all if key in sums_all else np.nan
                print(f"{v:>7.2f}", end='')
        print()

    print("\n2D MAP Peak B (thr=0.10):")
    print("  ms\\ml ", end='')
    for ml in PEAK_B_ML: print(f"{ml:>7}", end='')
    print()
    for ms in PEAK_B_MS:
        print(f"  {ms:<6}", end='')
        for ml in PEAK_B_ML:
            if ml <= ms: print(f"{'-':>7}", end='')
            else:
                key = f"ms{ms:02d}_ml{ml:03d}_thr0.10"
                v = sums_all.get(key, np.nan)/n_all if key in sums_all else np.nan
                print(f"{v:>7.2f}", end='')
        print()

    # Plateau test for top 5
    print("\n--- Plateau top 5 (±1 step ms/ml, same thr) ---")
    for k, rs in items[:5]:
        if k == 'B': continue
        parts = k.split('_')
        ms = int(parts[0][2:]); ml = int(parts[1][2:]); thr = float(parts[2][3:])
        # determine which peak region
        if ml <= 96:
            mlg = PEAK_A_ML; msg = PEAK_A_MS
        else:
            mlg = PEAK_B_ML; msg = PEAK_B_MS
        try:
            mi = msg.index(ms); li = mlg.index(ml)
        except ValueError: continue
        nbrs = []
        for dm in (-1, 0, 1):
            for dl in (-1, 0, 1):
                if dm == 0 and dl == 0: continue
                ni = mi + dm; nj = li + dl
                if 0 <= ni < len(msg) and 0 <= nj < len(mlg):
                    nms, nml = msg[ni], mlg[nj]
                    if nml > nms:
                        nk = f"ms{nms:02d}_ml{nml:03d}_thr{thr:.2f}"
                        if nk in sums_all: nbrs.append((nk, sums_all[nk]/n_all))
        if nbrs:
            avg = float(np.mean([v for _, v in nbrs]))
            spread = max(v for _, v in nbrs) - min(v for _, v in nbrs)
            print(f"  PEAK {k} ({rs/n_all:.2f}) → nbr avg={avg:.2f} spread={spread:.2f} n={len(nbrs)}")
            for nk, nv in nbrs: print(f"    {nk:<28} {nv:.3f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
