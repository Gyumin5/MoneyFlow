"""Dense grid around 2 peaks found in coarse:
  Peak A: (ms=30, ml=84) — short-mid
  Peak B: (ms=42, ml=210) — short-long

Plateau test: 3-neighbor avg rank near peak rank.
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


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    # Dense grids around 2 peaks
    peak_A = [(ms, ml, thr) for ms in [24, 27, 30, 33, 36, 39]
              for ml in [72, 78, 84, 90, 96, 105]
              for thr in [0.05, 0.08, 0.10] if ml > ms]
    peak_B = [(ms, ml, thr) for ms in [36, 39, 42, 45, 48, 51]
              for ml in [189, 200, 210, 220, 230]
              for thr in [0.05, 0.08, 0.10] if ml > ms]
    combos = list(set(peak_A + peak_B))
    print(f"# combos: {len(combos)} (Peak A {len(set(peak_A))} + Peak B {len(set(peak_B))})")

    all_periods = sorted(set([ms for ms,_,_ in combos] + [ml for _,ml,_ in combos]))
    print(f"Mom periods to compute: {all_periods}")
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
            key = f"ms{ms}_ml{ml}_thr{thr:.2f}"
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
    print(f"\nTOP 20 (lower rank_sum = better):")
    print(f"  {'cfg':<25} {'rank_sum':>10} {'avg_rank':>9} {'wins':>6} {'win%':>6}")
    for k, rs in items[:20]:
        print(f"  {k:<25} {rs:>10.0f} {rs/n_all:>9.3f} {wins_all[k]:>6d} {wins_all[k]/n_all*100:>5.1f}%")

    # find peaks within Peak A / Peak B subsets
    print("\n--- Peak A region (ms ≤ 39, ml ≤ 105) top 5 ---")
    a_items = [(k, v) for k, v in items if k != 'B' and any(f"_ml{ml}_" in k for ml in [72,78,84,90,96,105])]
    for k, rs in a_items[:5]:
        print(f"  {k:<25} rank_sum={rs:.0f} avg_rank={rs/n_all:.3f}")

    print("\n--- Peak B region (ms 36-51, ml 189-230) top 5 ---")
    b_items = [(k, v) for k, v in items if k != 'B' and any(f"_ml{ml}_" in k for ml in [189,200,210,220,230])]
    for k, rs in b_items[:5]:
        print(f"  {k:<25} rank_sum={rs:.0f} avg_rank={rs/n_all:.3f}")

    # Plateau test: for each top cfg, look at 4 immediate neighbors (±1 step ms, ±1 step ml)
    print("\n--- Plateau test ---")
    # Build cfg map
    rank_map = {k: v for k, v in items}
    def neighbors_of(ms, ml, thr, step_ms_list, step_ml_list):
        # find nearest ms/ml in lists
        neighbors = []
        try:
            ms_i = step_ms_list.index(ms)
            ml_i = step_ml_list.index(ml)
        except ValueError:
            return []
        for dms in (-1, 1):
            for dml in (-1, 1):
                ni = ms_i + dms; nj = ml_i + dml
                if 0 <= ni < len(step_ms_list) and 0 <= nj < len(step_ml_list):
                    nms, nml = step_ms_list[ni], step_ml_list[nj]
                    if nml > nms:
                        nk = f"ms{nms}_ml{nml}_thr{thr:.2f}"
                        if nk in rank_map:
                            neighbors.append((nk, rank_map[nk]))
        return neighbors

    print("\nPeak A neighbors (steps ms[24,27,30,33,36,39], ml[72,78,84,90,96,105]):")
    a_top = a_items[:3]
    for k, rs in a_top:
        parts = k.split('_')
        ms = int(parts[0][2:]); ml = int(parts[1][2:]); thr = float(parts[2][3:])
        nbrs = neighbors_of(ms, ml, thr, [24,27,30,33,36,39], [72,78,84,90,96,105])
        peak_rank = rs / n_all
        nbr_ranks = [v/n_all for _, v in nbrs]
        spread = max(nbr_ranks) - min(nbr_ranks) if nbr_ranks else 0
        avg = float(np.mean(nbr_ranks)) if nbr_ranks else 0
        print(f"  PEAK {k} ({peak_rank:.2f}) → neighbors avg={avg:.2f} spread={spread:.2f} count={len(nbrs)}")
        for nk, nv in nbrs: print(f"    {nk:<25} {nv/n_all:.3f}")

    print("\nPeak B neighbors (steps ms[36,39,42,45,48,51], ml[189,200,210,220,230]):")
    b_top = b_items[:3]
    for k, rs in b_top:
        parts = k.split('_')
        ms = int(parts[0][2:]); ml = int(parts[1][2:]); thr = float(parts[2][3:])
        nbrs = neighbors_of(ms, ml, thr, [36,39,42,45,48,51], [189,200,210,220,230])
        peak_rank = rs / n_all
        nbr_ranks = [v/n_all for _, v in nbrs]
        spread = max(nbr_ranks) - min(nbr_ranks) if nbr_ranks else 0
        avg = float(np.mean(nbr_ranks)) if nbr_ranks else 0
        print(f"  PEAK {k} ({peak_rank:.2f}) → neighbors avg={avg:.2f} spread={spread:.2f} count={len(nbrs)}")
        for nk, nv in nbrs: print(f"    {nk:<25} {nv/n_all:.3f}")

    print(f"\nB baseline rank: {sums_all.get('B', 0)/n_all:.3f}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
