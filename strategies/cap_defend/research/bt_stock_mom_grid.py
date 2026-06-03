"""Stock C mode — mom (ms, ml) 그리드 + thr — window rank-sum.

C base: multi-snap n=3 stag=23 int=69 + cap=1/3+Cash + buf 7%
Grid:
- ms ∈ {21, 30, 42, 63}
- ml ∈ {63, 84, 105, 126, 168, 210}  (ml > ms only)
- thr ∈ {0.05, 0.10}
B baseline: multi no mom, cap=1/3+Cash, buf 7%, thr=0.10
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute, half_t, CASH_KEY, TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_window_rank import _run_multi_eq, WIN_SIZES, STRIDES


def window_rank_sum_multi(eq_dict, win_sizes=WIN_SIZES, strides=STRIDES):
    common = None
    for s in eq_dict.values():
        if common is None: common = s.index
        else: common = common.intersection(s.index)
    common = sorted(common)
    if len(common) < max(win_sizes) + max(strides): return None
    sums = defaultdict(float); wins = defaultdict(int); total = 0
    keys = sorted(eq_dict.keys())
    for size in win_sizes:
        for stride in strides:
            starts = list(range(0, len(common) - size, stride))
            for s_idx in starts:
                d0 = common[s_idx]; d1 = common[s_idx + size - 1]
                cals = {}
                for k in keys:
                    seg = eq_dict[k].loc[d0:d1].dropna()
                    if len(seg) < 30: cals[k] = np.nan; continue
                    yrs = (seg.index[-1]-seg.index[0]).days/365.25
                    if yrs <= 0: cals[k] = np.nan; continue
                    cagr = (seg.iloc[-1]/seg.iloc[0])**(1/yrs)-1
                    peak = seg.cummax(); mdd = float((seg/peak-1).min())
                    cals[k] = cagr/abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()): continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1):
                    sums[mk] += r
                wins[ranked[0][0]] += 1
                total += 1
    return sums, wins, total


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    ms_list = [21, 30, 42, 63]
    ml_list = [63, 84, 105, 126, 168, 210]
    thr_list = [0.05, 0.10]
    combos = [(ms, ml, thr) for ms in ms_list for ml in ml_list if ml > ms for thr in thr_list]
    print(f"# combos: {len(combos)} + B baseline")

    # precompute mom for full grid — off_periods must include ALL mom values used
    all_periods = sorted(set(ms_list + ml_list))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    # build equity for each combo + B
    sums_all = defaultdict(float); wins_all = defaultdict(int); n_all = 0
    per_anchor_log = []
    print("Building equity per anchor + combo...")
    for anchor in range(0, 11):
        eqs = {}
        # B baseline
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eqs['B'] = eq_B
        for ms, ml, thr in combos:
            key = f"C_ms{ms}_ml{ml}_thr{thr:.2f}"
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
        per_anchor_log.append((anchor, dict(sums), dict(wins), n))

    print("\nWindow rank-sum (1518 win × 11 anchors), lower=better")
    print(f"  total windows: {n_all}, total cfgs: {len(combos)+1}")
    print(f"  {'cfg':<25} {'rank_sum':>10} {'avg_rank':>9} {'wins':>6} {'win%':>6}")
    items = sorted(sums_all.items(), key=lambda x: x[1])
    for k, rs in items:
        print(f"  {k:<25} {rs:>10.0f} {rs/n_all:>9.3f} {wins_all[k]:>6d} {wins_all[k]/n_all*100:>5.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
