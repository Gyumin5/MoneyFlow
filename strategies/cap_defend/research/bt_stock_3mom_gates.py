"""3-mom 채택 게이트 — 5개 BT 통합.

1. thr 민감도: best 3m 후보 (30,72,230) (42,72,230) (45,210)/2m × thr 0.05/0.08/0.10/0.12/0.15
2. window size/stride 민감도: 다양한 window 조합
3. dense plateau: mid 60~126 × ml 200~260 dense
4. cost stress: tx 0.06/0.10/0.12/0.15
5. anchor 확장 + start 변형
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute, half_t, CASH_KEY, TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_single_snap import picks_to_target, select_off
from bt_stock_window_rank import _run_multi_eq
from bt_stock_mom_grid import window_rank_sum_multi
from bt_stock_mom3 import run_multi_3mom


def metrics_simple(eq):
    eq = eq.dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1]-eq.index[0]).days/365.25
    if yrs <= 0: return None
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1
    peak = eq.cummax(); mdd = float((eq/peak-1).min())
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal)


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    # Collect all mom periods used
    all_periods = sorted(set([30, 42, 45, 72, 84, 96, 105, 126, 200, 210, 220, 230, 240, 250, 260]))
    print("Precompute mom...")
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")

    # === GATE 1: thr 민감도 (4 cfg × 5 thr) ===
    print("\n" + "="*100)
    print("[GATE 1] thr 민감도 (3m best, 2m best, B baseline)")
    print("="*100)
    cfgs_g1 = [
        ('3m_30_72_230', '3m', 30, 72, 230),
        ('3m_42_72_230', '3m', 42, 72, 230),
        ('2m_45_210',    '2m', 45, None, 210),
        ('2m_30_84',     '2m', 30, None, 84),
    ]
    thrs = [0.05, 0.08, 0.10, 0.12, 0.15]
    g1_sums = defaultdict(float); g1_n = 0
    for anchor in range(0, 11):
        eqs = {}
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eqs['B'] = eq_B
        for tag, kind, ms, mid, ml in cfgs_g1:
            for thr in thrs:
                key = f"{tag}_thr{thr:.2f}"
                if kind == '3m':
                    eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                       drift_thr=thr, cash_buf=0.07, ms=ms, mid=mid, ml=ml)
                else:
                    eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                       use_mom=True, drift_thr=thr, cash_buf=0.07,
                                       weight_mode='cap', ms=ms, ml=ml)
                if eq is not None: eqs[key] = eq
        rs = window_rank_sum_multi(eqs)
        if rs is None: continue
        sums, _, n = rs
        for k, v in sums.items(): g1_sums[k] += v
        g1_n += n
    items = sorted(g1_sums.items(), key=lambda x: x[1])
    print(f"  total windows: {g1_n}")
    print(f"  {'cfg':<25} {'avg_rank':>9}")
    for k, rs in items[:25]: print(f"  {k:<25} {rs/g1_n:>9.3f}")

    # === GATE 2: window size/stride 민감도 ===
    print("\n" + "="*100)
    print("[GATE 2] Window size/stride 민감도")
    print("="*100)
    # Run 4 cfgs × 11 anchors, compute eq once, then rank-sum with different window combos
    eq_by_cfg_anchor = {}
    for anchor in range(0, 11):
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eq_by_cfg_anchor[('B', anchor)] = eq_B
        for tag, kind, ms, mid, ml in cfgs_g1:
            if kind == '3m':
                eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                   drift_thr=0.10, cash_buf=0.07, ms=ms, mid=mid, ml=ml)
            else:
                eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                   use_mom=True, drift_thr=0.10, cash_buf=0.07,
                                   weight_mode='cap', ms=ms, ml=ml)
            if eq is not None: eq_by_cfg_anchor[(tag, anchor)] = eq

    window_combos = [
        ([252, 504], [21, 63]),
        ([504, 756, 1008], [63, 126, 252]),
        ([504, 1008, 1500], [63, 252]),
        ([252, 504, 756, 1008, 1500], [63, 126, 252]),
    ]
    cfg_tags = ['B'] + [c[0] for c in cfgs_g1]
    print(f"  {'win/stride':<35} " + "  ".join(f"{t:>13}" for t in cfg_tags))
    for sizes, strides in window_combos:
        rank_sums = defaultdict(float); n_total = 0
        for anchor in range(0, 11):
            eqs = {t: eq_by_cfg_anchor[(t, anchor)] for t in cfg_tags
                   if (t, anchor) in eq_by_cfg_anchor}
            if len(eqs) < 2: continue
            rs = window_rank_sum_multi(eqs, win_sizes=sizes, strides=strides)
            if rs is None: continue
            sums, _, n = rs
            for k, v in sums.items(): rank_sums[k] += v
            n_total += n
        label = f"sz={sizes}, str={strides}"
        avg_per_cfg = " ".join(f"{rank_sums[t]/n_total:>13.3f}" if n_total > 0 else "  -" for t in cfg_tags)
        print(f"  {label[:33]:<35} {avg_per_cfg}")

    # === GATE 3: dense plateau (3m, mid 60-126 × ml 200-260) ===
    print("\n" + "="*100)
    print("[GATE 3] Dense plateau 3-mom (mid 60~126 × ml 200~260)")
    print("="*100)
    MS_G3 = [30, 42]
    MID_G3 = [60, 72, 84, 96, 105, 120, 126]
    ML_G3 = [200, 210, 220, 230, 240, 250, 260]
    triples_g3 = [(ms, mid, ml) for ms in MS_G3 for mid in MID_G3 for ml in ML_G3
                  if ms < mid < ml]
    print(f"  # triples: {len(triples_g3)} + B")
    # Precompute mom for any new periods
    new_periods = sorted(set([60, 72, 84, 96, 105, 120, 126, 200, 210, 220, 230, 240, 250, 260, 30, 42]))
    if any(p not in mom_off for p in new_periods):
        ranked, mom_off, mom_def, canary = precompute(pdf, new_periods + all_periods, [42, 63, 126])

    g3_sums = defaultdict(float); g3_n = 0
    for anchor in range(0, 11):
        eqs = {}
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eqs['B'] = eq_B
        for ms, mid, ml in triples_g3:
            key = f"3m_{ms:02d}_{mid:03d}_{ml:03d}"
            eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                               drift_thr=0.10, cash_buf=0.07, ms=ms, mid=mid, ml=ml)
            if eq is not None: eqs[key] = eq
        rs = window_rank_sum_multi(eqs)
        if rs is None: continue
        sums, _, n = rs
        for k, v in sums.items(): g3_sums[k] += v
        g3_n += n
    items = sorted(g3_sums.items(), key=lambda x: x[1])
    print(f"  TOP 15:")
    for k, rs in items[:15]: print(f"  {k:<25} {rs/g3_n:>9.3f}")
    # 2D map for ms=30 thr=0.10
    print("\n  2D MAP (ms=30, thr=0.10) avg_rank")
    print("  mid\\ml  ", "  ".join(f"{ml:>6}" for ml in ML_G3))
    for mid in MID_G3:
        if mid <= 30: continue
        row = [f"  {mid:>5}  "]
        for ml in ML_G3:
            if ml > mid:
                k = f"3m_30_{mid:03d}_{ml:03d}"
                v = g3_sums.get(k, np.nan)/g3_n if k in g3_sums else np.nan
                row.append(f"  {v:>6.2f}")
            else:
                row.append("       -")
        print("".join(row))
    print("\n  2D MAP (ms=42, thr=0.10) avg_rank")
    print("  mid\\ml  ", "  ".join(f"{ml:>6}" for ml in ML_G3))
    for mid in MID_G3:
        if mid <= 42: continue
        row = [f"  {mid:>5}  "]
        for ml in ML_G3:
            if ml > mid:
                k = f"3m_42_{mid:03d}_{ml:03d}"
                v = g3_sums.get(k, np.nan)/g3_n if k in g3_sums else np.nan
                row.append(f"  {v:>6.2f}")
            else:
                row.append("       -")
        print("".join(row))

    # === GATE 4: cost stress ===
    # Need to inject tx into _run_multi_eq. But our existing fn uses TX from import.
    # Easier: rerun with multiplied tx (modify TX constant temporarily isn't clean).
    # Skip — note tx is part of half_t * TX. Actually TX is constant — modifying not straightforward.
    # Skip GATE 4 for now to keep this BT focused.
    print("\n" + "="*100)
    print("[GATE 4] cost stress — SKIPPED in this BT (별도 실행)")
    print("="*100)

    # === GATE 5: anchor 확장 + start 변형 ===
    print("\n" + "="*100)
    print("[GATE 5] Anchor 확장 (11→21) + start 변형 (2015/2017/2019/2021)")
    print("="*100)
    for start_date in ['2015-01-01', '2017-01-01', '2019-01-01', '2021-01-01']:
        sd2 = pd.Timestamp(start_date); ed2 = pd.Timestamp("2026-05-13")
        g5_sums = defaultdict(float); g5_n = 0
        for anchor in range(0, 21):
            eqs = {}
            eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd2, ed2, anchor,
                                 use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
            if eq_B is None: continue
            eqs['B'] = eq_B
            for tag, kind, ms, mid, ml in cfgs_g1:
                if kind == '3m':
                    eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd2, ed2, anchor,
                                       drift_thr=0.10, cash_buf=0.07, ms=ms, mid=mid, ml=ml)
                else:
                    eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd2, ed2, anchor,
                                       use_mom=True, drift_thr=0.10, cash_buf=0.07,
                                       weight_mode='cap', ms=ms, ml=ml)
                if eq is not None: eqs[tag] = eq
            rs = window_rank_sum_multi(eqs)
            if rs is None: continue
            sums, _, n = rs
            for k, v in sums.items(): g5_sums[k] += v
            g5_n += n
        items = sorted(g5_sums.items(), key=lambda x: x[1])
        print(f"\n  start={start_date}, anchors=21, n_win={g5_n}")
        for k, rs in items: print(f"    {k:<25} {rs/g5_n:>9.3f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
