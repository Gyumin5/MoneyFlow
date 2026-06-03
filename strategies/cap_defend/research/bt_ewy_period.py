"""EWY (Korea) period 분해 — 최근 1년 제외 시 우위 유지 여부.

비교: R7 vs R7+EWY
- 5.4yr 전체 (2017-01-01 ~ 2026-05-13)
- end=2025-05-13 (최근 1년 제외)
- end=2024-05-13 (최근 2년 제외)
- end=2023-05-13 (최근 3년 제외)
- end=2020-05-13 (코로나 이후 제외)

3-mom (30/72/230), cap+Cash, multi-snap n=3 stag=23 int=69, thr=0.05.
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
import bt_stock_coin_v3 as bcv3
from bt_stock_coin_v3 import precompute
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_mom_grid import window_rank_sum_multi
from bt_stock_mom3 import run_multi_3mom


R7_BASE = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")


def main():
    t0 = time.time()
    all_tickers = sorted(set(list(R7_BASE) + list(bcv3.DEF_TICKERS) + ['EEM', 'EWY'] + list(ALL_TICKERS)))
    pm = load_prices(all_tickers, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    periods = [
        ('전체', '2017-01-01', '2026-05-13'),
        ('최근1y제외', '2017-01-01', '2025-05-13'),
        ('최근2y제외', '2017-01-01', '2024-05-13'),
        ('최근3y제외', '2017-01-01', '2023-05-13'),
        ('코로나후제외', '2017-01-01', '2020-05-13'),
    ]

    cfgs = [
        ('R7', list(R7_BASE)),
        ('R7+EWY', list(R7_BASE) + ['EWY']),
    ]

    for label, sd_s, ed_s in periods:
        sd = pd.Timestamp(sd_s); ed = pd.Timestamp(ed_s)
        sums_all = defaultdict(float); n_all = 0
        for tag, universe in cfgs:
            bcv3.OFF_R7 = tuple(universe)
            try:
                ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
            except Exception as e:
                continue
            for anchor in range(0, 11):
                eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                   drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230)
                if eq is not None:
                    # 단일 mode/anchor 에서 R7 vs R7+EWY rank-sum 비교 위해 메모리 보관
                    pass
        # rerun with parallel eqs
        per_anchor = []
        for anchor in range(0, 11):
            eqs = {}
            for tag, universe in cfgs:
                bcv3.OFF_R7 = tuple(universe)
                ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
                eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                   drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230)
                if eq is not None:
                    eqs[tag] = eq
            if len(eqs) < 2: continue
            rs = window_rank_sum_multi(eqs)
            if rs is None: continue
            sums, _, n = rs
            for k, v in sums.items(): sums_all[k] += v
            n_all += n
        if n_all == 0:
            print(f"\n[{label}] {sd_s}~{ed_s}: 데이터 부족")
            continue
        print(f"\n[{label}] {sd_s}~{ed_s}: n_windows={n_all}")
        for k, rs in sorted(sums_all.items(), key=lambda x: x[1]):
            print(f"  {k:<10} avg_rank={rs/n_all:.3f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
