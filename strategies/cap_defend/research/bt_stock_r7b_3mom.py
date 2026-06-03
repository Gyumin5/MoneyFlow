"""R7B universe (EWJ instead of VNQ) — 3-mom 결과 재검증."""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
import bt_stock_coin_v3 as bcv3
from bt_stock_coin_v3 import precompute, CASH_KEY
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_window_rank import _run_multi_eq
from bt_stock_mom_grid import window_rank_sum_multi
from bt_stock_mom3 import run_multi_3mom


def main():
    t0 = time.time()
    # Monkey-patch OFF_R7 to R7B (EWJ instead of VNQ)
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "EWJ", "GLD", "PDBC")
    print(f"OFF universe → {bcv3.OFF_R7}")

    pm = load_prices(list(ALL_TICKERS) + ['EWJ'], start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    print(f"Universe in pdf: {[t for t in bcv3.OFF_R7 if t in pdf.columns]}")
    if 'EWJ' not in pdf.columns:
        print("ERROR: EWJ not in price data")
        return

    all_periods = sorted(set([30, 42, 45, 72, 84, 210, 230]))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    cfgs = [
        ('3m_30_72_230', '3m', 30, 72, 230),
        ('3m_42_72_230', '3m', 42, 72, 230),
        ('2m_45_210',    '2m', 45, None, 210),
        ('2m_30_84',     '2m', 30, None, 84),
    ]

    sums_all = defaultdict(float); wins_all = defaultdict(int); n_all = 0
    for anchor in range(0, 11):
        eqs = {}
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eqs['B'] = eq_B
        for tag, kind, ms, mid, ml in cfgs:
            if kind == '3m':
                eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                   0.05, 0.07, ms, mid, ml)
            else:
                eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                   use_mom=True, drift_thr=0.05, cash_buf=0.07,
                                   weight_mode='cap', ms=ms, ml=ml)
            if eq is not None: eqs[tag] = eq
        rs = window_rank_sum_multi(eqs)
        if rs is None: continue
        sums, wins, n = rs
        for k, v in sums.items(): sums_all[k] += v
        for k, v in wins.items(): wins_all[k] += v
        n_all += n

    items = sorted(sums_all.items(), key=lambda x: x[1])
    print(f"\nR7B (EWJ) — n_windows={n_all}, n_cfgs={len(cfgs)+1}")
    print(f"  {'cfg':<25} {'avg_rank':>9} {'win%':>6}")
    for k, rs in items:
        print(f"  {k:<25} {rs/n_all:>9.3f} {wins_all[k]/n_all*100:>5.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
