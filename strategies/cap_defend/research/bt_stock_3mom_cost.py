"""GATE 4 — cost stress (tx 0.06%, 0.10%, 0.12%, 0.15%, 0.20%)."""
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
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    all_periods = sorted(set([30, 42, 45, 72, 84, 210, 230]))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    cfgs = [
        ('3m_30_72_230', '3m', 30, 72, 230),
        ('3m_42_72_230', '3m', 42, 72, 230),
        ('2m_45_210',    '2m', 45, None, 210),
        ('2m_30_84',     '2m', 30, None, 84),
    ]

    tx_list = [0.0006, 0.0010, 0.0012, 0.0015, 0.0020, 0.0030]
    print(f"{'tx':<8} " + "  ".join(f"{n:>15}" for n in ['B'] + [c[0] for c in cfgs]))
    print(f"{'(bp)':<8} " + "  ".join(f"{'avg_rank':>15}" for _ in range(len(cfgs)+1)))
    print("-" * 110)

    for tx_val in tx_list:
        # monkey-patch TX in dependency modules
        bcv3.TX = tx_val
        # also bt_stock_window_rank, bt_stock_mom3
        import bt_stock_window_rank as bwr
        import bt_stock_mom3 as bm3
        bwr.TX = tx_val
        bm3.TX = tx_val

        sums_all = defaultdict(float); n_all = 0
        for anchor in range(0, 11):
            eqs = {}
            eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                 use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
            if eq_B is None: continue
            eqs['B'] = eq_B
            for tag, kind, ms, mid, ml in cfgs:
                if kind == '3m':
                    eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                                       sd, ed, anchor,
                                       0.05, 0.07, ms, mid, ml)
                else:
                    eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                                       use_mom=True, drift_thr=0.05, cash_buf=0.07,
                                       weight_mode='cap', ms=ms, ml=ml)
                if eq is not None: eqs[tag] = eq
            rs = window_rank_sum_multi(eqs)
            if rs is None: continue
            sums, _, n = rs
            for k, v in sums.items(): sums_all[k] += v
            n_all += n
        row = " ".join(f"{sums_all[k]/n_all:>15.3f}" for k in ['B'] + [c[0] for c in cfgs])
        print(f"{tx_val*1e4:>5.1f}bp  " + row)

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
