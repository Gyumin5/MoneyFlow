"""주식 single-snap + mom + refill — PRIME stagger only.

stagger prime 후보 (exclude 19=fut, 31=spot): {7, 11, 13, 17, 23, 29, 37, 41, 43, 47, 53}
multi-snap 도 같이 (n=3) — 현행 V24 (n=3 stag=23 int=69) 와 비교.
single-snap: n=1, snap_int = stagger.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, combine, half_t,
                              CASH_KEY, TX, PERIODS, N_ANCHORS)
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_single_snap import run_one

PRIMES = [7, 11, 13, 17, 23, 29, 37, 41, 43, 47, 53]  # exclude 19/31


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 84], [42, 63, 126])

    print("=" * 100)
    print("Stock single-snap (N=1, snap_int=stagger=prime) + mom30/84 + refill + cap=1/3+Cash + 7%buf")
    print("=" * 100)
    print(f"  {'cfg':<35} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>7} {'MDD':>7}")
    print("-" * 100)
    results = []
    for stagger in PRIMES:
        for thr in (0.05, 0.10):
            for buf in (0.0, 0.07):
                for mode in ('cap', 'ew'):
                    cls = []; cas = []; mds = []
                    for start, end in PERIODS:
                        sd = pd.Timestamp(start); ed = pd.Timestamp(end)
                        rs = []
                        for a in range(N_ANCHORS):
                            r = run_one(pdf, ranked, mom_off, mom_def, canary,
                                       sd, ed, a, 30, 84, thr, buf, mode, stagger)
                            if r: rs.append(r)
                        if rs:
                            cls.append(float(np.mean([r['Cal'] for r in rs])))
                            cas.append(float(np.mean([r['CAGR'] for r in rs])))
                            mds.append(float(np.mean([r['MDD'] for r in rs])))
                    if len(cls) == 4:
                        avg = float(np.mean(cls))
                        label = f"stag={stagger} thr={thr:.2f} buf={int(buf*100)}% w={mode}"
                        print(f"  {label:<35} {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} {avg:>6.2f} {np.mean(cas)*100:>6.1f}% {np.mean(mds)*100:>7.1f}%")
                        results.append((stagger, thr, buf, mode, cls, avg, float(np.mean(cas)), float(np.mean(mds))))

    print()
    print("=" * 100)
    print("TOP 10 by avg Cal")
    print("=" * 100)
    results.sort(key=lambda r: -r[5])
    for r in results[:10]:
        stagger, thr, buf, mode, cls, avg, cagr, mdd = r
        print(f"  stag={stagger} thr={thr:.2f} buf={int(buf*100)}% w={mode}  avg={avg:.2f}  per-period={cls}  CAGR={cagr*100:.1f}% MDD={mdd*100:.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
