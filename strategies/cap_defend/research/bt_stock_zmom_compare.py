"""주식 Z랭킹 모멘텀 정의 비교: weighted(0.5/0.3/0.2 of 63/126/252, 채택BT) vs pure252(라이브 현행).

질문: 어느 랭킹이 풀 BT 에서 실제로 나은가? (가중이 BT 최선이었나 확인)
state machine = refill 모드(라이브 동일). 11 anchor 평균, 비용 1x/3x/5x.
ranking 만 STOCK_ZMOM 환경변수로 토글 (bt_stock_coin_v3.precompute).
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
import bt_stock_coin_v3 as v3
from bt_stock_coin_v3 import TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_refill_vs_anchor import run, metrics

MS, MID, ML = 30, 72, 230
DRIFT, BUF = 0.05, 0.07
SNAP_INT, N_SNAPS = 69, 3


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    all_periods = sorted(set([MS, MID, ML, 30, 45, 84, 210]))
    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")

    print(f"# 주식 Z랭킹 모멘텀 비교 (refill 모드=라이브, 11 anchor, base TX={TX})")
    print(f"# weighted=채택BT / pure252=라이브현행. drift={DRIFT} buf={BUF} snap={SNAP_INT} n={N_SNAPS}\n")
    print(f"  {'cost':>5} {'zmom':<9} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'turnover/yr':>12}")
    for mult in (1, 3, 5):
        tx = TX * mult
        for zmom in ('weighted', 'pure252'):
            os.environ['STOCK_ZMOM'] = zmom
            ranked, mom_off, mom_def, canary = v3.precompute(pdf, all_periods, [42, 63, 126])
            agg, tov = [], []
            for anchor in range(0, 11):
                r = run(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                        DRIFT, BUF, MS, MID, ML, SNAP_INT, N_SNAPS, 'refill', tx=tx)
                if r is None:
                    continue
                eq, turnover, swaps = r
                m = metrics(eq)
                if m is None:
                    continue
                agg.append(m); tov.append(turnover)
            if not agg:
                print(f"  {mult}x    {zmom:<9} (no data)"); continue
            cagr = np.mean([x[0] for x in agg]); dd = np.mean([x[1] for x in agg])
            cal = np.mean([x[2] for x in agg]); yrs = agg[0][3]
            tvy = np.mean(tov) / yrs
            star = '  *채택*' if zmom == 'weighted' else '  *라이브*'
            print(f"  {mult}x    {zmom:<9} {cagr*100:>6.1f}% {dd*100:>6.1f}% {cal:>7.2f} {tvy:>11.2f}x{star}")
        print()
    os.environ.pop('STOCK_ZMOM', None)
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
