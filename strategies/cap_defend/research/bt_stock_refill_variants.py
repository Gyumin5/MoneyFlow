"""주식 refill 방식 교차 테스트: anchor-only vs refill(전량재선정,현행) vs slot(코인식 슬롯교체).

질문: 주식도 코인처럼 "3-mom 통과 보유는 유지, 탈락 슬롯만 교체"가 현행 전량재선정보다 나은가?
라이브 V25 주식 파라미터: ms=30 mid=72 ml=230, drift=0.05, buf=0.07, snap=69, n=3.
11 anchor 평균, 비용 1x/3x/5x (base TX).
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute, TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_refill_vs_anchor import run, metrics


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    MS, MID, ML = 30, 72, 230
    DRIFT, BUF = 0.05, 0.07
    SNAP_INT, N_SNAPS = 69, 3
    all_periods = sorted(set([MS, MID, ML, 30, 45, 84, 210]))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])
    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")

    MODES = ['anchor', 'refill', 'slot']
    print(f"# 주식 refill 방식 교차: ms={MS} mid={MID} ml={ML} drift={DRIFT} buf={BUF} snap={SNAP_INT} n={N_SNAPS}")
    print(f"# anchor=교체없음 / refill=전량재선정(현행) / slot=슬롯교체(코인식). 11 anchor 평균, base TX={TX}\n")
    print(f"  {'cost':>5} {'mode':<8} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'turnover/yr':>12} {'swaps':>7}")
    for mult in (1, 3, 5):
        tx = TX * mult
        for mode in MODES:
            agg, tov, swp = [], [], []
            for anchor in range(0, 11):
                r = run(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                        DRIFT, BUF, MS, MID, ML, SNAP_INT, N_SNAPS, mode, tx=tx)
                if r is None:
                    continue
                eq, turnover, swaps = r
                m = metrics(eq)
                if m is None:
                    continue
                agg.append(m); tov.append(turnover); swp.append(swaps)
            if not agg:
                print(f"  {mult}x    {mode:<8} (no data)"); continue
            cagr = np.mean([x[0] for x in agg]); dd = np.mean([x[1] for x in agg])
            cal = np.mean([x[2] for x in agg]); yrs = agg[0][3]
            tvy = np.mean(tov) / yrs; sw = np.mean(swp)
            star = '  *현행*' if mode == 'refill' else ''
            print(f"  {mult}x    {mode:<8} {cagr*100:>6.1f}% {dd*100:>6.1f}% {cal:>7.2f} {tvy:>11.2f}x {sw:>7.0f}{star}")
        print()
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
