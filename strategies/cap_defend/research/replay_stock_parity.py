"""주식 replay-diff: 라이브 선정함수(stock_strategy_v25.compute_offense) vs
BT 선정(bt_stock_coin_v3.precompute + bt_stock_mom3.fresh_pick_3mom) 일별 대조.

같은 가격 패널, 같은 날짜 컷오프(d 까지)로 두 코드의 offense picks 를 매일 산출해 diff.
canary-on(offense) 날만 비교. 불일치 일자/패턴 보고.

알려진 잠재 차이: Z-rank 모멘텀
  라이브 = 순수 12M(252d) return
  BT     = 가중 0.5*63 + 0.3*126 + 0.2*252
3-mom 필터(30/72/230)·top3·cap·universe·canary 는 동일.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_mom3 import fresh_pick_3mom
import stock_strategy_v25 as live

MS, MID, ML = 30, 72, 230


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    all_periods = sorted(set([MS, MID, ML, 30, 45, 84, 210]))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    dates = pdf.index[(pdf.index >= sd) & (pdf.index <= ed)]

    # 라이브 입력용 시리즈 (offense universe)
    series = {t: pdf[t] for t in live.OFFENSIVE_STOCK_UNIVERSE if t in pdf.columns}

    n_on = 0
    n_match_set = 0
    n_match_order = 0
    mismatches = []
    for d in dates:
        if not bool(canary.at[d]) if d in canary.index else True:
            continue  # canary off (defense) — offense 비교 대상 아님
        n_on += 1
        # BT picks
        bt = fresh_pick_3mom(ranked.at[d], mom_off[MS].loc[d], mom_off[MID].loc[d], mom_off[ML].loc[d])
        # LIVE picks (가격 d 까지 컷오프)
        ap = {t: s.loc[:d] for t, s in series.items()}
        lv, _, _ = live.compute_offense(ap)
        if set(bt) == set(lv):
            n_match_set += 1
            if list(bt) == list(lv):
                n_match_order += 1
        else:
            if len(mismatches) < 25:
                mismatches.append((d.date(), list(bt), list(lv)))

    print(f"# 주식 선정 replay-diff (canary-on 일 {n_on}일, {sd.date()}~{ed.date()})")
    print(f"  종목집합 일치: {n_match_set}/{n_on} ({n_match_set/n_on*100:.1f}%)")
    print(f"  순서까지 일치: {n_match_order}/{n_on} ({n_match_order/n_on*100:.1f}%)")
    print(f"  불일치: {n_on - n_match_set}일")
    if mismatches:
        print("\n  불일치 샘플 (날짜 | BT picks | LIVE picks):")
        for dt_, bt, lv in mismatches:
            print(f"   {dt_} | BT={bt} | LIVE={lv}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
