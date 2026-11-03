"""현물 종목 둘의 효율 이득이 전체 합성에서도 살아나나 — 제한된 측정.

왜 재나. 현물 슬리브를 종목 셋에서 둘로 줄이면 상한 1/3 이 그대로라 최대 노출이
2/3 로 줄고 나머지는 현금이 된다. 노출당 수익은 그래서 올라간다. 그런데 그 이득이
전체 합성 수익으로 살아나는지는 별개다 — 줄어든 노출을 현물 배분 비중으로 메우면
그 몫만큼 다른 슬리브를 포기하기 때문이다. history #147 이 "제한된 합성 측정은
값어치 있다" 로 닫은 자리다.

무엇을 고정하고 시작하나 (결과를 보기 전에 정한다 — 가장 좋은 칸을 골라 격자로
넓히면 누적 선택 편향이 결론을 부풀린다).

  칸 여섯 = 배분 셋 × 종목수 둘
    60/20/20  (현행 라이브, 현물:선물 1대1)
    60/25/15  (현물을 1.25배)
    60/30/10  (현물을 1.5배 — 종목 둘의 노출 손실 1/3 을 정확히 메우는 비율)
    각각 현물 종목 셋(universe_size=3)과 둘(universe_size=2)

  트리거 하나 = 반회전율 25%, 상대부족 20% (현행 라이브)
  앵커 일곱, 송금 지연 5영업일, 기본 비용 — 11-01 합동 격자와 같은 조건
  상한은 둘 다 1/3 그대로 (종목 둘에서 상한을 올리면 종목수 효과가 아니라 집중도
  효과가 섞인다)

판정 규칙 (이것도 결과 전에 고정)
  기준선은 60/20/20 × 종목 셋 = 현행 라이브.
  같은 창 짝 비교에서 수익과 칼마 두 기준 모두 기준선을 이기는 칸만 후보로 본다.
  종목 둘이 같은 배분의 셋을 못 이기면 종결한다 — 효율 이득이 합성에서 안 살아난다.
  종목수 효과는 같은 배분 안의 둘 대 셋 차이로만 읽고, 배분 효과는 종목 셋 줄로만
  읽는다. 둘을 섞어 해석하지 않는다.

읽기 전용이다. 라이브도 상태파일도 건드리지 않는다.

출력
  state/<job>/n2_a<앵커>.npz   (칸별 지표와 창별 등수)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAP))
for p in (ROOT, CAP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# 배분 셋. 주식은 60 고정이고 남은 40 을 현물:선물로 나눈다.
MIXES = [
    ('60/20/20', 0.60, 0.20, 0.20),
    ('60/25/15', 0.60, 0.25, 0.15),
    ('60/30/10', 0.60, 0.30, 0.10),
]
HT = 0.25      # 반회전율 문턱 (현행 라이브)
T3U = 0.20     # 상대부족 문턱 (현행 라이브)
NSIZES = [3, 2]


def main() -> int:
    ap = argparse.ArgumentParser(description='현물 종목 둘 대 셋 합성 비교')
    ap.add_argument('--anchor', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--delay', type=int, default=5)
    a = ap.parse_args()

    import bt_coin_param_resweep as R
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    from alloc_mix_real_rebal import canary_from_trace
    from alloc_joint_grid import (build_alloc_real_many, curve_metrics,
                                  window_ranksum)

    t0 = time.time()
    R._bars, R._funding = load_data('D')
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)

    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pdf = pd.DataFrame(load_prices(ALL_TICKERS, start="2005-01-01"))
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    _r, _mo, _md, can_st = precompute(pdf, [30, 72, 230], [42, 63, 126])
    can_st = can_st.astype(bool)
    print(f'[a{a.anchor}] 입력 준비 ({time.time() - t0:.0f}초)', flush=True)

    st = R.run_stock(a.anchor, False)
    tr_fu: list = []
    fu = R.run_fut({}, R.PH_FUT[a.anchor], False, trace=tr_fu)
    can_fu = canary_from_trace(tr_fu)

    # 현물 슬리브를 종목 수마다 따로 만든다. 기존 곡선에 상수를 곱해 대신하지 않는다
    # (history #147 설계 제약 ⑵) — 종목 수가 바뀌면 선정·리필·드리프트가 다 달라진다.
    sp_by_n = {}
    can_sp_by_n = {}
    for n in NSIZES:
        tr: list = []
        eq = R.run_spot({'universe_size': n}, R.PH_SPOT[a.anchor], False, trace=tr)
        if eq is None:
            print(f'[a{a.anchor}] 현물 종목 {n} 슬리브를 못 만들었다')
            return 1
        sp_by_n[n] = eq
        can_sp_by_n[n] = canary_from_trace(tr)
    if st is None or fu is None:
        print(f'[a{a.anchor}] 주식·선물 슬리브를 못 만들었다')
        return 1
    print(f'[a{a.anchor}] 슬리브 넷 ({time.time() - t0:.0f}초)', flush=True)

    # 종목 수마다 합성이 따로 돈다 — 현물 곡선과 현물 카나리가 둘 다 바뀌기 때문이다.
    labels, cals, shps, cagrs, mdds, fpys = [], [], [], [], [], []
    curve_list, dates_ref = [], None
    for n in NSIZES:
        tgt = np.array([[m[1], m[2], m[3]] for m in MIXES], dtype=float)
        ht = np.full(len(MIXES), HT)
        t3u = np.full(len(MIXES), T3U)
        dates, curves, fires = build_alloc_real_many(
            st, sp_by_n[n], fu, can_st, can_sp_by_n[n], can_fu,
            tgt, ht, t3u, delay=a.delay, transfer_cost=0.0)
        cal, sharpe, cagr, mdd, yrs = curve_metrics(curves, dates)
        if dates_ref is None:
            dates_ref = dates
        elif len(dates) != len(dates_ref) or dates[0] != dates_ref[0]:
            print(f'[a{a.anchor}] 종목 {n} 의 날짜축이 다르다 — 창 비교를 못 한다')
            return 1
        for i, m in enumerate(MIXES):
            labels.append(f'{m[0]} n{n}')
            cals.append(cal[i]); shps.append(sharpe[i])
            cagrs.append(cagr[i]); mdds.append(mdd[i]); fpys.append(fires[i] / yrs)
            curve_list.append(curves[i])
    print(f'[a{a.anchor}] 칸 {len(labels)}개 합성 ({time.time() - t0:.0f}초)', flush=True)

    allc = np.stack(curve_list)
    det: list = []
    rs, wins, rs_cagr, wins_cagr, nw = window_ranksum(
        allc, dates_ref, R.WIN_SIZES, R.STRIDES, detail=det)
    print(f'[a{a.anchor}] 창 {nw} ({time.time() - t0:.0f}초)', flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        label=np.array(labels),
        cal=np.array(cals), sharpe=np.array(shps), cagr=np.array(cagrs),
        mdd=np.array(mdds), fires_per_year=np.array(fpys),
        rs=rs, wins=wins, rs_cagr=rs_cagr, wins_cagr=wins_cagr, n=nw,
        rank_cal=np.stack([d[3] for d in det]),
        rank_cg=np.stack([d[4] for d in det]),
        cagr_w=np.stack([d[5] for d in det]),
        mdd_w=np.stack([d[6] for d in det]),
        anchor=a.anchor, delay=a.delay,
        d0=str(dates_ref[0].date()), d1=str(dates_ref[-1].date()),
        ndays=len(dates_ref),
    )
    print(f'[a{a.anchor}] 저장 {a.out} ({time.time() - t0:.0f}초)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
