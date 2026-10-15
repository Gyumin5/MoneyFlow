"""주식 슬리브도 같은 종류의 시점 결함을 갖고 있나, 갖고 있다면 얼마나 (2026-10-15).

bt_spot_order_causality.py 가 현물 주문 경로의 시간 누수를 확인한 다음 단계다. 주식
슬리브는 합성의 60% 라 같은 종류를 안 보고 넘어갈 자리가 아니다.

## 무엇이 다른가 — 현물 누수와 같은 것이 아니다

현물 쪽 결함은 체결가보다 뒤에 알려지는 값으로 주문 크기를 정한 것이다. 명백한 미래 정보다.

주식 쪽은 성질이 다르다. 채택 백테스트(bt_stock_mom3.run_multi_3mom)는 그날 종가로
신호를 만들고 그날 종가로 체결한다. 체결 시점 기준으로 미래 정보를 쓰지는 않는다. 대신
"방금 보고 계산한 그 종가에 그대로 살 수 있다" 를 가정한다.

문제는 라이브가 그렇게 안 돈다는 것이다. executor_stock 은 한국시각 23시 35분에 전 거래일
확정 종가로 신호를 만들고 그 뒤에 열리는 장에서 체결한다. 즉 라이브는 신호와 체결이 한 봉
떨어져 있는데 백테스트는 붙어 있다. CLAUDE.md 의 백테스트 필수 규칙도 주식은
sig_date = prev_trading_date 라고 적어 뒀다 — 채택 하니스가 그 규칙과 어긋나 있다.

## 어떻게 재나

bt_stock_mom3 에 STOCK_SIG_LAG 토글을 넣었다. 체결·평가 시점은 그대로 두고 신호를 읽는
날짜만 바꾼다.

- 현행(lag 0): 그날 신호 → 그날 체결.
- 수정(lag 1): 전 거래일 신호 → 그날 체결. 라이브 정합이고 CLAUDE.md 규칙과도 맞는다.

앵커 일곱 곳. 주식 단독과 합성 60/25/15 둘 다 찍는다. 현물·선물 슬리브는 이 축과 무관해서
두 팔이 같은 곡선을 쓴다 — 현물은 아직 누수가 있는 현행 그대로다. 두 결함을 한 판에 섞으면
어느 쪽이 얼마를 설명하는지 못 가른다.

## 판정 (결과 보기 전에 못박는다)

채택 판정이 아니다. 둘 중 좋은 쪽을 고르는 게 아니라 지금까지 적어 온 주식 숫자가 얼마나
낙관적이었나를 재는 것이다. 수정안이 나쁘게 나와도 수정안이 라이브가 실제로 하는 일이다.

보고할 것: 주식 단독과 합성의 네 지표(칼마·샤프·연복리·최대낙폭) 두 팔 차이, 앵커 일곱 곳의
분포. 창 순위합은 참고로만 찍는다 — 두 팔은 경쟁 후보가 아니다.

차이가 작게 나와도 "사소하다" 로 읽지 않는다. 이 측정이 답하는 것은 크기 하나이고, 그
숫자로 고른 파라미터가 올바른 시점에서도 1등인지는 따로 재야 한다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_stock_sig_lag_impact.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import bt_coin_param_resweep as R  # noqa: E402

ARMS = [("현행(당일 신호 당일 체결)", "0"), ("수정(전 거래일 신호)", "1")]


def run_stock_lag(lag: str, anchor: int) -> pd.Series:
    os.environ["STOCK_SIG_LAG"] = lag
    try:
        return R.run_stock(anchor, False)
    finally:
        os.environ["STOCK_SIG_LAG"] = "0"


def fmt(m: dict) -> str:
    return (f"칼마 {m['Cal']:.3f}  샤프 {m['Sharpe']:.3f}  "
            f"연복리 {m['CAGR']:+.2f}%  최대낙폭 {m['MDD']:.2f}%")


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    print("사전 확인 — 두 팔이 실제로 갈리나 (앵커 0)")
    probe = {lag: run_stock_lag(lag, R.ANCHORS[0]) for _, lag in ARMS}
    common = probe["0"].index.intersection(probe["1"].index)
    gap = float((probe["0"].loc[common] - probe["1"].loc[common]).abs().max())
    print(f"  두 곡선 최대 차이 {gap:.6g}")
    if gap <= 1e-9:
        print("  멈춘다 — 토글이 안 걸렸다.")
        return
    print("  갈린다. 본 측정으로 간다.\n", flush=True)

    spot_eq, fut_eq, stock_eq = {}, {}, {}
    for a in R.ANCHORS:
        spot_eq[a] = R.run_spot({}, R.PH_SPOT[a], False)
        fut_eq[a] = R.run_fut({}, R.PH_FUT[a], False)
        for _, lag in ARMS:
            stock_eq[(lag, a)] = run_stock_lag(lag, a)
        print(f"  앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)

    for scope in ("주식 단독", "합성 60/25/15"):
        print(f"\n{'=' * 64}\n[{scope}] 앵커 {len(R.ANCHORS)}곳 평균\n{'=' * 64}")
        per_arm = {}
        for label, lag in ARMS:
            ms = []
            for a in R.ANCHORS:
                st = stock_eq[(lag, a)]
                eq = st if scope == "주식 단독" else R.build_alloc(st, spot_eq[a], fut_eq[a])
                ms.append(R.metrics(eq))
            per_arm[lag] = ms
            avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            print(f"  {label:24s} {fmt(avg)}")
        keys = ('Cal', 'Sharpe', 'CAGR', 'MDD')
        a0 = {k: float(np.mean([m[k] for m in per_arm['0']])) for k in keys}
        a1 = {k: float(np.mean([m[k] for m in per_arm['1']])) for k in keys}
        print(f"  {'차이(수정-현행)':24s} 칼마 {a1['Cal'] - a0['Cal']:+.3f}  "
              f"샤프 {a1['Sharpe'] - a0['Sharpe']:+.3f}  "
              f"연복리 {a1['CAGR'] - a0['CAGR']:+.2f}%p  "
              f"최대낙폭 {a1['MDD'] - a0['MDD']:+.2f}%p")
        if scope == "주식 단독":
            print("  앵커별 칼마 (현행 → 수정)")
            for i, a in enumerate(R.ANCHORS):
                c0, c1 = per_arm['0'][i]['Cal'], per_arm['1'][i]['Cal']
                print(f"    앵커 {a}: {c0:.3f} → {c1:.3f}  ({c1 - c0:+.3f})")

    comp = {}
    for _, lag in ARMS:
        for a in R.ANCHORS:
            comp[f"{lag}|{a}"] = R.build_alloc(stock_eq[(lag, a)], spot_eq[a], fut_eq[a])
    rs, _wins, _n = R.window_rs(comp)
    for label, lag in ARMS:
        vals = [rs[f"{lag}|{a}"] for a in R.ANCHORS]
        print(f"\n창 순위합(참고) {label:24s} {float(np.mean(vals)):.3f}")

    print(f"\n총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
