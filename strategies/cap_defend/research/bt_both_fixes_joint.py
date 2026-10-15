"""시점 결함 둘을 동시에 고친 판 — 앞으로 쓸 기준선 (2026-10-15).

따로 잰 두 값을 더하면 안 된다. 합성 곡선은 세 슬리브 수익률의 가중합이고 최대낙폭은
합산되지 않으므로, 두 수정이 동시에 걸린 판을 직접 돌려야 한다.

## 네 팔

  ⑴ 현행      — 현물 당일 종가 평가, 주식 당일 신호
  ⑵ 현물만 수정 — 현물 전일 종가 평가, 주식 당일 신호
  ⑶ 주식만 수정 — 현물 당일 종가 평가, 주식 전 거래일 신호
  ⑷ 둘 다 수정  — 현물 전일 종가 평가, 주식 전 거래일 신호

⑵⑶ 은 이미 따로 쟀지만 같은 판에서 다시 센다. 비교 대상 수가 다른 값을 이어 붙이지
않는다는 2026-10-15 의 규약(history #53 계기) 때문이다.

선물 슬리브는 두 축 모두와 무관하다 — backtest_futures_v25 는 신호도 평가도 전일
기준이라 네 팔이 같은 곡선을 쓴다.

앵커 일곱 곳 × 비용 조건 둘.

## 판정 (결과 보기 전에 못박는다)

채택 판정이 아니다. ⑷ 가 나쁘게 나와도 ⑷ 가 맞는 값이다 — 시점은 성능 축으로 고를
대상이 아니다. 이 측정이 답하는 것은 둘이다.

  (가) 지금까지 적어 온 합성 숫자가 얼마나 부풀어 있었나. ⑴ 대 ⑷ 의 네 지표 차이.
  (나) 두 결함이 서로 상쇄하나 겹치나. ⑷-⑴ 이 (⑵-⑴)+(⑶-⑴) 와 얼마나 다른가.
       차이가 크면 따로 잰 값으로 합계를 추정할 수 없다는 뜻이고, 그 자체가 기록할
       사실이다.

앵커별 부호 일관성도 같이 센다. 일곱 곳이 한 방향이면 체계적 편향, 흩어지면 잡음에
가깝다 — 크기보다 이쪽이 판단에 중요하다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_both_fixes_joint.py
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

ARMS = [
    ("현행", "same", "0"),
    ("현물만 수정", "prev", "0"),
    ("주식만 수정", "same", "1"),
    ("둘 다 수정", "prev", "1"),
]
KEYS = ("Cal", "Sharpe", "CAGR", "MDD")


def run_spot_pv(pv_mode: str, phase: int, stress: bool) -> pd.Series:
    os.environ["SPOT_PV_DATE"] = pv_mode
    try:
        return R.run_spot({}, phase, stress)
    finally:
        os.environ["SPOT_PV_DATE"] = "same"


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

    print("사전 확인 — 네 팔이 실제로 갈리나 (앵커 0, 기본 비용)")
    sp = {m: run_spot_pv(m, R.PH_SPOT[R.ANCHORS[0]], False) for m in ("same", "prev")}
    st = {lg: run_stock_lag(lg, R.ANCHORS[0]) for lg in ("0", "1")}
    g_sp = float((sp["same"] - sp["prev"]).abs().max())
    c = st["0"].index.intersection(st["1"].index)
    g_st = float((st["0"].loc[c] - st["1"].loc[c]).abs().max())
    print(f"  현물 토글 곡선 차이 {g_sp:.6g} · 주식 토글 곡선 차이 {g_st:.6g}")
    if g_sp <= 1e-9 or g_st <= 1e-9:
        print("  멈춘다 — 토글 하나가 안 걸렸다.")
        return
    print("  둘 다 갈린다. 본 측정으로 간다.\n", flush=True)

    spot_eq, stock_eq, fut_eq = {}, {}, {}
    for a in R.ANCHORS:
        for s in (False, True):
            fut_eq[(a, s)] = R.run_fut({}, R.PH_FUT[a], s)
            for m in ("same", "prev"):
                spot_eq[(m, a, s)] = run_spot_pv(m, R.PH_SPOT[a], s)
        for lg in ("0", "1"):
            stock_eq[(lg, a)] = run_stock_lag(lg, a)
        print(f"  앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)

    for stress in (False, True):
        tag = "5배 비용" if stress else "기본 비용"
        print(f"\n{'=' * 66}\n=== 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 ===\n{'=' * 66}")

        per_arm = {}
        for label, pv, lag in ARMS:
            ms = []
            for a in R.ANCHORS:
                eq = R.build_alloc(stock_eq[(lag, a)], spot_eq[(pv, a, stress)],
                                   fut_eq[(a, stress)])
                ms.append(R.metrics(eq))
            per_arm[label] = ms
            avg = {k: float(np.mean([m[k] for m in ms])) for k in KEYS}
            print(f"  {label:12s} {fmt(avg)}")

        base = {k: float(np.mean([m[k] for m in per_arm["현행"]])) for k in KEYS}
        print()
        for label, _, _ in ARMS[1:]:
            cur = {k: float(np.mean([m[k] for m in per_arm[label]])) for k in KEYS}
            signs = [per_arm[label][i]["Cal"] - per_arm["현행"][i]["Cal"]
                     for i in range(len(R.ANCHORS))]
            down = sum(1 for v in signs if v < 0)
            print(f"  {label + ' - 현행':20s} 칼마 {cur['Cal']-base['Cal']:+.3f}  "
                  f"샤프 {cur['Sharpe']-base['Sharpe']:+.3f}  "
                  f"연복리 {cur['CAGR']-base['CAGR']:+.2f}%p  "
                  f"최대낙폭 {cur['MDD']-base['MDD']:+.2f}%p  "
                  f"(칼마 내려간 앵커 {down}/{len(R.ANCHORS)})")

        a2 = {k: float(np.mean([m[k] for m in per_arm["현물만 수정"]])) for k in KEYS}
        a3 = {k: float(np.mean([m[k] for m in per_arm["주식만 수정"]])) for k in KEYS}
        a4 = {k: float(np.mean([m[k] for m in per_arm["둘 다 수정"]])) for k in KEYS}
        print("\n  겹침 검사 — 따로 잰 값을 더하면 맞나")
        for k in KEYS:
            add = (a2[k] - base[k]) + (a3[k] - base[k])
            joint = a4[k] - base[k]
            print(f"    {k:7s} 따로 합 {add:+.3f} · 같이 돌림 {joint:+.3f} · 어긋남 {joint-add:+.3f}")

    print(f"\n총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
