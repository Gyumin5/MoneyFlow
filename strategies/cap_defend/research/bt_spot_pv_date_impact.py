"""현물 주문 시점 누수를 고쳤을 때 성과가 얼마나 내려가나 (2026-10-15).

bt_spot_order_causality.py 가 누수를 확인한 다음 단계다. 거기서는 "있나" 를 쟀고
여기서는 "얼마나" 를 잰다.

## 무엇을 재나

unified_backtest.py 의 _execute_rebalance 가 주문 수량을 정할 때 쓰는 계좌 평가 시점을
두 가지로 놓고 전 구간을 다시 돈다.

- 현행(same): 그날 종가로 평가. 체결은 그날 시가. 체결 시점에 알 수 없는 값을 쓴다.
- 수정(prev): 전일 종가로 평가. 선물 엔진(backtest_futures_v25.py:505)이 이미 쓰는 방식이다.

앵커 일곱 곳 × 비용 조건 둘. 현물 단독과 합성 60/25/15 둘 다 찍는다.
주식·선물 슬리브는 이 축과 무관해서 두 팔이 같은 곡선을 쓴다.

## 판정 (결과 보기 전에 못박는다)

이건 채택 판정이 아니다. 두 팔 중 성과가 좋은 쪽을 고르는 측정이 아니라, 지금까지 적어 온
숫자가 얼마나 부풀어 있었나를 재는 것이다. 수정안이 더 나쁘게 나와도 수정안이 맞는 값이다 —
시간 누수는 성능 축으로 고를 대상이 아니기 때문이다.

그래서 결론은 숫자 자체로 쓴다.
  · 현물 단독 네 지표(칼마·샤프·연복리·최대낙폭)의 두 팔 차이.
  · 합성 네 지표의 두 팔 차이.
  · 앵커 일곱 곳의 분포 — 차이가 한두 앵커에 몰렸나 고르게 퍼졌나.
  · 창 순위합은 참고로만 찍는다. 두 팔은 경쟁 후보가 아니다.

읽는 법 하나. 차이가 작다고 누수가 사소한 것은 아니다. 누수는 크기와 무관하게 채택 심사의
근거를 무효로 만든다 — 그 숫자로 고른 파라미터가 그 숫자 없이도 1등인지는 따로 재야 한다.
이 측정은 "지금까지 쓴 숫자가 얼마나 틀렸나" 만 답한다.

## 선행조건

정본 엔진에 SPOT_PV_DATE 토글이 있어야 한다(규칙상 측정 뒤 되돌린다).
  cd /home/gmoh/mon/251229
  git apply --3way strategies/cap_defend/research/bt_spot_pv_date.engine.patch
안 얹고 돌리면 두 팔이 같은 곡선을 내는데 그건 "차이 없음" 과 눈으로 구분이 안 된다
(history #52). 그래서 재기 전에 한 벌로 확인하고 안 갈리면 멈춘다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_spot_pv_date_impact.py
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

ARMS = [("현행(당일 종가로 평가)", "same"), ("수정(전일 종가로 평가)", "prev")]


def run_spot_pv(pv_mode: str, phase: int, stress: bool) -> pd.Series:
    os.environ["SPOT_PV_DATE"] = pv_mode
    try:
        return R.run_spot({}, phase, stress)
    finally:
        os.environ["SPOT_PV_DATE"] = "same"


def fmt(m: dict) -> str:
    return (f"칼마 {m['Cal']:.3f}  샤프 {m['Sharpe']:.3f}  "
            f"연복리 {m['CAGR']:+.2f}%  최대낙폭 {m['MDD']:.2f}%")


def main() -> None:
    t0 = time.time()
    os.makedirs(R.EQDIR, exist_ok=True)
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    print("사전 확인 — 두 팔이 실제로 갈리나 (앵커 0, 기본 비용)")
    probe = {mode: run_spot_pv(mode, R.PH_SPOT[R.ANCHORS[0]], False) for _, mode in ARMS}
    gap = float((probe["same"] - probe["prev"]).abs().max())
    print(f"  두 곡선 최대 차이 {gap:.6g}")
    if gap <= 1e-9:
        print("  멈춘다 — 토글이 안 걸렸다. 엔진 패치를 먼저 얹어라.")
        return
    print("  갈린다. 본 측정으로 간다.\n")

    stock_eq, fut_eq = {}, {}
    for a in R.ANCHORS:
        for s in (False, True):
            fut_eq[(a, s)] = R.run_fut({}, R.PH_FUT[a], s)
        _p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(_p):
            stock_eq[a] = pd.read_csv(_p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(_p, header=["eq"])

    spot_eq: dict = {}
    for a in R.ANCHORS:
        for s in (False, True):
            for _, mode in ARMS:
                spot_eq[(mode, a, s)] = run_spot_pv(mode, R.PH_SPOT[a], s)

    for stress in (False, True):
        tag = "5배 비용" if stress else "기본 비용"
        print(f"\n{'=' * 64}\n=== {tag} · 앵커 {len(R.ANCHORS)}곳 ===\n{'=' * 64}")

        for scope in ("현물 단독", "합성 60/25/15"):
            print(f"\n[{scope}] 앵커 평균")
            per_anchor = {}
            for label, mode in ARMS:
                ms = []
                for a in R.ANCHORS:
                    sp = spot_eq[(mode, a, stress)]
                    eq = sp if scope == "현물 단독" else R.build_alloc(
                        stock_eq[a], sp, fut_eq[(a, stress)])
                    ms.append(R.metrics(eq))
                per_anchor[mode] = ms
                avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                print(f"  {label:22s} {fmt(avg)}")
            a0 = {k: float(np.mean([m[k] for m in per_anchor['same']])) for k in ('Cal', 'Sharpe', 'CAGR', 'MDD')}
            a1 = {k: float(np.mean([m[k] for m in per_anchor['prev']])) for k in ('Cal', 'Sharpe', 'CAGR', 'MDD')}
            print(f"  {'차이(수정-현행)':22s} 칼마 {a1['Cal'] - a0['Cal']:+.3f}  "
                  f"샤프 {a1['Sharpe'] - a0['Sharpe']:+.3f}  "
                  f"연복리 {a1['CAGR'] - a0['CAGR']:+.2f}%p  "
                  f"최대낙폭 {a1['MDD'] - a0['MDD']:+.2f}%p")
            if scope == "현물 단독":
                print("  앵커별 칼마 (현행 → 수정)")
                for i, a in enumerate(R.ANCHORS):
                    c0 = per_anchor['same'][i]['Cal']
                    c1 = per_anchor['prev'][i]['Cal']
                    print(f"    앵커 {a}: {c0:.3f} → {c1:.3f}  ({c1 - c0:+.3f})")

        comp = {}
        for label, mode in ARMS:
            for a in R.ANCHORS:
                comp[f"{mode}|{a}"] = R.build_alloc(
                    stock_eq[a], spot_eq[(mode, a, stress)], fut_eq[(a, stress)])
        rs, _wins, _n = R.window_rs(comp)
        for label, mode in ARMS:
            vals = [rs[f"{mode}|{a}"] for a in R.ANCHORS]
            print(f"\n  창 순위합(참고) {label:22s} {float(np.mean(vals)):.3f}")

    print(f"\n총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
