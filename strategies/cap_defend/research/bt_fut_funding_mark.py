"""선물 펀딩비 시점을 고치면 무엇이 달라지나 (2026-10-15, 운영자 지시 8974).

결함. backtest_futures_v25.py 가 펀딩비 명목을 그날 종가로 재서 capital 을 먼저 깎고,
그 뒤에 전 거래일 평가액을 읽는다. 그래서 그날 시가에 낼 주문 크기에 그날 종가가 아주
조금 섞인다. 종가 흔들기 검사에서 1.10e-5 로 나왔고 원인은 펀딩비로 좁혀졌다(6c0c38d).

앞서 "크기가 작아 판정에 안 닿는다" 고 미뤘다. 그 판단을 수치로 확인한다.

재는 것. 토글 FUT_FUNDING_MARK=prev 로 펀딩비 명목을 전 거래일 종가로 바꾸고, 선물 단독과
합성 60/25/15 를 앵커 일곱 곳에서 두 벌 돌려 네 지표를 비교한다. 현물·주식 슬리브는 두 팔이
같은 것을 쓰므로(저장소 현행 엔진) 차이는 전부 선물에서 온다.

판정 (결과 보기 전에 못박는다).
  합성 칼마 차이가 0.01 미만이면 판정에 안 닿는 것으로 보고 안 고친다. 그 경우에도
  토글과 이 측정은 재현 가능하게 남긴다.
  0.01 이상이면 고치는 쪽으로 간다 — 그때는 별도 결정이다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_fut_funding_mark.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import bt_coin_param_resweep as R  # noqa: E402


def main() -> int:
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    t0 = time.time()
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"일봉 준비 {time.time()-t0:.0f}초 · 심볼 {len(R._bars)}종", flush=True)

    rows: dict[str, list] = {"선물 단독": [], "합성": []}
    for mark in ("same", "prev"):
        os.environ["FUT_FUNDING_MARK"] = mark
        fu_list, comp_list = [], []
        for a in R.ANCHORS:
            fu = R.run_fut({}, R.PH_FUT[a], False)
            sp = R.run_spot({}, R.PH_SPOT[a], False)
            st = R.run_stock(a, False)
            fu_list.append(fu)
            comp_list.append(R.build_alloc(st, sp, fu))
            print(f"  [{mark}] 앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)
        rows["선물 단독"].append((mark, fu_list))
        rows["합성"].append((mark, comp_list))
    os.environ["FUT_FUNDING_MARK"] = "same"

    g = float((rows["선물 단독"][0][1][0] - rows["선물 단독"][1][1][0]).abs().max())
    print(f"\n사전 확인 — 두 팔 선물 곡선 최대 차이 {g:.6g}")
    if g <= 1e-9:
        print("  멈춘다 — 토글이 안 걸려 같은 것을 두 번 재고 있다.")
        return 1

    out = {}
    for label, arms in rows.items():
        print(f"\n[{label}]  앵커 {len(R.ANCHORS)}곳 평균")
        print(f"  {'팔':<16}{'칼마':>9}{'샤프':>9}{'연복리':>10}{'최대낙폭':>10}")
        vals = {}
        for mark, curves in arms:
            ms = [R.metrics(c) for c in curves]
            avg = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
            vals[mark] = avg
            nm = "현행(그날 종가)" if mark == "same" else "수정(전일 종가)"
            print(f"  {nm:<16}{avg['Cal']:9.3f}{avg['Sharpe']:9.3f}"
                  f"{avg['CAGR']:9.2f}%{avg['MDD']:9.2f}%")
        d = {k: vals["prev"][k] - vals["same"][k] for k in vals["same"]}
        print(f"  {'차이':<16}{d['Cal']:+9.3f}{d['Sharpe']:+9.3f}"
              f"{d['CAGR']:+9.2f}%{d['MDD']:+9.2f}%")
        out[label] = d

    print("\n" + "=" * 62)
    dc = abs(out["합성"]["Cal"])
    print(f"합성 칼마 차이 절대값 {dc:.4f} · 사전등록 문턱 0.01")
    if dc < 0.01:
        print("  문턱 미만. 판정에 안 닿는다 — 고치지 않고 사실만 남긴다.")
    else:
        print("  문턱 이상. 고치는 쪽으로 간다 — 별도 결정이다.")
    print(f"\n총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
