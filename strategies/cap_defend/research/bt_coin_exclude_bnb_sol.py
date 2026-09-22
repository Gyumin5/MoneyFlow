"""BNB 와 SOL 을 후보군에서 빼고 같은 측정을 다시 한다 (2026-09-22).

운영자 지적: 그 둘이 아웃라이어다. 지금 성적이 두 종목의 큰 상승에 얹혀 있는 것이라면
선물 드리프트 문턱을 옮길지 정하는 근거 자체가 흔들린다.

기여도를 빼서 보는 방식은 쓰지 않는다. 두 종목이 후보에서 사라지면 그날 선정이 바뀌고
그 뒤 전부가 달라지기 때문에, 아예 없는 세계로 처음부터 다시 돌려야 답이 나온다.

제외는 엔진을 고치지 않고 입력에서 한다 — bars 에서 두 키를 지우면 두 엔진 모두
후보 루프의 bars.get(coin) is None 에서 걸러낸다(unified_backtest.py:557,
backtest_futures_v25.py 같은 위치). 선물 엔진에는 exclude_assets 인자가 없어서
그쪽에 인자를 새로 다는 대신 이 방법을 쓴다. SSoT 엔진은 한 줄도 안 고친다.

네 팔을 같은 자로 잰다.
  포함 현행(0.03) · 포함 0.0075 — 앞선 측정의 캐시를 그대로 쓴다.
  제외 현행(0.03) · 제외 0.0075 — 여기서 새로 돈다.

물어야 할 것은 "성적이 떨어지나" 가 아니다. 그건 당연하다. 물음은 셋이다.
  하나. 두 종목을 빼면 얼마나 내려앉나.
  둘. 내려앉은 뒤에도 문턱을 조이는 쪽이 여전히 이기나.
  셋. 이기는 앵커 수가 유지되나.
둘째가 무너지면 그 우위는 전략이 아니라 두 종목이 만든 것이고, 그럼 안 바꾼다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_coin_exclude_bnb_sol.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402  (하니스·캐시·순위합 재사용)

DROP = ("BNB", "SOL")
FUT_NEW = 0.0075

INC_CUR = R.CUR              # 포함 · 현행 0.03 (캐시 있음)
INC_NEW = "선물 0.0075"       # 포함 · 0.0075 (캐시 있음)
EXC_CUR = "제외 현행"
EXC_NEW = "제외 0.0075"


def build_k2(bars: dict):
    from backtest_futures_v25 import build_K2_signal
    return build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                           btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                           l_min=2.0, l_mid=3.0, l_max=4.0)


def main() -> None:
    os.makedirs(R.EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    print("코인 일봉 로드 중...")
    bars_full, funding = load_data("D")
    print(f"  전체 {len(bars_full)}종 ({time.time()-t0:.0f}초)")

    bars_cut = {k: v for k, v in bars_full.items() if k not in DROP}
    missing = [c for c in DROP if c not in bars_full]
    if missing:
        raise SystemExit(f"제외 대상이 데이터에 없다: {missing} — 심볼 표기를 확인해라")
    print(f"  제외 후 {len(bars_cut)}종 (뺀 것: {', '.join(DROP)})")

    print("주식 슬리브(합성용 고정 팔) 앵커별 불러오는 중...")
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])
    print(f"  완료 ({time.time()-t0:.0f}초)")

    # ── 포함 세계: 캐시가 다 있어야 한다. 없으면 전체 심볼로 채운다. ──
    R._bars, R._funding = bars_full, funding
    R._k2 = build_k2(bars_full)
    for name, fut_over in ((INC_CUR, {}), (INC_NEW, dict(drift_threshold=FUT_NEW))):
        for a in R.ANCHORS:
            for stress in (False, True):
                R.ensure(name, {}, fut_over, a, stress)
    print(f"포함 세계 캐시 확인 완료 ({time.time()-t0:.0f}초)")

    # ── 제외 세계: 여기서 새로 돈다. 현물 설정은 두 팔이 같으므로 곡선을 공유한다. ──
    R._bars, R._funding = bars_cut, funding
    R._k2 = build_k2(bars_cut)
    todo = [(a, s) for a in R.ANCHORS for s in (False, True)]
    for i, (a, stress) in enumerate(todo, 1):
        k_cur = R._key(EXC_CUR, a, stress)
        k_new = R._key(EXC_NEW, a, stress)
        if R._load("sp", k_cur) is None:
            R._save("sp", k_cur, R.run_spot({}, R.PH_SPOT[a], stress))
        if R._load("sp", k_new) is None:
            R._save("sp", k_new, R._load("sp", k_cur))   # 현물 설정 동일 → 같은 곡선
        if R._load("fu", k_cur) is None:
            R._save("fu", k_cur, R.run_fut({}, R.PH_FUT[a], stress))
        if R._load("fu", k_new) is None:
            R._save("fu", k_new, R.run_fut(dict(drift_threshold=FUT_NEW),
                                           R.PH_FUT[a], stress))
        print(f"  제외 세계 {i}/{len(todo)} ({time.time()-t0:.0f}초)")

    # ── 1. 절대 성적: 두 종목을 빼면 얼마나 내려앉나 ──
    print("\n=== 두 종목 제외가 성적에 미치는 영향 (앵커 7곳 평균) ===")
    print(f"{'팔':<14}{'비용':<8}{'합성CAGR':>10}{'합성MDD':>9}{'합성Cal':>9}"
          f"{'합성Sharpe':>11}{'현물Cal':>9}{'선물Cal':>9}")
    for stress in (False, True):
        for name in (INC_CUR, EXC_CUR, INC_NEW, EXC_NEW):
            cag = mdd = cal = shp = spc = fuc = 0.0
            n = 0
            for a in R.ANCHORS:
                k = R._key(name, a, stress)
                sp, fu = R._load("sp", k), R._load("fu", k)
                if sp is None or fu is None:
                    continue
                m = R.metrics(R.build_alloc(stock_eq[a], sp, fu))
                cag += m["CAGR"]; mdd += m["MDD"]; cal += m["Cal"]; shp += m["Sharpe"]
                spc += R.metrics(sp)["Cal"]; fuc += R.metrics(fu)["Cal"]
                n += 1
            if not n:
                continue
            tag = "5배" if stress else "기본"
            print(f"{name:<14}{tag:<8}{cag/n:>10.2f}{mdd/n:>9.2f}{cal/n:>9.2f}"
                  f"{shp/n:>11.2f}{spc/n:>9.2f}{fuc/n:>9.2f}")

    # ── 2. 제외 세계 안에서 문턱 우위가 살아남나 (핵심) ──
    exc_arms = [(EXC_CUR, {}, {}), (EXC_NEW, {}, dict(drift_threshold=FUT_NEW))]
    inc_arms = [(INC_CUR, {}, {}), (INC_NEW, {}, dict(drift_threshold=FUT_NEW))]
    for stress in (False, True):
        R.report("문턱 우위 · BNB SOL 포함", inc_arms, stress, stock_eq)
        R.report("문턱 우위 · BNB SOL 제외", exc_arms, stress, stock_eq)

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
