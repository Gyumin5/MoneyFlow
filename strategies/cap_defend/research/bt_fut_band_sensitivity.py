"""체결 밴드를 바꿔가며 선물 드리프트 문턱 결론이 버티는지 본다 (2026-09-22).

물음의 출처: 0.0075 채택 근거의 핵심이 "발화는 1.20배 느는데 체결은 1.08배만 는다,
체결 밴드가 자잘한 조정을 삼키기 때문" 이다. 그런데 그 밴드가 백테스트와 라이브에서
같은 의미로 걸리지 않으면 이 근거가 통째로 무너진다.

코드 대조로는 "같아 보인다" 까지밖에 못 간다(라이브에는 거래소 최소주문·수량 반올림·
레버리지 하향 사전매도가 더 있고 백테스트에는 자본 부족 스킵이 있다). 그래서 방향을
바꾼다 — 밴드가 실제로 몇 퍼센트인지 맞히는 대신, 밴드가 얼마이든 결론이 같은지를 잰다.

밴드를 0(밴드 없음)·0.025·0.05(현행 정합)·0.10·0.15 로 놓고 각각에서 현행 0.03 과
0.0075 를 맞붙인다. 판정은 이렇다.
  다섯 밴드 전부에서 0.0075 가 이기면 — 밴드 모델링이 틀려도 채택 결론은 안 무너진다.
  특정 밴드에서만 이기면 — 그 우위는 전략이 아니라 밴드가 만든 것이고 채택하면 안 된다.

밴드 0 이 특히 중요하다. 밴드가 없으면 발화가 곧 체결이라 "회전율로 설명되는 우위"
여부가 가장 가혹하게 드러난다. 현물에서 같은 축이 무너진 것이 정확히 그 조건이었다.

엔진은 env 토글 FUT_REBAL_BAND 로만 건드린다(backtest_futures_v25.py 의 _band).
실험이 끝나면 되돌리고 재현 패치만 남긴다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 -u bt_fut_band_sensitivity.py
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

BANDS = [0.0, 0.025, 0.05, 0.10, 0.15]
CUR_THR = 0.03      # 현행
NEW_THR = 0.0075    # 채택 후보
LIVE_BAND = 0.05    # 라이브 DELTA_THRESHOLD 와 정합된 값

# 밴드 0.05 는 이미 돈 캐시가 있다(현행 / 선물 0.0075). 그 이름을 그대로 쓴다.
def arm_name(band: float, thr: float) -> str:
    if band == LIVE_BAND:
        return R.CUR if thr == CUR_THR else "선물 0.0075"
    tag = f"{band:g}"
    return f"밴드{tag} 현행" if thr == CUR_THR else f"밴드{tag} 0.0075"


def with_band(band: float, fn):
    """엔진의 체결 밴드를 band 로 놓고 fn() 을 부른다."""
    old = os.environ.get("FUT_REBAL_BAND")
    os.environ["FUT_REBAL_BAND"] = repr(band)
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("FUT_REBAL_BAND", None)
        else:
            os.environ["FUT_REBAL_BAND"] = old


def selfcheck() -> None:
    """토글이 기본값에서 기존 캐시와 한 치도 다르지 않은지 먼저 본다.

    이걸 안 하면 뒤의 모든 수치가 '토글을 넣어서 달라진 것' 인지
    '밴드를 바꿔서 달라진 것' 인지 구분이 안 된다.
    """
    k = R._key(R.CUR, 0, False)
    cached = R._load("fu", k)
    if cached is None:
        print("자체검사 건너뜀 — 기준 캐시가 없다")
        return
    fresh = with_band(LIVE_BAND, lambda: R.run_fut({}, R.PH_FUT[0], False))
    common = cached.index.intersection(fresh.index)
    gap = float((cached.reindex(common) - fresh.reindex(common)).abs().max())
    print(f"자체검사 · 밴드 {LIVE_BAND} 재계산 대 기존 캐시 최대 괴리 {gap:.10f} "
          f"({len(common)}봉) → {'동일' if gap < 1e-9 else '어긋남 — 아래 수치 신뢰 불가'}")


def count_fires(band: float, thr: float) -> dict:
    """발화(Rebal)와 체결(Trades). 엔진 반환값이라 계측 코드가 필요 없다."""
    from backtest_futures_v25 import run as fu_run
    over = {} if thr == CUR_THR else dict(drift_threshold=thr)

    def _go():
        return fu_run(R._bars, R._funding, interval="D", leverage=R._k2,
                      tx_cost=R.TX_FUT, maint_rate=0.004, vol_days=90,
                      vol_mode="daily", post_flip_delay=5,
                      phase_offset_bars=R.PH_FUT[0], health_mode="mom2vol",
                      start_date=R.START, end_date=R.END,
                      **{**R.FUT_BASE, **over})

    m = with_band(band, _go)
    return dict(rebal=int(m.get("Rebal", 0)), trades=int(m.get("Trades", 0)),
                cagr=m["CAGR"], mdd=m["MDD"], cal=m["Cal"])


def main() -> None:
    os.makedirs(R.EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...")
    bars, funding = load_data("D")
    R._bars, R._funding = bars, funding
    R._k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  {len(bars)}종 ({time.time()-t0:.0f}초)")

    selfcheck()
    print(f"  ({time.time()-t0:.0f}초)")

    # 주식은 합성용 고정 팔이다. 밴드와 무관하므로 앵커별로 한 번만 불러온다.
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])

    # 현물도 밴드와 무관하다(현물 엔진에는 밴드가 없다). 현행 곡선을 전 팔이 공유한다.
    todo = [(b, thr, a, s) for b in BANDS for thr in (CUR_THR, NEW_THR)
            for a in R.ANCHORS for s in (False, True)]
    print(f"\n선물 곡선 {len(todo)}건 계산 (밴드 {LIVE_BAND} 는 기존 캐시 재사용)")
    for i, (band, thr, a, stress) in enumerate(todo, 1):
        name = arm_name(band, thr)
        k = R._key(name, a, stress)
        if R._load("sp", k) is None:
            base = R._load("sp", R._key(R.CUR, a, stress))
            R._save("sp", k, base if base is not None else R.run_spot({}, R.PH_SPOT[a], stress))
        if R._load("fu", k) is None:
            over = {} if thr == CUR_THR else dict(drift_threshold=thr)
            R._save("fu", k, with_band(band, lambda: R.run_fut(over, R.PH_FUT[a], stress)))
            print(f"  {i}/{len(todo)} 밴드{band:g} {'현행' if thr == CUR_THR else '0.0075'} "
                  f"앵커{a} {'5배' if stress else '기본'} ({time.time()-t0:.0f}초)")

    # ── 1. 밴드별로 문턱 우위가 살아남나 (핵심) ──
    for stress in (False, True):
        for band in BANDS:
            arms = [(arm_name(band, CUR_THR), {}, {}),
                    (arm_name(band, NEW_THR), {}, dict(drift_threshold=NEW_THR))]
            R.report(f"체결 밴드 {band:g}", arms, stress, stock_eq)

    # ── 2. 밴드가 실제로 무엇을 삼키나 (기제 확인) ──
    print("\n=== 밴드가 삼키는 양 (앵커 0 · 기본 비용 · 선물 슬리브 단독) ===")
    print(f"{'밴드':<8}{'문턱':<10}{'발화':>7}{'체결':>7}{'체결÷발화':>11}"
          f"{'수익%':>10}{'낙폭%':>9}{'칼마':>8}")
    counts = {}
    for band in BANDS:
        for thr in (CUR_THR, NEW_THR):
            c = count_fires(band, thr)
            counts[(band, thr)] = c
            ratio = c["trades"] / c["rebal"] if c["rebal"] else 0.0
            print(f"{band:<8g}{thr:<10g}{c['rebal']:>7}{c['trades']:>7}{ratio:>11.2f}"
                  f"{c['cagr']:>10.1f}{c['mdd']:>9.2f}{c['cal']:>8.2f}")

    print("\n=== 문턱을 조였을 때 늘어나는 양 (밴드별) ===")
    print(f"{'밴드':<8}{'발화배수':>10}{'체결배수':>10}{'차이':>9}{'칼마 변화':>12}")
    for band in BANDS:
        a, b = counts[(band, CUR_THR)], counts[(band, NEW_THR)]
        fr = b["rebal"] / a["rebal"] if a["rebal"] else 0.0
        tr = b["trades"] / a["trades"] if a["trades"] else 0.0
        print(f"{band:<8g}{fr:>10.2f}{tr:>10.2f}{tr-fr:>+9.2f}"
              f"{b['cal']-a['cal']:>+12.2f}")

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
