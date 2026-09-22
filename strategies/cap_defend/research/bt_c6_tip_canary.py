"""C6 — 주식 카나리를 신흥국주식에서 물가연동국채로 바꿔본다 (2026-09-22).

계기: 2026-09-22 논문 조사에서 추린 백테스트 후보 세 건 중 둘째다. 자산군 로테이션
문헌이 방어 전환 신호로 쓰는 자산이 우리 것과 다르다 — 우리는 신흥국주식(EEM)의
200일선을 쓰는데, 여러 논문은 채권 쪽 자산을 같이 본다. 그 계열에서 우리가 안 써본
축 하나가 물가연동국채(TIP)다.

바꾸는 축은 하나다. 카나리 자산만 EEM -> TIP 으로 바꾼다. 히스테리시스 0.5%,
선정(Z점수 랭킹 + 3모멘텀 필터), 상한 1/3, 스냅 3개·간격 69·스태거 23, 드리프트 5%,
현금버퍼 7%, 거래비용 전부 현행 그대로다. 채택 백테스트 엔진에는 손대지 않는다 —
카나리 계열을 밖에서 만들어 인자로 넣는다(run_multi_3mom 이 canary 를 인자로 받는다).

방향에 대한 사전 선언 (결과 보기 전에 적는다):
  가설은 자산군 로테이션 문헌의 표준형 그대로다. 카나리 자산이 자기 추세 위에 있으면
  위험선호, 아래면 방어. TIP 도 같은 방향으로 읽는다.
  뒤집은 방향(TIP 이 추세 위면 방어)도 같이 재지만 그건 진단용이다. 뒤집은 쪽이
  이겨도 채택하지 않는다 — 부호를 데이터에서 고른 것이 되기 때문이다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 다. 주식 슬리브만 좋고 합성이 반대면 채택하지 않는다.
  2. 윈도우 순위합에서 이겨야 한다. 전체기간 수치 하나로 판정하지 않는다.
  3. 거래비용 5배에서 순서가 뒤집히면 현행 유지다.
  4. 이동평균 기간 인접값(150·200·250)에 플래토가 있어야 한다. 200 만 좋으면 기각.
  5. 앵커 0~10 에서 분산이 커지면 기각. 특정 앵커에서만 좋은 건 우연이다.
  6. 개선이 회전수 감소로 설명되면 채택하지 않는다. 카나리가 켜진 날 비율을 같이 본다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_c6_tip_canary.py
출력은 state/20260922-c6-tip-canary/ 로 리다이렉트한다.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

START = "2020-10-01"
END = os.environ.get("C6_END", "2026-09-18")
BASE_TX = 0.001          # 주식 편도 0.1% (프로젝트 규약)
STRESS_TX = 0.005        # 5배
SMA_PERIODS = [150, 200, 250]
HYST = 0.005
ANCHORS = list(range(0, 11))


def build_canary(px: pd.Series, sma: int, hyst: float, invert: bool = False) -> pd.Series:
    """자산 하나의 이동평균 대비 위치 + 히스테리시스 사표(dead zone) 유지."""
    ma = px.rolling(sma).mean()
    dist = px / ma - 1
    raw = (dist > hyst).astype(float) - (dist < -hyst).astype(float)
    if invert:
        raw = -raw
    out, prev = [], False
    for v in raw.fillna(0).values:
        if v == 1:
            prev = True
        elif v == -1:
            prev = False
        out.append(prev)
    return pd.Series(out, index=px.index)


def main() -> None:
    import bt_fut_rebal_band as base
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3

    base.START, base.END = START, END
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")

    print(f"기간 {START} ~ {END}")
    pm = load_prices(list(ALL_TICKERS) + ["TIP"], start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    if "TIP" not in pdf.columns or pdf["TIP"].dropna().empty:
        raise SystemExit("TIP 가격이 없다 — data/TIP.csv 를 먼저 채워라")
    pdf = pdf.dropna(subset=["EEM", "TIP"])
    print(f"가격 {len(pdf)}일: {pdf.index[0].date()} ~ {pdf.index[-1].date()}")

    ranked, mom_off, mom_def, _ = precompute(pdf, [30, 72, 230], [42, 63, 126])

    arms: dict[str, pd.Series] = {}
    for sma in SMA_PERIODS:
        arms[f"EEM {sma}일(현행계열)"] = build_canary(pdf["EEM"], sma, HYST)
        arms[f"TIP {sma}일"] = build_canary(pdf["TIP"], sma, HYST)
    arms["TIP 200일(부호반전·진단용)"] = build_canary(pdf["TIP"], 200, HYST, invert=True)

    sim = pdf.index[(pdf.index >= START) & (pdf.index <= END)]
    print("\n카나리가 켜져 있던 날 비율 (구간 내)")
    for name, can in arms.items():
        on = can.reindex(sim).fillna(False)
        flips = int((on.astype(int).diff().abs() == 1).sum())
        print(f"  {name:<28} {on.mean()*100:>5.1f}%   전환 {flips:>3}회")

    print("\n슬리브 계산 — 코인 현물·선물은 고정 팔이다")
    eq_sp = base.run_spot_live()
    m_fu = base.run_fut(0.05)
    eq_fu = m_fu["_equity"]
    for nm, eq in (("코인현물", eq_sp), ("코인선물", eq_fu)):
        mt = base.metrics(eq)
        print(f"  {nm:<8} CAGR {mt['CAGR']:>7.1f}%  MDD {mt['MDD']:>7.1f}%  Cal {mt['Cal']:>5.2f}")

    # 거래비용은 bt_stock_mom3 이 import 시점에 값으로 받아간다(from ... import TX).
    # 그래서 원본 모듈이 아니라 실제로 쓰는 모듈의 전역을 바꿔야 한다.
    import bt_stock_mom3 as bsm3

    def stock_eq(can: pd.Series, anchor: int, tx: float):
        old_tx = bsm3.TX
        bsm3.TX = tx
        try:
            m = run_multi_3mom(pdf, ranked, mom_off, mom_def, can,
                               pd.Timestamp(START), pd.Timestamp(END), anchor=anchor,
                               drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                               snap_int=69, n_snaps=3)
        finally:
            bsm3.TX = old_tx
        if m is None:
            return None, None
        eq = m["_equity"] if isinstance(m, dict) else m
        turn = m.get("_turnover") if isinstance(m, dict) else None
        return eq, turn

    # ── 1) 앵커 0 기준 전체 성과 ──────────────────────────────────────────
    print("\n주식 슬리브 단독 (앵커 0, 거래비용 기본)")
    print(f"{'팔':<28}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
    stock_eqs: dict[str, pd.Series] = {}
    for name, can in arms.items():
        eq, _ = stock_eq(can, 0, BASE_TX)
        if eq is None:
            print(f"  {name:<26} 실패")
            continue
        stock_eqs[name] = eq
        mt = base.metrics(eq)
        print(f"{name:<28}{mt['CAGR']:>8.1f}%{mt['MDD']:>8.1f}%"
              f"{mt['Sharpe']:>8.2f}{mt['Cal']:>8.2f}")

    print("\n합성 60/25/15 (앵커 0, 거래비용 기본)")
    print(f"{'팔':<28}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
    comps: dict[str, pd.Series] = {}
    for name, eq in stock_eqs.items():
        c = base.build_alloc(eq, eq_sp, eq_fu)
        comps[name] = c
        mt = base.metrics(c)
        print(f"{name:<28}{mt['CAGR']:>8.1f}%{mt['MDD']:>8.1f}%"
              f"{mt['Sharpe']:>8.2f}{mt['Cal']:>8.2f}")

    # ── 2) 윈도우 순위합 ─────────────────────────────────────────────────
    sums, wins, n = base.window_rs(comps)
    print(f"\n윈도우 순위합 — 합성 기준, 창 {base.WIN_SIZES} × 보폭 {base.STRIDES}, n={n}")
    print(f"{'팔':<28}{'평균순위':>10}{'1위횟수':>9}")
    for k, v in sorted(sums.items(), key=lambda x: x[1] / max(n, 1)):
        print(f"{k:<28}{v/max(n,1):>10.3f}{wins[k]:>9}")

    # ── 3) 거래비용 5배 ─────────────────────────────────────────────────
    print(f"\n거래비용 5배({STRESS_TX*100:.1f}%) — 주식 슬리브만 가중, 코인 팔 고정")
    comps_s: dict[str, pd.Series] = {}
    for name, can in arms.items():
        eq, _ = stock_eq(can, 0, STRESS_TX)
        if eq is None:
            continue
        comps_s[name] = base.build_alloc(eq, eq_sp, eq_fu)
    if comps_s:
        s2, w2, n2 = base.window_rs(comps_s)
        print(f"{'팔':<28}{'평균순위':>10}{'Calmar':>9}")
        for k, v in sorted(s2.items(), key=lambda x: x[1] / max(n2, 1)):
            print(f"{k:<28}{v/max(n2,1):>10.3f}{base.metrics(comps_s[k])['Cal']:>9.2f}")

    # ── 4) 앵커 분산 ────────────────────────────────────────────────────
    print(f"\n앵커 분산 — 앵커 {ANCHORS[0]}~{ANCHORS[-1]}, 합성 Calmar")
    print(f"{'팔':<28}{'평균':>8}{'표준편차':>10}{'최소':>8}{'최대':>8}")
    for name, can in arms.items():
        cals = []
        for a in ANCHORS:
            eq, _ = stock_eq(can, a, BASE_TX)
            if eq is None:
                continue
            cals.append(base.metrics(base.build_alloc(eq, eq_sp, eq_fu))["Cal"])
        if not cals:
            continue
        arr = np.array(cals)
        print(f"{name:<28}{arr.mean():>8.2f}{arr.std():>10.3f}"
              f"{arr.min():>8.2f}{arr.max():>8.2f}")

    print("\n판정은 보고서에서 한다. 위 여섯 기준을 순서대로 대조할 것.")


if __name__ == "__main__":
    main()
