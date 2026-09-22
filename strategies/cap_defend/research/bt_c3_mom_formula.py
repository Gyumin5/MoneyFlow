"""C3 — 주식 Z점수 모멘텀 산식을 단순평균으로 바꿔 플래토를 점검한다 (2026-09-22).

계기: 2026-09-22 논문 조사에서 추린 백테스트 후보 세 건 중 셋째다. 상장지수펀드
모멘텀 문헌 다수가 여러 기간 수익률의 단순평균을 쓰는데, 우리는 63·126·252일에
0.5·0.3·0.2 가중을 준다. 그 가중치가 어디서 왔는지 지금 근거가 남아 있지 않다.

이건 개선 실험이 아니라 플래토 점검이다. 기대하는 결과는 "거의 같다" 다.
  - 거의 같으면 통과. 가중치가 특정 값에 걸려 있지 않다는 뜻이고, 현행을 유지한다.
  - 단순평균이 뚜렷이 나쁘면 현행 가중에 근거가 생긴 것이므로 역시 현행 유지다.
  - 단순평균이 뚜렷이 좋으면 그게 문제다. 축 하나 바꿔 크게 좋아진다는 건 현행
    가중치가 과거 데이터에 맞춰져 있었다는 신호이므로, 채택이 아니라 원인 조사로 간다.
세 갈래 전부 결론은 "이번에 라이브를 바꾸지 않는다" 다. 그 점을 미리 적어 둔다.

바꾸는 축은 하나다. Z점수 랭킹에 들어가는 모멘텀 정의만
  0.5×63일 + 0.3×126일 + 0.2×252일  ->  (63일 + 126일 + 189일 + 252일) / 4
로 바꾼다. 3모멘텀 필터(30·72·230), 카나리, 상한, 스냅, 스태거, 드리프트, 비용은 고정.
구현은 채택 BT 에 이미 있던 환경변수 STOCK_ZMOM 에 갈래 하나(avg4)를 더한 것이고
기본값은 그대로 weighted 다 — 아무것도 지정하지 않으면 현행과 동일하게 돈다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 다.
  2. 윈도우 순위합 평균순위 차이가 0.1 이내면 "같다" 로 본다.
  3. 앵커 0~10 분산을 같이 본다. 평균이 비슷해도 분산이 크게 다르면 같다고 하지 않는다.
  4. 스윕하지 않는다. 가중치 격자를 돌리는 순간 이 점검의 목적이 사라진다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_c3_mom_formula.py
출력은 state/20260922-c3-mom-formula/ 로 리다이렉트한다.
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
END = os.environ.get("C3_END", "2026-09-18")
BASE_TX = 0.001
STRESS_TX = 0.005
ANCHORS = list(range(0, 11))


def main() -> None:
    import bt_fut_rebal_band as base
    import bt_stock_mom3 as bsm3
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3

    base.START, base.END = START, END
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")

    print(f"기간 {START} ~ {END}")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    print(f"가격 {len(pdf)}일: {pdf.index[0].date()} ~ {pdf.index[-1].date()}")

    pre: dict[str, tuple] = {}
    for mode, label in (("weighted", "가중 0.5/0.3/0.2 (현행)"),
                        ("avg4", "단순평균 63·126·189·252")):
        os.environ["STOCK_ZMOM"] = mode
        pre[label] = precompute(pdf, [30, 72, 230], [42, 63, 126])
    os.environ["STOCK_ZMOM"] = "weighted"

    # 랭킹이 실제로 얼마나 갈리는지 먼저 본다. 안 갈리면 성과 비교는 의미가 없다.
    ra, rb = pre["가중 0.5/0.3/0.2 (현행)"][0], pre["단순평균 63·126·189·252"][0]
    sim = pdf.index[(pdf.index >= START) & (pdf.index <= END)]
    diff_top1 = diff_top3 = tot = 0
    for d in sim:
        if d not in ra.index or d not in rb.index:
            continue
        a, b = ra.at[d], rb.at[d]
        if not a or not b:
            continue
        tot += 1
        if a[0] != b[0]:
            diff_top1 += 1
        if set(a[:3]) != set(b[:3]):
            diff_top3 += 1
    print(f"\n랭킹 차이: 1위가 다른 날 {diff_top1}/{tot} ({diff_top1/max(tot,1)*100:.1f}%), "
          f"상위 3 집합이 다른 날 {diff_top3}/{tot} ({diff_top3/max(tot,1)*100:.1f}%)")

    print("\n슬리브 계산 — 코인 현물·선물은 고정 팔이다")
    eq_sp = base.run_spot_live()
    eq_fu = base.run_fut(0.05)["_equity"]

    def stock_eq(key: str, anchor: int, tx: float):
        ranked, mom_off, mom_def, canary = pre[key]
        old = bsm3.TX
        bsm3.TX = tx
        try:
            m = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                               pd.Timestamp(START), pd.Timestamp(END), anchor=anchor,
                               drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                               snap_int=69, n_snaps=3)
        finally:
            bsm3.TX = old
        if m is None:
            return None
        return m["_equity"] if isinstance(m, dict) else m

    print("\n주식 슬리브 단독 (앵커 0)")
    print(f"{'팔':<28}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
    comps: dict[str, pd.Series] = {}
    for key in pre:
        eq = stock_eq(key, 0, BASE_TX)
        if eq is None:
            print(f"{key:<28} 실패")
            continue
        mt = base.metrics(eq)
        print(f"{key:<28}{mt['CAGR']:>8.1f}%{mt['MDD']:>8.1f}%"
              f"{mt['Sharpe']:>8.2f}{mt['Cal']:>8.2f}")
        comps[key] = base.build_alloc(eq, eq_sp, eq_fu)

    print("\n합성 60/25/15 (앵커 0)")
    print(f"{'팔':<28}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'Calmar':>8}")
    for key, c in comps.items():
        mt = base.metrics(c)
        print(f"{key:<28}{mt['CAGR']:>8.1f}%{mt['MDD']:>8.1f}%"
              f"{mt['Sharpe']:>8.2f}{mt['Cal']:>8.2f}")

    sums, wins, n = base.window_rs(comps)
    print(f"\n윈도우 순위합 — 합성, 창 {base.WIN_SIZES} × 보폭 {base.STRIDES}, n={n}")
    print(f"{'팔':<28}{'평균순위':>10}{'1위횟수':>9}")
    ranks = {}
    for k, v in sorted(sums.items(), key=lambda x: x[1] / max(n, 1)):
        ranks[k] = v / max(n, 1)
        print(f"{k:<28}{ranks[k]:>10.3f}{wins[k]:>9}")
    if len(ranks) == 2:
        gap = abs(list(ranks.values())[0] - list(ranks.values())[1])
        print(f"  평균순위 차이 {gap:.3f} — 0.1 이내면 같다고 본다")

    print(f"\n거래비용 5배({STRESS_TX*100:.1f}%)")
    comps_s = {}
    for key in pre:
        eq = stock_eq(key, 0, STRESS_TX)
        if eq is not None:
            comps_s[key] = base.build_alloc(eq, eq_sp, eq_fu)
    if len(comps_s) == 2:
        s2, w2, n2 = base.window_rs(comps_s)
        for k, v in sorted(s2.items(), key=lambda x: x[1] / max(n2, 1)):
            print(f"{k:<28}{v/max(n2,1):>10.3f}"
                  f"{base.metrics(comps_s[k])['Cal']:>9.2f}")

    print(f"\n앵커 분산 — 앵커 {ANCHORS[0]}~{ANCHORS[-1]}, 합성 Calmar")
    print(f"{'팔':<28}{'평균':>8}{'표준편차':>10}{'최소':>8}{'최대':>8}")
    for key in pre:
        cals = []
        for a in ANCHORS:
            eq = stock_eq(key, a, BASE_TX)
            if eq is not None:
                cals.append(base.metrics(base.build_alloc(eq, eq_sp, eq_fu))["Cal"])
        if cals:
            arr = np.array(cals)
            print(f"{key:<28}{arr.mean():>8.2f}{arr.std():>10.3f}"
                  f"{arr.min():>8.2f}{arr.max():>8.2f}")

    print("\n판정은 보고서에서 한다. 스윕하지 않는다.")


if __name__ == "__main__":
    main()
