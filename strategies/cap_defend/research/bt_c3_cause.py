"""C3 후속 — 단순평균이 이긴 이유가 무엇인지 가른다 (2026-09-22).

C3 플래토 점검에서 기대와 달리 단순평균이 이겼다(합성 윈도우 순위합 평균순위
1.254 vs 현행 1.746). 사전 선언대로 채택이 아니라 원인 조사로 왔다.

가르려는 것 둘이다.
  1. 이긴 원인이 189일 항목을 새로 넣은 것인가, 아니면 가중을 균등으로 바꾼 것인가.
     둘을 분리하려면 189일 없는 균등평균(63·126·252)을 같이 재면 된다.
  2. 우위가 구간에 고르게 퍼져 있나, 아니면 몇 구간에 얹혀 있나.
     한 구간을 빼면 부호가 뒤집히는 우위는 규칙이 아니라 사건이다.

이건 최적화가 아니다. 가중치 격자를 돌리지 않는다 — 돌리는 순간 "현행이 과거에
맞춰져 있었나" 라는 물음이 "어느 가중치가 과거에 제일 맞나" 로 바뀐다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_c3_cause.py
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


def main() -> None:
    import bt_fut_rebal_band as base
    import bt_stock_mom3 as bsm3
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3

    base.START, base.END = START, END
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")

    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    print(f"기간 {START} ~ {END} · 가격 {len(pdf)}일")

    pre = {}
    for mode, label in (("weighted", "가중 0.5/0.3/0.2 (현행)"),
                        ("avg3", "균등평균 63·126·252"),
                        ("avg4", "균등평균 63·126·189·252")):
        os.environ["STOCK_ZMOM"] = mode
        pre[label] = precompute(pdf, [30, 72, 230], [42, 63, 126])
    os.environ["STOCK_ZMOM"] = "weighted"

    eq_sp = base.run_spot_live()
    eq_fu = base.run_fut(0.05)["_equity"]

    def stock_eq(key: str):
        ranked, mom_off, mom_def, canary = pre[key]
        old = bsm3.TX
        bsm3.TX = BASE_TX
        try:
            m = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                               pd.Timestamp(START), pd.Timestamp(END), anchor=0,
                               drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                               snap_int=69, n_snaps=3)
        finally:
            bsm3.TX = old
        return m["_equity"] if isinstance(m, dict) else m

    print("\n[1] 189일 항목인가, 균등 가중인가")
    print(f"{'팔':<28}{'주식CAGR':>10}{'주식Cal':>9}{'합성CAGR':>10}{'합성Cal':>9}")
    comps, stocks = {}, {}
    for key in pre:
        eq = stock_eq(key)
        stocks[key] = eq
        c = base.build_alloc(eq, eq_sp, eq_fu)
        comps[key] = c
        ms, mc = base.metrics(eq), base.metrics(c)
        print(f"{key:<28}{ms['CAGR']:>9.1f}%{ms['Cal']:>9.2f}"
              f"{mc['CAGR']:>9.1f}%{mc['Cal']:>9.2f}")

    sums, wins, n = base.window_rs(comps)
    print(f"\n합성 윈도우 순위합 n={n}")
    print(f"{'팔':<28}{'평균순위':>10}{'1위횟수':>9}")
    for k, v in sorted(sums.items(), key=lambda x: x[1] / max(n, 1)):
        print(f"{k:<28}{v/max(n,1):>10.3f}{wins[k]:>9}")

    print("\n[2] 우위가 어디에 얹혀 있나 — 연도별 주식 슬리브 수익률")
    cur = "가중 0.5/0.3/0.2 (현행)"
    alt = "균등평균 63·126·189·252"
    idx = stocks[cur].index.intersection(stocks[alt].index)
    print(f"{'연도':<8}{'현행':>10}{'단순평균':>10}{'차이':>10}{'거래일':>8}")
    yearly = {}
    for y, seg in pd.Series(idx, index=idx).groupby(idx.year):
        d = seg.index
        a = stocks[cur].loc[d]
        b = stocks[alt].loc[d]
        ra = a.iloc[-1] / a.iloc[0] - 1
        rb = b.iloc[-1] / b.iloc[0] - 1
        yearly[y] = (ra, rb)
        print(f"{y:<8}{ra*100:>9.1f}%{rb*100:>9.1f}%{(rb-ra)*100:>+9.1f}%p{len(d):>8}")

    print("\n[3] 한 해를 빼면 순위가 뒤집히나 — 합성 윈도우 순위합 재계산")
    print(f"{'제외한 해':<12}{'현행 평균순위':>14}{'단순평균 평균順':>16}{'승자':>8}")
    for y in sorted(yearly):
        keep = {k: v[v.index.year != y] for k, v in
                {cur: comps[cur], alt: comps[alt]}.items()}
        keep = {k: (v / v.iloc[0]) for k, v in keep.items()}
        s2, w2, n2 = base.window_rs(keep)
        if n2 == 0:
            continue
        ra, rb = s2[cur] / n2, s2[alt] / n2
        print(f"{y:<12}{ra:>14.3f}{rb:>16.3f}{('단순평균' if rb < ra else '현행'):>8}")

    print("\n[4] 선정이 실제로 갈린 날의 성격")
    ra_, rb_ = pre[cur][0], pre[alt][0]
    sim = pdf.index[(pdf.index >= START) & (pdf.index <= END)]
    rows = []
    for d in sim:
        if d not in ra_.index or d not in rb_.index:
            continue
        a, b = ra_.at[d], rb_.at[d]
        if not a or not b:
            continue
        rows.append((d, a[0] != b[0], set(a[:3]) != set(b[:3]), a[0], b[0]))
    df = pd.DataFrame(rows, columns=["date", "top1", "top3", "cur1", "alt1"])
    print(f"  전체 {len(df)}일 · 1위 다름 {df.top1.sum()}일 · 상위3 집합 다름 {df.top3.sum()}일")
    print("  1위가 갈릴 때 현행이 고른 종목 상위:")
    for t, c in df[df.top1].cur1.value_counts().head(5).items():
        print(f"    {t:<6}{c:>5}일")
    print("  1위가 갈릴 때 단순평균이 고른 종목 상위:")
    for t, c in df[df.top1].alt1.value_counts().head(5).items():
        print(f"    {t:<6}{c:>5}일")
    print("  연도별 1위 불일치 일수:")
    for y, c in df[df.top1].groupby(df[df.top1].date.dt.year).size().items():
        print(f"    {y}  {c}일")


if __name__ == "__main__":
    main()
