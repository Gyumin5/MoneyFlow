"""C3 판정을 가른다 — 단순평균 우위가 앵커 하나짜리인가 (2026-09-22).

bt_c3_mom_formula.py 에서 두 게이트가 서로 다른 말을 했다.
  · 윈도우 순위합(앵커 0): 단순평균 1.254 vs 현행 1.746 — 사전 문턱 0.1 을 크게 넘어 단순평균 우세.
  · 앵커 0~10 합성 칼마: 현행 평균 3.21, 단순평균 3.20. 표준편차·최소·최대가 셋 다 같다.

순위합은 앵커 0 한 곳에서만 쟀고, 앵커 평균은 수준값만 봤다. 프로젝트 채택 기준은
"앵커 하나에서만 유난히 좋으면 기각" 이므로 이 둘이 갈리면 판정을 못 한다.

그래서 앵커마다 순위합을 따로 낸다. 물음은 하나다 — 단순평균이 앵커 11곳에서 다 이기나,
아니면 앵커 0 에서만 이기나.
  · 다 이기면 앵커 의존이 아니다. 수준 차이가 작을 뿐 방향은 일관된 것이고, 채택 후보로 남는다.
  · 앵커마다 승자가 갈리면 앵커 0 결과는 대표성이 없다. 기각한다.

바꾸는 축은 없다. bt_c3_mom_formula 와 같은 팔·같은 기간·같은 비용이고, 앵커만 돌린다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_c3_anchor_rs.py
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
ANCHORS = list(range(0, 11))

CUR = "가중 0.5/0.3/0.2 (현행)"
ALT = "단순평균 63·126·189·252"


def main() -> None:
    import bt_fut_rebal_band as base
    import bt_stock_mom3 as bsm3
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3

    base.START, base.END = START, END
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")

    print(f"기간 {START} ~ {END} · 앵커 {ANCHORS[0]}~{ANCHORS[-1]} · 비용 {BASE_TX*100:.1f}%")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    pre: dict[str, tuple] = {}
    for mode, label in (("weighted", CUR), ("avg4", ALT)):
        os.environ["STOCK_ZMOM"] = mode
        pre[label] = precompute(pdf, [30, 72, 230], [42, 63, 126])
    os.environ["STOCK_ZMOM"] = "weighted"

    eq_sp = base.run_spot_live()
    eq_fu = base.run_fut(0.05)["_equity"]

    def stock_eq(key: str, anchor: int):
        ranked, mom_off, mom_def, canary = pre[key]
        old = bsm3.TX
        bsm3.TX = BASE_TX
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

    print(f"\n앵커별 — 합성 60/25/15 윈도우 순위합 (창 {base.WIN_SIZES} × 보폭 {base.STRIDES})")
    print(f"{'앵커':<6}{'현행 평균순위':>14}{'단순평균 평균순위':>18}{'승자':>10}"
          f"{'현행 Cal':>11}{'단순평균 Cal':>14}")
    wins = {CUR: 0, ALT: 0}
    gaps, dcal = [], []
    for a in ANCHORS:
        comps = {}
        for key in (CUR, ALT):
            eq = stock_eq(key, a)
            if eq is not None:
                comps[key] = base.build_alloc(eq, eq_sp, eq_fu)
        if len(comps) != 2:
            print(f"{a:<6} 실패")
            continue
        sums, _, n = base.window_rs(comps)
        rc, ra_ = sums[CUR] / max(n, 1), sums[ALT] / max(n, 1)
        cc = base.metrics(comps[CUR])["Cal"]
        ca = base.metrics(comps[ALT])["Cal"]
        w = ALT if ra_ < rc else CUR
        wins[w] += 1
        gaps.append(rc - ra_)
        dcal.append(ca - cc)
        label = "단순평균" if w == ALT else "현행"
        print(f"{a:<6}{rc:>14.3f}{ra_:>18.3f}{label:>10}{cc:>11.2f}{ca:>14.2f}")

    g = np.array(gaps)
    d = np.array(dcal)
    print(f"\n앵커 {len(gaps)}곳 중 단순평균 승 {wins[ALT]}곳 · 현행 승 {wins[CUR]}곳")
    print(f"평균순위 차이(현행 − 단순평균) 평균 {g.mean():+.3f} · 최소 {g.min():+.3f} · 최대 {g.max():+.3f}")
    print(f"  (양수면 단순평균 우세. 사전 문턱은 0.1 이다)")
    print(f"합성 Calmar 차이(단순평균 − 현행) 평균 {d.mean():+.3f} · 최소 {d.min():+.3f} · 최대 {d.max():+.3f}")

    print("\n판정은 보고서에서 한다.")


if __name__ == "__main__":
    main()
