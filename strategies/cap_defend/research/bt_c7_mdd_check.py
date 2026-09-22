"""C7 의 최대낙폭이 다섯 팔에서 똑같이 나온 까닭을 확인한다 (2026-09-22).

bt_c7_etf_regime.py 의 C 절에서 합성 최대낙폭이 게이트 네 형태와 현행 모두 -15.5% 로
소수점까지 같게 나왔다. 두 갈래로 읽힌다.
  (가) 덮개가 실제로는 안 걸렸다 — 측정 오류다. 그러면 C7 기각 근거가 통째로 무너진다.
  (나) 덮개는 걸렸는데 낙폭이 생긴 구간에 게이트가 꺼져 있었다 — 그러면 "상관이 높을 때
       노출을 줄인다" 는 규칙이 정작 가장 아팠던 구간을 못 잡았다는 뜻이고, 기각이 더 강해진다.

가르는 법: 팔마다 최대낙폭의 고점일·저점일을 뽑고, 그 구간에서 게이트가 켜져 있던 날의
비율을 센다. 덮개가 걸렸는지는 그 구간의 평균 노출 배수로 직접 확인한다.
배수가 1.0 이면 (나)이고, 1.0 보다 작은데도 낙폭이 같으면 다시 봐야 한다.

bt_c7_etf_regime 의 함수를 그대로 가져다 쓴다 — 덮개 식을 다시 적으면 다른 것을 재게 된다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_c7_mdd_check.py
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


def dd_span(eq: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """최대낙폭의 고점일·저점일·깊이."""
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return peak, trough, float(dd.min()) * 100


def main() -> None:
    import bt_c7_etf_regime as c7

    END = os.environ.get("C7_END", "2026-09-18")

    # A 절에서 쓰는 알트끼리 평균 상관을 그대로 다시 만든다.
    basket = c7.alt_basket()
    cols = {}
    for t in basket:
        s = c7.load_close(t)
        if s is not None:
            cols[t] = s
    rets = pd.DataFrame(cols).pct_change().dropna(how="all")
    corr = c7.avg_pairwise_corr(rets, c7.ROLL)

    print(f"알트 바스켓 {len(cols)}종 · 상관 창 {c7.ROLL}일 · 끝 {END}")

    eqs = c7.part_b(END)["_eq"]
    eq_st, eq_sp, eq_fu = eqs["주식"], eqs["코인현물"], eqs["코인선물"]
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    r_st = eq_st.loc[common].pct_change().fillna(0)
    r_sp = eq_sp.loc[common].pct_change().fillna(0)
    r_fu = eq_fu.loc[common].pct_change().fillna(0)

    def arm(f: pd.Series) -> pd.Series:
        f = f.reindex(common).fillna(1.0)
        ret = 0.60 * r_st + 0.25 * f * r_sp + 0.15 * f * r_fu
        turn = f.diff().abs().fillna(0.0)
        cost = turn * (0.25 * c7.TX_SPOT + 0.15 * c7.TX_FUT)
        return (1 + ret - cost).cumprod()

    ones = pd.Series(1.0, index=common)
    arms = {"현행(게이트 없음)": (arm(ones), ones)}
    for q in c7.GATE_Q:
        on = c7.gate_series(corr, q).reindex(common).fillna(False)
        for fmul in c7.GATE_F:
            name = f"상위{int((1-q)*100)}% 넘으면 ×{fmul:.2f}"
            fs = pd.Series(np.where(on.values, fmul, 1.0), index=common)
            arms[name] = (arm(fs), fs)

    print("\n[1] 팔마다 최대낙폭 구간과 그 구간의 실제 노출 배수")
    print(f"{'팔':<26}{'MDD':>8}{'고점일':>13}{'저점일':>13}{'일수':>7}"
          f"{'구간 평균배수':>13}{'배수<1 인 날':>13}")
    for name, (eq, fs) in arms.items():
        peak, trough, mdd = dd_span(eq)
        seg = fs.loc[peak:trough]
        below = float((seg < 1.0).mean()) * 100
        print(f"{name:<26}{mdd:>7.2f}%{str(peak.date()):>13}{str(trough.date()):>13}"
              f"{len(seg):>7}{seg.mean():>13.3f}{below:>12.1f}%")

    print("\n[2] 덮개가 실제로 걸렸는지 — 전 구간 대조")
    print(f"{'팔':<26}{'전체 평균배수':>13}{'배수<1 인 날':>13}{'최종 배수전환':>14}{'최종 자산':>11}")
    for name, (eq, fs) in arms.items():
        below = float((fs < 1.0).mean()) * 100
        flips = int((fs.diff().abs() > 0).sum())
        print(f"{name:<26}{fs.mean():>13.3f}{below:>12.1f}%{flips:>14}{eq.iloc[-1]:>11.2f}")

    print("\n[3] 현행 팔의 최대낙폭 구간에서 게이트는 어땠나")
    peak, trough, mdd = dd_span(arms["현행(게이트 없음)"][0])
    print(f"  현행 최대낙폭 {mdd:.2f}% · {peak.date()} ~ {trough.date()} ({(trough-peak).days}일)")
    for q in c7.GATE_Q:
        on = c7.gate_series(corr, q).reindex(common).fillna(False)
        seg = on.loc[peak:trough]
        print(f"  상위{int((1-q)*100)}% 문턱 — 그 구간에서 켜진 날 {float(seg.mean())*100:.1f}% "
              f"({int(seg.sum())}/{len(seg)}일), 전 구간 켜진 날 {float(on.mean())*100:.1f}%")

    print("\n[4] 게이트가 켜진 날 코인 두 팔의 그날 수익은 실제로 나빴나")
    r_coin = (0.25 * r_sp + 0.15 * r_fu) / 0.40
    for q in c7.GATE_Q:
        on = c7.gate_series(corr, q).reindex(common).fillna(False)
        a, b = r_coin[on.values], r_coin[~on.values]
        print(f"  상위{int((1-q)*100)}% — 켜진 날 {len(a)}일 평균 {a.mean()*100:+.3f}% "
              f"· 꺼진 날 {len(b)}일 평균 {b.mean()*100:+.3f}%")
    print("  (켜진 날이 더 나쁘지 않으면 게이트 신호 자체에 정보가 없는 것이다)")


if __name__ == "__main__":
    main()
