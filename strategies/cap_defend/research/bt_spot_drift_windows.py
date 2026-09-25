"""느슨한 쪽 문턱의 이득이 한 구간에 묶여 있나 (2026-09-25).

교정된 현금버퍼 1% 위에서 현물 드리프트 문턱을 다시 재니 0.15·0.20 이 두 비용 조건 모두에서
현행 0.10 을 이겼다. 착수 전 고정 기준으로는 대안 후보 자격이 있다.

그런데 이번 주에 같은 함정을 두 번 봤다. 앵커 일곱 곳 전승은 독립 검증 일곱 번이 아니라
같은 사건을 위상만 바꿔 일곱 번 센 것일 수 있고, 최대낙폭 통계는 가장 깊은 구간 하나만
기록해 다른 구간의 악화를 감춘다.

그래서 채택 판단 전에 둘을 본다.
  1. 평가창을 현행 최대낙폭 최저점을 품은 것과 안 품은 것으로 가르고, 각 무리에서
     칼마 승률과 차이 분포를 센다. 한쪽에서만 이기면 그 이득은 사건 하나다.
  2. 낙폭 구간별로 깊이를 대조한다. 어떤 구간이 더 깊어졌는지 최대낙폭만으로는 안 보인다.
  3. 초과 로그수익 합 — 이득이 수익에서 오나 낙폭에서 오나.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_drift_windows.py
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

import bt_coin_param_resweep as R  # noqa: E402

R.OUT = "/home/gmoh/mon/251229/state/20260925-spotdrift-buf"
R.EQDIR = os.path.join(R.OUT, "eq")

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]
ARMS = ["현물 0.15", "현물 0.20"]


def _cal(eq: pd.Series) -> float:
    if len(eq) < 30 or eq.iloc[0] <= 0:
        return float('nan')
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0:
        return float('nan')
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    return float('nan') if mdd >= -1e-9 else cagr / abs(mdd)


def _alloc(name: str, a: int, stock_eq: dict) -> pd.Series | None:
    sp = R._load("sp", R._key(name, a, False))
    fu = R._load("fu", R._key(name, a, False))
    if sp is None or fu is None:
        return None
    return R.build_alloc(stock_eq[a], sp, fu)


def episodes(eq: pd.Series, n: int = 5) -> list[tuple[pd.Timestamp, float]]:
    """깊은 낙폭 구간 상위 n개의 최저점 날짜와 깊이."""
    dd = eq / eq.cummax() - 1
    out, used = [], set()
    d = dd.copy()
    for _ in range(n):
        if d.empty or d.min() >= -1e-9:
            break
        t = d.idxmin()
        out.append((t, float(d.loc[t])))
        lo = eq.index.get_indexer([t])[0]
        i, j = lo, lo
        while i > 0 and dd.iloc[i - 1] < -1e-9:
            i -= 1
        while j + 1 < len(dd) and dd.iloc[j + 1] < -1e-9:
            j += 1
        seg = set(eq.index[i:j + 1])
        used |= seg
        d = dd.drop(index=[x for x in seg if x in d.index])
    return out


def main() -> None:
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if not os.path.exists(p):
            print(f"주식 곡선 없음 a{a} — 스윕을 먼저 돌려라")
            return
        stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]

    base0 = _alloc(R.CUR, 0, stock_eq)
    if base0 is None:
        print("현행 곡선 없음 — 스윕을 먼저 돌려라")
        return
    ddb = base0 / base0.cummax() - 1
    event = ddb.idxmin()
    print(f"기준 사건: {pd.Timestamp(event).date()} (현행 합성 최대낙폭 최저점, "
          f"{float(ddb.min()):.2%})\n")

    for arm in ARMS:
        wi = {True: [], False: []}
        exc = []
        for a in R.ANCHORS:
            b = _alloc(R.CUR, a, stock_eq)
            t = _alloc(arm, a, stock_eq)
            if b is None or t is None:
                continue
            i = b.index.intersection(t.index)
            b, t = b.loc[i], t.loc[i]
            exc.append(float(np.log(t.iloc[-1] / t.iloc[0]) - np.log(b.iloc[-1] / b.iloc[0])))
            for w in WIN_SIZES:
                for s in STRIDES:
                    for k in range(0, len(i) - w, s):
                        seg = i[k:k + w]
                        cb, ct = _cal(b.loc[seg]), _cal(t.loc[seg])
                        if np.isnan(cb) or np.isnan(ct):
                            continue
                        wi[bool(seg[0] <= event <= seg[-1])].append(ct - cb)

        print(f"=== {arm} 대 현행 ===")
        print(f"  5.9년 초과 로그수익(앵커별): "
              f"{' '.join(f'{v:+.4f}' for v in exc)}")
        for has in (True, False):
            d = np.array(wi[has], dtype=float)
            if not len(d):
                continue
            lbl = "그 날을 품은 창" if has else "그 날이 없는 창"
            print(f"  [{lbl}] 창 {len(d)}개 · 이긴 창 {int((d > 0).sum())}개 "
                  f"({(d > 0).mean():.0%}) · 평균 {d.mean():+.3f} 중앙 {np.median(d):+.3f} "
                  f"최소 {d.min():+.3f} 최대 {d.max():+.3f}")

        ta = _alloc(arm, 0, stock_eq)
        if ta is None:
            continue
        print("  낙폭 구간별 깊이(앵커 0) — 현행 기준 상위 5개")
        for t0, depth in episodes(base0):
            i0 = max(0, base0.index.get_indexer([t0])[0] - 120)
            i1 = min(len(base0) - 1, base0.index.get_indexer([t0])[0] + 120)
            seg = base0.index[i0:i1 + 1]
            sa = ta.loc[ta.index.intersection(seg)]
            da = float((sa / sa.cummax() - 1).min()) if len(sa) else float('nan')
            print(f"    {pd.Timestamp(t0).date()}  현행 {depth:+.2%}  {arm} {da:+.2%}  "
                  f"차이 {(da - depth) * 100:+.2f}%p")
        print()


if __name__ == '__main__':
    main()
