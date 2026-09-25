"""겹치지 않는 기간으로 잘라 0.15 를 현행과 직접 비교한다 (2026-09-25).

앞선 두 검증이 남긴 마지막 자리다. 문턱 재측정(423c60a)은 겹치는 창의 순위합으로 판정했고,
그 방식은 작은 차이를 큰 등수로 벌린다는 지적을 받았다. 집중 위험 집계(e43f855)는 꼬리로
0.20·끔을 떨어뜨리고 0.15 만 남겼다. 그래서 0.15 하나를, 창이 서로 겹치지 않는 구간으로
잘라 순위가 아니라 대응 수익률과 칼마 차이로 직접 본다.

겹치지 않게 자르는 이유는 하나다 — 겹치는 창에서는 같은 날이 여러 창에 들어가므로 한 구간의
우세가 여러 번 세어진다. 겹치지 않으면 각 구간이 독립 표본 한 개다(표본 수가 줄어드는 대신).

셋을 본다.
  1. 합성 60/25/15 를 겹치지 않는 블록으로 자르고 블록마다 연수익·최대낙폭·칼마 차이.
  2. 앵커 7곳 × 블록을 합쳐 부호 일관성(0.15 가 이긴 블록 비율).
  3. 현물 슬리브 단독으로도 같은 것 — 합성은 주식·선물이 희석한다.

블록 길이는 라운드 넘버로 둘 쓴다. 504일(약 2년, 5.9년을 4블록)과 378일(약 1.5년, 5블록).
곡선은 이미 계산된 캐시를 쓴다(state/20260925-spotdrift-buf/eq) — 새로 돌리는 것이 없다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_drift_disjoint.py
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

ARM = "현물 0.15"
BLOCKS = [504, 378]


def _stats(eq: pd.Series) -> tuple[float, float, float]:
    """연수익·최대낙폭·칼마."""
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return float('nan'), float('nan'), float('nan')
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd < -1e-9 else float('nan')
    return cagr, mdd, cal


def _alloc(name: str, a: int, stock_eq: dict, stress: bool) -> pd.Series | None:
    sp = R._load("sp", R._key(name, a, stress))
    fu = R._load("fu", R._key(name, a, stress))
    if sp is None or fu is None:
        return None
    return R.build_alloc(stock_eq[a], sp, fu)


def compare(label, get, stress: bool, back: bool = False) -> None:
    """back=True 면 블록을 끝에서부터 자른다 — 앞에서 자르면 나머지가 최근 쪽에서
    잘려나가 가장 최근 구간이 표본에서 빠지기 때문이다(504일 기준 약 1.8년)."""
    side = "끝에서 자름" if back else "앞에서 자름"
    print(f"\n=== {label} · {'비용 5배' if stress else '기본 비용'} · {side} ===")
    for w in BLOCKS:
        cal_d, cagr_d, mdd_d = [], [], []
        per_block: dict[int, list[float]] = {}
        for a in R.ANCHORS:
            b, t = get(R.CUR, a, stress), get(ARM, a, stress)
            if b is None or t is None:
                continue
            i = b.index.intersection(t.index)
            b, t = b.loc[i], t.loc[i]
            nb = len(i) // w
            off = len(i) - nb * w if back else 0
            for k in range(nb):
                seg = i[off + k * w:off + (k + 1) * w]
                cb, ct = _stats(b.loc[seg]), _stats(t.loc[seg])
                if np.isnan(cb[2]) or np.isnan(ct[2]):
                    continue
                cal_d.append(ct[2] - cb[2])
                cagr_d.append(ct[0] - cb[0])
                mdd_d.append(ct[1] - cb[1])
                per_block.setdefault(k, []).append(ct[2] - cb[2])
        if not cal_d:
            continue
        d = np.array(cal_d)
        print(f"  블록 {w}일 · 표본 {len(d)}개(앵커 7 × 블록 {len(per_block)})")
        print(f"    칼마 차 평균 {d.mean():+.3f} 중앙 {np.median(d):+.3f} "
              f"최소 {d.min():+.3f} 최대 {d.max():+.3f} · 이긴 표본 "
              f"{int((d > 0).sum())}/{len(d)} ({(d > 0).mean():.0%})")
        print(f"    연수익 차 평균 {np.mean(cagr_d):+.2%} · "
              f"최대낙폭 차 평균 {np.mean(mdd_d):+.2f}%p "
              f"(양수면 0.15 가 덜 깊다)")
        print("    블록별(시간순) 칼마 차 평균 · 이긴 앵커")
        for k in sorted(per_block):
            v = np.array(per_block[k])
            print(f"      {k + 1}번째 {v.mean():+.3f}  {int((v > 0).sum())}/{len(v)}")


def main() -> None:
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if not os.path.exists(p):
            print("주식 곡선 없음 — bt_spot_drift_buffer.py 를 먼저 돌려라")
            return
        stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]

    for stress in (False, True):
        for back in (False, True):
            compare("합성 60/25/15",
                    lambda n, a, s: _alloc(n, a, stock_eq, s), stress, back)
            compare("현물 슬리브 단독",
                    lambda n, a, s: R._load("sp", R._key(n, a, s)), stress, back)


if __name__ == '__main__':
    main()
