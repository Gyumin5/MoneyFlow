"""주식 V25 채택 심사를 시점 고친 판에서 다시 매긴다 (2026-10-15).

운영자 물음 = "중요한건 전략간 순위가 변했는가". 코인 파라미터 축은
bt_rank_shift_check.py 가 센다. 이 파일은 그보다 윗층을 센다 — 주식 슬리브를
V24 에서 V25 로 바꾼 그 결정이다.

그 결정이 제일 위태롭다. 합성 성과가 시점 때문에 내려간 분(-0.086)은 사실상 전부
주식에서 왔고(주식만 고치면 -0.087, 현물만 고치면 +0.007), 주식은 합성의 60% 다.
채택 심사를 통째로 당일 신호 당일 체결 위에서 돌렸다면, 뽑힌 것이 전략이 아니라
시점일 수 있다.

## 재는 것

채택 심사 첫 관문(bt_stock_3mom_gates.py GATE 1)을 그대로 다시 돌린다. 팔도 앵커도
창도 안 바꾼다 — 바꾸는 것은 신호 시점 하나뿐이다.

  B      = 현행(V24) 다중 스냅, 모멘텀 필터 없음, 드리프트 0.10
  3m/2m  = 후보 넷 × 드리프트 문턱 다섯 = 스무 팔
  앵커 11곳(채택 당시 그대로), 창 순위합은 그때 쓰던 scorer 그대로

두 벌을 돌린다. 현행(STOCK_SIG_LAG=0)과 수정(=1).

## 판정 (결과 보기 전에 못박는다)

  (가) 채택된 팔(3m_30_72_230 thr 0.05)이 두 벌 모두에서 B 를 이기나.
       한쪽에서만 이기면 그 채택은 시점에 기대고 있었다는 뜻이다.
  (나) 1등이 바뀌나. 바뀌면 지금 라이브 파라미터가 1등이 아니게 된다.
  (다) 채택된 팔의 등수가 몇에서 몇으로 가나.

채택 판정이 아니다. 순위가 바뀌어도 이 측정이 라이브를 바꾸자고 말하지 않는다 —
사실만 적고 재선정은 별도 결정이다.

## 선행조건

bt_stock_mom3 과 bt_stock_window_rank 두 곳에 STOCK_SIG_LAG 토글이 있어야 한다
(둘 다 0 이면 기존 동작과 같다). 측정이 끝나면 되돌리고 재현 패치만 남긴다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_stock_adoption_rank.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_stock_coin_v3 import precompute  # noqa: E402
from bt_stock_mom3 import run_multi_3mom  # noqa: E402
from bt_stock_mom_grid import window_rank_sum_multi  # noqa: E402
from bt_stock_window_rank import _run_multi_eq  # noqa: E402
from stock_engine import ALL_TICKERS, load_prices  # noqa: E402

import bt_stock_coin_v3 as bcv3  # noqa: E402

# 채택 심사가 쓰던 구간·모멘텀 기간 그대로. 다른 값을 쓰면 그 결정의 재현이 아니다.
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2026-05-13")
ALL_PERIODS = sorted({30, 42, 45, 72, 84, 96, 105, 126, 200, 210, 220, 230, 240, 250, 260})
ANCHORS = range(0, 11)
THRS = [0.05, 0.08, 0.10, 0.12, 0.15]
CFGS = [
    ("3m_30_72_230", "3m", 30, 72, 230),
    ("3m_42_72_230", "3m", 42, 72, 230),
    ("2m_45_210", "2m", 45, None, 210),
    ("2m_30_84", "2m", 30, None, 84),
]
ADOPTED = "3m_30_72_230_thr0.05"


def build(pdf, ranked, mom_off, mom_def, canary, anchor) -> dict:
    eqs = {}
    b = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, START, END, anchor,
                      use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode="cap")
    if b is None:
        return {}
    eqs["B"] = b
    for tag, kind, ms, mid, ml in CFGS:
        for thr in THRS:
            key = f"{tag}_thr{thr:.2f}"
            if kind == "3m":
                eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, START, END,
                                    anchor, drift_thr=thr, cash_buf=0.07,
                                    ms=ms, mid=mid, ml=ml)
            else:
                eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, START, END,
                                   anchor, use_mom=True, drift_thr=thr, cash_buf=0.07,
                                   weight_mode="cap", ms=ms, ml=ml)
            if eq is not None:
                eqs[key] = eq
    return eqs


def score(pdf, ranked, mom_off, mom_def, canary, lag: str) -> tuple[dict, int]:
    os.environ["STOCK_SIG_LAG"] = lag
    try:
        sums: dict[str, float] = defaultdict(float)
        n_tot = 0
        for anchor in ANCHORS:
            eqs = build(pdf, ranked, mom_off, mom_def, canary, anchor)
            if not eqs:
                continue
            rs = window_rank_sum_multi(eqs)
            if rs is None:
                continue
            s, _w, n = rs
            for k, v in s.items():
                sums[k] += v
            n_tot += n
    finally:
        os.environ["STOCK_SIG_LAG"] = "0"
    return ({k: v / n_tot for k, v in sums.items()} if n_tot else {}), n_tot


def main() -> int:
    t0 = time.time()
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, ALL_PERIODS, [42, 63, 126])
    print(f"가격 준비 {time.time()-t0:.1f}초", flush=True)

    print("사전 확인 — 토글이 두 하니스 모두에 걸리나")
    probe = {}
    for lg in ("0", "1"):
        os.environ["STOCK_SIG_LAG"] = lg
        try:
            probe[("3mom", lg)] = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                                                 START, END, 0, drift_thr=0.05,
                                                 cash_buf=0.07, ms=30, mid=72, ml=230)
            probe[("B", lg)] = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary,
                                             START, END, 0, use_mom=False,
                                             drift_thr=0.10, cash_buf=0.07,
                                             weight_mode="cap")
        finally:
            os.environ["STOCK_SIG_LAG"] = "0"
    for nm in ("3mom", "B"):
        g = float((probe[(nm, "0")] - probe[(nm, "1")]).abs().max())
        print(f"  {nm} 두 팔 최대 차이 {g:.6g}")
        if g <= 1e-9:
            print("  멈춘다 — 토글이 안 걸린 하니스가 있다.")
            return 1
    print("  둘 다 걸린다. 본 측정으로 간다.\n", flush=True)

    out = {}
    for label, lag in (("현행(당일 신호)", "0"), ("수정(전 거래일 신호)", "1")):
        r, n = score(pdf, ranked, mom_off, mom_def, canary, lag)
        out[label] = r
        print(f"[{label}] 창 {n}개 · 팔 {len(r)}개 ({time.time()-t0:.0f}초)", flush=True)
        for k, v in sorted(r.items(), key=lambda x: x[1])[:6]:
            star = " ←채택" if k == ADOPTED else (" ←V24" if k == "B" else "")
            print(f"    {k:<24} 평균순위 {v:.3f}{star}")
        print(flush=True)

    a, b = out["현행(당일 신호)"], out["수정(전 거래일 신호)"]
    if not a or not b:
        print("측정 실패 — 한쪽이 비었다.")
        return 1

    def pos(r, k):
        return sorted(r, key=lambda x: r[x]).index(k) + 1

    print("=" * 66)
    w_a = min(a, key=lambda k: a[k])
    w_b = min(b, key=lambda k: b[k])
    print(f"1등  현행 {w_a}  →  수정 {w_b}   ({'바뀜' if w_a != w_b else '같음'})")
    for k in (ADOPTED, "B"):
        if k in a and k in b:
            nm = "채택 팔" if k == ADOPTED else "V24(B)"
            print(f"{nm:8s} {k:<24} 평균순위 {a[k]:.3f} → {b[k]:.3f} · "
                  f"등수 {pos(a,k)} → {pos(b,k)} (전체 {len(a)})")
    if ADOPTED in a and "B" in a:
        beat_a = a[ADOPTED] < a["B"]
        beat_b = b[ADOPTED] < b["B"]
        print(f"\n채택 팔이 V24 를 이기나 — 현행 {'예' if beat_a else '아니오'} · "
              f"수정 {'예' if beat_b else '아니오'}")
        if beat_a and beat_b:
            print("  두 벌 모두에서 이긴다. V25 채택은 시점에 기대고 있지 않다.")
        elif beat_a and not beat_b:
            print("  현행에서만 이긴다. 그 채택은 시점 착오 위에서 났다는 뜻이다.")
        else:
            print("  현행에서도 안 이긴다 — 채택 당시 조건을 이 하니스가 그대로 재현하지")
            print("  못하고 있다는 뜻일 수 있다. 결론 내기 전에 재현부터 의심한다.")

    swaps = 0
    keys = [k for k in a if k in b]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = keys[i], keys[j]
            if (a[x] - a[y]) * (b[x] - b[y]) < 0:
                swaps += 1
    print(f"\n뒤바뀐 쌍 {swaps}/{len(keys)*(len(keys)-1)//2}")
    print(f"\n총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
