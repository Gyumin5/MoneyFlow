"""변동성 가중 — 이득이 몇 날에 몰렸나, 회전율은 얼마나 느나 (2026-09-25).

ai-debate run-20260924T154821Z 의 검증 계획 중 남은 둘을 잰다.
  1. 초과수익 집중도 — 상위 5·10·20일이 전체 초과수익에서 차지하는 몫, 그 날을 뺀 결과,
     해별 초과수익. 2026-09-24 배수 축을 기각시킨 검사와 같은 잣대다.
  2. 실매매 마찰 — 리밸 횟수와 회전율, 그리고 종목별 주문금액이 업비트 최소주문 5,000원에
     걸리는 건수. 슬리브 규모를 몇 가지 가정해 센다.

읽는 값: /home/gmoh/mon/251229/state/20260925-invvol-control/eq/ (대조 실험이 저장한 곡선)

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_invvol_detail.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_fut_spike_hold import (ANCHORS, PH_SPOT, START, END, SPOT_BASE,  # noqa: E402
                               TX_SPOT, build_alloc, metrics)

EQ = "/home/gmoh/mon/251229/state/20260925-invvol-control/eq"
ARM = "역수_20"
CURK = "현행"
SLEEVE_SIZES = [5_000_000, 20_000_000, 100_000_000]  # 코인 현물 슬리브 원화 규모 가정


def _load(kind: str, key: str):
    p = os.path.join(EQ, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)["eq"]


def concentration() -> None:
    print("\n[초과수익이 몇 날에 몰렸나 — 합성 60/25/15, 기본 비용]")
    print(f"  {'앵커':<5s}{'초과수익합':>11s}{'상위5일':>9s}{'상위10일':>9s}{'상위20일':>9s}"
          f"{'상위20일 뺀 합':>14s}")
    for a in ANCHORS:
        st = _load("st", f"주식__a{a}")
        fu = _load("fu", f"선물__a{a}")
        cur = _load("sp", f"{CURK}__a{a}")
        arm = _load("sp", f"{ARM}__a{a}")
        if any(x is None for x in (st, fu, cur, arm)):
            print(f"  {a:<5d}(자료 없음)")
            continue
        base = build_alloc(st, cur, fu)
        test = build_alloc(st, arm, fu)
        i = base.index.intersection(test.index)
        d = (np.log(test.loc[i]).diff() - np.log(base.loc[i]).diff()).dropna()
        tot = float(d.sum())
        srt = d.sort_values(ascending=False)
        t5, t10, t20 = float(srt[:5].sum()), float(srt[:10].sum()), float(srt[:20].sum())
        print(f"  {a:<5d}{tot:>+11.4f}{t5/tot if tot else 0:>9.0%}"
              f"{t10/tot if tot else 0:>9.0%}{t20/tot if tot else 0:>9.0%}"
              f"{tot - t20:>+14.4f}")

    print("\n[해별 초과수익 — 앵커 0, 합성, 로그수익 합]")
    st = _load("st", "주식__a0")
    fu = _load("fu", "선물__a0")
    cur = _load("sp", f"{CURK}__a0")
    arm = _load("sp", f"{ARM}__a0")
    if any(x is None for x in (st, fu, cur, arm)):
        return
    base = build_alloc(st, cur, fu)
    test = build_alloc(st, arm, fu)
    i = base.index.intersection(test.index)
    d = (np.log(test.loc[i]).diff() - np.log(base.loc[i]).diff()).dropna()
    for y, v in d.groupby(d.index.year).sum().items():
        n_pos = int((d[d.index.year == y] > 0).sum())
        n_neg = int((d[d.index.year == y] < 0).sum())
        print(f"  {y}  {v:>+8.4f}   이긴날 {n_pos:>4d}  진날 {n_neg:>4d}")


def _trace(mode: str, days: int, bars, funding) -> list:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_SPOT"] = mode
    os.environ["CD_WEIGHT_VOL_DAYS"] = str(days)
    tr: list = []
    try:
        bt_run(bars, funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=TX_SPOT, start_date=START, end_date=END,
               vol_mode="daily", phase_offset_bars=PH_SPOT[0],
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", _trace=tr, **SPOT_BASE)
    finally:
        os.environ["CD_WEIGHT_MODE_SPOT"] = 'ew'
    return tr


def friction() -> None:
    from unified_backtest import load_data
    bars, funding = load_data("D")
    print("\n[실매매 마찰 — 앵커 0, 현물 슬리브]")
    for label, mode, days in (("현행 등분", 'ew', 90), ("역수 창 20", 'invvol', 20)):
        tr = _trace(mode, days, bars, funding)
        prev = {}
        n_rebal = 0
        turn = []
        small = {s: 0 for s in SLEEVE_SIZES}
        orders = 0
        for row in tr:
            if not row.get("rebal"):
                continue
            tgt = {k: v for k, v in (row.get("target") or {}).items()
                   if k not in ("CASH", "Cash")}
            keys = set(tgt) | set(prev)
            deltas = {k: tgt.get(k, 0.0) - prev.get(k, 0.0) for k in keys}
            ht = sum(abs(v) for v in deltas.values()) / 2
            if ht > 1e-9:
                n_rebal += 1
                turn.append(ht)
                for k, v in deltas.items():
                    if abs(v) <= 1e-9:
                        continue
                    orders += 1
                    for s in SLEEVE_SIZES:
                        if abs(v) * s < 5000:
                            small[s] += 1
            prev = tgt
        yrs = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
        tv = np.array(turn) if turn else np.array([0.0])
        print(f"  {label}")
        print(f"    리밸 {n_rebal}회 ({n_rebal/yrs:.1f}회/년), 주문 {orders}건")
        print(f"    회전율(half-turnover) 평균 {tv.mean():.3f} 중앙 {np.median(tv):.3f} "
              f"합 {tv.sum():.1f} (연 {tv.sum()/yrs:.2f})")
        for s in SLEEVE_SIZES:
            print(f"    슬리브 {s//10000:,}만원일 때 5,000원 미만 주문 {small[s]}건 "
                  f"({small[s]/max(orders,1):.1%})")


def _dd(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def drawdowns() -> None:
    """초과수익이 0 이면 이득은 전부 낙폭 쪽이다. 그 낙폭 개선이 한 사건인지 본다."""
    print("\n[낙폭 비교 — 합성 60/25/15, 기본 비용]")
    print(f"  {'앵커':<5s}{'현행 낙폭':>10s}{'역수20 낙폭':>12s}{'차':>8s}"
          f"{'현행 총수익':>12s}{'역수20 총수익':>14s}   현행 최저점일")
    for a in ANCHORS:
        st = _load("st", f"주식__a{a}")
        fu = _load("fu", f"선물__a{a}")
        cur = _load("sp", f"{CURK}__a{a}")
        arm = _load("sp", f"{ARM}__a{a}")
        if any(x is None for x in (st, fu, cur, arm)):
            continue
        base, test = build_alloc(st, cur, fu), build_alloc(st, arm, fu)
        mb, mt = metrics(base), metrics(test)
        db = _dd(base)
        trough = db.idxmin()
        print(f"  {a:<5d}{mb['MDD']:>+10.2f}{mt['MDD']:>+12.2f}"
              f"{mt['MDD']-mb['MDD']:>+8.2f}"
              f"{base.iloc[-1]/base.iloc[0]:>11.2f}x{test.iloc[-1]/test.iloc[0]:>13.2f}x"
              f"   {str(trough)[:10]}")

    print("\n[낙폭 구간별 — 앵커 0, 현행 기준 깊은 구간 5개에서 두 팔의 깊이]")
    st = _load("st", "주식__a0")
    fu = _load("fu", "선물__a0")
    cur = _load("sp", f"{CURK}__a0")
    arm = _load("sp", f"{ARM}__a0")
    if any(x is None for x in (st, fu, cur, arm)):
        return
    base, test = build_alloc(st, cur, fu), build_alloc(st, arm, fu)
    i = base.index.intersection(test.index)
    base, test = base.loc[i], test.loc[i]
    db, dt = _dd(base), _dd(test)
    seen, used = [], pd.Series(False, index=i)
    for _ in range(5):
        cand = db.where(~used)
        if cand.isna().all():
            break
        t = cand.idxmin()
        lo = max(0, i.get_loc(t) - 120)
        hi = min(len(i), i.get_loc(t) + 120)
        win = i[lo:hi]
        seen.append((t, float(db.loc[t]), float(dt.loc[win].min())))
        used.loc[win] = True
    for t, b, x in seen:
        print(f"  {str(t)[:10]}  현행 {b:>+7.2%}   역수20 {x:>+7.2%}   차 {x-b:>+6.2%}")


def main() -> None:
    concentration()
    drawdowns()
    friction()


if __name__ == '__main__':
    main()
