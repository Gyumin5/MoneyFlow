#!/usr/bin/env python3
"""Group D — 포지션 사이징.

D1. 연속 손실 서킷: 최근 N회 연속 손실 후 다음 K회 skip
D2. Win-rate 기반 cap scaling: 최근 N개 win rate에 따라 cap 스케일
D3. Rolling MAE cap: 최근 N개 평균 MAE 크면 cap 축소
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common3 import (
    load_all, load_cached_events, run_splits,
    CAP_SPOT, CAP_FUT_OPTS, OUT,
)
import pandas as pd


def filter_loss_streak(ev: pd.DataFrame, streak: int, skip: int) -> pd.DataFrame:
    """현재 entry_ts 이전에 이미 exit 된 이벤트 중 연속 streak 이상 loss → 이후 skip개 제외.
    exit_ts < entry_ts 조건으로 look-ahead 방지.
    """
    if len(ev) == 0:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    ev = ev.sort_values("entry_ts").reset_index(drop=True)
    keep = []
    blocked_until_idx = -1  # skip 카운트 index (ev index 기준 pseudo)
    for i, r in ev.iterrows():
        past = ev[ev["exit_ts"] < r["entry_ts"]].sort_values("exit_ts")
        # 가장 최근 streak개가 모두 음수인지 확인
        block = False
        if len(past) >= streak:
            tail = past["pnl_pct"].iloc[-streak:]
            if (tail <= 0).all():
                # skip 개수만큼 블록 (직전 streak 마지막 exit 이후 skip 거래만큼 차단)
                # 현재 진입 이전 exit된 이벤트 수 중 해당 streak 이후 이벤트 수
                streak_end_idx = past.index[-1]
                events_after_streak_end = past[past.index > streak_end_idx]
                # 블록 조건: 그 뒤로 skip개 이내면 차단
                if len(events_after_streak_end) < skip:
                    block = True
        keep.append(not block)
    return ev[keep].reset_index(drop=True)


def filter_winrate_scale(ev: pd.DataFrame, window: int,
                          low_wr: float, high_wr: float,
                          low_scale: float = 0.5,
                          high_scale: float = 1.5) -> tuple[pd.DataFrame, pd.Series]:
    """과거 '이미 exit된' 이벤트들 중 최근 window개의 win rate에 따라 pnl_pct 스케일.
    exit_ts < entry_ts 조건으로 look-ahead 방지.
    ⚠ pnl만 스케일 — slot 자본 효율성은 근사 (실제 cap 줄이는 것과 다름).
    """
    if len(ev) == 0:
        return ev, pd.Series(dtype=float)
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    ev = ev.sort_values("entry_ts").reset_index(drop=True)
    scale_vals = []
    for _, r in ev.iterrows():
        past = ev[ev["exit_ts"] < r["entry_ts"]].sort_values("exit_ts")
        if len(past) < max(3, window // 2):
            scale_vals.append(1.0); continue
        tail = past.tail(window)
        wr = float((tail["pnl_pct"] > 0).mean())
        if wr < low_wr:
            scale_vals.append(low_scale)
        elif wr > high_wr:
            scale_vals.append(high_scale)
        else:
            scale_vals.append(1.0)
    scale = pd.Series(scale_vals, index=ev.index)
    ev["pnl_pct"] = ev["pnl_pct"] * scale.values
    return ev, scale


def main():
    v21_spot, v21_fut, hist, _, cd = load_all()
    ev_spot = load_cached_events("spot")
    ev_fut  = load_cached_events("fut")

    rows = []
    rows += run_splits("baseline", "spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_splits("baseline", "fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    # D1 손실 스트릭 skip
    for (streak, skip) in [(2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = filter_loss_streak(ev, streak, skip)
            lab = f"D1_lossstreak_{streak}L_skip{skip}"
            print(f"  {lab} {kind}: kept {len(ev2)}/{len(ev)}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # D2 winrate scale (pnl을 스케일)
    for window in [10, 20]:
        for (low_wr, high_wr) in [(0.4, 0.7), (0.5, 0.8)]:
            for kind, ev, v21, caps in [
                ("spot", ev_spot, v21_spot, [CAP_SPOT]),
                ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
            ]:
                ev2, _ = filter_winrate_scale(ev, window, low_wr, high_wr)
                lab = f"D2_wr_w{window}_lo{low_wr:.1f}_hi{high_wr:.1f}"
                rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_sizing.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")
    piv = df[df["span"] == "holdout"].pivot_table(
        index="label", columns=["kind", "cap"], values="Cal")
    print("\n=== Holdout Cal ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
