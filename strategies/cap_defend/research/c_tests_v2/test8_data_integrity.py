#!/usr/bin/env python3
"""Test 8: Data integrity — 데이터 결손 경고.

Codex 권고 반영: 조용히 진행되는 결손 사례를 로그로 명시.
- 코인별 1h bar row 누락 (gap > 3시간)
- 일별 coin_daily close 결손
- historical_universe 누락 날짜 (snapshot gap)
- FULL_END 이전 stale 데이터
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all, load_bars_1h, FULL_END

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def check_1h_gaps(coin: str, max_gap_h: int = 3) -> dict:
    bars = load_bars_1h(coin)
    if bars is None:
        return {"coin": coin, "status": "missing_csv"}
    idx = pd.to_datetime(bars.index)
    if len(idx) < 2:
        return {"coin": coin, "status": "too_short", "n": len(idx)}
    # 연속된 시간 gap 체크
    diffs = idx.to_series().diff().dt.total_seconds().div(3600).dropna()
    gaps = diffs[diffs > max_gap_h]
    return {
        "coin": coin, "status": "ok",
        "n": len(idx),
        "first": str(idx[0]),
        "last": str(idx[-1]),
        "n_gaps": len(gaps),
        "max_gap_hours": float(gaps.max()) if len(gaps) else 0.0,
        "stale_days_from_full_end": (FULL_END - idx[-1]).days,
    }


def check_daily(cd_dict) -> list[dict]:
    rows = []
    for coin, s in cd_dict.items():
        if s is None or len(s) == 0:
            rows.append({"coin": coin, "status": "empty_daily"})
            continue
        rows.append({
            "coin": coin, "status": "ok",
            "n": len(s),
            "first": str(s.index[0]),
            "last": str(s.index[-1]),
            "n_nan": int(s.isna().sum()),
            "stale_days_from_full_end": (FULL_END - s.index[-1]).days,
        })
    return rows


def check_universe(hist) -> dict:
    dates = sorted(pd.Timestamp(d) for d in hist.keys())
    if len(dates) < 2:
        return {"status": "too_few_snapshots"}
    diffs = pd.Series(dates).diff().dt.days.dropna()
    gaps = diffs[diffs > 45]  # 월간 예상, 45일 초과면 결손
    return {
        "status": "ok",
        "n_snapshots": len(dates),
        "first": str(dates[0]),
        "last": str(dates[-1]),
        "n_gaps_over_45d": len(gaps),
        "max_gap_days": int(diffs.max()) if len(diffs) else 0,
        "stale_days_from_full_end": (FULL_END - dates[-1]).days,
    }


def main():
    v21_s, v21_f, hist, avail, cd = load_all()

    print(f"=== Universe snapshots ===")
    uni = check_universe(hist)
    print(uni)
    pd.DataFrame([uni]).to_csv(os.path.join(OUT, "test8_universe.csv"), index=False)

    print(f"\n=== 1h bar gaps (max 3h) per coin ===")
    rows_1h = []
    for c in avail:
        r = check_1h_gaps(c)
        rows_1h.append(r)
    df_1h = pd.DataFrame(rows_1h)
    df_1h.to_csv(os.path.join(OUT, "test8_1h_gaps.csv"), index=False)
    # 경고만 출력
    warn = df_1h[(df_1h["status"] != "ok") | (df_1h.get("n_gaps", 0) > 10)
                  | (df_1h.get("stale_days_from_full_end", 0) > 30)]
    print(f"  ok: {(df_1h['status']=='ok').sum()}, warn: {len(warn)}")
    if len(warn):
        print(warn.head(20).to_string(index=False))

    print(f"\n=== coin_daily ===")
    rows_d = check_daily(cd)
    df_d = pd.DataFrame(rows_d)
    df_d.to_csv(os.path.join(OUT, "test8_daily.csv"), index=False)
    warn_d = df_d[(df_d["status"] != "ok") | (df_d.get("n_nan", 0) > 0)
                   | (df_d.get("stale_days_from_full_end", 0) > 30)]
    print(f"  ok: {(df_d['status']=='ok').sum()}, warn: {len(warn_d)}")
    if len(warn_d):
        print(warn_d.head(20).to_string(index=False))

    # V21 equity 자체 결손
    print(f"\n=== V21 equity ===")
    for lbl, v21 in [("spot", v21_s), ("fut", v21_f)]:
        print(f"  V21_{lbl}: {v21.index[0]} → {v21.index[-1]} "
              f"(stale={(FULL_END - v21.index[-1]).days}d)")

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
