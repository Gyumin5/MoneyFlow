#!/usr/bin/env python3
"""Test 7: Intrabar MAE (Maximum Adverse Excursion) 분포.

Gemini 지적: stop-loss 없는 C는 진입 후 봉 내부 최저점에서
선물 3x 레버리지 wipeout -33.3% 근접할 수 있음.
각 event의 entry_px → hold 구간 최저 Low → MAE_pct 측정.

지표:
- MAE_pct 분포 (avg/p5/p50/worst)
- 3x 기준 실질 wipeout 근접도 (MAE × 3 ≤ -0.95 = 즉시 청산 위험)
- event별 원금 대비 최악 미실현 손실
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_all, load_cached_events, load_bars_1h, HOLDOUT_START, FULL_END)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def compute_mae(ev: pd.DataFrame) -> pd.DataFrame:
    """각 event별로 entry_ts ~ exit_ts 사이 1h Low의 최저값 → MAE%."""
    if len(ev) == 0:
        return pd.DataFrame()
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"] = pd.to_datetime(ev["exit_ts"])

    # coin별 1h bar 캐시
    bar_cache = {}
    maes = []
    for _, row in ev.iterrows():
        coin = row["coin"]
        if coin not in bar_cache:
            bar_cache[coin] = load_bars_1h(coin)
        bars = bar_cache[coin]
        if bars is None:
            maes.append(np.nan)
            continue
        # entry_ts 직후 ~ exit_ts 구간 (entry bar는 포함)
        idx = bars.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
            bars = bars.copy()
            bars.index = idx
        mask = (bars.index >= row["entry_ts"]) & (bars.index <= row["exit_ts"])
        sub = bars[mask]
        if len(sub) == 0:
            maes.append(np.nan)
            continue
        min_low = float(sub["Low"].min())
        mae_pct = (min_low / float(row["entry_px"])) - 1.0
        maes.append(round(mae_pct, 5))
    ev["mae_pct"] = maes
    return ev


def analyze(ev_mae: pd.DataFrame, label: str, lev: float = 1.0) -> dict:
    valid = ev_mae.dropna(subset=["mae_pct"])
    if len(valid) == 0:
        return {"label": label, "n": 0}
    m = valid["mae_pct"].values
    lev_m = m * lev
    return {
        "label": label, "lev": lev, "n": len(valid),
        "avg_mae": round(float(np.mean(m)), 4),
        "p50_mae": round(float(np.percentile(m, 50)), 4),
        "p5_mae": round(float(np.percentile(m, 5)), 4),
        "worst_mae": round(float(min(m)), 4),
        "lev_avg_mae": round(float(np.mean(lev_m)), 4),
        "lev_worst_mae": round(float(min(lev_m)), 4),
        "n_wipeout_95": int((lev_m <= -0.95).sum()),  # 레버리지 적용 시 -95% 이하
        "n_wipeout_90": int((lev_m <= -0.90).sum()),
        "n_deep_50": int((lev_m <= -0.50).sum()),
    }


def main():
    v21_s, v21_f, hist, avail, cd = load_all()

    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    # MAE 계산
    print("[1/2] Computing MAE for spot events...")
    ev_s_mae = compute_mae(ev_s)
    ev_s_mae.to_csv(os.path.join(OUT, "test7_mae_spot.csv"), index=False)
    print("[2/2] Computing MAE for fut events...")
    ev_f_mae = compute_mae(ev_f)
    ev_f_mae.to_csv(os.path.join(OUT, "test7_mae_fut.csv"), index=False)

    rows = []
    # Spot 1x
    for period, mask in [
        ("전구간", slice(None)),
        ("Holdout", ev_s_mae["entry_ts"] >= HOLDOUT_START),
    ]:
        sub = ev_s_mae[mask] if not isinstance(mask, slice) else ev_s_mae
        rows.append(analyze(sub, f"spot_{period}", lev=1.0))

    # Fut 3x
    for period, mask in [
        ("전구간", slice(None)),
        ("Holdout", ev_f_mae["entry_ts"] >= HOLDOUT_START),
    ]:
        sub = ev_f_mae[mask] if not isinstance(mask, slice) else ev_f_mae
        rows.append(analyze(sub, f"fut_{period}", lev=3.0))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test7_mae_summary.csv"), index=False)
    print("\n=== MAE 요약 ===")
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
