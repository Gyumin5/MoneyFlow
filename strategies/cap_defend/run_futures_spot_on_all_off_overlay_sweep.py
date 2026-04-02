#!/usr/bin/env python3
"""현물 ON + 선물 all OFF 숏 오버레이 스윕."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data
from run_futures_spot_on_all_off_overlay_combo_test import (
    prepare_overlay_context,
    build_overlay_target_series,
)
from run_short_cash_gate_test import LongShortSingleAccountEngine


WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CANDIDATES = []
for w in WEIGHTS:
    CANDIDATES.append(dict(name=f"top3_donchian50_{int(w*100)}", mode="dynamic", short_weight=w, topn=3, rule="donchian50"))
    CANDIDATES.append(dict(name=f"top5_donchian50_{int(w*100)}", mode="dynamic", short_weight=w, topn=5, rule="donchian50"))
    CANDIDATES.append(dict(name=f"top3_sma200_{int(w*100)}", mode="dynamic", short_weight=w, topn=3, rule="sma200"))
    CANDIDATES.append(dict(name=f"top5_sma200_{int(w*100)}", mode="dynamic", short_weight=w, topn=5, rule="sma200"))


def main():
    t0 = time.time()
    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["4h", "1h"]}
    bars_1h, funding_1h = data["1h"]
    print("Preparing overlay context...")
    ctx = prepare_overlay_context(data)

    rows = []
    for cand in CANDIDATES:
        target_series = build_overlay_target_series(ctx, cand)
        engine = LongShortSingleAccountEngine(
            bars_1h,
            funding_1h,
            leverage=5.0,
            stop_kind="prev_close_pct",
            stop_pct=0.15,
            stop_gate="cash_guard",
            stop_gate_cash_threshold=0.34,
            per_coin_leverage_mode="cap_mom_blend_543_cash",
            leverage_floor=3.0,
            leverage_mid=4.0,
            leverage_ceiling=5.0,
            leverage_cash_threshold=0.34,
            leverage_partial_cash_threshold=0.0,
            leverage_count_floor_max=2,
            leverage_count_mid_max=4,
            leverage_canary_floor_gap=0.015,
            leverage_canary_mid_gap=0.04,
            leverage_canary_high_gap=0.08,
            leverage_canary_sma_bars=1200,
            leverage_mom_lookback_bars=24 * 30,
            leverage_vol_lookback_bars=24 * 90,
        )
        m = engine.run(target_series)
        row = {
            "name": cand["name"],
            "Cal": m.get("Cal", 0),
            "CAGR": m.get("CAGR", 0),
            "MDD": m.get("MDD", 0),
            "Sharpe": m.get("Sharpe", 0),
            "Rebal": m.get("Rebal", 0),
        }
        rows.append(row)
        print(f"{row['name']:<20} Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%}")

    rows.sort(key=lambda r: (-r["Cal"], -r["Sharpe"], r["name"]))
    print("\nTop candidates")
    for row in rows[:20]:
        print(f"- {row['name']}: Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, MDD={row['MDD']:+.1%}")
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
