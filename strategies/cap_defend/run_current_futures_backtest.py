#!/usr/bin/env python3
"""현재 실거래 선물 전략 단독 백테스트."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data
from futures_live_config import CURRENT_LIVE_COMBO, CURRENT_STRATEGIES
from run_ensemble import SingleAccountEngine, combine_targets
from run_stoploss_test import START, END


def generate_trace(data, cfg):
    from backtest_futures_full import run

    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    trace = []
    run(
        bars,
        funding,
        interval=interval,
        leverage=1.0,
        start_date=START,
        end_date=END,
        _trace=trace,
        **run_cfg,
    )
    return trace


def main():
    t0 = time.time()
    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["4h", "1h"]}
    bars_1h, funding_1h = data["1h"]
    all_dates = bars_1h["BTC"].index[(bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)]

    print("Generating traces...")
    traces = {name: generate_trace(data, cfg) for name, cfg in CURRENT_STRATEGIES.items()}
    combined = combine_targets({k: traces[k] for k in CURRENT_LIVE_COMBO}, CURRENT_LIVE_COMBO, all_dates)

    engine = SingleAccountEngine(
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
    m = engine.run(combined)

    print("\nResults")
    print(
        f"current_live     "
        f"Cal={m['Cal']:.2f} "
        f"CAGR={m['CAGR']:+.1%} "
        f"MDD={m['MDD']:+.1%} "
        f"Liq={m['Liq']} "
        f"Stops={m.get('Stops', 0)} "
        f"Rebal={m.get('Rebal', 0)}"
    )
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
