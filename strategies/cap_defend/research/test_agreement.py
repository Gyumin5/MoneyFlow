#!/usr/bin/env python3
"""선물 Agreement Risk Budget 실험.

3전략 합의도에 따라 combined target을 스케일링:
- 3/3 ON → 100% target
- 2/3 ON → 66% target (나머지 CASH)
- 1/3 ON → 33% target (나머지 CASH)
- 0/3 ON → 100% CASH
"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from futures_live_config import CURRENT_LIVE_COMBO, CURRENT_STRATEGIES, START, END
from futures_ensemble_engine import SingleAccountEngine, combine_targets


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **run_cfg)
    return trace


def combine_targets_agreement(traces, weights, all_dates_1h, scale_mode='proportional'):
    """Agreement-aware combine: scale target by fraction of strategies that are ON."""
    sorted_traces = {}
    for key, trace in traces.items():
        entries = [(e['date'], e['target']) for e in trace]
        entries.sort(key=lambda x: x[0])
        sorted_traces[key] = entries

    trace_idx = {key: 0 for key in traces}
    combined_series = []
    latest_targets = {key: {'CASH': 1.0} for key in traces}

    for date in all_dates_1h:
        for key in traces:
            entries = sorted_traces[key]
            idx = trace_idx[key]
            while idx < len(entries) and entries[idx][0] <= date:
                latest_targets[key] = entries[idx][1]
                idx += 1
            trace_idx[key] = idx

        # Count how many strategies are "ON" (have non-CASH positions)
        n_on = 0
        n_total = len(traces)
        for key in traces:
            tgt = latest_targets[key]
            cash_w = tgt.get('CASH', 0)
            if cash_w < 0.99:  # has real positions
                n_on += 1

        # Scale factor based on agreement
        if scale_mode == 'proportional':
            scale = n_on / n_total  # 0, 0.33, 0.67, 1.0
        elif scale_mode == 'threshold2':
            scale = 1.0 if n_on >= 2 else (0.33 if n_on == 1 else 0.0)
        elif scale_mode == 'threshold3':
            scale = 1.0 if n_on >= 3 else 0.0
        else:
            scale = 1.0

        # Merge with scale
        merged = {}
        for key, w in weights.items():
            for coin, cw in latest_targets[key].items():
                merged[coin] = merged.get(coin, 0) + cw * w

        if scale < 1.0:
            # Scale down non-CASH, put remainder in CASH
            non_cash_total = sum(v for k, v in merged.items() if k != 'CASH')
            if non_cash_total > 0:
                for coin in list(merged.keys()):
                    if coin != 'CASH':
                        merged[coin] *= scale
                merged['CASH'] = merged.get('CASH', 0) + non_cash_total * (1 - scale)

        combined_series.append((date, merged))

    return combined_series


ENGINE_KWARGS = dict(
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


def run_with_mode(data, traces, all_dates, mode):
    combo = {k: traces[k] for k in CURRENT_LIVE_COMBO}
    if mode == 'baseline':
        combined = combine_targets(combo, CURRENT_LIVE_COMBO, all_dates)
    else:
        combined = combine_targets_agreement(combo, CURRENT_LIVE_COMBO, all_dates, scale_mode=mode)

    bars_1h, funding_1h = data["1h"]
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    return engine.run(combined)


def main():
    t0 = time.time()
    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["4h", "1h"]}
    bars_1h, funding_1h = data["1h"]
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)
    ]

    print("Generating traces...")
    traces = {name: generate_trace(data, cfg) for name, cfg in CURRENT_STRATEGIES.items()}

    modes = [
        ('baseline', '원본 (스케일링 없음)'),
        ('proportional', '비례 (0/33/67/100%)'),
        ('threshold2', '2개이상 ON→100%, 1개→33%, 0→0%'),
        ('threshold3', '3개 모두 ON→100%, 아니면 0%'),
    ]

    print(f"\n{'모드':<35s} {'Cal':>5s} {'CAGR':>8s} {'MDD':>8s} {'Liq':>4s} {'Stops':>5s} {'Rebal':>5s}")
    print("-" * 75)

    for mode, desc in modes:
        m = run_with_mode(data, traces, all_dates, mode)
        print(
            f"{desc:<35s} "
            f"{m['Cal']:>5.2f} "
            f"{m['CAGR']:>+8.1%} "
            f"{m['MDD']:>+8.1%} "
            f"{m['Liq']:>4d} "
            f"{m.get('Stops', 0):>5d} "
            f"{m.get('Rebal', 0):>5d}"
        )

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
