#!/usr/bin/env python3
"""현재 선물 최종 전략 + 현물 ON / 선물 all OFF 숏 오버레이 결합 테스트."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import get_mcap, load_data, run
from run_ensemble import combine_targets
from run_short_cash_gate_test import LongShortSingleAccountEngine
from run_short_spot_on_all_futures_off_overlay_test import (
    _coin_spot_risk_on_daily,
    _build_canary_off_series,
    _build_rule_cache,
    _dynamic_short_target,
)
from run_spot_futures_allocation import FUT_STRATEGIES, FUT_WEIGHTS
from run_stoploss_test import END, START


CANDIDATES = [
    dict(name="baseline", mode="none", short_weight=0.0),
    dict(name="btc_25", mode="btc", short_weight=0.25),
    dict(name="btc_50", mode="btc", short_weight=0.50),
    dict(name="top3_donchian50_25", mode="dynamic", short_weight=0.25, topn=3, rule="donchian50"),
    dict(name="top3_donchian50_50", mode="dynamic", short_weight=0.50, topn=3, rule="donchian50"),
    dict(name="top5_sma200_25", mode="dynamic", short_weight=0.25, topn=5, rule="sma200"),
    dict(name="top5_sma200_50", mode="dynamic", short_weight=0.50, topn=5, rule="sma200"),
]


def generate_trace(data, cfg):
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


def prepare_overlay_context(data):
    traces = {name: generate_trace(data, cfg) for name, cfg in FUT_STRATEGIES.items()}
    bars_1h, _ = data["1h"]
    all_dates = bars_1h["BTC"].index[(bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)]
    combined = combine_targets(traces, FUT_WEIGHTS, all_dates)

    spot_on_daily = _coin_spot_risk_on_daily()
    off_4h1 = _build_canary_off_series(data, "4h1", all_dates)
    off_4h2 = _build_canary_off_series(data, "4h2", all_dates)
    off_1h1 = _build_canary_off_series(data, "1h1", all_dates)
    all_off = off_4h1 & off_4h2 & off_1h1
    rule_cache = _build_rule_cache(bars_1h, "1h")
    return combined, all_dates, spot_on_daily, all_off, rule_cache


def build_overlay_target_series(ctx, candidate):
    combined, all_dates, spot_on_daily, all_off, rule_cache = ctx
    out = []
    for date, base_target in combined:
        target = dict(base_target)
        spot_on = bool(spot_on_daily.get(date.normalize(), False))
        if not (spot_on and bool(all_off.loc[date])):
            out.append((date, target))
            continue
        if abs(target.get("CASH", 0.0) - 1.0) > 1e-9 or len([c for c, w in target.items() if c != "CASH" and abs(w) > 1e-9]) > 0:
            out.append((date, target))
            continue

        if candidate["mode"] == "none":
            out.append((date, target))
            continue
        if candidate["mode"] == "btc":
            target = {"BTC": -candidate["short_weight"], "CASH": 1.0 - candidate["short_weight"]}
            out.append((date, target))
            continue

        picks = _dynamic_short_target(date, candidate["topn"], candidate["rule"], rule_cache)
        if not picks:
            out.append((date, {"CASH": 1.0}))
            continue
        w = candidate["short_weight"] / len(picks)
        overlay = {coin: -w for coin in picks}
        overlay["CASH"] = 1.0 - candidate["short_weight"]
        out.append((date, overlay))
    return out


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
            "Liq": m.get("Liq", 0),
            "Stops": m.get("Stops", 0),
            "Rebal": m.get("Rebal", 0),
        }
        rows.append(row)
        print(
            f"{row['name']:<22} Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']} Rebal={row['Rebal']}"
        )

    rows.sort(key=lambda r: (-r["Cal"], -r["Sharpe"], r["name"]))
    print("\nTop candidates")
    for row in rows:
        print(
            f"- {row['name']}: Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, "
            f"MDD={row['MDD']:+.1%}, Stops={row['Stops']}, Rebal={row['Rebal']}"
        )
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
