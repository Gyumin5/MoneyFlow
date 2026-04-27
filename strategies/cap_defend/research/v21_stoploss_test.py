#!/usr/bin/env python3
"""V21 선물 앙상블 + stop-loss 추가 효과 측정.

V21 스펙 (futures_live_config.py / V21_OPERATION_MANUAL.md):
- 4h_S240_SN120: SMA240, Mom20/720, daily 5%, snap120
- 4h_S240_SN30:  SMA240, Mom20/480, daily 5%, snap30
- 4h_S120_SN120: SMA120, Mom20/720, daily 5%, snap120
- L3 고정 3x, stop 없음, cash_guard 없음

Stop 옵션:
- none (baseline)
- prev_close_pct (-5/-10/-12/-15/-20)
- highest_close_since_entry_pct (-5/-10/-15)
- highest_high_since_entry_pct (-10/-15/-20)
- rolling_high_close_pct (-10/-15, lookback 24/72)

stop_gate:
- always
- cash_guard (cash >= 0.34 이상일 때만 발동)
"""
from __future__ import annotations
import os, sys, time
from itertools import product
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from backtest_futures_full import load_data, run
from futures_ensemble_engine import SingleAccountEngine, combine_targets

OUT = os.path.join(HERE, "v21_stoploss_out")
os.makedirs(OUT, exist_ok=True)

START = "2020-10-01"
END = "2026-03-28"

V21_MEMBERS = {
    "4h_S240_SN120": dict(
        interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
    "4h_S240_SN30": dict(
        interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=480,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=30,
    ),
    "4h_S120_SN120": dict(
        interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
}
V21_WEIGHTS = {k: 1/3 for k in V21_MEMBERS}
LEVERAGE = 3.0


def build_traces(data):
    traces = {}
    for name, params in V21_MEMBERS.items():
        iv = params["interval"]
        bars, funding = data[iv]
        rp = dict(params); del rp["interval"]
        trace = []
        # V21 spec: universe_size=3 (Top 3 coin)
        run(bars, funding, interval=iv, leverage=1.0,
            start_date=START, end_date=END, _trace=trace,
            universe_size=3, **rp)
        traces[name] = trace
    return traces


def run_config(args):
    traces, all_dates, bars_1h, funding_1h, cfg = args
    combined = combine_targets(traces, V21_WEIGHTS, all_dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=LEVERAGE, leverage_mode="fixed",
        per_coin_leverage_mode="none",
        stop_kind=cfg["stop_kind"],
        stop_pct=cfg["stop_pct"],
        stop_lookback_bars=cfg.get("stop_lookback_bars", 0),
        stop_gate=cfg["stop_gate"],
        stop_gate_cash_threshold=cfg.get("stop_gate_cash_threshold", 0.0),
    )
    m = engine.run(combined)
    return {
        **cfg,
        "Cal": round(m.get("Cal", 0), 3),
        "CAGR": round(m.get("CAGR", 0), 4),
        "MDD": round(m.get("MDD", 0), 4),
        "Sh": round(m.get("Sharpe", 0), 3),
        "Liq": m.get("Liq", 0),
        "Stops": m.get("Stops", 0),
        "Rebal": m.get("Rebal", 0),
    }


def main():
    print("Loading data...")
    intervals = {p["interval"] for p in V21_MEMBERS.values()} | {"1h"}
    data = {iv: load_data(iv) for iv in intervals}
    bars_1h, funding_1h = data["1h"]
    all_dates = bars_1h["BTC"].index
    all_dates = all_dates[(all_dates >= START) & (all_dates <= END)]

    print("Building traces...")
    t0 = time.time()
    traces = build_traces(data)
    print(f"  done ({time.time()-t0:.0f}s)")

    # configs
    configs = [
        {"stop_kind": "none", "stop_pct": 0.0, "stop_gate": "always"},
    ]
    for pct in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        configs.append({"stop_kind": "prev_close_pct", "stop_pct": pct, "stop_gate": "always"})
        configs.append({"stop_kind": "prev_close_pct", "stop_pct": pct,
                         "stop_gate": "cash_guard", "stop_gate_cash_threshold": 0.34})
    for pct in [0.10, 0.15, 0.20]:
        configs.append({"stop_kind": "highest_close_since_entry_pct", "stop_pct": pct, "stop_gate": "always"})
        configs.append({"stop_kind": "highest_high_since_entry_pct", "stop_pct": pct, "stop_gate": "always"})
    for pct, lb in product([0.10, 0.15], [24, 72]):
        configs.append({"stop_kind": "rolling_high_close_pct", "stop_pct": pct,
                         "stop_lookback_bars": lb, "stop_gate": "always"})
    print(f"Total configs: {len(configs)}")

    # sequential (engine is heavy, parallel 어려움 — traces 공유)
    rows = []
    for i, cfg in enumerate(configs, 1):
        t1 = time.time()
        r = run_config((traces, all_dates, bars_1h, funding_1h, cfg))
        rows.append(r)
        print(f"[{i}/{len(configs)}] kind={cfg['stop_kind']:36s} pct={cfg['stop_pct']} "
              f"gate={cfg['stop_gate']:12s} → Cal={r['Cal']:.2f} MDD={r['MDD']:.2%} "
              f"CAGR={r['CAGR']:.2%} Liq={r['Liq']} Stops={r['Stops']} ({time.time()-t1:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v21_stoploss.csv"), index=False)

    base = df[df["stop_kind"] == "none"].iloc[0]
    df["dCal"] = df["Cal"] - base["Cal"]
    df["dMDD"] = df["MDD"] - base["MDD"]
    df["dCAGR"] = df["CAGR"] - base["CAGR"]

    print("\n=== V21 단독 baseline ===")
    print(f"  Cal={base['Cal']:.2f}, CAGR={base['CAGR']:.2%}, MDD={base['MDD']:.2%}, Liq={base['Liq']}")

    print("\n=== Top by Cal (baseline 포함) ===")
    print(df.sort_values("Cal", ascending=False).head(10).to_string(index=False))

    print("\n=== 개선 (dCal > 0) ===")
    good = df[df["dCal"] > 0].sort_values("dCal", ascending=False)
    if len(good):
        print(good.to_string(index=False))
    else:
        print("  없음 — 모든 stop 설정이 baseline(none) 못 이김")

    print(f"\n저장: {OUT}/v21_stoploss.csv")


if __name__ == "__main__":
    main()
