#!/usr/bin/env python3
"""선물에서 gap exclusion 최적값 탐색 (1D 단독 / 4h 단독, L3). 병렬 워커 버전."""
from __future__ import annotations
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from backtest_futures_full import load_data, run as bt_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from run_futures_fixedlev_search import FIXED_CFG

START = "2020-10-01"
END = "2026-04-13"
WINNERS = os.path.join(HERE, "guard_search_runs/guard_v3_floor_mmdd/winners/winners.json")
PHASE_B = os.path.join(HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv")

BL_DROPS = [-0.05, -0.08, -0.10, -0.12, -0.15, -0.18, -0.20, -0.25, -0.30]
BL_DAYS = [3, 7, 14, 21, 30, 45, 60]

_DATA = None
_BARS_1H = None
_FUNDING_1H = None
_ALL_DATES = None


def _init_worker():
    """Fork 기반: 부모에서 로드된 전역 data 공유."""
    pass


def _run_one(args):
    label, member, bl_drop, bl_days = args
    params = dict(FIXED_CFG)
    params.update({
        "sma_bars": int(member["sma_bars"]),
        "mom_short_bars": int(member["mom_short_bars"]),
        "mom_long_bars": int(member["mom_long_bars"]),
        "vol_threshold": float(member["vol_threshold"]),
        "snap_interval_bars": int(member["snap_interval_bars"]),
        "bl_drop": bl_drop,
        "bl_days": bl_days,
    })
    bars, funding = _DATA[member["interval"]]
    trace = []
    bt_run(bars, funding, interval=member["interval"], leverage=1.0,
           start_date=START, end_date=END, _trace=trace, **params)
    dates = _ALL_DATES[(_ALL_DATES >= START) & (_ALL_DATES <= END)]
    combined = combine_targets({member["case_id"]: trace}, {member["case_id"]: 1.0}, dates)
    engine = SingleAccountEngine(
        _BARS_1H, _FUNDING_1H,
        leverage=3.0, leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always",
    )
    m = engine.run(combined)
    return {
        "member": label, "bl_drop": bl_drop, "bl_days": bl_days,
        "Cal": round(m.get("Cal", 0), 3),
        "CAGR": round(m.get("CAGR", 0), 4),
        "MDD": round(m.get("MDD", 0), 4),
        "Sh": round(m.get("Sharpe", 0), 3),
        "Liq": int(m.get("Liq", 0)),
    }


def enrich(case_id, phase_b_df, winners):
    w = next(x for x in winners if x["case_id"] == case_id)
    row = phase_b_df[phase_b_df["case_id"] == case_id].iloc[0].to_dict()
    w.update({
        "sma_bars": row["sma_bars"], "mom_short_bars": row["mom_short_bars"],
        "mom_long_bars": row["mom_long_bars"], "vol_threshold": row["vol_threshold"],
        "snap_interval_bars": row["snap_interval_bars"],
    })
    return w


def main():
    global _DATA, _BARS_1H, _FUNDING_1H, _ALL_DATES

    ws = json.load(open(WINNERS))["winners"]
    df = pd.read_csv(PHASE_B, on_bad_lines="skip")
    df = df[df["error"].fillna("") == ""]

    l3_4h = enrich("590f6ac76a1acb4f", df, ws)
    l3_1D = enrich("8a34a940aeba307b", df, ws)

    print("Loading data (once, shared via fork)...")
    _DATA = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    _BARS_1H, _FUNDING_1H = _DATA["1h"]
    _ALL_DATES = _BARS_1H["BTC"].index

    # baselines
    base_4h = _run_one(("4h_S240", l3_4h, 0, 0))
    base_1D = _run_one(("1D_S40", l3_1D, 0, 0))
    print(f"baseline 4h: Cal={base_4h['Cal']:.2f} CAGR={base_4h['CAGR']:+.1%} MDD={base_4h['MDD']:+.1%}")
    print(f"baseline 1D: Cal={base_1D['Cal']:.2f} CAGR={base_1D['CAGR']:+.1%} MDD={base_1D['MDD']:+.1%}")

    tasks = []
    for label, member in [("4h_S240", l3_4h), ("1D_S40", l3_1D)]:
        for d in BL_DROPS:
            for days in BL_DAYS:
                tasks.append((label, member, d, days))

    n_workers = int(os.environ.get("SWEEP_WORKERS", "8"))
    print(f"\n=== Running {len(tasks)} combos × parallel workers={n_workers} ===")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {r['member']} drop={r['bl_drop']} days={r['bl_days']} Cal={r['Cal']}")

    baselines = {"4h_S240": base_4h["Cal"], "1D_S40": base_1D["Cal"]}
    for r in results:
        r["dCal"] = round(r["Cal"] - baselines[r["member"]], 3)

    out = pd.DataFrame(results).sort_values(["member", "Cal"], ascending=[True, False])
    out.to_csv(os.path.join(HERE, "futures_gap_sweep.csv"), index=False)

    for mem in ["4h_S240", "1D_S40"]:
        sub = out[out["member"] == mem]
        print(f"\n--- TOP 10 {mem} by Cal (baseline={baselines[mem]:.2f}) ---")
        print(sub.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
