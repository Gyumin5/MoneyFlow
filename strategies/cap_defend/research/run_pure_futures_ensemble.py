#!/usr/bin/env python3
"""Pure futures ensemble scan.

1. From phase_b_results.csv (error==""), pick top-K singles per interval
   to get a pool of size ~8 with interval diversity.
2. Enumerate all 2/3/4-member EW combinations.
3. Run full-period backtest (2020-10 ~ 2026-04-13), L3 fixed, guard=none.
4. Output pure_ens_results.csv + top.json (top-10 by Cal improvement vs d005).

Pool selection: interval top-3 (1D, 2h, 4h) sorted by Cal.
Combos: C(9,2)+C(9,3)+C(9,4)=36+84+126=246 (manageable).
Can reduce via --top-per-iv and --max-members.
"""
from __future__ import annotations
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from backtest_futures_full import load_data, run as bt_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from run_futures_fixedlev_search import FIXED_CFG
from futures_live_config import CURRENT_STRATEGIES

FULL_START = "2020-10-01"
FULL_END = "2026-04-13"
LEV = 3.0


def build_trace(data, interval, params, start, end):
    bars, funding = data[interval]
    trace: list = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end, _trace=trace, **params)
    return trace


def run_combo(member_rows, data, bars_1h, funding_1h, all_dates_1h, trace_cache):
    traces, weights = {}, {}
    w = 1.0 / len(member_rows)
    for m in member_rows:
        cid = m["case_id"]
        if cid not in trace_cache:
            params = dict(FIXED_CFG)
            params.update({
                "sma_bars": int(m["sma_bars"]),
                "mom_short_bars": int(m["mom_short_bars"]),
                "mom_long_bars": int(m["mom_long_bars"]),
                "vol_threshold": float(m["vol_threshold"]),
                "snap_interval_bars": int(m["snap_interval_bars"]),
            })
            trace_cache[cid] = build_trace(data, m["interval"], params, FULL_START, FULL_END)
        traces[cid] = trace_cache[cid]
        weights[cid] = w
    combined = combine_targets(traces, weights, all_dates_1h)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=LEV, leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_gate="always",
    )
    return engine.run(combined)


def run_d005_baseline(data, bars_1h, funding_1h, all_dates_1h):
    traces, weights = {}, {}
    w = 1.0 / len(CURRENT_STRATEGIES)
    for name, cfg in CURRENT_STRATEGIES.items():
        params = dict(FIXED_CFG)
        params["vol_mode"] = cfg["vol_mode"]
        params.update({
            "sma_bars": cfg["sma_bars"],
            "mom_short_bars": cfg["mom_short_bars"],
            "mom_long_bars": cfg["mom_long_bars"],
            "vol_threshold": cfg["vol_threshold"],
            "snap_interval_bars": cfg["snap_interval_bars"],
        })
        tr = build_trace(data, cfg["interval"], params, FULL_START, FULL_END)
        traces[f"d005_{name}"] = tr
        weights[f"d005_{name}"] = w
    combined = combine_targets(traces, weights, all_dates_1h)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=LEV, leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_gate="always",
    )
    return engine.run(combined)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--phase-b-csv", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "pure_futures_ens"))
    p.add_argument("--top-per-iv", type=int, default=3)
    p.add_argument("--max-members", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.phase_b_csv, on_bad_lines="skip")
    df = df[df["error"].fillna("") == ""].copy()
    # dedupe: same interval+sma+ms+ml+vol+snap => one entry (take best Cal)
    df = df.sort_values("Cal", ascending=False).drop_duplicates(
        subset=["interval", "sma_bars", "mom_short_bars", "mom_long_bars",
                "vol_threshold", "snap_interval_bars"])
    pool_rows = []
    for iv in ["D", "2h", "4h"]:
        sub = df[df["interval"] == iv].sort_values("Cal", ascending=False).head(args.top_per_iv)
        for _, r in sub.iterrows():
            pool_rows.append(r.to_dict())
    print(f"Pool size: {len(pool_rows)} (top-{args.top_per_iv} per iv × 3)")
    for r in pool_rows:
        print(f"  {r['case_id']:<18s} {r['interval']:<3s} {r['label'][:40]:<40s} Cal={r['Cal']:.2f}")

    print("\nLoading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index
    all_dates_1h = all_dates_1h[(all_dates_1h >= FULL_START) & (all_dates_1h <= FULL_END)]

    print("\nRunning d005 baseline at L3...")
    t0 = time.time()
    d005_m = run_d005_baseline(data, bars_1h, funding_1h, all_dates_1h)
    print(f"  d005 L3: Cal={d005_m.get('Cal',0):.2f} CAGR={d005_m.get('CAGR',0):+.1%} "
          f"MDD={d005_m.get('MDD',0):+.1%} Sh={d005_m.get('Sharpe',0):.2f} "
          f"({time.time()-t0:.1f}s)")

    trace_cache: dict = {}
    rows = []
    combos = []
    for n in range(2, args.max_members + 1):
        combos.extend(itertools.combinations(range(len(pool_rows)), n))
    print(f"\nCombos to evaluate: {len(combos)}")

    t_start = time.time()
    for idx, idxs in enumerate(combos, 1):
        members = [pool_rows[i] for i in idxs]
        ids = [m["case_id"] for m in members]
        labels = [m["label"] for m in members]
        ivs = [m["interval"] for m in members]
        try:
            t0 = time.time()
            metrics = run_combo(members, data, bars_1h, funding_1h, all_dates_1h, trace_cache)
            el = time.time() - t0
            row = {
                "combo_id": "+".join(ids),
                "n_members": len(ids),
                "intervals": "+".join(ivs),
                "labels": " | ".join(labels),
                "Cal": metrics.get("Cal", 0),
                "CAGR": metrics.get("CAGR", 0),
                "MDD": metrics.get("MDD", 0),
                "Sharpe": metrics.get("Sharpe", 0),
                "Liq": metrics.get("Liq", 0),
                "Rebal": metrics.get("Rebal", 0),
                "elapsed_sec": el,
                "error": "",
            }
            rows.append(row)
            if idx % 10 == 0 or idx == len(combos):
                elapsed = time.time() - t_start
                eta = elapsed / idx * (len(combos) - idx)
                print(f"  [{idx}/{len(combos)}] Cal={row['Cal']:.2f} "
                      f"MDD={row['MDD']:+.1%} n={row['n_members']} "
                      f"({row['intervals']}) elapsed={elapsed:.0f}s eta={eta:.0f}s")
        except Exception as e:
            rows.append({
                "combo_id": "+".join(ids),
                "n_members": len(ids),
                "intervals": "+".join(ivs),
                "labels": " | ".join(labels),
                "Cal": 0, "CAGR": 0, "MDD": 0, "Sharpe": 0, "Liq": 0, "Rebal": 0,
                "elapsed_sec": 0, "error": str(e),
            })
            print(f"  [{idx}/{len(combos)}] ERROR {e}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(args.out_dir, "pure_ens_results.csv"), index=False)

    # d005 baseline reference
    d005_cal = d005_m.get("Cal", 0)
    d005_mdd = d005_m.get("MDD", 0)

    # filter: Cal > d005_cal AND MDD better (less negative) than d005_mdd
    improved = res_df[(res_df["error"] == "") &
                      (res_df["Cal"] > d005_cal) &
                      (res_df["MDD"] > d005_mdd)].copy()
    improved = improved.sort_values("Cal", ascending=False).head(10)

    top_json = {
        "d005_baseline": {
            "Cal": d005_cal, "CAGR": d005_m.get("CAGR", 0),
            "MDD": d005_mdd, "Sharpe": d005_m.get("Sharpe", 0),
            "lev": LEV, "period": f"{FULL_START} ~ {FULL_END}",
        },
        "top_improved": improved.to_dict(orient="records"),
        "n_total": len(res_df), "n_improved": len(improved),
    }
    with open(os.path.join(args.out_dir, "top.json"), "w") as f:
        json.dump(top_json, f, indent=2, default=str)

    print(f"\n=== TOP improved vs d005 (Cal={d005_cal:.2f} MDD={d005_mdd:+.1%}) ===")
    for _, r in improved.iterrows():
        print(f"  Cal={r['Cal']:.2f} MDD={r['MDD']:+.1%} Sh={r['Sharpe']:.2f} "
              f"n={r['n_members']} iv={r['intervals']}")


if __name__ == "__main__":
    main()
