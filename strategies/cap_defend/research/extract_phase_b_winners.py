#!/usr/bin/env python3
"""Phase B 결과에서 lev별 winner 선정 + equity csv 저장.

산출:
- winners.json: [{lev, case_id, label, guard_name, ...}]
- {case_id}_equity.csv: Date, Value 시계열 (Phase D mix용)
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from backtest_futures_full import load_data
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from futures_live_config import START, END
from run_futures_fixedlev_search import FIXED_CFG


def pick_winner_per_lev(df: pd.DataFrame, max_liq: int = 1) -> pd.DataFrame:
    df = df[df["error"].fillna("") == ""].copy()
    df = df[df["Liq"] <= max_liq]
    out = []
    for lev in sorted(df["leverage"].unique()):
        sub = df[df["leverage"] == lev]
        if sub.empty:
            continue
        out.append(sub.nlargest(1, "Cal").iloc[0])
    return pd.DataFrame(out)


def pick_top_per_lev(df: pd.DataFrame, max_liq: int = 1, top_k: int = 3) -> pd.DataFrame:
    """lev별 top_k. interval 다양성 우선 (interval당 1개씩 채운 후 추가)."""
    df = df[df["error"].fillna("") == ""].copy()
    df = df[df["Liq"] <= max_liq]
    out = []
    for lev in sorted(df["leverage"].unique()):
        sub = df[df["leverage"] == lev].sort_values("Cal", ascending=False)
        if sub.empty:
            continue
        chosen = []
        seen_iv = set()
        for _, row in sub.iterrows():
            if row["interval"] not in seen_iv:
                chosen.append(row)
                seen_iv.add(row["interval"])
                if len(chosen) >= top_k:
                    break
        if len(chosen) < top_k:
            chosen_cids = {r["case_id"] for r in chosen}
            for _, row in sub.iterrows():
                if row["case_id"] in chosen_cids:
                    continue
                chosen.append(row)
                chosen_cids.add(row["case_id"])
                if len(chosen) >= top_k:
                    break
        out.extend(chosen)
    return pd.DataFrame(out)


def build_trace(data, interval, params, start, end):
    from backtest_futures_full import run as bt_run
    bars, funding = data[interval]
    trace: list = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end, _trace=trace, **params)
    return trace


def emit_equity(winner, data, bars_1h, funding_1h, all_dates_1h, out_dir, start, end):
    params = dict(FIXED_CFG)
    params.update({
        "sma_bars": int(winner["sma_bars"]),
        "mom_short_bars": int(winner["mom_short_bars"]),
        "mom_long_bars": int(winner["mom_long_bars"]),
        "vol_threshold": float(winner["vol_threshold"]),
        "snap_interval_bars": int(winner["snap_interval_bars"]),
    })
    trace = build_trace(data, winner["interval"], params, start, end)
    cid = winner["case_id"]
    combined = combine_targets({cid: trace}, {cid: 1.0}, all_dates_1h)
    stop_pct = float(winner["stop_pct_actual"])
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(winner["leverage"]), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind=str(winner["stop_kind"]), stop_pct=stop_pct,
        stop_lookback_bars=int(winner["stop_lookback_bars"]), stop_gate="always",
    )
    metrics = engine.run(combined)
    eq_series = metrics.get("_equity")
    if eq_series is None:
        raise RuntimeError(f"engine returned no _equity for {cid}")
    eq_df = pd.DataFrame({"Date": eq_series.index, "Value": eq_series.values})
    out_csv = os.path.join(out_dir, f"{cid}_equity.csv")
    eq_df.to_csv(out_csv, index=False)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-b-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-liq", type=int, default=1)
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.phase_b_csv, on_bad_lines="skip")
    top_df = pick_top_per_lev(df, max_liq=args.max_liq, top_k=3)
    print(f"Picked {len(top_df)} top entries across levs (top-3 per lev w/ iv diversity)")
    if top_df.empty:
        with open(os.path.join(args.out_dir, "winners.json"), "w") as f:
            json.dump({"winners": [], "fixed_cfg": dict(FIXED_CFG)}, f, indent=2)
        print("WARN: no winners (all liquidated?). Wrote empty winners.json")
        return

    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index

    winners_list = []
    by_lev: dict = {}
    for _, w in top_df.iterrows():
        try:
            csv_path = emit_equity(w, data, bars_1h, funding_1h, all_dates_1h, args.out_dir,
                                    args.start, args.end)
            print(f"L{int(w['leverage'])} {w['case_id']} equity -> {csv_path}")
        except Exception as e:
            print(f"WARN equity emit failed for L{int(w['leverage'])} {w['case_id']}: {e}")
            csv_path = None
        entry = {
            "lev": float(w["leverage"]),
            "case_id": str(w["case_id"]),
            "label": str(w["label"]),
            "interval": str(w["interval"]),
            "guard_name": str(w["guard_name"]),
            "stop_kind": str(w["stop_kind"]),
            "stop_pct_actual": float(w["stop_pct_actual"]),
            "stop_lookback_bars": int(w["stop_lookback_bars"]),
            "Cal": float(w["Cal"]),
            "CAGR": float(w["CAGR"]),
            "MDD": float(w["MDD"]),
            "Sharpe": float(w["Sharpe"]),
            "Liq": int(w["Liq"]),
            "Stops": int(w["Stops"]),
            "equity_csv": csv_path,
        }
        winners_list.append(entry)
        by_lev.setdefault(float(w["leverage"]), []).append(entry)

    # Build true ensembles via SingleAccountEngine with MULTIPLE guard variants per lev
    # (단일계정 합산 + guard sweep)
    ENS_GUARD_VARIANTS = [
        {"name": "none", "stop_kind": "none", "stop_pct": 0.0, "stop_lookback_bars": 0, "eqloss": 0.0},
        {"name": "prev_close_eq10", "stop_kind": "prev_close_pct", "eqloss": 0.10, "stop_lookback_bars": 0},
        {"name": "prev_close_eq20", "stop_kind": "prev_close_pct", "eqloss": 0.20, "stop_lookback_bars": 0},
        {"name": "trail_N5_eq20", "stop_kind": "rolling_high_close_pct", "eqloss": 0.20, "stop_lookback_bars": 5},
        {"name": "trail_N10_eq30", "stop_kind": "rolling_high_close_pct", "eqloss": 0.30, "stop_lookback_bars": 10},
    ]

    for lev, members in by_lev.items():
        if len(members) < 2:
            print(f"L{int(lev)} ensemble skipped (only {len(members)} members)")
            continue
        # Build combined target once (guard 무관)
        traces = {}
        weights = {}
        w = 1.0 / len(members)
        for m in members:
            params = dict(FIXED_CFG)
            w_row = top_df[top_df["case_id"] == m["case_id"]].iloc[0]
            params.update({
                "sma_bars": int(w_row["sma_bars"]),
                "mom_short_bars": int(w_row["mom_short_bars"]),
                "mom_long_bars": int(w_row["mom_long_bars"]),
                "vol_threshold": float(w_row["vol_threshold"]),
                "snap_interval_bars": int(w_row["snap_interval_bars"]),
            })
            tr = build_trace(data, m["interval"], params, args.start, args.end)
            traces[m["case_id"]] = tr
            weights[m["case_id"]] = w
        combined = combine_targets(traces, weights, all_dates_1h)

        for gv in ENS_GUARD_VARIANTS:
            try:
                stop_pct = gv["eqloss"] / lev if gv["stop_kind"] != "none" else 0.0
                engine = SingleAccountEngine(
                    bars_1h, funding_1h,
                    leverage=float(lev), leverage_mode="fixed", per_coin_leverage_mode="none",
                    stop_kind=gv["stop_kind"],
                    stop_pct=stop_pct,
                    stop_lookback_bars=gv["stop_lookback_bars"],
                    stop_gate="always",
                )
                metrics = engine.run(combined)
                eq_series = metrics.get("_equity")
                if eq_series is None:
                    raise RuntimeError("engine returned no _equity for ensemble")
                ens_cid = f"ENS_L{int(lev)}_{gv['name']}"
                out_csv = os.path.join(args.out_dir, f"{ens_cid}_equity.csv")
                pd.DataFrame({"Date": eq_series.index, "Value": eq_series.values}).to_csv(out_csv, index=False)
                ens_entry = {
                    "lev": float(lev),
                    "case_id": ens_cid,
                    "label": f"ensemble_top{len(members)}_guard={gv['name']}",
                    "interval": "mixed",
                    "guard_name": gv["name"],
                    "stop_kind": gv["stop_kind"],
                    "stop_pct_actual": stop_pct,
                    "stop_lookback_bars": gv["stop_lookback_bars"],
                    "Cal": float(metrics.get("Cal", 0)),
                    "CAGR": float(metrics.get("CAGR", 0)),
                    "MDD": float(metrics.get("MDD", 0)),
                    "Sharpe": float(metrics.get("Sharpe", 0)),
                    "Cal_m": float(metrics.get("Cal_m", 0)),
                    "MDD_m_avg": float(metrics.get("MDD_m_avg", 0)),
                    "Liq": int(metrics.get("Liq", 0)),
                    "Stops": int(metrics.get("Stops", 0)),
                    "equity_csv": out_csv,
                    "members": [m["case_id"] for m in members],
                    "is_ensemble": True,
                }
                winners_list.append(ens_entry)
                print(f"L{int(lev)} ENS guard={gv['name']:18s}: "
                      f"Cal={ens_entry['Cal']:.2f} CAGR={ens_entry['CAGR']:+.1%} "
                      f"MDD={ens_entry['MDD']:+.1%} Liq={ens_entry['Liq']}")
            except Exception as e:
                print(f"WARN ensemble build failed L{int(lev)} guard={gv['name']}: {e}")

    out_json = os.path.join(args.out_dir, "winners.json")
    with open(out_json, "w") as f:
        json.dump({"winners": winners_list}, f, indent=2)
    print(f"Wrote {out_json} ({len(winners_list)} entries)")


if __name__ == "__main__":
    main()
