#!/usr/bin/env python3
"""Phase B: 가드 그리드 탐색 러너.

phase A에서 추출된 seeds (lev × tf × params)에 다양한 stop guard 변형을 적용해
어느 guard가 lev별로 최적인지 비교한다.

Guard 그리드 (21):
- none
- prev_close_pct × eqloss {5,10,20,30,40}   = 5
- rolling_high_close N=3 × eqloss 5종         = 5
- rolling_high_close N=5 × eqloss 5종         = 5
- rolling_high_close N=10 × eqloss 5종        = 5

eqloss = stop_pct × leverage (자본손실 환산). lev별 stop_pct = eqloss / lev.

같은 run-name으로 재실행 시 case_id 누적/skip → resume 지원.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from typing import List

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "research"))

from backtest_futures_full import load_data
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from futures_live_config import END, START
from run_futures_fixedlev_search import FIXED_CFG, _build_trace, append_journal, ensure_dir

RESULTS_ROOT = os.path.join(HERE, "research", "guard_search_runs")

# Guard variants
EQLOSS_GRID = [0.05, 0.10, 0.20, 0.30, 0.40]
TRAIL_LOOKBACKS = [3, 5, 10]


def build_guard_variants() -> List[dict]:
    variants = [{"name": "none", "stop_kind": "none", "stop_pct": 0.0, "stop_lookback_bars": 0, "eqloss": 0.0}]
    for eq in EQLOSS_GRID:
        variants.append({
            "name": f"prev_close_eq{int(eq*100):02d}",
            "stop_kind": "prev_close_pct",
            "stop_pct": eq,  # placeholder; set per-lev below
            "stop_lookback_bars": 0,
            "eqloss": eq,
        })
    for n in TRAIL_LOOKBACKS:
        for eq in EQLOSS_GRID:
            variants.append({
                "name": f"trail_close_N{n}_eq{int(eq*100):02d}",
                "stop_kind": "rolling_high_close_pct",
                "stop_pct": eq,
                "stop_lookback_bars": n,
                "eqloss": eq,
            })
    return variants


def case_hash(parts: list) -> str:
    raw = "|".join(str(p) for p in parts).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


FIELDS = [
    "case_id", "stage", "leverage", "interval", "label",
    "sma_bars", "mom_short_bars", "mom_long_bars", "vol_threshold", "snap_interval_bars",
    "guard_name", "stop_kind", "stop_pct_actual", "stop_lookback_bars", "eqloss_target",
    "Sharpe", "CAGR", "MDD", "Cal", "MDD_m_avg", "MDD_m_min", "MDD_m_max", "Cal_m",
    "Liq", "Stops", "Rebal",
    "elapsed_sec", "error",
]

# worker globals
_WORK_DATA = None
_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_ALL_DATES_1H = None


def _init_worker(data, bars_1h, funding_1h, all_dates_1h):
    global _WORK_DATA, _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_ALL_DATES_1H
    _WORK_DATA = data
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_ALL_DATES_1H = all_dates_1h


def _build_trace_local(interval: str, params: dict, start_date: str, end_date: str):
    from backtest_futures_full import run as bt_run
    bars, funding = _WORK_DATA[interval]
    trace: list = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start_date, end_date=end_date, _trace=trace, **params)
    return trace


def _run_case(work_item):
    case, start_date, end_date = work_item
    t0 = time.time()
    try:
        params = dict(FIXED_CFG)
        params.update({
            "sma_bars": case["sma_bars"],
            "mom_short_bars": case["mom_short_bars"],
            "mom_long_bars": case["mom_long_bars"],
            "vol_threshold": case["vol_threshold"],
            "snap_interval_bars": case["snap_interval_bars"],
        })
        trace = _build_trace_local(case["interval"], params, start_date, end_date)
        combined = combine_targets({case["case_id"]: trace}, {case["case_id"]: 1.0}, _WORK_ALL_DATES_1H)
        # eqloss → stop_pct per lev
        stop_pct_actual = case["eqloss_target"] / case["leverage"] if case["stop_kind"] != "none" else 0.0
        engine = SingleAccountEngine(
            _WORK_BARS_1H, _WORK_FUNDING_1H,
            leverage=case["leverage"], leverage_mode="fixed", per_coin_leverage_mode="none",
            stop_kind=case["stop_kind"], stop_pct=stop_pct_actual,
            stop_lookback_bars=case["stop_lookback_bars"], stop_gate="always",
        )
        m = engine.run(combined)
        row = {
            "case_id": case["case_id"], "stage": "phase_b",
            "leverage": case["leverage"], "interval": case["interval"], "label": case["label"],
            "sma_bars": params["sma_bars"], "mom_short_bars": params["mom_short_bars"],
            "mom_long_bars": params["mom_long_bars"], "vol_threshold": params["vol_threshold"],
            "snap_interval_bars": params["snap_interval_bars"],
            "guard_name": case["guard_name"], "stop_kind": case["stop_kind"],
            "stop_pct_actual": stop_pct_actual, "stop_lookback_bars": case["stop_lookback_bars"],
            "eqloss_target": case["eqloss_target"],
            "Sharpe": m.get("Sharpe", 0), "CAGR": m.get("CAGR", 0), "MDD": m.get("MDD", 0),
            "Cal": m.get("Cal", 0),
            "MDD_m_avg": m.get("MDD_m_avg", 0), "MDD_m_min": m.get("MDD_m_min", 0),
            "MDD_m_max": m.get("MDD_m_max", 0), "Cal_m": m.get("Cal_m", 0),
            "Liq": m.get("Liq", 0), "Stops": m.get("Stops", 0),
            "Rebal": m.get("Rebal", 0),
            "elapsed_sec": time.time() - t0, "error": "",
        }
        prog = (f"[B] L{int(case['leverage'])} {case['interval']} {case['label'][:30]} "
                f"{case['guard_name']} Cal={row['Cal']:.2f} MDD={row['MDD']:+.1%} "
                f"Liq={row['Liq']} Stops={row['Stops']}")
        return row, prog
    except Exception as exc:
        row = {f: "" for f in FIELDS}
        row.update({
            "case_id": case["case_id"], "stage": "phase_b",
            "leverage": case["leverage"], "interval": case["interval"], "label": case["label"],
            "guard_name": case["guard_name"], "stop_kind": case["stop_kind"],
            "eqloss_target": case["eqloss_target"], "Cal": -999, "CAGR": -999, "MDD": -1,
            "elapsed_sec": time.time() - t0, "error": str(exc),
        })
        return row, f"[B] ERROR {case['case_id']}: {exc}"


def load_existing(out_csv: str) -> set:
    if not os.path.isfile(out_csv):
        return set()
    done = set()
    with open(out_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row.get("case_id")
            if cid:
                done.add(cid)
    return done


def append_row(out_csv: str, row: dict):
    write_header = not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def build_cases(seeds: list, variants: list) -> list:
    cases = []
    for seed in seeds:
        for v in variants:
            cid = case_hash([
                seed["interval"], seed["sma_bars"], seed["mom_short_bars"], seed["mom_long_bars"],
                seed["vol_threshold"], seed["snap_interval_bars"], seed["lev"],
                v["name"], v["eqloss"],
            ])
            cases.append({
                "case_id": cid, "leverage": seed["lev"], "interval": seed["interval"],
                "label": seed["label"], "sma_bars": seed["sma_bars"],
                "mom_short_bars": seed["mom_short_bars"], "mom_long_bars": seed["mom_long_bars"],
                "vol_threshold": seed["vol_threshold"], "snap_interval_bars": seed["snap_interval_bars"],
                "guard_name": v["name"], "stop_kind": v["stop_kind"],
                "stop_lookback_bars": v["stop_lookback_bars"], "eqloss_target": v["eqloss"],
            })
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True, help="phase_b_seeds.json path")
    p.add_argument("--run-name", default="guard_v1")
    p.add_argument("--results-root", default=RESULTS_ROOT)
    p.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    args = p.parse_args()

    run_dir = os.path.join(args.results_root, args.run_name)
    ensure_dir(run_dir)
    out_csv = os.path.join(run_dir, "phase_b_results.csv")
    journal = os.path.join(run_dir, "journal.log")
    meta = os.path.join(run_dir, "metadata.json")

    with open(args.seeds) as f:
        seed_data = json.load(f)
    seeds = seed_data["seeds"]

    variants = build_guard_variants()
    all_cases = build_cases(seeds, variants)

    done = load_existing(out_csv)
    pending = [c for c in all_cases if c["case_id"] not in done]
    append_journal(journal, f"Run start workers={args.workers} total={len(all_cases)} done={len(done)} pending={len(pending)}")
    print(f"Total {len(all_cases)} cases, done {len(done)}, pending {len(pending)}")

    with open(meta, "w") as f:
        json.dump({"run_name": args.run_name, "n_seeds": len(seeds), "n_variants": len(variants),
                   "total_cases": len(all_cases), "start": args.start, "end": args.end,
                   "started_at": datetime.now().isoformat()}, f, indent=2)

    if not pending:
        append_journal(journal, "Nothing to do.")
        print("All done.")
        return

    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index

    work_items = [(c, args.start, args.end) for c in pending]
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(data, bars_1h, funding_1h, all_dates_1h)) as pool:
        for i, (row, prog) in enumerate(pool.imap_unordered(_run_case, work_items, chunksize=1), 1):
            append_row(out_csv, row)
            if i % 20 == 0 or i == len(work_items):
                append_journal(journal, f"{prog} ({i}/{len(work_items)})")
                print(f"{prog} ({i}/{len(work_items)})")

    append_journal(journal, "Phase B done.")
    print("Phase B complete.")


if __name__ == "__main__":
    main()
