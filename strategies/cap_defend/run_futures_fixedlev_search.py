#!/usr/bin/env python3
"""고정 레버리지 선물 전략 탐색 러너.

목표
- 레버리지 world 분리: 2x / 3x / 4x / 5x
- Top3 종목 고정, 종목당 cap=1/3
- 단일 전략 탐색: 1D / 4h / 2h
- 각 world + 가드 트랙별 시간봉 Top3 선발
- 최대 4개 전략 EW 앙상블 탐색
- 결과/진행상황 지속 저장 + resume 지원

가드 트랙
- no_guard: stop 없음
- stop_only: prev_close_pct stop만 항상 적용
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backtest_futures_full import load_data, run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from futures_live_config import END, START

RESULTS_ROOT = os.path.join(HERE, "research", "fixedlev_search_runs")
LEVERAGES = [2.0, 3.0, 4.0, 5.0]
GUARD_MODES = ["no_guard", "stop_only"]
TIMEFRAME_ORDER = ["D", "4h", "2h"]
TIMEFRAME_LABELS = {"D": "1D", "4h": "4h", "2h": "2h"}

FIXED_CFG = dict(
    universe_size=3,
    selection="greedy",
    cap=1 / 3,
    tx_cost=0.0004,
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    health_mode="mom2vol",
    vol_mode="daily",
    n_snapshots=3,
)

PARAM_GRID = {
    "D": dict(
        sma_bars=[40, 50, 60, 80],
        mom_short_bars=[10, 20, 30],
        mom_long_bars=[60, 90, 120, 180],
        snap_interval_bars=[15, 21, 30, 45],
    ),
    "4h": dict(
        sma_bars=[120, 180, 240, 360],
        mom_short_bars=[10, 20, 30, 40],
        mom_long_bars=[120, 240, 480, 720],
        snap_interval_bars=[21, 30, 42, 60, 84],
    ),
    "2h": dict(
        sma_bars=[120, 180, 240, 360],
        mom_short_bars=[10, 20, 30, 40],
        mom_long_bars=[120, 240, 480, 720],
        snap_interval_bars=[42, 60, 84, 120, 168],
    ),
}
VOL_THRESHOLDS = [0.05, 0.07]

SINGLE_FIELDS = [
    "case_id",
    "world_key",
    "stage",
    "leverage",
    "guard_mode",
    "interval",
    "timeframe_label",
    "label",
    "sma_bars",
    "mom_short_bars",
    "mom_long_bars",
    "vol_threshold",
    "snap_interval_bars",
    "Sharpe",
    "CAGR",
    "MDD",
    "Cal",
    "Liq",
    "Stops",
    "Rebal",
    "elapsed_sec",
    "error",
]

ENSEMBLE_FIELDS = [
    "case_id",
    "world_key",
    "stage",
    "leverage",
    "guard_mode",
    "n_members",
    "members",
    "member_labels",
    "member_intervals",
    "Sharpe",
    "CAGR",
    "MDD",
    "Cal",
    "Liq",
    "Stops",
    "Rebal",
    "elapsed_sec",
    "error",
]

_WORK_DATA = {}
_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_ALL_DATES_1H = None
_WORK_TRACE_MAP = None
_WORK_STOP_PCT = 0.15


@dataclass(frozen=True)
class StrategyCase:
    case_id: str
    interval: str
    leverage: float
    guard_mode: str
    params: dict
    label: str


def guard_engine_kwargs(guard_mode: str, stop_pct: float) -> dict:
    if guard_mode == "stop_only":
        return dict(
            stop_kind="prev_close_pct",
            stop_pct=stop_pct,
            stop_gate="always",
        )
    return dict(stop_kind="none", stop_pct=0.0, stop_gate="always")


def world_key(leverage: float, guard_mode: str) -> str:
    return f"L{int(leverage)}_{guard_mode}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_journal(path: str, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_csv(path: str, fieldnames: List[str], row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done_ids(path: str) -> set[str]:
    return {row["case_id"] for row in read_csv_rows(path) if row.get("case_id")}


def case_hash(parts: Iterable[str]) -> str:
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def strategy_label(interval: str, params: dict) -> str:
    return (
        f"{TIMEFRAME_LABELS[interval]}"
        f"_S{params['sma_bars']}"
        f"_MS{params['mom_short_bars']}"
        f"_ML{params['mom_long_bars']}"
        f"_V{int(params['vol_threshold'] * 100):02d}"
        f"_SN{params['snap_interval_bars']}"
    )


def build_single_cases(
    leverages: List[float],
    guard_modes: List[str],
) -> List[StrategyCase]:
    cases: List[StrategyCase] = []
    for leverage in leverages:
        for guard_mode in guard_modes:
            for interval in TIMEFRAME_ORDER:
                grid = PARAM_GRID[interval]
                keys = list(grid.keys())
                values = [grid[k] for k in keys]
                for combo in itertools.product(*values):
                    params = dict(zip(keys, combo))
                    for vol_threshold in VOL_THRESHOLDS:
                        full = dict(FIXED_CFG)
                        full.update(params)
                        full["vol_threshold"] = vol_threshold
                        cid = case_hash(
                            [
                                "single",
                                interval,
                                f"{leverage:.1f}",
                                guard_mode,
                                str(full["sma_bars"]),
                                str(full["mom_short_bars"]),
                                str(full["mom_long_bars"]),
                                f"{full['vol_threshold']:.2f}",
                                str(full["snap_interval_bars"]),
                            ]
                        )
                        cases.append(
                            StrategyCase(
                                case_id=cid,
                                interval=interval,
                                leverage=leverage,
                                guard_mode=guard_mode,
                                params=full,
                                label=strategy_label(interval, full),
                            )
                        )
    return cases


def metric_sort_key(row: dict) -> tuple:
    cal = float(row.get("Cal", -999) or -999)
    sharpe = float(row.get("Sharpe", -999) or -999)
    mdd = float(row.get("MDD", -1) or -1)
    cagr = float(row.get("CAGR", -999) or -999)
    liq = int(float(row.get("Liq", 999999) or 999999))
    stops = int(float(row.get("Stops", 999999) or 999999))
    rebal = int(float(row.get("Rebal", 999999) or 999999))
    return (-cal, -sharpe, abs(mdd), -cagr, liq, stops, rebal, row.get("label", ""))


def init_run_dir(run_dir: str, args: argparse.Namespace) -> None:
    ensure_dir(run_dir)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "script": os.path.abspath(__file__),
        "start": args.start,
        "end": args.end,
        "workers": args.workers,
        "leverages": args.leverages,
        "guard_modes": args.guard_modes,
        "top_k_per_timeframe": args.top_k_per_timeframe,
        "max_ensemble_size": args.max_ensemble_size,
        "stop_pct": args.stop_pct,
        "notes": {
            "n_picks": 3,
            "cap": 1 / 3,
            "canary_hyst": 0.015,
            "vol_mode": "daily",
            "vol_thresholds": VOL_THRESHOLDS,
            "guard_tracks": {
                "no_guard": "stop 없음, liq만 존재",
                "stop_only": "prev_close_pct stop만 항상 적용",
            },
        },
    }
    write_json(os.path.join(run_dir, "metadata.json"), metadata)
    readme_path = os.path.join(run_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "# Fixed Leverage Futures Search\n\n"
                "이 디렉터리는 고정 레버리지 선물 탐색 결과를 저장한다.\n\n"
                "구성:\n"
                "- `single_results.csv`: 단일 전략 결과 누적\n"
                "- `selected_top3.json`: world별 시간봉 Top3 선발 결과\n"
                "- `ensemble_results.csv`: 앙상블 결과 누적\n"
                "- `journal.log`: 진행 로그\n"
                "- `metadata.json`: 실행 조건\n"
                "- `summary.json`: 단계별 요약\n"
            )


def _init_single_worker(data, bars_1h, funding_1h, all_dates_1h, stop_pct: float):
    global _WORK_DATA, _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_ALL_DATES_1H, _WORK_STOP_PCT
    _WORK_DATA = data
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_ALL_DATES_1H = all_dates_1h
    _WORK_STOP_PCT = stop_pct


def _build_trace(interval: str, params: dict, start_date: str, end_date: str):
    bars, funding = _WORK_DATA[interval]
    trace: List[dict] = []
    run(
        bars,
        funding,
        interval=interval,
        leverage=1.0,
        start_date=start_date,
        end_date=end_date,
        _trace=trace,
        **params,
    )
    return trace


def _run_single_case(work_item):
    case_dict, start_date, end_date = work_item
    t0 = time.time()
    try:
        interval = case_dict["interval"]
        params = dict(case_dict["params"])
        trace = _build_trace(interval, params, start_date, end_date)
        combined = combine_targets({case_dict["case_id"]: trace}, {case_dict["case_id"]: 1.0}, _WORK_ALL_DATES_1H)
        engine = SingleAccountEngine(
            _WORK_BARS_1H,
            _WORK_FUNDING_1H,
            leverage=case_dict["leverage"],
            leverage_mode="fixed",
            per_coin_leverage_mode="none",
            **guard_engine_kwargs(case_dict["guard_mode"], _WORK_STOP_PCT),
        )
        metrics = engine.run(combined)
        row = {
            "case_id": case_dict["case_id"],
            "world_key": world_key(case_dict["leverage"], case_dict["guard_mode"]),
            "stage": "single",
            "leverage": case_dict["leverage"],
            "guard_mode": case_dict["guard_mode"],
            "interval": interval,
            "timeframe_label": TIMEFRAME_LABELS[interval],
            "label": case_dict["label"],
            "sma_bars": params["sma_bars"],
            "mom_short_bars": params["mom_short_bars"],
            "mom_long_bars": params["mom_long_bars"],
            "vol_threshold": params["vol_threshold"],
            "snap_interval_bars": params["snap_interval_bars"],
            "Sharpe": metrics.get("Sharpe", 0),
            "CAGR": metrics.get("CAGR", 0),
            "MDD": metrics.get("MDD", 0),
            "Cal": metrics.get("Cal", 0),
            "Liq": metrics.get("Liq", 0),
            "Stops": metrics.get("Stops", 0),
            "Rebal": metrics.get("Rebal", 0),
            "elapsed_sec": time.time() - t0,
            "error": "",
        }
        progress = (
            f"[single] {row['world_key']} {row['label']} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
        )
        return row, progress
    except Exception as exc:
        row = {
            "case_id": case_dict["case_id"],
            "world_key": world_key(case_dict["leverage"], case_dict["guard_mode"]),
            "stage": "single",
            "leverage": case_dict["leverage"],
            "guard_mode": case_dict["guard_mode"],
            "interval": case_dict["interval"],
            "timeframe_label": TIMEFRAME_LABELS[case_dict["interval"]],
            "label": case_dict["label"],
            "sma_bars": case_dict["params"]["sma_bars"],
            "mom_short_bars": case_dict["params"]["mom_short_bars"],
            "mom_long_bars": case_dict["params"]["mom_long_bars"],
            "vol_threshold": case_dict["params"]["vol_threshold"],
            "snap_interval_bars": case_dict["params"]["snap_interval_bars"],
            "Sharpe": 0,
            "CAGR": -999,
            "MDD": -1,
            "Cal": -999,
            "Liq": 0,
            "Stops": 0,
            "Rebal": 0,
            "elapsed_sec": time.time() - t0,
            "error": str(exc),
        }
        progress = f"[single] ERROR {row['world_key']} {row['label']} {exc}"
        return row, progress


def select_top3(single_rows: List[dict], top_k_per_timeframe: int) -> dict:
    selected = {}
    for leverage in LEVERAGES:
        for guard_mode in GUARD_MODES:
            wk = world_key(leverage, guard_mode)
            selected[wk] = {}
            for interval in TIMEFRAME_ORDER:
                subset = [
                    row for row in single_rows
                    if row.get("world_key") == wk
                    and row.get("interval") == interval
                    and not row.get("error")
                ]
                subset.sort(key=metric_sort_key)
                selected[wk][interval] = subset[:top_k_per_timeframe]
    return selected


def _init_ensemble_worker(bars_1h, funding_1h, all_dates_1h, trace_map, stop_pct: float):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_ALL_DATES_1H, _WORK_TRACE_MAP, _WORK_STOP_PCT
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_ALL_DATES_1H = all_dates_1h
    _WORK_TRACE_MAP = trace_map
    _WORK_STOP_PCT = stop_pct


def _run_ensemble_case(work_item):
    payload = work_item
    t0 = time.time()
    try:
        member_ids = payload["member_ids"]
        weights = {mid: 1.0 / len(member_ids) for mid in member_ids}
        traces = {mid: _WORK_TRACE_MAP[mid] for mid in member_ids}
        combined = combine_targets(traces, weights, _WORK_ALL_DATES_1H)
        engine = SingleAccountEngine(
            _WORK_BARS_1H,
            _WORK_FUNDING_1H,
            leverage=payload["leverage"],
            leverage_mode="fixed",
            per_coin_leverage_mode="none",
            **guard_engine_kwargs(payload["guard_mode"], _WORK_STOP_PCT),
        )
        metrics = engine.run(combined)
        row = {
            "case_id": payload["case_id"],
            "world_key": world_key(payload["leverage"], payload["guard_mode"]),
            "stage": "ensemble",
            "leverage": payload["leverage"],
            "guard_mode": payload["guard_mode"],
            "n_members": len(member_ids),
            "members": "+".join(member_ids),
            "member_labels": " | ".join(payload["member_labels"]),
            "member_intervals": "+".join(payload["member_intervals"]),
            "Sharpe": metrics.get("Sharpe", 0),
            "CAGR": metrics.get("CAGR", 0),
            "MDD": metrics.get("MDD", 0),
            "Cal": metrics.get("Cal", 0),
            "Liq": metrics.get("Liq", 0),
            "Stops": metrics.get("Stops", 0),
            "Rebal": metrics.get("Rebal", 0),
            "elapsed_sec": time.time() - t0,
            "error": "",
        }
        progress = (
            f"[ensemble] {row['world_key']} n={row['n_members']} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
        )
        return row, progress
    except Exception as exc:
        row = {
            "case_id": payload["case_id"],
            "world_key": world_key(payload["leverage"], payload["guard_mode"]),
            "stage": "ensemble",
            "leverage": payload["leverage"],
            "guard_mode": payload["guard_mode"],
            "n_members": len(payload["member_ids"]),
            "members": "+".join(payload["member_ids"]),
            "member_labels": " | ".join(payload["member_labels"]),
            "member_intervals": "+".join(payload["member_intervals"]),
            "Sharpe": 0,
            "CAGR": -999,
            "MDD": -1,
            "Cal": -999,
            "Liq": 0,
            "Stops": 0,
            "Rebal": 0,
            "elapsed_sec": time.time() - t0,
            "error": str(exc),
        }
        progress = f"[ensemble] ERROR {row['world_key']} {row['members']} {exc}"
        return row, progress


def build_ensemble_cases(selected_top3: dict, max_ensemble_size: int) -> List[dict]:
    cases = []
    for leverage in LEVERAGES:
        for guard_mode in GUARD_MODES:
            wk = world_key(leverage, guard_mode)
            candidates = []
            for interval in TIMEFRAME_ORDER:
                candidates.extend(selected_top3.get(wk, {}).get(interval, []))
            if not candidates:
                continue
            for n_members in range(1, min(max_ensemble_size, len(candidates)) + 1):
                for combo in itertools.combinations(candidates, n_members):
                    member_ids = [row["case_id"] for row in combo]
                    case_id = case_hash(["ensemble", wk, *member_ids])
                    cases.append(
                        {
                            "case_id": case_id,
                            "leverage": leverage,
                            "guard_mode": guard_mode,
                            "member_ids": member_ids,
                            "member_labels": [row["label"] for row in combo],
                            "member_intervals": [row["interval"] for row in combo],
                        }
                    )
    return cases


def generate_trace_map(selected_top3: dict, data: dict, args: argparse.Namespace, journal_path: str) -> dict:
    trace_map = {}
    already = {}
    for wk in selected_top3.values():
        for rows in wk.values():
            for row in rows:
                already[row["case_id"]] = row
    ordered_rows = list(already.values())
    append_journal(journal_path, f"Generating traces for selected candidates: {len(ordered_rows)}")
    for idx, row in enumerate(ordered_rows, start=1):
        params = dict(FIXED_CFG)
        params.update(
            sma_bars=int(row["sma_bars"]),
            mom_short_bars=int(row["mom_short_bars"]),
            mom_long_bars=int(row["mom_long_bars"]),
            vol_threshold=float(row["vol_threshold"]),
            snap_interval_bars=int(row["snap_interval_bars"]),
        )
        interval = row["interval"]
        bars, funding = data[interval]
        trace: List[dict] = []
        run(
            bars,
            funding,
            interval=interval,
            leverage=1.0,
            start_date=args.start,
            end_date=args.end,
            _trace=trace,
            **params,
        )
        trace_map[row["case_id"]] = trace
        append_journal(journal_path, f"Trace ready {idx}/{len(ordered_rows)} {row['label']}")
    return trace_map


def run_parallel(
    work_items: List,
    worker_fn,
    init_fn,
    initargs: tuple,
    workers: int,
    append_row_fn,
    journal_path: str,
    progress_every: int = 25,
) -> int:
    if not work_items:
        return 0
    completed = 0
    total = len(work_items)
    if workers <= 1:
        init_fn(*initargs)
        for item in work_items:
            row, progress = worker_fn(item)
            append_row_fn(row)
            completed += 1
            print(progress)
            if completed == 1 or completed % progress_every == 0 or completed == total:
                append_journal(journal_path, f"{progress} ({completed}/{total})")
        return completed

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=init_fn, initargs=initargs) as pool:
        for row, progress in pool.imap_unordered(worker_fn, work_items, chunksize=1):
            append_row_fn(row)
            completed += 1
            print(progress)
            if completed == 1 or completed % progress_every == 0 or completed == total:
                append_journal(journal_path, f"{progress} ({completed}/{total})")
    return completed


def coerce_selection(selected: dict, leverages: List[float], guard_modes: List[str]) -> dict:
    result = {}
    for leverage in leverages:
        for guard_mode in guard_modes:
            wk = world_key(leverage, guard_mode)
            result[wk] = {}
            for interval in TIMEFRAME_ORDER:
                result[wk][interval] = selected.get(wk, {}).get(interval, [])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="고정 레버리지 선물 전략 탐색")
    parser.add_argument("--stage", choices=["single", "ensemble", "all"], default="all")
    parser.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--top-k-per-timeframe", type=int, default=3)
    parser.add_argument("--max-ensemble-size", type=int, default=4)
    parser.add_argument("--stop-pct", type=float, default=0.15)
    parser.add_argument("--single-limit", type=int, default=0)
    parser.add_argument("--ensemble-limit", type=int, default=0)
    parser.add_argument("--leverages", nargs="*", type=float, default=LEVERAGES)
    parser.add_argument("--guard-modes", nargs="*", default=GUARD_MODES)
    return parser


def main():
    args = build_parser().parse_args()
    args.leverages = [float(x) for x in args.leverages]
    args.guard_modes = list(args.guard_modes)

    run_dir = os.path.join(args.results_root, args.run_name)
    init_run_dir(run_dir, args)
    journal_path = os.path.join(run_dir, "journal.log")
    summary_path = os.path.join(run_dir, "summary.json")
    single_csv = os.path.join(run_dir, "single_results.csv")
    selected_json = os.path.join(run_dir, "selected_top3.json")
    ensemble_csv = os.path.join(run_dir, "ensemble_results.csv")

    append_journal(journal_path, f"Run start stage={args.stage} workers={args.workers}")
    t0 = time.time()

    intervals = sorted(set(TIMEFRAME_ORDER + ["1h"]))
    print(f"Loading data: {intervals}")
    data = {iv: load_data(iv) for iv in intervals}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= args.start) & (bars_1h["BTC"].index <= args.end)
    ]

    summary = {
        "started_at": datetime.now().isoformat(),
        "run_dir": run_dir,
        "single_completed": 0,
        "ensemble_completed": 0,
    }

    if args.stage in ("single", "all"):
        all_cases = build_single_cases(args.leverages, args.guard_modes)
        if args.single_limit > 0:
            all_cases = all_cases[: args.single_limit]
        done_ids = load_done_ids(single_csv)
        pending = [
            (
                {
                    "case_id": case.case_id,
                    "interval": case.interval,
                    "leverage": case.leverage,
                    "guard_mode": case.guard_mode,
                    "params": case.params,
                    "label": case.label,
                },
                args.start,
                args.end,
            )
            for case in all_cases
            if case.case_id not in done_ids
        ]
        append_journal(
            journal_path,
            f"Single stage total={len(all_cases)} done={len(done_ids)} pending={len(pending)}",
        )

        def append_single(row):
            append_csv(single_csv, SINGLE_FIELDS, row)

        completed = run_parallel(
            pending,
            _run_single_case,
            _init_single_worker,
            (data, bars_1h, funding_1h, all_dates_1h, args.stop_pct),
            args.workers,
            append_single,
            journal_path,
        )
        summary["single_completed"] = len(done_ids) + completed
        summary["single_total"] = len(all_cases)
        write_json(summary_path, summary)

    single_rows = read_csv_rows(single_csv)
    selected_top3 = select_top3(single_rows, args.top_k_per_timeframe)
    selected_top3 = coerce_selection(selected_top3, args.leverages, args.guard_modes)
    write_json(selected_json, selected_top3)
    append_journal(journal_path, "Top3 selection saved")

    if args.stage in ("ensemble", "all"):
        trace_map = generate_trace_map(selected_top3, data, args, journal_path)
        ensemble_cases = build_ensemble_cases(selected_top3, args.max_ensemble_size)
        if args.ensemble_limit > 0:
            ensemble_cases = ensemble_cases[: args.ensemble_limit]
        done_ids = load_done_ids(ensemble_csv)
        pending = [case for case in ensemble_cases if case["case_id"] not in done_ids]
        append_journal(
            journal_path,
            f"Ensemble stage total={len(ensemble_cases)} done={len(done_ids)} pending={len(pending)}",
        )

        def append_ensemble(row):
            append_csv(ensemble_csv, ENSEMBLE_FIELDS, row)

        completed = run_parallel(
            pending,
            _run_ensemble_case,
            _init_ensemble_worker,
            (bars_1h, funding_1h, all_dates_1h, trace_map, args.stop_pct),
            args.workers,
            append_ensemble,
            journal_path,
        )
        summary["ensemble_completed"] = len(done_ids) + completed
        summary["ensemble_total"] = len(ensemble_cases)
        write_json(summary_path, summary)

    summary["finished_at"] = datetime.now().isoformat()
    summary["elapsed_sec"] = time.time() - t0
    write_json(summary_path, summary)
    append_journal(journal_path, f"Run finished elapsed={summary['elapsed_sec']:.1f}s")

    print(f"Run dir: {run_dir}")
    print(f"Elapsed: {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
