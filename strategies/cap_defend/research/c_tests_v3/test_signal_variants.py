#!/usr/bin/env python3
"""Signal-level variants — 신호 파라미터 재추출 (c_engine_v5 호출).

F1. dip_bars 단축: 6/12/18h
A_ext. dip_thr 세분화: -0.12/-0.15/-0.22/-0.25 (spot용), 변형: spot/fut 각각 파라미터 그리드

공간이 큼 → joblib 병렬, n_jobs=24.
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "next_strategies")))

from _common3 import (
    load_all, run_splits, CAP_SPOT, CAP_FUT_OPTS, OUT,
)
from c_engine_v5 import run_c_v5
from joblib import Parallel, delayed
import pandas as pd

N_JOBS = 24


def _extract_one(coin: str, params: dict) -> list[dict]:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "next_strategies")))
    from c_engine_v5 import load_coin, run_c_v5
    df = load_coin(coin + "USDT")
    if df is None:
        return []
    _, evs = run_c_v5(df, **params)
    for e in evs:
        e["coin"] = coin
    return evs


def extract_events_parallel(avail, params, n_jobs=N_JOBS) -> pd.DataFrame:
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_one)(c, params) for c in sorted(avail))
    return pd.DataFrame([e for batch in results for e in batch])


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()
    print(f"universe: {len(avail)} coins")

    rows = []

    # SPOT configs
    spot_grid = [
        # baseline (dip_bars=24, dip_thr=-0.20, tp=0.04, tstop=24)
        # F1 dip_bars 단축 — spot 현행 24 → 6/12/18
        dict(dip_bars=6,  dip_thr=-0.20, tp=0.04, tstop=24, label="F1_dbars6_spot"),
        dict(dip_bars=12, dip_thr=-0.20, tp=0.04, tstop=24, label="F1_dbars12_spot"),
        dict(dip_bars=18, dip_thr=-0.20, tp=0.04, tstop=24, label="F1_dbars18_spot"),
        # Aext dip_thr 세분화
        dict(dip_bars=24, dip_thr=-0.12, tp=0.04, tstop=24, label="Aext_dthr-12_spot"),
        dict(dip_bars=24, dip_thr=-0.15, tp=0.04, tstop=24, label="Aext_dthr-15_spot"),
        dict(dip_bars=24, dip_thr=-0.22, tp=0.04, tstop=24, label="Aext_dthr-22_spot"),
        dict(dip_bars=24, dip_thr=-0.25, tp=0.04, tstop=24, label="Aext_dthr-25_spot"),
    ]

    # FUT configs (baseline dip_bars=24, dip_thr=-0.18, tp=0.08, tstop=48)
    fut_grid = [
        dict(dip_bars=6,  dip_thr=-0.18, tp=0.08, tstop=48, label="F1_dbars6_fut"),
        dict(dip_bars=12, dip_thr=-0.18, tp=0.08, tstop=48, label="F1_dbars12_fut"),
        dict(dip_bars=18, dip_thr=-0.18, tp=0.08, tstop=48, label="F1_dbars18_fut"),
        dict(dip_bars=24, dip_thr=-0.10, tp=0.08, tstop=48, label="Aext_dthr-10_fut"),
        dict(dip_bars=24, dip_thr=-0.14, tp=0.08, tstop=48, label="Aext_dthr-14_fut"),
        dict(dip_bars=24, dip_thr=-0.22, tp=0.08, tstop=48, label="Aext_dthr-22_fut"),
        dict(dip_bars=24, dip_thr=-0.25, tp=0.08, tstop=48, label="Aext_dthr-25_fut"),
    ]

    for cfg in spot_grid:
        t0 = time.time()
        params = {k: v for k, v in cfg.items() if k != "label"}
        ev = extract_events_parallel(avail, params)
        print(f"  {cfg['label']}: {len(ev)} events ({time.time()-t0:.0f}s)")
        rows += run_splits(cfg["label"], "spot", ev, v21_spot, cd, hist, [CAP_SPOT])

    for cfg in fut_grid:
        t0 = time.time()
        params = {k: v for k, v in cfg.items() if k != "label"}
        ev = extract_events_parallel(avail, params)
        print(f"  {cfg['label']}: {len(ev)} events ({time.time()-t0:.0f}s)")
        rows += run_splits(cfg["label"], "fut", ev, v21_fut, cd, hist, CAP_FUT_OPTS)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_signal_variants.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
