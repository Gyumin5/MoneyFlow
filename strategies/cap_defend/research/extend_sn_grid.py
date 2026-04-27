"""SN 인접 grid 확장 — fut 1D L3 의 unique (sma,ms,ml,vmode,vthr) × 신규 SN 추가 BT.

신규 SN: {66, 72, 75, 78, 84} — SN=73 인접 (모두 %3==0)
출력: redesign_univ3_raw.csv 에 append
"""
from __future__ import annotations
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

OUT_CSV = os.path.join(HERE, "redesign_univ3_raw.csv")

NEW_SN = [66, 72, 75, 78, 84]


def preload():
    from unified_backtest import load_data
    global _DATA
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def run_one(cfg):
    from unified_backtest import run as bt_run
    iv = cfg["iv"]
    bars, funding = _DATA[iv]
    try:
        m = bt_run(
            bars, funding,
            interval=iv, asset_type=cfg["asset"], leverage=float(cfg["lev"]),
            universe_size=3, cap=1 / 3, tx_cost=0.0004,
            sma_bars=int(cfg["sma"]), mom_short_bars=int(cfg["ms"]),
            mom_long_bars=int(cfg["ml"]),
            vol_mode=cfg["vmode"], vol_threshold=float(cfg["vthr"]),
            snap_interval_bars=int(cfg["snap"]), n_snapshots=3,
            phase_offset_bars=0,
            canary_hyst=0.015, health_mode="mom2vol",
            stop_kind="none", stop_pct=0.0,
            drift_threshold=0.10, post_flip_delay=5,
            dd_lookback=60, dd_threshold=-0.25,
            bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
            start_date="2020-10-01", end_date="2026-04-13",
        )
        return {
            "tag": cfg["tag"], "asset": cfg["asset"], "iv": iv, "lev": cfg["lev"],
            "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
            "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
            "Sh": float(m.get("Sharpe", 0) or 0),
            "Cal": float(m.get("Cal", 0) or 0),
            "CAGR": float(m.get("CAGR", 0) or 0),
            "MDD": float(m.get("MDD", 0) or 0),
            "rebal": int(m.get("Rebal", 0) or 0),
            "liq": int(m.get("Liq", 0) or 0),
        }
    except Exception as e:
        return {"tag": cfg["tag"], "asset": cfg["asset"], "iv": iv, "lev": cfg["lev"],
                "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
                "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
                "Sh": 0.0, "Cal": 0.0, "CAGR": 0.0, "MDD": 0.0, "rebal": 0, "liq": 0,
                "err": str(e)[:120]}


def main():
    df = pd.read_csv(OUT_CSV)
    fut_1d = df[(df["asset"] == "fut") & (df["iv"] == "D") & (df["lev"] == 3.0)].copy()
    print(f"[fut 1D L3] {len(fut_1d)} existing rows")
    base = fut_1d.drop_duplicates(subset=["sma", "ms", "ml", "vmode", "vthr"])
    print(f"[base unique] {len(base)} configs")

    tasks = []
    existing_tags = set(df["tag"].astype(str))
    for _, row in base.iterrows():
        for sn in NEW_SN:
            tag = f"fut_1D_S{int(row['sma'])}_M{int(row['ms'])}_{int(row['ml'])}" \
                  f"_d{float(row['vthr']):.2f}_SN{sn}_L3"
            if tag in existing_tags:
                continue
            tasks.append({
                "tag": tag, "asset": "fut", "iv": "D", "lev": 3.0,
                "sma": int(row["sma"]), "ms": int(row["ms"]), "ml": int(row["ml"]),
                "vmode": row["vmode"], "vthr": float(row["vthr"]),
                "snap": int(sn),
            })

    print(f"[new tasks] {len(tasks)}")
    if not tasks:
        return

    t0 = time.time()
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        out_df = pd.DataFrame(buf)
        out_df.to_csv(OUT_CSV, mode="a", header=False, index=False)
        buf = []

    with Pool(24, initializer=preload) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4), 1):
            buf.append(res)
            if len(buf) >= 200:
                flush()
            if i % 200 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"[{i}/{len(tasks)}] {rate:.2f}/s ETA {(len(tasks)-i)/max(rate,1e-6)/60:.1f}m",
                      flush=True)
    flush()
    print(f"[done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
