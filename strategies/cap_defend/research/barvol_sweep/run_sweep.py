"""Threshold sweep for vol_mode × threshold per interval, 10-anchor × L3 no stop.

Parallelized via multiprocessing.Pool (shared data via initializer).
"""
import os, sys, time, json
from multiprocessing import Pool
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
RES = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(RES))
sys.path.insert(0, RES)

from backtest_futures_full import load_data, run as bt_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from run_futures_fixedlev_search import FIXED_CFG

FULL_END = "2026-04-13"
ANCHORS = ["2020-10-01","2021-04-01","2021-10-01","2022-04-01","2022-10-01",
           "2023-04-01","2023-10-01","2024-04-01","2024-10-01","2025-04-01"]

# Config list
CONFIGS = [
    # (tag, interval, sma, mom_s, mom_l, vol_mode, vol_thr, snap)
    # 1D sweep
    ("1D_S40_M20/90_d0.03",  "D",  40,  20,  90, "daily", 0.03, 45),
    ("1D_S40_M20/90_d0.05",  "D",  40,  20,  90, "daily", 0.05, 45),
    ("1D_S40_M20/90_d0.07",  "D",  40,  20,  90, "daily", 0.07, 45),
    ("1D_S40_M20/90_b0.80",  "D",  40,  20,  90, "bar",   0.80, 45),
    ("1D_S40_M20/90_b1.20",  "D",  40,  20,  90, "bar",   1.20, 45),
    ("1D_S40_M20/90_b1.60",  "D",  40,  20,  90, "bar",   1.60, 45),
    # 4h sweep (long mom 720)
    ("4h_S240_M20/720_d0.03", "4h", 240, 20, 720, "daily", 0.03, 84),
    ("4h_S240_M20/720_d0.05", "4h", 240, 20, 720, "daily", 0.05, 84),
    ("4h_S240_M20/720_d0.07", "4h", 240, 20, 720, "daily", 0.07, 84),
    ("4h_S240_M20/720_b0.50", "4h", 240, 20, 720, "bar",   0.50, 84),
    ("4h_S240_M20/720_b0.70", "4h", 240, 20, 720, "bar",   0.70, 84),
    ("4h_S240_M20/720_b1.00", "4h", 240, 20, 720, "bar",   1.00, 84),
    # 2h sweep
    ("2h_S240_M20/720_d0.03", "2h", 240, 20, 720, "daily", 0.03, 120),
    ("2h_S240_M20/720_d0.05", "2h", 240, 20, 720, "daily", 0.05, 120),
    ("2h_S240_M20/720_b0.50", "2h", 240, 20, 720, "bar",   0.50, 120),
    ("2h_S240_M20/720_b0.70", "2h", 240, 20, 720, "bar",   0.70, 120),
    ("2h_S240_M20/720_b1.00", "2h", 240, 20, 720, "bar",   1.00, 120),
]

_DATA = None
def _init():
    global _DATA
    _DATA = {iv: load_data(iv) for iv in ["1h","2h","4h","D"]}

def _run_one(task):
    tag, iv, sma, ms, ml, vm, vt, sn, anchor = task
    global _DATA
    bars_1h, funding_1h = _DATA["1h"]
    dates = bars_1h["BTC"].index
    ddates = dates[(dates>=anchor) & (dates<=FULL_END)]
    params = dict(FIXED_CFG)
    params.update({"vol_mode":vm, "sma_bars":sma, "mom_short_bars":ms,
                   "mom_long_bars":ml, "vol_threshold":vt, "snap_interval_bars":sn})
    trace = []
    bars, funding = _DATA[iv]
    bt_run(bars, funding, interval=iv, leverage=1.0,
           start_date=anchor, end_date=FULL_END, _trace=trace, **params)
    combined = combine_targets({"x":trace}, {"x":1.0}, ddates)
    engine = SingleAccountEngine(bars_1h, funding_1h,
        leverage=3.0, leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always")
    try:
        r = engine.run(combined)
        m = r["metrics"] if "metrics" in r else r
        return {"tag": tag, "anchor": anchor,
                "Cal": float(m.get("Cal",0)),
                "CAGR": float(m.get("CAGR",0)),
                "MDD": float(m.get("MDD",0)),
                "Sharpe": float(m.get("Sharpe",0))}
    except Exception as e:
        return {"tag": tag, "anchor": anchor, "error": str(e)}


def main():
    tasks = []
    for c in CONFIGS:
        for a in ANCHORS:
            tasks.append((*c, a))
    print(f"Total tasks: {len(tasks)}  ({len(CONFIGS)} configs × {len(ANCHORS)} anchors)", flush=True)
    t0 = time.time()

    results = []
    with Pool(processes=6, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=1), 1):
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] elapsed={int(time.time()-t0)}s  last={r.get('tag')[:25]} Cal={r.get('Cal',0):.2f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(HERE, "raw.csv"), index=False)

    # Summarize per tag
    rows = []
    for tag, g in df.groupby("tag"):
        sh = g["Sharpe"].values
        cal = g["Cal"].values
        cagr = g["CAGR"].values
        mdd = g["MDD"].values
        rows.append({
            "tag": tag, "n": len(g),
            "mSh": float(sh.mean()), "sSh": float(sh.std()),
            "mCal": float(cal.mean()),
            "mCAGR": float(cagr.mean()),
            "wMDD": float(mdd.min()),
        })
    sdf = pd.DataFrame(rows).sort_values("mCal", ascending=False)
    sdf.to_csv(os.path.join(HERE, "summary.csv"), index=False)
    print("\n=== SUMMARY (sorted by mCal) ===")
    print(sdf.to_string(index=False))
    print(f"\nDone in {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
