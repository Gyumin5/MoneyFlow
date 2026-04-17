#!/usr/bin/env python3
"""현물 V20 gap exclusion 과적합 점검: 멤버별 (gap, days) 2D 스윕.

각 멤버를 스윕할 때 다른 멤버는 현재값 유지. plateau 존재 여부 확인.
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CD = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, CD)

import run_current_coin_v20_backtest as spot_bt

START = "2020-10-01"
END = "2026-04-13"

GAPS_D = [-0.08, -0.10, -0.12, -0.15, -0.18, -0.20, -0.25]
GAPS_4H = [-0.05, -0.07, -0.10, -0.12, -0.15, -0.18, -0.20]
DAYS = [3, 7, 10, 14, 21, 30, 45, 60]


def metrics(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd else 0
    return {"Sh": sh, "CAGR": float(cagr), "MDD": mdd, "Cal": float(cal)}


def _run_once(args):
    """(target_member, gap, days)로 해당 멤버만 바꾸고 백테스트."""
    target, gap, days = args
    # MEMBERS는 프로세스 로컬 import — 수정해도 타 프로세스 영향 없음
    import run_current_coin_v20_backtest as sb
    import importlib, trade.coin_live_engine as cle  # type: ignore
    importlib.reload(cle)  # ensure fresh defaults
    sb.MEMBERS = cle.MEMBERS

    sb.MEMBERS[target]["gap_threshold"] = gap
    sb.MEMBERS[target]["exclusion_days"] = days

    res = sb.run_backtest(start=START, end=END)
    m = metrics(res["equity"])
    return {
        "target": target, "gap": gap, "days": days,
        "Cal": round(m["Cal"], 3),
        "CAGR": round(m["CAGR"], 4),
        "MDD": round(m["MDD"], 4),
        "Sh": round(m["Sh"], 3),
    }


def main():
    tasks = []
    for g in GAPS_D:
        for d in DAYS:
            tasks.append(("D_SMA50", g, d))
    for g in GAPS_4H:
        for d in DAYS:
            tasks.append(("4h_SMA240", g, d))

    print(f"Total {len(tasks)} combos")

    n_workers = int(os.environ.get("SWEEP_WORKERS", "12"))
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_once, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                print(f"  FAIL {futs[fut]}: {e}")
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]")

    out = pd.DataFrame(results).sort_values(["target", "Cal"], ascending=[True, False])
    out.to_csv(os.path.join(HERE, "spot_gap_sweep.csv"), index=False)

    for t in ["D_SMA50", "4h_SMA240"]:
        sub = out[out["target"] == t]
        print(f"\n--- {t} TOP 10 ---")
        print(sub.head(10).to_string(index=False))
        print(f"--- {t} 현재값(-0.15/30d or -0.10/10d) 주변 ---")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
