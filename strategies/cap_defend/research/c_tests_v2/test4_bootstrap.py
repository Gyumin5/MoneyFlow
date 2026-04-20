#!/usr/bin/env python3
"""Test 4: Holdout block bootstrap — 실전 최악 시나리오 추정.

holdout daily returns를 60일 블록으로 리샘플링 × 500회.
Cal/CAGR/MDD의 p5/p50/p95/worst 산출.
출력: c_tests_v2/out/test4_bootstrap.csv
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS,
                     HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

N_BOOT = 500
BLOCK = 60
SEED = 42


def bootstrap(returns: np.ndarray, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n == 0:
        return {
            "CAGR_p5": np.nan, "CAGR_p50": np.nan, "CAGR_p95": np.nan, "CAGR_worst": np.nan,
            "MDD_p5": np.nan, "MDD_p50": np.nan, "MDD_p95": np.nan, "MDD_worst": np.nan,
            "Cal_p5": np.nan, "Cal_p50": np.nan, "Cal_p95": np.nan, "Cal_worst": np.nan,
            "n": 0,
        }
    block = min(block, n)
    cagr_l, mdd_l, cal_l = [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=(n // block) + 1)
        blocks = np.concatenate([returns[s:s+block] for s in starts])[:n]
        eq = (1 + blocks).cumprod()
        if eq[-1] <= 0: continue
        yrs = len(blocks) / 365.25
        cagr = eq[-1] ** (1 / yrs) - 1
        run_max = np.maximum.accumulate(eq)
        mdd = float(((eq - run_max) / run_max).min())
        cal = cagr / abs(mdd) if mdd < 0 else 0
        cagr_l.append(cagr); mdd_l.append(mdd); cal_l.append(cal)
    if not cagr_l:
        # 모든 iter에서 eq[-1] <= 0으로 걸러진 경우 방어 (Codex 지적)
        return {
            "CAGR_p5": np.nan, "CAGR_p50": np.nan, "CAGR_p95": np.nan, "CAGR_worst": np.nan,
            "MDD_p5": np.nan, "MDD_p50": np.nan, "MDD_p95": np.nan, "MDD_worst": np.nan,
            "Cal_p5": np.nan, "Cal_p50": np.nan, "Cal_p95": np.nan, "Cal_worst": np.nan,
            "n": 0,
        }
    return {
        "CAGR_p5": float(np.percentile(cagr_l, 5)),
        "CAGR_p50": float(np.percentile(cagr_l, 50)),
        "CAGR_p95": float(np.percentile(cagr_l, 95)),
        "CAGR_worst": float(min(cagr_l)),
        "MDD_p5": float(np.percentile(mdd_l, 5)),
        "MDD_p50": float(np.percentile(mdd_l, 50)),
        "MDD_p95": float(np.percentile(mdd_l, 95)),
        "MDD_worst": float(min(mdd_l)),
        "Cal_p5": float(np.percentile(cal_l, 5)),
        "Cal_p50": float(np.percentile(cal_l, 50)),
        "Cal_p95": float(np.percentile(cal_l, 95)),
        "Cal_worst": float(min(cal_l)),
        "n": len(cagr_l),
    }


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    v21_sH = slice_v21(v21_s, HOLDOUT_START, FULL_END)
    v21_fH = slice_v21(v21_f, HOLDOUT_START, FULL_END)

    rows = []

    # V21 단독
    for name, v21H in [("spot_V21_alone", v21_sH), ("fut_V21_alone", v21_fH)]:
        rets = v21H["equity"].pct_change().dropna().values
        res = bootstrap(rets)
        rows.append({"label": name, **res})

    # V21+C 현물
    ev_sH = ev_s[(ev_s["entry_ts"] >= HOLDOUT_START) & (ev_s["entry_ts"] <= FULL_END)]
    port, _ = run_spot_combo(ev_sH, cd, v21_sH, hist, CAP_SPOT)
    rets = port.pct_change().dropna().values
    rows.append({"label": "spot_V21+C_cap0.333", **bootstrap(rets)})

    # V21+C 선물 (cap 3종)
    ev_fH = ev_f[(ev_f["entry_ts"] >= HOLDOUT_START) & (ev_f["entry_ts"] <= FULL_END)]
    for cap in CAP_FUT_OPTS:
        port, _ = run_fut_combo(ev_fH, cd, v21_fH, hist, cap)
        rets = port.pct_change().dropna().values
        rows.append({"label": f"fut_V21+C_cap{cap}", **bootstrap(rets)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test4_bootstrap.csv"), index=False)

    # readable output
    print("=== Holdout Block Bootstrap (60d × {} iters) ===".format(N_BOOT))
    for _, r in df.iterrows():
        print(f"\n{r['label']}:")
        print(f"  CAGR p5/50/95/worst: {r['CAGR_p5']:+.2%} / {r['CAGR_p50']:+.2%} / {r['CAGR_p95']:+.2%} / {r['CAGR_worst']:+.2%}")
        print(f"  MDD  p5/50/95/worst: {r['MDD_p5']:+.2%} / {r['MDD_p50']:+.2%} / {r['MDD_p95']:+.2%} / {r['MDD_worst']:+.2%}")
        print(f"  Cal  p5/50/95/worst: {r['Cal_p5']:.2f} / {r['Cal_p50']:.2f} / {r['Cal_p95']:.2f} / {r['Cal_worst']:.2f}")
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
