"""V25 BNB/SOL 제외 — fut ms=12 ml=118 plateau dense grid + alloc.

ms ∈ {8,10,11,12,13,14,15,16}, ml ∈ {100,105,110,115,118,121,127,135}
baseline snap (sn=95 n=5). spot 고정 (ms=20 ml=127 sn=217 n=7).
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_final_verify import (run_spot_ms_ml, run_fut_ms_ml, run_stock,
                                  build_alloc, metrics, window_rs)

START = "2020-10-01"; END = "2026-05-29"


def main():
    t0 = time.time()
    eq_st = run_stock()
    eq_sp = run_spot_ms_ml(20, 127)  # spot 고정

    MS = [8, 10, 11, 12, 13, 14, 15, 16]
    ML = [100, 105, 110, 115, 118, 121, 127, 135]

    eq_dict = {}; results = []
    for ms in MS:
        for ml in ML:
            if ml <= ms * 2: continue
            tag = f"ms{ms:02d}_ml{ml:03d}"
            eq = run_fut_ms_ml(ms, ml)
            if eq is None: continue
            eq_dict[tag] = eq
            m = metrics(eq)
            alloc = build_alloc(eq_st, eq_sp, eq)
            m_al = metrics(alloc)
            results.append((tag, ms, ml, m, m_al))

    # 표 출력 (ms × ml grid)
    print("\n[fut sleeve Cal grid]")
    print("       " + " ".join(f"ml{m:>4}" for m in ML))
    for ms in MS:
        row = [f"ms{ms:>3}"]
        for ml in ML:
            val = next((r[3][3] for r in results if r[1] == ms and r[2] == ml), None)
            row.append(f"{val:5.2f}" if val is not None else "  -  ")
        print("  " + "  ".join(row))

    print("\n[alloc Cal grid]")
    print("       " + " ".join(f"ml{m:>4}" for m in ML))
    for ms in MS:
        row = [f"ms{ms:>3}"]
        for ml in ML:
            val = next((r[4][3] for r in results if r[1] == ms and r[2] == ml), None)
            row.append(f"{val:5.2f}" if val is not None else "  -  ")
        print("  " + "  ".join(row))

    # rank by alloc Cal
    print("\n[Top 15 by alloc Cal]")
    sorted_by_alloc = sorted(results, key=lambda r: -r[4][3])[:15]
    print(f"  {'cfg':<14} {'fut_Cal':>8} {'alloc_CAGR':>11} {'alloc_MDD':>10} {'alloc_Sh':>9} {'alloc_Cal':>10}")
    for tag, ms, ml, m_fu, m_al in sorted_by_alloc:
        print(f"  {tag:<14} {m_fu[3]:>7.2f} {m_al[0]:>10.1f}% {m_al[1]:>+9.1f}% {m_al[2]:>8.2f} {m_al[3]:>9.2f}")

    # baseline 위치
    print(f"\n  baseline ms18 ml127 alloc Cal: {next((r[4][3] for r in results if r[1]==18 and r[2]==127), 'N/A')}")
    # window rank-sum (sleeve)
    print("\n[fut sleeve window rank-sum]")
    rs = window_rs(eq_dict)
    if rs:
        sums, wins, n_w = rs
        for tag, v in sorted(sums.items(), key=lambda x: x[1])[:10]:
            print(f"    {tag:<14} avg_rank={v/n_w:.2f} win={wins[tag]/n_w*100:.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
