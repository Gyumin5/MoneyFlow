"""V25 BNB/SOL 제외 — 광범위 snap grid (3단계 확장).

spot: sn 217 ~ 851, n 7 / 11 / 13 / 17 (prime stagger 만)
fut:  sn 65 ~ 247, n 3 / 5 / 7 / 11 (prime stagger 만)
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_excl_fine import (run_spot, run_fut, metrics, window_rs, gen_pairs)

START = "2020-10-01"; END = "2026-05-29"


def main():
    t0 = time.time()
    # spot wider
    spot_snaps = [217, 287, 319, 351, 391, 427, 451, 481, 533, 583, 611, 689, 737, 781, 851]
    spot_ns = [7, 11, 13, 17]
    # fut wider
    fut_snaps = [65, 78, 91, 95, 105, 115, 119, 133, 161, 209, 247, 299, 371]
    fut_ns = [3, 5, 7, 11, 13]

    sp_pairs = gen_pairs(spot_snaps, spot_ns)
    fu_pairs = gen_pairs(fut_snaps, fut_ns)
    print(f"spot pairs: {len(sp_pairs)}, fut pairs: {len(fu_pairs)}")

    print("\n[spot wider]")
    eq_sp = {}
    for sn, n, st in sp_pairs:
        tag = f"sn{sn}_n{n}_st{st}"
        eq = run_spot(sn, n)
        if eq is None: continue
        eq_sp[tag] = eq
        m = metrics(eq)
        if m: print(f"  {tag:<22} CAGR {m[0]:5.1f}% MDD {m[1]:+6.1f}% Cal {m[3]:.2f}")
    rs = window_rs(eq_sp)
    if rs:
        sums, wins, n_w = rs
        print(f"\n  spot top 10 (n_w={n_w}):")
        for tag, v in sorted(sums.items(), key=lambda x: x[1])[:10]:
            print(f"    {tag:<22} avg_rank={v/n_w:.2f} win={wins[tag]/n_w*100:.1f}%")

    print("\n[fut wider]")
    eq_fu = {}
    for sn, n, st in fu_pairs:
        tag = f"sn{sn}_n{n}_st{st}"
        eq = run_fut(sn, n)
        if eq is None: continue
        eq_fu[tag] = eq
        m = metrics(eq)
        if m: print(f"  {tag:<22} CAGR {m[0]:5.1f}% MDD {m[1]:+6.1f}% Cal {m[3]:.2f}")
    rs = window_rs(eq_fu)
    if rs:
        sums, wins, n_w = rs
        print(f"\n  fut top 10 (n_w={n_w}):")
        for tag, v in sorted(sums.items(), key=lambda x: x[1])[:10]:
            print(f"    {tag:<22} avg_rank={v/n_w:.2f} win={wins[tag]/n_w*100:.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
