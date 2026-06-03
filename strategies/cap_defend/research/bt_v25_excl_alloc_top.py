"""V25 BNB/SOL 제외 — wider grid winner 적용 시 합성 alloc 비교.

비교 (모두 BNB/SOL 제외, 60/25/15):
- baseline: spot sn=217 n=7 + fut sn=95 n=5
- A: spot sn=781 n=11 + fut sn=371 n=7 (wider top)
- B: spot sn=481 n=13 + fut sn=247 n=13
- C: spot sn=583 n=11 + fut sn=209 n=11
"""
from __future__ import annotations
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_excl_fine import (run_spot, run_fut, run_stock, build_alloc, metrics)


def main():
    t0 = time.time()
    print("[stock V25]")
    eq_st = run_stock()

    cfgs = {
        'baseline (sn217+sn95)':       ((217, 7), (95, 5)),
        'A wider1 (sn781+sn371)':      ((781, 11), (371, 7)),
        'B wider2 (sn481+sn247)':      ((481, 13), (247, 13)),
        'C wider3 (sn583+sn209)':      ((583, 11), (209, 11)),
        'D fine (sn319+sn133)':        ((319, 11), (133, 7)),
        'E mixed (sn781+sn95)':        ((781, 11), (95, 5)),
        'F mixed (sn217+sn371)':       ((217, 7), (371, 7)),
    }

    for tag, ((sp_sn, sp_n), (fu_sn, fu_n)) in cfgs.items():
        eq_sp = run_spot(sp_sn, sp_n)
        eq_fu = run_fut(fu_sn, fu_n)
        if eq_sp is None or eq_fu is None: continue
        alloc = build_alloc(eq_st, eq_sp, eq_fu)
        m_sp = metrics(eq_sp); m_fu = metrics(eq_fu); m_al = metrics(alloc)
        print(f"\n{tag}")
        print(f"  spot: CAGR {m_sp[0]:5.1f}% MDD {m_sp[1]:+6.1f}% Cal {m_sp[3]:.2f}")
        print(f"  fut:  CAGR {m_fu[0]:5.1f}% MDD {m_fu[1]:+6.1f}% Cal {m_fu[3]:.2f}")
        print(f"  alloc 60/25/15: CAGR {m_al[0]:5.1f}% MDD {m_al[1]:+6.1f}% Sharpe {m_al[2]:.2f} Cal {m_al[3]:.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
