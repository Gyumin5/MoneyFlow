#!/usr/bin/env python3
"""V21 현물 (코인) 가드 sweep 테스트.

V21 현재 설정:
- gap_threshold = -0.15 (15% 갭 하락 감지)
- exclusion_days = 30 (30일간 해당 코인 제외)

테스트:
- gap_threshold: -1.0 (비활성), -0.10, -0.15 (baseline), -0.20, -0.25
- exclusion_days: 0 (즉시 복구), 7, 30 (baseline), 60

총 5 × 4 = 20 configs.
"""
from __future__ import annotations
import os, sys, time, copy
from itertools import product
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in [HERE, os.path.dirname(HERE), ROOT,
          os.path.join(ROOT, "trade"), os.path.join(ROOT, "strategies", "cap_defend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_current_coin_v20_backtest import run_backtest, calc_metrics
import coin_live_engine

OUT = os.path.join(HERE, "v21_spot_guard_out")
os.makedirs(OUT, exist_ok=True)

START = "2020-10-01"
END = "2026-03-30"


def set_members_guard(gap_thr: float, excl_days: int):
    """coin_live_engine.MEMBERS 복사하여 가드 조정."""
    orig_members = coin_live_engine.MEMBERS
    new = {}
    for name, cfg in orig_members.items():
        c = dict(cfg)
        c["gap_threshold"] = gap_thr
        c["exclusion_days"] = excl_days
        new[name] = c
    coin_live_engine.MEMBERS = new


def restore_members(orig):
    coin_live_engine.MEMBERS = orig


def main():
    orig_members = copy.deepcopy(coin_live_engine.MEMBERS)

    configs = []
    for gt in [-1.0, -0.10, -0.15, -0.20, -0.25]:
        for ed in [0, 7, 30, 60]:
            configs.append({"gap_threshold": gt, "exclusion_days": ed})
    print(f"Total configs: {len(configs)}")

    rows = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        set_members_guard(cfg["gap_threshold"], cfg["exclusion_days"])
        try:
            result = run_backtest(START, END)
            eq = result["equity"]
            m = calc_metrics(eq)
            rebal = result.get("rebal_count", 0)
        except Exception as e:
            m = {"Cal": 0, "CAGR": 0, "MDD": 0}
            rebal = 0
            print(f"  error: {str(e)[:80]}")
        elapsed = time.time() - t0
        row = {
            **cfg,
            "Cal": round(m.get("Cal", 0), 3),
            "CAGR": round(m.get("CAGR", 0), 4),
            "MDD": round(m.get("MDD", 0), 4),
            "Sh": round(m.get("Sharpe", 0), 3),
            "rebal": rebal,
            "elapsed": round(elapsed, 1),
        }
        rows.append(row)
        print(f"[{i}/{len(configs)}] gap={cfg['gap_threshold']:+.2f} ex={cfg['exclusion_days']:3d}d "
              f"→ Cal={row['Cal']:.2f} MDD={row['MDD']:.2%} CAGR={row['CAGR']:.2%} "
              f"rebal={rebal} ({elapsed:.0f}s)")

    restore_members(orig_members)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "v21_spot_guard.csv"), index=False)

    # baseline (현재 설정) 찾기
    base = df[(df["gap_threshold"] == -0.15) & (df["exclusion_days"] == 30)].iloc[0]
    df["dCal"] = df["Cal"] - base["Cal"]
    df["dMDD"] = df["MDD"] - base["MDD"]

    print(f"\n=== V21 spot baseline (gap=-0.15, ex=30d) ===")
    print(f"  Cal={base['Cal']:.2f}, MDD={base['MDD']:.2%}, CAGR={base['CAGR']:.2%}")

    print("\n=== Top 10 by Cal ===")
    print(df.sort_values("Cal", ascending=False).head(10).to_string(index=False))

    print("\n=== 개선 (dCal > 0) ===")
    good = df[df["dCal"] > 0].sort_values("dCal", ascending=False)
    if len(good) == 0:
        print("  없음 — baseline 유지가 최선")
    else:
        print(good.to_string(index=False))

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print(f"저장: {OUT}/v21_spot_guard.csv")


if __name__ == "__main__":
    main()
