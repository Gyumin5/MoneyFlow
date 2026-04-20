#!/usr/bin/env python3
"""tz 버그 우회 — 재계산. 각 tag을 다시 돌리기보단 14후보만 빠르게 재시뮬."""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from phase_common import parse_tag, run_single_target

ANCHOR = "2020-10-01"
END = "2026-04-13"
OUT = os.path.join(HERE, "test_holdout_fairness")

df = pd.read_csv(os.path.join(OUT, "kequal1_fair.csv"))
equities = {}
for i, r in df.iterrows():
    tag = r["tag"]
    meta = parse_tag(tag)
    cfg = {k: meta[k] for k in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
    print(f"[{i+1}/{len(df)}] {tag}")
    res = run_single_target(r["asset"], cfg, lev=float(r["lev"]),
                            anchor=ANCHOR, end=END, want_equity=True)
    eq = res.get("_equity")
    if eq is None:
        continue
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    idx = pd.to_datetime(eq.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    eq.index = idx
    eq_d = eq.resample("D").last().dropna()
    equities[tag] = eq_d.pct_change().fillna(0)

ret_df = pd.DataFrame(equities).dropna(how="all")
corr = ret_df.corr()
corr.to_csv(os.path.join(OUT, "pairwise_corr.csv"))

tags = df["tag"].tolist()
asset = df.set_index("tag")["asset"].to_dict()
print("\n=== Pairwise corr (same asset) ===")
for a in ("spot","fut"):
    ts = [t for t in tags if asset.get(t) == a and t in corr.columns]
    pairs = []
    for i in range(len(ts)):
        for j in range(i+1, len(ts)):
            pairs.append((ts[i], ts[j], float(corr.loc[ts[i], ts[j]])))
    pairs.sort(key=lambda x: -x[2])
    print(f"\n-- {a} ({len(pairs)} pairs) --")
    for t1,t2,c in pairs[:15]:
        flag=" ⚠️" if c>=0.85 else ""
        print(f"  {c:.3f}{flag}  {t1[-40:]}  ×  {t2[-40:]}")
    n95 = sum(1 for _,_,c in pairs if c>=0.95)
    n85 = sum(1 for _,_,c in pairs if c>=0.85)
    print(f"  >=0.85: {n85}/{len(pairs)},  >=0.95: {n95}/{len(pairs)}")

print("\nsaved pairwise_corr.csv")
