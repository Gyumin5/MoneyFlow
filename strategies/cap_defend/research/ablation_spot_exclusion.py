#!/usr/bin/env python3
"""Spot V20 exclusion 규칙 ablation."""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CD = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, CD)

import run_current_coin_v20_backtest as spot_bt
cle = spot_bt  # MEMBERS is direct import; spot_bt.MEMBERS is the live copy

START = "2020-10-01"
END = "2026-04-13"


def metrics(eq):
    import numpy as np
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd else 0
    return {"Sh": sh, "CAGR": float(cagr), "MDD": mdd, "Cal": float(cal)}


def run(label, gap_d, gap_4h):
    orig_d = cle.MEMBERS["D_SMA50"]["gap_threshold"]
    orig_4h = cle.MEMBERS["4h_SMA240"]["gap_threshold"]
    cle.MEMBERS["D_SMA50"]["gap_threshold"] = gap_d
    cle.MEMBERS["4h_SMA240"]["gap_threshold"] = gap_4h
    try:
        res = spot_bt.run_backtest(start=START, end=END)
    finally:
        cle.MEMBERS["D_SMA50"]["gap_threshold"] = orig_d
        cle.MEMBERS["4h_SMA240"]["gap_threshold"] = orig_4h
    eq = res["equity"]
    m = metrics(eq)
    print(f"{label}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Sh={m['Sh']:.2f}")
    return m


if __name__ == "__main__":
    print("V20 (앙상블) exclusion ablation")
    m_on = run("ON  (D=-15%/4h=-10%, 기본)", -0.15, -0.10)
    m_off = run("OFF (gap=-99%, 실질 비활성)", -0.99, -0.99)
    m_strict = run("STRICT (D=-10%/4h=-7%)", -0.10, -0.07)
    m_loose = run("LOOSE  (D=-25%/4h=-20%)", -0.25, -0.20)

    import pandas as pd
    df = pd.DataFrame({
        "ON": m_on, "OFF": m_off, "STRICT": m_strict, "LOOSE": m_loose
    }).T
    print()
    print(df.to_string())
