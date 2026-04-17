#!/usr/bin/env python3
"""현물 V20 전략을 선물에 얹어보면? 간이 시뮬.

방법: spot V20 일별 수익률에 레버리지 L배 적용 + 연율 funding/fee 차감.
보수적 가정: funding 연 10%, 레버리지 부분에만 적용. 거래비용 tx 연 추가 2%.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(CD))
sys.path.insert(0, REPO)
sys.path.insert(0, CD)

import run_current_coin_v20_backtest as spot_bt

START = "2020-10-01"
END = "2026-04-13"


def metrics(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if eq.iloc[0] <= 0 or yrs <= 0:
        return {"Sh": 0, "CAGR": 0, "MDD": 0, "Cal": 0, "Liq": 0}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd else 0
    liq = int((eq <= 0).any())
    return {"Sh": sh, "CAGR": float(cagr), "MDD": mdd, "Cal": float(cal), "Liq": liq}


def lever_equity(base_eq, L, funding_annual=0.10, extra_tx_annual=0.02):
    """naive L배 레버리지: 일별 수익률 × L, 매일 (L-1)배에 대한 funding/비용 차감."""
    dr = base_eq.pct_change().fillna(0)
    daily_cost = (funding_annual + extra_tx_annual) * (L - 1) / 365.0
    eq = 1.0
    out = []
    for r in dr:
        # maintenance margin 0.4% — 심플하게 eq 0 이하면 청산
        eq *= (1 + L * r - daily_cost)
        if eq <= 0:
            eq = 0
        out.append(eq)
    return pd.Series(out, index=base_eq.index)


def main():
    print("Running spot V20 baseline...")
    res = spot_bt.run_backtest(start=START, end=END)
    eq = res["equity"]
    if getattr(eq.index, "tz", None) is not None:
        eq.index = eq.index.tz_localize(None)
    m = metrics(eq)
    print(f"Spot V20 (L1): Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Sh={m['Sh']:.2f}")

    print("\n--- V20 strategy with leverage (naive daily-return scaling) ---")
    for L in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        eqL = lever_equity(eq, L)
        m = metrics(eqL)
        tag = "LIQ!" if m["Liq"] else ""
        print(f"L={L:.1f}x  Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Sh={m['Sh']:>5.2f} {tag}")

    print("\n--- 참고: 선물 d005 전략 실제 백테스트 ---")
    print("L2 ENS_none: Cal 2.58 CAGR+88% MDD-34% Sh 1.73")
    print("L3 ENS_none: Cal 2.98 CAGR+143% MDD-48% Sh 1.73")
    print("L4 ENS_none: Cal 2.47 CAGR+155% MDD-63% Sh 1.54")


if __name__ == "__main__":
    main()
