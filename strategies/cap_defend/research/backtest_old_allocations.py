#!/usr/bin/env python3
"""주식포트폴리오.xlsx의 과거 정적 배분들을 백테스트해서 V20 전략과 비교.

각 정적 배분은 월간 밴드 리밸 (드리프트 ±5% 시 복원) 가정.
기간: 2020-10 ~ 2026-04 공통.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

START = "2020-10-01"
END = "2026-04-13"

# 옛날 방식들 (target ratio)
ALLOCATIONS = {
    "2020-12 (12자산 GTAA)": {
        "VTI": 0.15, "VNQ": 0.075, "VCLT": 0.05, "VWO": 0.075, "VEA": 0.075,
        "EMLC": 0.05, "GLD": 0.075, "SCHP": 0.10, "MCHI": 0.10, "TLT": 0.05,
        "EWY": 0.05, "BTC-USD": 0.15,  # TIGER200 → EWY proxy
    },
    "2021-01 (8자산 축소)": {
        "VTI": 0.225, "GLD": 0.075, "SCHP": 0.15, "MCHI": 0.075, "TLT": 0.20,
        "DBC": 0.075, "EWY": 0.05, "BTC-USD": 0.15,
    },
    "2022-01 (10자산)": {
        "VTI": 0.225, "VNQ": 0.05, "GLD": 0.075, "SCHP": 0.125, "MCHI": 0.075,
        "AAXJ": 0.05, "TLT": 0.125, "DBC": 0.075, "EWY": 0.05, "BTC-USD": 0.15,
    },
    "2023-01 (9자산 +Quant)": {
        # Quant는 proxy 불가 → BTC 2배로 대체 (근사)
        "VTI": 0.20, "VNQ": 0.10, "GLD": 0.075, "SCHP": 0.125, "TLT": 0.125,
        "PDBC": 0.075, "EWY": 0.05, "BTC-USD": 0.25,
    },
    "2024-10 (8자산 no Quant)": {
        "VTI": 0.20, "VNQ": 0.10, "GLD": 0.10, "SCHP": 0.15, "TLT": 0.15,
        "PDBC": 0.10, "EWY": 0.05, "BTC-USD": 0.15,
    },
    "2025-12 (Cash+3주식)": {
        # Cash는 무위험 (0 return) 가정
        "CASH": 0.40, "EEM": 0.20, "GLD": 0.20, "VEA": 0.20,
    },
}

TICKERS = set()
for alloc in ALLOCATIONS.values():
    for t in alloc:
        if t != "CASH":
            TICKERS.add(t)


def load_prices(tickers, start, end):
    # Try local first, fallback yfinance
    out = {}
    missing = []
    for t in tickers:
        p = os.path.join("/home/gmoh/mon/251229/data", f"{t}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
            col = "Adj_Close" if "Adj_Close" in df.columns else (
                  "Adj Close" if "Adj Close" in df.columns else "Close")
            s = df[col].astype(float)
            if s.index.min() <= pd.Timestamp(start):
                out[t] = s
                continue
        missing.append(t)
    if missing:
        print(f"yfinance fetching: {missing}")
        df = yf.download(missing, start="2010-01-01", end=end, auto_adjust=True,
                         progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(missing[0])
        for t in missing:
            if t in df.columns:
                out[t] = df[t].dropna()
    return out


def metrics(eq):
    eq = eq.dropna()
    if len(eq) < 2:
        return {"Sh": 0, "CAGR": 0, "MDD": 0, "Cal": 0}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd else 0
    return {"Sh": sh, "CAGR": float(cagr), "MDD": mdd, "Cal": float(cal)}


def backtest_static(alloc, prices, start, end, band=0.05, tx=0.001):
    tickers = [t for t in alloc if t != "CASH"]
    df = pd.concat({t: prices[t] for t in tickers}, axis=1)
    df = df.loc[start:end].ffill().dropna()
    if df.empty:
        return None
    rets = df.pct_change().fillna(0)
    cash_w = alloc.get("CASH", 0.0)
    target = {t: alloc[t] for t in tickers}
    # cur weights
    cur = dict(target)
    cur_cash = cash_w
    eq = 1.0
    out = []
    for dt, row in rets.iterrows():
        # apply returns
        new_vals = {t: eq * cur[t] * (1 + row[t]) for t in tickers}
        new_cash = eq * cur_cash  # 0 return
        new_eq = sum(new_vals.values()) + new_cash
        if new_eq > 0:
            cur = {t: new_vals[t] / new_eq for t in tickers}
            cur_cash = new_cash / new_eq
        # band check
        max_drift = max([abs(cur[t] - target[t]) for t in tickers] + [abs(cur_cash - cash_w)])
        if max_drift >= band:
            turnover = sum(abs(cur[t] - target[t]) for t in tickers) + abs(cur_cash - cash_w)
            new_eq *= (1 - tx * turnover / 2)
            cur = dict(target)
            cur_cash = cash_w
        eq = new_eq
        out.append(eq)
    return pd.Series(out, index=df.index)


def main():
    print(f"Loading {len(TICKERS)} tickers...")
    prices = load_prices(TICKERS, START, END)
    print(f"Loaded: {list(prices.keys())}")

    results = []
    eqs = {}
    for name, alloc in ALLOCATIONS.items():
        # Check all tickers available
        missing = [t for t in alloc if t != "CASH" and t not in prices]
        if missing:
            print(f"SKIP {name}: missing {missing}")
            continue
        eq = backtest_static(alloc, prices, START, END)
        if eq is None or eq.empty:
            print(f"SKIP {name}: no data")
            continue
        m = metrics(eq)
        results.append({"name": name, **m,
                        "period": f"{eq.index[0].date()} ~ {eq.index[-1].date()}"})
        eqs[name] = eq
        print(f"{name}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Sh={m['Sh']:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(HERE, "old_allocations_results.csv"), index=False)
    print(f"\nWrote {os.path.join(HERE, 'old_allocations_results.csv')}")

    # V20+V17 참고 (이미 계산된 값)
    print("\n--- 참고: 현재 전략 솔로 (2020-10~2026-04) ---")
    print("주식 V17:      Cal 1.12 CAGR+14% MDD-13% Sh 1.30")
    print("현물 spot_V20: Cal 1.83 CAGR+47% MDD-26% Sh 1.69")
    print("선물 L3_ENS:   Cal 3.02 CAGR+143% MDD-47% Sh 1.73")
    print("조합 rank_sum#1: Cal 3.82 CAGR+90% MDD-24% Sh 2.41")

    # 랭킹
    df_sorted = df.sort_values("Cal", ascending=False)
    print("\n=== 옛날 방식 Cal 순위 ===")
    print(df_sorted[["name", "Cal", "CAGR", "MDD", "Sh"]].to_string(index=False))


if __name__ == "__main__":
    main()
