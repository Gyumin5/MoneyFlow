"""Baseline + 다양한 전략 비교 BT.

비교군:
  Buy & Hold:
    - SPY 100% B&H
    - BTC 100% B&H
  Classic portfolios:
    - 60/40 (SPY/TLT)
    - All-Weather (SPY 30 / TLT 40 / IEF 15 / GLD 7.5 / DBC 7.5)
    - Permanent (SPY 25 / TLT 25 / GLD 25 / IEF 25)
  우리 전략 (new ensembles):
    - 주식 100% (stock new ens k=2)
    - 60 stock / 40 spot (no fut)
    - 50/45/5
    - 55/40/5
    - 60/35/5 (현재)
    - 70/25/5
  V21 비교:
    - 60/35/5 V21

기간: 2020-10-01 ~ 2025-12-31 (stock 데이터 한계)
출력: baseline_compare_results.csv
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, REPO)

START = "2020-10-01"
END = "2025-12-31"


def load_yahoo_close(ticker, start=START, end=END):
    """stock_engine.load_prices 활용해서 yahoo close 가져오기."""
    from stock_engine import load_prices
    p = load_prices([ticker], start=start)
    if ticker not in p:
        return None
    s = p[ticker]
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s


def load_btc_close(start=START, end=END):
    from unified_backtest import load_data
    bars, _ = load_data("D")
    df = bars.get("BTC")
    if df is None or df.empty:
        df = bars.get("BTCUSDT")
    if df is None or df.empty:
        # try lowercase or any BTC* key
        for k in bars:
            if k.upper().startswith("BTC"):
                df = bars[k]
                break
    if df is None or df.empty:
        return None
    col = "Close" if "Close" in df.columns else "close"
    s = df[col]
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s


def bh_equity(price_series):
    if price_series is None or len(price_series) < 2:
        return None
    return price_series / price_series.iloc[0]


def fixed_portfolio(prices_dict, weights, rebal="monthly"):
    """prices_dict: ticker → close series. weights: ticker → weight."""
    df = pd.concat({k: v for k, v in prices_dict.items() if v is not None}, axis=1).ffill().dropna()
    if df.empty or len(df) < 2:
        return None
    rets = df.pct_change().fillna(0)
    cols = list(df.columns)
    w_arr = np.array([weights[c] for c in cols])
    cur_w = w_arr.copy()
    nav = [1.0]
    last_month = df.index[0].to_period("M")
    for i in range(1, len(df)):
        r = rets.iloc[i].values
        cur_w = cur_w * (1 + r)
        s = cur_w.sum()
        cur_w = cur_w / s
        nav.append(nav[-1] * s)
        cm = df.index[i].to_period("M")
        if rebal == "monthly" and cm != last_month:
            cur_w = w_arr.copy()
            last_month = cm
    return pd.Series(nav, index=df.index)


def metrics_from_eq(eq):
    if eq is None or len(eq) < 2:
        return {"Cal": None, "CAGR": None, "MDD": None, "Sh": None}
    eq = eq.dropna()
    rets = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[0] <= 0:
        return {"Cal": None, "CAGR": None, "MDD": None, "Sh": None}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak).min())
    sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    cal = float(cagr / abs(mdd)) if mdd != 0 else 0
    return {"Cal": cal, "CAGR": float(cagr), "MDD": mdd, "Sh": sh}


def main():
    t0 = time.time()
    print("[baseline compare BT]", flush=True)

    rows = []

    # ====== Buy & Hold ======
    print("[bh] SPY ...", flush=True)
    spy = load_yahoo_close("SPY")
    print("[bh] BTC ...", flush=True)
    btc = load_btc_close()

    spy_eq = bh_equity(spy)
    btc_eq = bh_equity(btc)
    rows.append({"label": "SPY_BH", **metrics_from_eq(spy_eq)})
    rows.append({"label": "BTC_BH", **metrics_from_eq(btc_eq)})

    # ====== Classic portfolios ======
    print("[classic] loading TLT, IEF, GLD, DBC ...", flush=True)
    tlt = load_yahoo_close("TLT")
    ief = load_yahoo_close("IEF")
    gld = load_yahoo_close("GLD")
    dbc = load_yahoo_close("DBC")

    if spy is not None and tlt is not None:
        eq = fixed_portfolio({"SPY": spy, "TLT": tlt}, {"SPY": 0.60, "TLT": 0.40})
        rows.append({"label": "SPY_TLT_60_40", **metrics_from_eq(eq)})
    if all(s is not None for s in [spy, tlt, ief, gld, dbc]):
        eq = fixed_portfolio(
            {"SPY": spy, "TLT": tlt, "IEF": ief, "GLD": gld, "DBC": dbc},
            {"SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075})
        rows.append({"label": "AllWeather", **metrics_from_eq(eq)})
    if all(s is not None for s in [spy, tlt, gld, ief]):
        eq = fixed_portfolio(
            {"SPY": spy, "TLT": tlt, "GLD": gld, "IEF": ief},
            {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "IEF": 0.25})
        rows.append({"label": "Permanent", **metrics_from_eq(eq)})

    # ====== 우리 전략 (allocation_sweep_results.csv 에서 재사용) ======
    print("[ours] loading allocation_sweep_results.csv ...", flush=True)
    sweep_csv = os.path.join(HERE, "allocation_sweep_results.csv")
    if os.path.exists(sweep_csv):
        sweep = pd.read_csv(sweep_csv)
        labels_to_keep = [
            "stock_new",        # stock 100%
            "spot_new",         # spot 100%
            "fut_new",          # fut 100%
            "stock_v21", "spot_v21", "fut_v21",
            "port_new_50_45_5", "port_new_55_40_5", "port_new_60_35_5",
            "port_new_70_25_5", "port_new_50_40_10",
            "port_v21_60_35_5",
        ]
        for _, r in sweep[sweep["label"].isin(labels_to_keep)].iterrows():
            rows.append({"label": r["label"], "Cal": r["Cal"],
                         "CAGR": r["CAGR"], "MDD": r["MDD"], "Sh": r["Sh"]})

    # ====== stock 100% + 60/40 stock+spot (no fut) — fresh BT ======
    print("[ours] stock_new + spot_new equity 추출 ...", flush=True)
    from allocation_sweep import (fut_member_trace, spot_member_trace, stock_member_trace,
                                   get_equity, portfolio_eq)
    spot_new_traces = [spot_member_trace(c) for c in [
        {"sma": 39, "ms": 20, "ml": 303, "vmode": "daily", "vthr": 0.05, "snap": 60},
        {"sma": 39, "ms": 7, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 60},
    ]]
    stock_new_traces = [
        stock_member_trace("stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual"),
        stock_member_trace("stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual"),
    ]
    spot_new_eq = get_equity(spot_new_traces, "spot", leverage=1.0)
    stock_new_eq = get_equity([t for t in stock_new_traces if t], "stock", leverage=1.0)

    if stock_new_eq is not None and spot_new_eq is not None:
        eq = portfolio_eq({"stock": stock_new_eq, "spot": spot_new_eq},
                           {"stock": 0.60, "spot": 0.40}, rebal="monthly")
        rows.append({"label": "port_new_60_40_0_no_fut", **metrics_from_eq(eq)})
    if stock_new_eq is not None:
        rows.append({"label": "port_new_100_0_0_stock_only", **metrics_from_eq(stock_new_eq)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "baseline_compare_results.csv"), index=False)
    print(df.to_string(index=False), flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
