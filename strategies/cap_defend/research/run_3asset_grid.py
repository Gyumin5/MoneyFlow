#!/usr/bin/env python3
"""3-asset (주식/현물/선물) 조합 그리드.

Stock V17 (고정) × Spot(3후보) × Futures(4후보) × 비율 × 밴드.

Spot 후보:
- spot_D: V20 엔진에서 D_SMA50 단독
- spot_4h: V20 엔진에서 4h_SMA240 단독
- spot_V20: D+4h 50:50 앙상블 (default)

Futures 후보 (L2, from winners.json):
- fut_4h: 4h_S240_MS40_ML720 single
- fut_1D: 1D_S40_MS20_ML90 single
- fut_2h: 2h_S240_MS40_ML240 single
- fut_ENS: ENS_L2_none

출력: research/3asset_grid/3asset_results.csv + top.json
"""
from __future__ import annotations
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CD = os.path.dirname(HERE)
sys.path.insert(0, CD)
sys.path.insert(0, HERE)

OUT_DIR = os.path.join(HERE, "3asset_grid")
os.makedirs(OUT_DIR, exist_ok=True)

START = "2020-10-01"
END = "2026-04-01"


# ─── Stock V17 equity ───
def load_stock_v17():
    from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS, run_bt
    import stock_engine as tsi

    OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")

    def check_crash_vt(params, ind, date):
        if params.crash == "vt":
            ret = tsi.get_val(ind, "VT", date, "ret")
            return not np.isnan(ret) and ret <= -params.crash_thresh
        return False

    print("Loading stock data...")
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    _init(prices, ind)
    tsi.check_crash = check_crash_vt

    V17 = SP(offensive=OFF_R7, defensive=DEF, canary_assets=("EEM",), canary_sma=200,
             canary_hyst=0.005, select="zscore3", weight="ew", defense="top3",
             def_mom_period=126, health="none", tx_cost=0.001, crash="vt",
             crash_thresh=0.03, crash_cool=3, sharpe_lookback=252,
             start=START, end=END)

    dfs = []
    for a in range(1, 12):
        sp = replace(V17, _anchor=a)
        df = run_bt(tsi._g_prices, tsi._g_ind, sp)
        if df is not None:
            dfs.append(df)
    print(f"Stock anchors: {len(dfs)}")

    norm = []
    for df in dfs:
        v = df["Value"]
        norm.append(v / v.iloc[0])
    st_df = pd.concat(norm, axis=1).ffill().dropna()
    eq = st_df.mean(axis=1)
    eq.index = pd.to_datetime(eq.index)
    if getattr(eq.index, "tz", None) is not None:
        eq.index = eq.index.tz_localize(None)
    return eq


# ─── Spot equity (single member or ensemble) ───
def run_spot_variant(members_keep: list[str]):
    """members_keep = ['D_SMA50'] or ['4h_SMA240'] or both."""
    import run_current_coin_v20_backtest as spot_bt
    import trade.coin_live_engine as cle

    # monkey-patch in spot_bt module imports
    orig_members = spot_bt.MEMBERS
    orig_weights = spot_bt.ENSEMBLE_WEIGHTS

    new_members = {k: orig_members[k] for k in members_keep}
    n = len(new_members)
    new_weights = {k: 1.0 / n for k in members_keep}

    spot_bt.MEMBERS = new_members
    spot_bt.ENSEMBLE_WEIGHTS = new_weights

    try:
        res = spot_bt.run_backtest(start=START, end=END)
    finally:
        spot_bt.MEMBERS = orig_members
        spot_bt.ENSEMBLE_WEIGHTS = orig_weights

    eq = res["equity"]
    eq.index = pd.to_datetime(eq.index)
    if getattr(eq.index, "tz", None) is not None:
        eq.index = eq.index.tz_localize(None)
    return eq


def load_futures_eq(case_id: str):
    path = os.path.join(HERE, "guard_search_runs/guard_v3_floor_mmdd/winners",
                         f"{case_id}_equity.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")["Value"].astype(float)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


# ─── Metrics + mix ───
def metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"Sh": 0, "CAGR": 0, "MDD": 0, "Cal": 0}
    # resample daily to avoid different sampling freq issues
    ed = eq.resample("D").last().dropna()
    yrs = (ed.index[-1] - ed.index[0]).days / 365.25
    if yrs <= 0 or ed.iloc[0] <= 0:
        return {"Sh": 0, "CAGR": 0, "MDD": 0, "Cal": 0}
    cagr = (ed.iloc[-1] / ed.iloc[0]) ** (1 / yrs) - 1
    dr = ed.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0
    mdd = float((ed / ed.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd else 0
    return {"Sh": sh, "CAGR": float(cagr), "MDD": mdd, "Cal": float(cal)}


def mix_eq(series_dict: dict, weights: dict, band, init=1.0):
    df = pd.concat([s.rename(k) for k, s in series_dict.items()], axis=1).dropna()
    rets = {k: df[k].pct_change().fillna(0).values for k in series_dict}
    cur = dict(weights)
    eq = init
    out = np.empty(len(df))
    keys = list(series_dict.keys())
    for i in range(len(df)):
        total = 0.0
        vals = {}
        for k in keys:
            v = eq * cur[k] * (1 + rets[k][i])
            vals[k] = v
            total += v
        eq = total
        if eq > 0:
            for k in keys:
                cur[k] = vals[k] / eq
        if band is not None:
            max_drift = max(abs(cur[k] - weights[k]) for k in keys)
            if max_drift >= band:
                cur = dict(weights)
        out[i] = eq
    return pd.Series(out, index=df.index)


# ─── Grid ───
SPOT_VARIANTS = {
    "spot_D": ["D_SMA50"],
    "spot_4h": ["4h_SMA240"],
    "spot_V20": ["D_SMA50", "4h_SMA240"],
}

FUT_CASE = {
    "L2_4h": "b95466ef23e7a88b",
    "L2_1D": "b9f0e326ae94aa6b",
    "L2_ENS": "ENS_L2_none",
    "L3_4h": "590f6ac76a1acb4f",
    "L3_1D": "8a34a940aeba307b",
    "L3_ENS": "ENS_L3_none",
    "L4_4h": "809a390ee60511ed",
    "L4_1D": "e1dd5f74a7e197eb",
    "L4_ENS": "ENS_L4_none",
}

WEIGHT_GRID = []
for st in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    for sp in [0.10, 0.20, 0.30, 0.40, 0.50]:
        fu = round(1.0 - st - sp, 2)
        if 0.05 <= fu <= 0.50:
            WEIGHT_GRID.append((st, sp, fu))

BANDS = [None, 0.03, 0.05, 0.08, 0.10, 0.15]


def main():
    stock_eq = load_stock_v17()
    print(f"Stock V17: {stock_eq.index[0]} ~ {stock_eq.index[-1]}")
    print(f"  metrics: {metrics(stock_eq)}")

    print("Loading 3 spot variants...")
    spot_eqs = {}
    for name, members in SPOT_VARIANTS.items():
        print(f"  running {name}...")
        eq = run_spot_variant(members)
        spot_eqs[name] = eq
        print(f"    {name}: {metrics(eq)}")

    print("Loading 4 futures equity csvs...")
    fut_eqs = {name: load_futures_eq(cid) for name, cid in FUT_CASE.items()}
    for name, eq in fut_eqs.items():
        print(f"  {name}: {metrics(eq)}")

    rows = []
    for sp_name, sp_eq in spot_eqs.items():
        for fu_name, fu_eq in fut_eqs.items():
            for st_w, sp_w, fu_w in WEIGHT_GRID:
                for band in BANDS:
                    try:
                        mix = mix_eq(
                            {"st": stock_eq, "sp": sp_eq, "fut": fu_eq},
                            {"st": st_w, "sp": sp_w, "fut": fu_w},
                            band,
                        )
                        m = metrics(mix)
                        rows.append({
                            "spot": sp_name, "fut": fu_name,
                            "st_w": st_w, "sp_w": sp_w, "fu_w": fu_w,
                            "band": "no_rebal" if band is None else f"{band:.2f}",
                            "Cal": round(m["Cal"], 3),
                            "CAGR": round(m["CAGR"], 4),
                            "MDD": round(m["MDD"], 4),
                            "Sh": round(m["Sh"], 3),
                        })
                    except Exception as e:
                        print(f"FAIL {sp_name}/{fu_name} {st_w}/{sp_w}/{fu_w} {band}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "3asset_results.csv"), index=False)
    print(f"\nWrote {len(df)} rows")

    # Rankings: Cal, Sh, CAGR, rank-sum
    df["rank_Cal"] = df["Cal"].rank(ascending=False, method="min")
    df["rank_Sh"] = df["Sh"].rank(ascending=False, method="min")
    df["rank_CAGR"] = df["CAGR"].rank(ascending=False, method="min")
    df["rank_sum"] = df["rank_Cal"] + df["rank_Sh"] + df["rank_CAGR"]
    df.to_csv(os.path.join(OUT_DIR, "3asset_results.csv"), index=False)

    top_cal = df.sort_values("Cal", ascending=False).head(15)
    top_sh = df.sort_values("Sh", ascending=False).head(15)
    top_cagr = df.sort_values("CAGR", ascending=False).head(15)
    top_ranksum = df.sort_values("rank_sum").head(15)
    safe = df[df["MDD"] >= -0.29].sort_values("rank_sum").head(15)

    cols = ["spot", "fut", "st_w", "sp_w", "fu_w", "band", "Cal", "CAGR", "MDD", "Sh", "rank_sum"]
    print("\n=== TOP 15 by Cal ===")
    print(top_cal[cols].to_string(index=False))
    print("\n=== TOP 15 by Sh ===")
    print(top_sh[cols].to_string(index=False))
    print("\n=== TOP 15 by CAGR ===")
    print(top_cagr[cols].to_string(index=False))
    print("\n=== TOP 15 by rank_sum ===")
    print(top_ranksum[cols].to_string(index=False))
    print("\n=== TOP 15 by rank_sum (MDD >= -29%) ===")
    print(safe[cols].to_string(index=False))

    with open(os.path.join(OUT_DIR, "top.json"), "w") as f:
        json.dump({
            "top_cal": top_cal[cols].to_dict("records"),
            "top_sh": top_sh[cols].to_dict("records"),
            "top_cagr": top_cagr[cols].to_dict("records"),
            "top_ranksum": top_ranksum[cols].to_dict("records"),
            "top_safe_ranksum": safe[cols].to_dict("records"),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
