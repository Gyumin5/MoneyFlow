"""자산배분 sweep — stock/spot/fut 3자산 portfolio.

배분 profile (stock/spot/fut):
  50/45/5, 50/40/10, 55/40/5, 60/35/5, 60/30/10, 65/30/5, 70/25/5

각 profile × {V21 baseline, new ensembles} 조합으로 portfolio NAV 측정.
출력: allocation_sweep_results.csv
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


def fut_member_trace(cfg):
    from unified_backtest import run as bt_run, load_data
    bars, fnd = load_data(cfg["iv"])
    trace = []
    bt_run(bars, fnd, interval=cfg["iv"], asset_type="fut",
           leverage=cfg["lev"], universe_size=3, cap=1/3, tx_cost=0.0004,
           sma_bars=cfg["sma"], mom_short_bars=cfg["ms"], mom_long_bars=cfg["ml"],
           vol_mode=cfg["vmode"], vol_threshold=cfg["vthr"],
           snap_interval_bars=cfg["snap"], n_snapshots=3,
           phase_offset_bars=0, canary_hyst=0.015, health_mode="mom2vol",
           stop_kind="none", stop_pct=0.0,
           drift_threshold=0.10, post_flip_delay=5,
           dd_lookback=60, dd_threshold=-0.25,
           bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
           start_date=START, end_date=END, _trace=trace)
    return trace


def spot_member_trace(cfg):
    from unified_backtest import run as bt_run, load_data
    iv = cfg.get("iv", "D")
    bars, fnd = load_data(iv)
    trace = []
    bt_run(bars, fnd, interval=iv, asset_type="spot",
           leverage=1.0, universe_size=3, cap=1/3, tx_cost=0.004,
           sma_bars=cfg["sma"], mom_short_bars=cfg["ms"], mom_long_bars=cfg["ml"],
           vol_mode=cfg["vmode"], vol_threshold=cfg["vthr"],
           snap_interval_bars=cfg["snap"], n_snapshots=3,
           phase_offset_bars=0, canary_hyst=0.015, health_mode="mom2vol",
           stop_kind="none", stop_pct=0.0,
           drift_threshold=0.10, post_flip_delay=5,
           dd_lookback=60, dd_threshold=-0.25,
           bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
           start_date=START, end_date=END, _trace=trace)
    return trace


def stock_member_trace(tag):
    from redesign_common import parse_cfg
    from redesign_stock_adapter import run_stock_from_cfg
    rank = pd.read_csv(os.path.join(HERE, "redesign_rank_stock.csv"))
    rank["tag"] = rank["tag"].astype(str)
    row = rank[rank["tag"] == tag]
    if row.empty:
        return None
    cfg = parse_cfg("stock", row.iloc[0])
    if cfg is None:
        return None
    r = run_stock_from_cfg(cfg, phase_offset=0, tx_cost=0.0025,
                            start=START, end=END, with_trace=True)
    if r.get("status") != "ok":
        return None
    return r.get("_trace")


def get_equity(traces, asset, leverage=1.0):
    from redesign_ensemble_bt import ensemble_from_traces, ensemble_fut_sae, _DATA
    # _DATA preload for non-stock
    if asset != "stock" and _DATA is None:
        from unified_backtest import load_data
        import redesign_ensemble_bt as reb
        reb._DATA = {iv: load_data(iv) for iv in ("D", "4h")}
    if asset == "fut":
        m = ensemble_fut_sae(traces, leverage=leverage, tx_cost=0.0004)
        if m is None:
            return None
        return m.get("equity")
    m = ensemble_from_traces(traces, asset, leverage=leverage)
    if m is None:
        return None
    return m.get("equity")


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
    dd = (eq - peak) / peak
    mdd = float(dd.min())
    sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    cal = float(cagr / abs(mdd)) if mdd != 0 else 0
    return {"Cal": cal, "CAGR": float(cagr), "MDD": mdd, "Sh": sh}


def portfolio_eq(asset_eqs, weights, rebal="monthly"):
    """asset_eqs: dict[name → equity series]. weights: dict[name → weight]."""
    df = pd.concat(asset_eqs, axis=1).ffill().dropna()
    rets = df.pct_change().fillna(0)
    w_arr = np.array([weights[c] for c in df.columns])
    cur_w = w_arr.copy()
    nav = [1.0]
    last_month = df.index[0].to_period("M")
    for i, ts in enumerate(df.index[1:], 1):
        r = rets.iloc[i].values
        cur_w = cur_w * (1 + r)
        s = cur_w.sum()
        cur_w = cur_w / s
        nav.append(nav[-1] * s)
        cm = ts.to_period("M")
        if rebal == "monthly" and cm != last_month:
            cur_w = w_arr.copy()
            last_month = cm
    return pd.Series(nav, index=df.index)


def main():
    t0 = time.time()
    print("[allocation sweep BT]", flush=True)

    # ====== V21 baseline traces ======
    print("[v21] fut traces ...", flush=True)
    fut_v21_traces = [fut_member_trace(c) for c in [
        {"iv": "4h", "sma": 240, "ms": 20, "ml": 720, "vmode": "daily", "vthr": 0.05, "snap": 120, "lev": 3.0},
        {"iv": "4h", "sma": 240, "ms": 20, "ml": 480, "vmode": "daily", "vthr": 0.05, "snap": 30, "lev": 3.0},
        {"iv": "4h", "sma": 120, "ms": 20, "ml": 720, "vmode": "daily", "vthr": 0.05, "snap": 120, "lev": 3.0},
    ]]
    print("[v21] spot traces ...", flush=True)
    spot_v21_traces = [spot_member_trace(c) for c in [
        {"sma": 50, "ms": 20, "ml": 90, "vmode": "daily", "vthr": 0.05, "snap": 90},
        {"sma": 100, "ms": 20, "ml": 120, "vmode": "daily", "vthr": 0.05, "snap": 90},
        {"sma": 150, "ms": 20, "ml": 60, "vmode": "daily", "vthr": 0.05, "snap": 90},
    ]]
    # stock V21 baseline = stock new single best (V17 production tag 다음 best, Cal 1.03)
    print("[v21] stock baseline trace (single best as V17 proxy) ...", flush=True)
    stock_v21_trace = stock_member_trace("stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdefault")

    # ====== New traces ======
    print("[new] fut new k=2 traces ...", flush=True)
    fut_new_traces = [fut_member_trace(c) for c in [
        {"iv": "D", "sma": 44, "ms": 18, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 24, "lev": 3.0},
        {"iv": "D", "sma": 39, "ms": 18, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 24, "lev": 3.0},
    ]]
    print("[new] spot new k=2 traces ...", flush=True)
    spot_new_traces = [spot_member_trace(c) for c in [
        {"sma": 39, "ms": 20, "ml": 303, "vmode": "daily", "vthr": 0.05, "snap": 60},
        {"sma": 39, "ms": 7, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 60},
    ]]
    print("[new] stock new k=2 traces ...", flush=True)
    stock_new_traces = [
        stock_member_trace("stk_sn105_sma300_h0.020_sma_sh_dm63_sma200_sh252_mstdual"),
        stock_member_trace("stk_sn105_sma300_h0.020_sma_comp_sort_dm63_sma200_sh252_mstdual"),
    ]

    # ====== Equity 추출 ======
    print("[equity extract]", flush=True)
    fut_v21_eq = get_equity(fut_v21_traces, "fut", leverage=3.0)
    fut_new_eq = get_equity(fut_new_traces, "fut", leverage=3.0)
    spot_v21_eq = get_equity(spot_v21_traces, "spot", leverage=1.0)
    spot_new_eq = get_equity(spot_new_traces, "spot", leverage=1.0)
    stock_v21_eq = get_equity([stock_v21_trace], "stock", leverage=1.0) if stock_v21_trace else None
    stock_new_eq = get_equity([t for t in stock_new_traces if t], "stock", leverage=1.0)

    print("[base metrics]")
    rows = []
    for label, eq in [("fut_v21", fut_v21_eq), ("fut_new", fut_new_eq),
                      ("spot_v21", spot_v21_eq), ("spot_new", spot_new_eq),
                      ("stock_v21", stock_v21_eq), ("stock_new", stock_new_eq)]:
        m = metrics_from_eq(eq)
        m["label"] = label
        rows.append(m)
        print(f"  {label}: Cal={m['Cal']} CAGR={m['CAGR']} MDD={m['MDD']} Sh={m['Sh']}", flush=True)

    # ====== Allocation profiles ======
    profiles = [
        (0.50, 0.45, 0.05),
        (0.50, 0.40, 0.10),
        (0.55, 0.40, 0.05),
        (0.55, 0.35, 0.10),
        (0.60, 0.35, 0.05),
        (0.60, 0.30, 0.10),
        (0.65, 0.30, 0.05),
        (0.70, 0.25, 0.05),
    ]
    for stk_w, sp_w, fu_w in profiles:
        for variant, stk_eq, sp_eq, fu_eq in [
            ("v21", stock_v21_eq, spot_v21_eq, fut_v21_eq),
            ("new", stock_new_eq, spot_new_eq, fut_new_eq),
        ]:
            if stk_eq is None or sp_eq is None or fu_eq is None:
                continue
            eq = portfolio_eq(
                {"stock": stk_eq, "spot": sp_eq, "fut": fu_eq},
                {"stock": stk_w, "spot": sp_w, "fut": fu_w},
                rebal="monthly")
            m = metrics_from_eq(eq)
            m["label"] = f"port_{variant}_{int(stk_w*100)}_{int(sp_w*100)}_{int(fu_w*100)}"
            rows.append(m)
            print(f"  {m['label']}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:.1%} MDD={m['MDD']:.1%} Sh={m['Sh']:.2f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(HERE, "allocation_sweep_results.csv"), index=False)
    print(f"[done] {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
