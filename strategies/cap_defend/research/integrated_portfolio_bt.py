"""통합 portfolio backtest — 자산 3개 (stock/spot/fut) × 전략 2종 (V21 vs new).

각 (asset, strategy) 조합의 daily equity 를 BT 로 추출 → 공통 캘린더 forward fill →
KRW 통합 (USD/KRW FX 시계열 적용) → 60/35/5 weight 적용한 portfolio NAV 계산.

비교:
- baseline_v21: stock V17 + spot V21 ens + fut V21 ens
- new_ens:      stock new k=2 + spot new k=2 + fut new k=2 (SAE)

출력: integrated_portfolio_results.csv
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
END = "2025-12-31"  # stock 데이터 한계


def fut_ensemble_equity(member_cfgs, lev=3.0):
    """fut SAE ensemble equity (USDT 기준, 1h grid)."""
    from unified_backtest import run as bt_run, load_data
    from futures_ensemble_engine import SingleAccountEngine
    bars_4h, funding_4h = load_data("4h")
    bars_d, funding_d = load_data("D")
    bars_1h, funding_1h = load_data("1h")
    traces = []
    for cfg in member_cfgs:
        bars = bars_4h if cfg["iv"] == "4h" else bars_d
        fnd = funding_4h if cfg["iv"] == "4h" else funding_d
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
        traces.append(trace)
    ts_1h = next(iter(bars_1h.values())).index
    member_events = []
    for tr in traces:
        rows = [(pd.Timestamp(t["date"]), dict(t["target"] or {})) for t in tr]
        rows.sort(key=lambda x: x[0])
        member_events.append(rows)
    starts = [e[0][0] for e in member_events]
    ends = [e[-1][0] for e in member_events]
    t0, t1 = max(starts), min(ends)
    ts_1h = ts_1h[(ts_1h >= t0) & (ts_1h <= t1)]
    k = len(member_events)
    indices = [0] * k
    cur = [{} for _ in range(k)]
    target_series = []
    for ts in ts_1h:
        for mi in range(k):
            ev = member_events[mi]
            while indices[mi] < len(ev) and ev[indices[mi]][0] <= ts:
                cur[mi] = ev[indices[mi]][1]
                indices[mi] += 1
        merged = {}
        for tgt in cur:
            for asset, w in tgt.items():
                key = str(asset).upper()
                if key == "CASH": continue
                merged[key] = merged.get(key, 0.0) + float(w) / k
        cw = max(0.0, 1.0 - sum(merged.values()))
        if cw > 1e-9: merged["CASH"] = cw
        target_series.append((ts, merged))
    sae = SingleAccountEngine(bars_1h, funding_1h, leverage=lev,
                               tx_cost=0.0004, stop_kind="none", leverage_mode="fixed")
    metrics = sae.run(target_series)
    eq = metrics.get("_equity")
    if eq is None: return None
    return pd.Series(eq).resample("D").last().ffill()


def spot_ensemble_equity(member_cfgs):
    """spot L1 ensemble equity (KRW 기준, daily). trace-based portfolio sim."""
    from unified_backtest import run as bt_run, load_data
    bars_d, _ = load_data("D")
    member_eqs = []
    for cfg in member_cfgs:
        m = bt_run(bars_d, None, interval="D", asset_type="spot",
                   leverage=1.0, universe_size=3, cap=1/3, tx_cost=0.004,
                   sma_bars=cfg["sma"], mom_short_bars=cfg["ms"], mom_long_bars=cfg["ml"],
                   vol_mode=cfg["vmode"], vol_threshold=cfg["vthr"],
                   snap_interval_bars=cfg["snap"], n_snapshots=3,
                   phase_offset_bars=0, canary_hyst=0.015, health_mode="mom2vol",
                   stop_kind="none", stop_pct=0.0,
                   drift_threshold=0.10, post_flip_delay=5,
                   dd_lookback=60, dd_threshold=-0.25,
                   bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
                   start_date=START, end_date=END)
        eq = m.get("_equity")
        if eq is not None:
            member_eqs.append(pd.Series(eq))
    if not member_eqs:
        return None
    # simple EW of members (not true single-account, 근사값)
    df = pd.concat(member_eqs, axis=1).ffill()
    rets = df.pct_change().fillna(0)
    avg_ret = rets.mean(axis=1)
    eq = (1 + avg_ret).cumprod()
    return eq


def stock_ensemble_equity(member_tags):
    """stock 측정 보류 (stock_engine 내부 equity 비공개) — 분석 단계 None 처리."""
    return None


def metrics_from_eq(eq):
    if eq is None or len(eq) < 2:
        return {"Cal": None, "CAGR": None, "MDD": None, "Sh": None}
    eq = eq.dropna()
    rets = eq.pct_change().dropna()
    n = len(eq)
    years = n / 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = dd.min()
    sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {"Cal": float(cal), "CAGR": float(cagr), "MDD": float(mdd), "Sh": float(sh)}


def portfolio_eq(asset_eqs, weights, rebal="monthly"):
    """asset_eqs: dict[name → equity series]. weights: dict[name → weight].
    공통 캘린더 forward fill → 월말/band 리밸런싱.
    """
    df = pd.concat(asset_eqs, axis=1).ffill().dropna()
    rets = df.pct_change().fillna(0)
    weights = {k: weights[k] for k in df.columns}
    w_arr = np.array([weights[c] for c in df.columns])
    cur_w = w_arr.copy()
    nav = [1.0]
    last_rebal_month = df.index[0].to_period("M")
    for i, ts in enumerate(df.index[1:], 1):
        r = rets.iloc[i].values
        cur_w = cur_w * (1 + r)
        nav_i = cur_w.sum()
        cur_w = cur_w / nav_i
        nav.append(nav[-1] * nav_i)
        cur_month = ts.to_period("M")
        if rebal == "monthly" and cur_month != last_rebal_month:
            cur_w = w_arr.copy()
            last_rebal_month = cur_month
    return pd.Series(nav, index=df.index)


def main():
    t0 = time.time()
    print("[integrated portfolio BT]", flush=True)

    # ====== V21 baseline ======
    print("[v21] fut V21 ensemble ...", flush=True)
    fut_v21 = fut_ensemble_equity([
        {"iv": "4h", "sma": 240, "ms": 20, "ml": 720, "vmode": "daily", "vthr": 0.05, "snap": 120, "lev": 3.0},
        {"iv": "4h", "sma": 240, "ms": 20, "ml": 480, "vmode": "daily", "vthr": 0.05, "snap": 30, "lev": 3.0},
        {"iv": "4h", "sma": 120, "ms": 20, "ml": 720, "vmode": "daily", "vthr": 0.05, "snap": 120, "lev": 3.0},
    ])
    print("[v21] spot V21 ensemble ...", flush=True)
    spot_v21 = spot_ensemble_equity([
        {"iv": "D", "sma": 50, "ms": 20, "ml": 90, "vmode": "daily", "vthr": 0.05, "snap": 90},
        {"iv": "D", "sma": 100, "ms": 20, "ml": 120, "vmode": "daily", "vthr": 0.05, "snap": 90},
        {"iv": "D", "sma": 150, "ms": 20, "ml": 60, "vmode": "daily", "vthr": 0.05, "snap": 90},
    ])

    # ====== New candidates ======
    print("[new] fut new ensemble k=2 ...", flush=True)
    fut_new = fut_ensemble_equity([
        {"iv": "D", "sma": 44, "ms": 18, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 24, "lev": 3.0},
        {"iv": "D", "sma": 39, "ms": 18, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 24, "lev": 3.0},
    ])
    print("[new] spot new ensemble k=2 ...", flush=True)
    spot_new = spot_ensemble_equity([
        {"iv": "D", "sma": 39, "ms": 20, "ml": 303, "vmode": "daily", "vthr": 0.05, "snap": 60},
        {"iv": "D", "sma": 39, "ms": 7, "ml": 127, "vmode": "daily", "vthr": 0.05, "snap": 60},
    ])

    # ====== Stock skip — V17 baseline 별도 측정 필요 (시간 부족 시 stock 미포함 portfolio) ======

    print("[metrics]", flush=True)
    rows = []
    for label, eq in [("fut_v21", fut_v21), ("fut_new", fut_new),
                      ("spot_v21", spot_v21), ("spot_new", spot_new)]:
        m = metrics_from_eq(eq)
        m["label"] = label
        rows.append(m)
        print(f"  {label}: Cal {m['Cal']} CAGR {m['CAGR']} MDD {m['MDD']} Sh {m['Sh']}", flush=True)

    # 통합 portfolio 65/35 (spot/fut, stock 제외 — spot+fut 만)
    if spot_v21 is not None and fut_v21 is not None:
        port_v21 = portfolio_eq({"spot": spot_v21, "fut": fut_v21},
                                 {"spot": 0.875, "fut": 0.125}, rebal="monthly")  # 35:5 → 87.5:12.5
        m = metrics_from_eq(port_v21); m["label"] = "port_v21_spot+fut_87.5_12.5"
        rows.append(m); print(f"  {m['label']}: {m}", flush=True)
    if spot_new is not None and fut_new is not None:
        port_new = portfolio_eq({"spot": spot_new, "fut": fut_new},
                                 {"spot": 0.875, "fut": 0.125}, rebal="monthly")
        m = metrics_from_eq(port_new); m["label"] = "port_new_spot+fut_87.5_12.5"
        rows.append(m); print(f"  {m['label']}: {m}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(HERE, "integrated_portfolio_results.csv"), index=False)
    print(f"[done] {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
