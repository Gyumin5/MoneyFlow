"""V21 fut (ENS_fut_L3_k3_12652d57) 3 members SAE 재BT.

V21 정확 cfg (CLAUDE.md):
- 4h_S240_SN120: SMA240, Mom20/720, daily vol 5%, snap120, L3
- 4h_S240_SN30 : SMA240, Mom20/480, daily vol 5%, snap30,  L3
- 4h_S120_SN120: SMA120, Mom20/720, daily vol 5%, snap120, L3

Output:
- v21_fut_sae_members.csv (single member SAE)
- v21_fut_sae_ensemble.csv (3-member EW SAE)
"""
from __future__ import annotations
import os
import sys
import time

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, REPO)

V21_MEMBERS = [
    {"name": "4h_S240_SN120", "iv": "4h", "sma": 240, "ms": 20, "ml": 720,
     "vmode": "daily", "vthr": 0.05, "snap": 120, "lev": 3.0},
    {"name": "4h_S240_SN30",  "iv": "4h", "sma": 240, "ms": 20, "ml": 480,
     "vmode": "daily", "vthr": 0.05, "snap": 30,  "lev": 3.0},
    {"name": "4h_S120_SN120", "iv": "4h", "sma": 120, "ms": 20, "ml": 720,
     "vmode": "daily", "vthr": 0.05, "snap": 120, "lev": 3.0},
]


def run_member_with_trace(cfg):
    from unified_backtest import run as bt_run, load_data
    bars, funding = load_data(cfg["iv"])
    trace = []
    m = bt_run(
        bars, funding, interval=cfg["iv"], asset_type="fut",
        leverage=cfg["lev"], universe_size=3, cap=1/3, tx_cost=0.0004,
        sma_bars=cfg["sma"], mom_short_bars=cfg["ms"], mom_long_bars=cfg["ml"],
        vol_mode=cfg["vmode"], vol_threshold=cfg["vthr"],
        snap_interval_bars=cfg["snap"], n_snapshots=3,
        phase_offset_bars=0,
        canary_hyst=0.015, health_mode="mom2vol",
        stop_kind="none", stop_pct=0.0,
        drift_threshold=0.10, post_flip_delay=5,
        dd_lookback=60, dd_threshold=-0.25,
        bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
        start_date="2020-10-01", end_date="2026-04-13",
        _trace=trace,
    )
    return m, trace


def member_sae(trace, lev=3.0):
    """단일 member trace 를 SAE 1h 로 재시뮬."""
    from unified_backtest import load_data
    from futures_ensemble_engine import SingleAccountEngine
    bars_1h, funding_1h = load_data("1h")
    ts_1h = next(iter(bars_1h.values())).index
    events = [(pd.Timestamp(t["date"]), dict(t["target"] or {})) for t in trace]
    events.sort(key=lambda x: x[0])
    if not events:
        return None
    t0, t1 = events[0][0], events[-1][0]
    ts_1h = ts_1h[(ts_1h >= t0) & (ts_1h <= t1)]
    target_series = []
    idx = 0
    cur = {}
    for ts in ts_1h:
        while idx < len(events) and events[idx][0] <= ts:
            cur = events[idx][1]
            idx += 1
        target_series.append((ts, dict(cur)))
    sae = SingleAccountEngine(bars_1h, funding_1h, leverage=lev,
                               tx_cost=0.0004, stop_kind="none", leverage_mode="fixed")
    return sae.run(target_series)


def ensemble_sae(traces, lev=3.0):
    """3 member trace 를 1h grid 위에 EW merge → SAE."""
    from unified_backtest import load_data
    from futures_ensemble_engine import SingleAccountEngine
    bars_1h, funding_1h = load_data("1h")
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
                if key == "CASH":
                    continue
                merged[key] = merged.get(key, 0.0) + float(w) / k
        cw = max(0.0, 1.0 - sum(merged.values()))
        if cw > 1e-9:
            merged["CASH"] = cw
        target_series.append((ts, merged))
    sae = SingleAccountEngine(bars_1h, funding_1h, leverage=lev,
                               tx_cost=0.0004, stop_kind="none", leverage_mode="fixed")
    return sae.run(target_series)


def main():
    t0 = time.time()
    print("[V21 fut SAE rebt]", flush=True)
    member_rows = []
    traces = []
    for cfg in V21_MEMBERS:
        print(f"  member {cfg['name']} ...", flush=True)
        m, tr = run_member_with_trace(cfg)
        traces.append(tr)
        sae = member_sae(tr, lev=cfg["lev"])
        member_rows.append({
            "name": cfg["name"],
            "cal_unified": float(m.get("Cal") or 0),
            "cagr_unified": float(m.get("CAGR") or 0),
            "mdd_unified": float(m.get("MDD") or 0),
            "cal_sae": float(sae.get("Cal", 0) or 0) if sae else None,
            "cagr_sae": float(sae.get("CAGR", 0) or 0) if sae else None,
            "mdd_sae": float(sae.get("MDD", 0) or 0) if sae else None,
            "sh_sae": float(sae.get("Sharpe", 0) or 0) if sae else None,
        })
    pd.DataFrame(member_rows).to_csv(os.path.join(HERE, "v21_fut_sae_members.csv"), index=False)
    print("[ensemble 3-member EW SAE]", flush=True)
    ens = ensemble_sae(traces, lev=3.0)
    pd.DataFrame([{
        "members": "|".join(m["name"] for m in V21_MEMBERS),
        "cal_sae": float(ens.get("Cal", 0) or 0) if ens else None,
        "cagr_sae": float(ens.get("CAGR", 0) or 0) if ens else None,
        "mdd_sae": float(ens.get("MDD", 0) or 0) if ens else None,
        "sh_sae": float(ens.get("Sharpe", 0) or 0) if ens else None,
    }]).to_csv(os.path.join(HERE, "v21_fut_sae_ensemble.csv"), index=False)
    print(f"[done] {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
