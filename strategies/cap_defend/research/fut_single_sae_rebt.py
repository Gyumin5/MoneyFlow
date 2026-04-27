"""fut top 단일 멤버를 SAE 로 재BT — unified_backtest Cal 과 비교.

진행
1. redesign_rank_fut.csv 읽기 (status==ok 후보)
2. 각 cfg 로 unified_backtest.run(_trace=trace) 후 SAE.run(trace) 시뮬
3. unified Cal vs SAE Cal 비교 → fut_single_sae_compare.csv

ensemble engine 과 동일 기준 (1h bars, leverage/funding/liquidation 정확).
"""
from __future__ import annotations
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, REPO)

OUT_CSV = os.path.join(HERE, "fut_single_sae_compare.csv")

_BARS_1H = None
_FUNDING_1H = None
_BARS_D = None
_FUNDING_D = None


def preload():
    global _BARS_1H, _FUNDING_1H, _BARS_D, _FUNDING_D
    from unified_backtest import load_data
    _BARS_1H, _FUNDING_1H = load_data("1h")
    _BARS_D, _FUNDING_D = load_data("D")


def run_one(task):
    cfg = task["cfg"]
    tag = task["tag"]
    try:
        from unified_backtest import run as bt_run
        from futures_ensemble_engine import SingleAccountEngine
        bars_d = _BARS_D
        funding_d = _FUNDING_D
        trace = []
        m_unified = bt_run(
            bars_d, funding_d, interval="D", asset_type="fut",
            leverage=float(cfg["lev"]),
            universe_size=3, cap=1 / 3, tx_cost=0.0004,
            sma_bars=int(cfg["sma"]), mom_short_bars=int(cfg["ms"]),
            mom_long_bars=int(cfg["ml"]),
            vol_mode=cfg["vmode"], vol_threshold=float(cfg["vthr"]),
            snap_interval_bars=int(cfg["snap"]), n_snapshots=3,
            phase_offset_bars=0,
            canary_hyst=0.015, health_mode="mom2vol",
            stop_kind="none", stop_pct=0.0,
            drift_threshold=0.10, post_flip_delay=5,
            dd_lookback=60, dd_threshold=-0.25,
            bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
            start_date="2020-10-01", end_date="2026-04-13",
            _trace=trace,
        )
        cal_unified = float(m_unified.get("Cal") or 0)
        cagr_unified = float(m_unified.get("CAGR") or 0)
        mdd_unified = float(m_unified.get("MDD") or 0)

        # SAE 시뮬: trace 를 1h grid 위에 ffill, EW (k=1)
        ts_1h = next(iter(_BARS_1H.values())).index
        events = [(pd.Timestamp(t["date"]), dict(t["target"] or {})) for t in trace]
        events.sort(key=lambda x: x[0])
        if not events:
            return {"tag": tag, "status": "error", "error": "empty trace"}
        t0 = events[0][0]
        t1 = events[-1][0]
        ts_1h = ts_1h[(ts_1h >= t0) & (ts_1h <= t1)]
        target_series = []
        idx = 0
        cur_target = {}
        for ts in ts_1h:
            while idx < len(events) and events[idx][0] <= ts:
                cur_target = events[idx][1]
                idx += 1
            target_series.append((ts, dict(cur_target)))
        sae = SingleAccountEngine(
            _BARS_1H, _FUNDING_1H,
            leverage=float(cfg["lev"]), tx_cost=0.0004,
            stop_kind="none", leverage_mode="fixed",
        )
        m_sae = sae.run(target_series)
        return {
            "tag": tag, "status": "ok",
            "cal_unified": cal_unified, "cagr_unified": cagr_unified, "mdd_unified": mdd_unified,
            "cal_sae": float(m_sae.get("Cal", 0) or 0),
            "cagr_sae": float(m_sae.get("CAGR", 0) or 0),
            "mdd_sae": float(m_sae.get("MDD", 0) or 0),
            "sh_sae": float(m_sae.get("Sharpe", 0) or 0),
        }
    except Exception as e:
        return {"tag": tag, "status": "error", "error": str(e)[:200]}


def main():
    df = pd.read_csv(os.path.join(HERE, "redesign_rank_fut.csv"))
    df = df[df.get("hard_gate_pass", True) == True] if "hard_gate_pass" in df.columns else df
    df = df.head(25)
    print(f"[build] {len(df)} fut single SAE re-BT tasks", flush=True)

    tasks = []
    for _, row in df.iterrows():
        cfg = {
            "iv": row.get("iv", "D"),
            "sma": row["sma"], "ms": row["ms"], "ml": row["ml"],
            "vmode": row["vmode"], "vthr": row["vthr"],
            "snap": row["snap"], "lev": row.get("lev", 3.0),
        }
        tasks.append({"tag": str(row["tag"]), "cfg": cfg})

    t0 = time.time()
    with Pool(4, initializer=preload) as pool:
        rows = []
        for i, res in enumerate(pool.imap_unordered(run_one, tasks), 1):
            rows.append(res)
            if i % 5 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"[{i}/{len(tasks)}] {rate:.2f}/s ETA {(len(tasks)-i)/max(rate,1e-6):.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[done] {(time.time()-t0)/60:.1f}m → {OUT_CSV}")


if __name__ == "__main__":
    main()
