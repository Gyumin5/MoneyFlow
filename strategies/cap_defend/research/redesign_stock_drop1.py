"""주식 Top 3 drop-1 수동 stress (AI 라운드 합의 옵션 c).

stock_engine_snap 수정 없이 V17 universe 에서 1종 제외하고 재평가.
Top 3 최종 후보에만 적용.

입력: redesign_rank_stock.csv (Top 3)
출력: redesign_stock_drop1.csv
  tag, excluded_asset, Cal, CAGR, MDD, Cal_decay, verdict

사용:
  python redesign_stock_drop1.py --top 3
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from redesign_common import parse_cfg
from redesign_stock_adapter import (
    STOCK_OFFENSE, STOCK_DEFENSE, STOCK_CANARY, _init_once,
)

# drop-1: universe 1종 제거 후 재평가
# canary(EEM) 는 제외 안 함 (신호 자산 유지. AI 라운드1 "drop tradable only" 권고)


def run_drop1_for_cfg(cfg, excluded: str, tx_cost: float = 0.0025):
    """excluded 자산을 offensive 에서 제거하고 재평가."""
    from stock_engine import SP
    import stock_engine as tsi
    from stock_engine_snap import run_snapshot_ensemble

    _init_once()
    # excluded 가 canary 면 skip (의미 없음)
    if excluded in STOCK_CANARY:
        return {"status": "skip", "reason": "canary asset"}

    offense = tuple(t for t in STOCK_OFFENSE if t != excluded)
    defense = tuple(t for t in STOCK_DEFENSE if t != excluded)
    if len(offense) < 3:
        return {"status": "skip", "reason": "offense universe<3"}

    params = SP(
        offensive=offense, defensive=defense, canary_assets=STOCK_CANARY,
        canary_sma=int(cfg["canary_sma"]),
        canary_hyst=float(cfg["canary_hyst"]),
        canary_type=str(cfg.get("canary_type", "sma")),
        health=str(cfg.get("health", "none")),
        defense="top2",
        def_mom_period=int(cfg.get("def_mom", 252)),
        select=str(cfg.get("select", "mom3_sh3")),
        n_mom=3, n_sh=3, weight="ew",
        tranche_days=(1,), tx_cost=tx_cost,
        start="2017-04-01", end="2025-12-31",
        capital=10000.0,
    )
    prices, ind = tsi._g_prices, tsi._g_ind
    try:
        eq_df = run_snapshot_ensemble(prices, ind, params,
                                       snap_days=int(cfg["snap"]),
                                       n_snap=3,
                                       monthly_anchor_mode=False,
                                       phase_offset=0)
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}
    if eq_df is None or eq_df.empty:
        return {"status": "error", "error": "empty equity"}
    col = next((c for c in ("PV", "Value") if c in eq_df.columns), None)
    if not col:
        return {"status": "error", "error": "no equity col"}
    eq = eq_df[col]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[-1] <= 0:
        return {"status": "error", "error": "invalid equity"}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {"status": "ok",
            "Cal": float(cal), "CAGR": float(cagr),
            "MDD": float(mdd), "Sh": float(sh)}


def run_baseline(cfg, tx_cost=0.0025):
    from redesign_stock_adapter import run_stock_from_cfg
    return run_stock_from_cfg(cfg, phase_offset=0, tx_cost=tx_cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3, help="Top N 후보")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Cal 열화 허용 (예: 0.30 = 30% 이내면 pass)")
    args = ap.parse_args()

    rank_csv = os.path.join(HERE, "redesign_rank_stock.csv")
    if not os.path.exists(rank_csv):
        print(f"missing: {rank_csv}")
        return
    rank = pd.read_csv(rank_csv).head(args.top)
    print(f"[stock drop-1] Top {len(rank)} 후보 대상")

    rows = []
    for _, r in rank.iterrows():
        tag = str(r["tag"])
        cfg = parse_cfg("stock", r)
        if cfg is None:
            rows.append({"tag": tag, "status": "error",
                         "error": "cfg parse fail"})
            continue
        baseline = run_baseline(cfg)
        if baseline.get("status") != "ok":
            rows.append({"tag": tag, "status": "error",
                         "error": f"baseline: {baseline.get('error','')[:100]}"})
            continue
        base_cal = baseline["Cal"]
        # offense + defense 합집합 (GLD/PDBC 양쪽 포함 → dedup)
        universe_union = []
        seen = set()
        for a in STOCK_OFFENSE + STOCK_DEFENSE:
            if a in seen:
                continue
            seen.add(a)
            universe_union.append(a)
        for excluded in universe_union:
            if excluded in STOCK_CANARY:
                continue
            res = run_drop1_for_cfg(cfg, excluded)
            if res.get("status") != "ok":
                rows.append({"tag": tag, "excluded": excluded,
                             "status": res.get("status"),
                             "error": res.get("error", "") or res.get("reason", "")})
                continue
            cal = res["Cal"]
            decay = (cal - base_cal) / base_cal if abs(base_cal) > 1e-9 else None
            verdict = ("pass" if decay is not None and decay >= -args.threshold
                       else "fail")
            rows.append({
                "tag": tag, "excluded": excluded,
                "status": "ok",
                "baseline_Cal": base_cal,
                "Cal": cal, "CAGR": res["CAGR"],
                "MDD": res["MDD"], "Sh": res["Sh"],
                "Cal_decay": decay, "verdict": verdict,
            })
            print(f"  {tag} / drop {excluded}: Cal {cal:.3f} "
                  f"(base {base_cal:.3f}, decay {decay:+.1%}) → {verdict}")

    out_csv = os.path.join(HERE, "redesign_stock_drop1.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[stock drop-1] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
