#!/usr/bin/env python3
"""Phase D: spot V20 + futures best 혼합 sweep.

목적: 현물 코인 V20 단독 대비 위험을 비슷하게 유지하면서 효율성을 끌어올리는
spot/futures 비율 + 밴드 리밸런싱 조합을 찾는다.

입력:
- spot V20 equity 시계열 (run_current_coin_v20_backtest.run() 호출)
- futures winner equity 시계열 (lev별, phase C 후보)

그리드:
- spot 비율: {60, 70, 80, 85, 90, 95}%
- 밴드: {0.05, 0.08, 0.10, no_rebal}
- futures winners: lev 4종 (per lev 1개)

합격선: MDD_mix <= MDD_spot * 1.10 AND Cal_mix >= Cal_spot * 1.05

출력: phase_d_results.csv + phase_d_top.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SPOT_RATIOS = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
BANDS = [0.05, 0.08, 0.10, None]  # None = no rebal


def metrics_from_equity(eq: pd.Series) -> dict:
    if eq.empty or len(eq) < 2:
        return {"Sharpe": 0, "CAGR": 0, "MDD": 0, "Cal": 0, "Final": 0}
    rets = eq.pct_change().dropna()
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    if n_years <= 0 or eq.iloc[0] <= 0:
        return {"Sharpe": 0, "CAGR": 0, "MDD": 0, "Cal": 0, "Final": float(eq.iloc[-1])}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1
    # daily basis assumed; if 4h or 1h, scale
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    cal = abs(cagr / dd) if dd != 0 else 0
    return {"Sharpe": float(sharpe), "CAGR": float(cagr),
            "MDD": float(dd), "Cal": float(cal), "Final": float(eq.iloc[-1])}


def simulate_mix(spot_eq: pd.Series, fut_eq: pd.Series, w_spot: float,
                 band: float | None, init_cap: float = 1.0) -> pd.Series:
    """일자 인덱스 정렬 후 비중 가중 합. band 초과 시 리밸 (전량 목표비중 복원)."""
    s = spot_eq.copy(); f = fut_eq.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    if getattr(f.index, "tz", None) is not None:
        f.index = f.index.tz_localize(None)
    df = pd.concat([s.rename("spot"), f.rename("fut")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    sret = df["spot"].pct_change().fillna(0)
    fret = df["fut"].pct_change().fillna(0)
    w_fut = 1.0 - w_spot
    cur_w_spot = w_spot
    cur_w_fut = w_fut
    eq = init_cap
    out = []
    for s, f in zip(sret, fret):
        # 자산별 가치 변화
        spot_val = eq * cur_w_spot * (1 + s)
        fut_val = eq * cur_w_fut * (1 + f)
        eq = spot_val + fut_val
        if eq > 0:
            new_w_spot = spot_val / eq
            new_w_fut = fut_val / eq
        else:
            new_w_spot, new_w_fut = w_spot, w_fut
        # 밴드 리밸
        if band is not None and abs(new_w_spot - w_spot) >= band:
            cur_w_spot = w_spot
            cur_w_fut = w_fut
        else:
            cur_w_spot = new_w_spot
            cur_w_fut = new_w_fut
        out.append(eq)
    return pd.Series(out, index=df.index)


def load_spot_v20_equity(start: str, end: str) -> Tuple[pd.Series, dict]:
    """spot V20 백테스트 1회 실행해서 equity 시계열 + metrics 반환."""
    sys.path.insert(0, os.path.dirname(HERE))
    from run_current_coin_v20_backtest import run_backtest
    print("Running spot V20 backtest...")
    res = run_backtest(start=start, end=end)
    eq = res["equity"]
    eq.index = pd.to_datetime(eq.index)
    return eq, res["metrics"]


def load_futures_equity(case_id: str, futures_csv_dir: str) -> pd.Series:
    """phase C에서 저장된 futures winner equity 시계열 csv 로드."""
    path = os.path.join(futures_csv_dir, f"{case_id}_equity.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return df["Value"].astype(float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--winners", required=True, help="phase C winners json (per lev best)")
    p.add_argument("--futures-equity-dir", required=True, help="phase C equity csv 디렉터리")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--start", default="2020-10-01")
    p.add_argument("--end", default="2026-04-13")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    spot_eq, spot_m = load_spot_v20_equity(args.start, args.end)
    print(f"Spot V20: Cal={spot_m['Cal']:.2f} CAGR={spot_m['CAGR']:+.1%} MDD={spot_m['MDD']:+.1%}")

    with open(args.winners) as f:
        winners = json.load(f)["winners"]  # [{lev, case_id, label, ...}]

    rows = []
    for w in winners:
        try:
            fut_eq = load_futures_equity(w["case_id"], args.futures_equity_dir)
        except Exception as e:
            print(f"WARN no futures equity for {w['case_id']}: {e}")
            continue
        for ratio in SPOT_RATIOS:
            for band in BANDS:
                mix_eq = simulate_mix(spot_eq, fut_eq, ratio, band)
                m = metrics_from_equity(mix_eq)
                accept = (abs(m["MDD"]) <= abs(spot_m["MDD"]) * 1.10 and
                          m["Cal"] >= spot_m["Cal"] * 1.05)
                rows.append({
                    "lev": w["lev"], "fut_label": w["label"],
                    "spot_ratio": ratio, "fut_ratio": 1 - ratio,
                    "band": band if band else "no_rebal",
                    "mix_Sharpe": round(m["Sharpe"], 3),
                    "mix_CAGR": round(m["CAGR"], 4),
                    "mix_MDD": round(m["MDD"], 4),
                    "mix_Cal": round(m["Cal"], 3),
                    "spot_Sharpe": round(spot_m["Sharpe"], 3),
                    "spot_CAGR": round(spot_m["CAGR"], 4),
                    "spot_MDD": round(spot_m["MDD"], 4),
                    "spot_Cal": round(spot_m["Cal"], 3),
                    "MDD_ratio": round(abs(m["MDD"]) / abs(spot_m["MDD"]), 3) if spot_m["MDD"] else None,
                    "Cal_ratio": round(m["Cal"] / spot_m["Cal"], 3) if spot_m["Cal"] else None,
                    "accept": accept,
                })

    out_csv = os.path.join(args.out_dir, "phase_d_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} mix rows to {out_csv}")

    df = pd.DataFrame(rows)
    accepted = df[df["accept"]].sort_values("mix_Cal", ascending=False)
    top = accepted.head(20).to_dict("records")
    with open(os.path.join(args.out_dir, "phase_d_top.json"), "w") as f:
        json.dump({"spot_baseline": spot_m, "accepted_top": top, "total": len(rows),
                   "n_accepted": len(accepted)}, f, indent=2, default=str)
    print(f"Accepted: {len(accepted)} / Total: {len(rows)}")


if __name__ == "__main__":
    main()
