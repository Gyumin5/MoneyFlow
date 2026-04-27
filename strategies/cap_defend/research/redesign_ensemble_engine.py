"""True single-account ensemble engine.

각 멤버의 daily target weights 시계열을 _trace 로 추출 → EW merge → 단일 portfolio 시뮬.
거래비용 netting 자동 반영 (멤버간 동일 자산 매수/매도 상쇄).

live engine (coin_live_engine.combine_ensemble) 동일 원리.

API
  ensemble_single_account(cfgs, weights, asset, start, end, tx_cost, leverage) → metrics dict
    - cfgs: list of cfg dict (parse_cfg output)
    - weights: list of float (각 멤버 비중, 정규화됨)
    - asset: 'fut' or 'spot' or 'stock'

내부
  1. 각 멤버 cfg 로 unified_backtest.run 실행, _trace 로 daily target 추출
  2. 모든 멤버 trace 의 date 합집합 → 각 date 에서 멤버 target weighted merge
  3. portfolio simulator: merged target 따라 매일 rebal, tx_cost 적용
"""
from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CAP)


def _to_target_dict(trace_row) -> Dict[str, float]:
    tgt = trace_row.get("target") or {}
    out = {}
    for k, v in tgt.items():
        if k.upper() == "CASH":
            continue
        out[str(k).upper()] = float(v)
    cash = max(0.0, 1.0 - sum(out.values()))
    if cash > 1e-9:
        out["CASH"] = cash
    return out


def _build_target_series(trace: List[dict]) -> pd.DataFrame:
    """trace → DataFrame(index=date, cols=assets) of weights."""
    rows = []
    for t in trace:
        d = pd.Timestamp(t["Date"])
        tgt = _to_target_dict(t)
        rows.append({"date": d, **tgt})
    df = pd.DataFrame(rows).set_index("date").fillna(0.0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _merge_targets(target_dfs: List[pd.DataFrame],
                   weights: List[float]) -> pd.DataFrame:
    """멤버별 target series 를 union date 위에서 ffill + EW merge."""
    all_dates = sorted(set().union(*[df.index for df in target_dfs]))
    aligned = []
    for df in target_dfs:
        df2 = df.reindex(all_dates).ffill().fillna(0.0)
        aligned.append(df2)
    all_assets = sorted(set().union(*[set(df.columns) for df in aligned]))
    merged = pd.DataFrame(0.0, index=all_dates, columns=all_assets)
    wsum = sum(weights)
    for df, w in zip(aligned, weights):
        df3 = df.reindex(columns=all_assets, fill_value=0.0)
        merged += df3 * (w / wsum)
    # 정규화
    row_sum = merged.sum(axis=1)
    nz = row_sum > 1e-9
    merged.loc[nz] = merged.loc[nz].div(row_sum[nz], axis=0)
    return merged


def _simulate_portfolio(prices: pd.DataFrame, targets: pd.DataFrame,
                         leverage: float = 1.0, tx_cost: float = 0.0004,
                         capital: float = 10000.0) -> pd.DataFrame:
    """단일 계좌 portfolio 시뮬. prices: index=date, cols=asset close prices.
    targets: 같은 index, 멤버 합산 target weights (자산 + CASH 행).
    매 date 당 target 변화 시 rebal — tx_cost = | delta_w | × pv × tx_cost (양방향 합).
    """
    # 가격/타겟 align
    common_dates = prices.index.intersection(targets.index)
    prices = prices.loc[common_dates]
    targets = targets.loc[common_dates]
    assets = [c for c in targets.columns if c != "CASH" and c in prices.columns]

    pv = capital
    holdings = {a: 0.0 for a in assets}  # 수량
    cash = capital
    pv_hist = []
    prev_target = {a: 0.0 for a in assets}

    for date in common_dates:
        pxs = prices.loc[date]
        # mark-to-market
        eq_assets = sum(holdings[a] * pxs[a] for a in assets if not pd.isna(pxs.get(a)))
        pv = cash + eq_assets * leverage  # leverage > 1 simplification
        # rebal to target
        cur_target = {a: float(targets.loc[date].get(a, 0.0)) for a in assets}
        changed = any(abs(cur_target[a] - prev_target[a]) > 1e-6 for a in assets)
        if changed and pv > 0:
            new_holdings = {}
            for a in assets:
                w = cur_target[a]
                px = pxs.get(a)
                if pd.isna(px) or px <= 0:
                    new_holdings[a] = holdings[a]
                    continue
                target_qty = (pv * w) / px
                turnover_qty = abs(target_qty - holdings[a])
                tx = turnover_qty * px * tx_cost
                cash -= (target_qty - holdings[a]) * px + tx
                new_holdings[a] = target_qty
            holdings = new_holdings
            prev_target = dict(cur_target)
        pv_hist.append({"Date": date, "PV": pv})
    return pd.DataFrame(pv_hist).set_index("Date")


def _metrics_from_pv(pv: pd.Series) -> Dict[str, float]:
    if len(pv) < 2 or pv.iloc[0] <= 0:
        return {"Cal": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sh": 0.0}
    yrs = (pv.index[-1] - pv.index[0]).days / 365.25
    if yrs <= 0:
        return {"Cal": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sh": 0.0}
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (1 / yrs) - 1
    dr = pv.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    mdd = float((pv / pv.cummax() - 1).min())
    cal = float(cagr / abs(mdd)) if mdd != 0 else 0.0
    return {"Cal": cal, "CAGR": float(cagr), "MDD": mdd, "Sh": sh}


def ensemble_single_account(member_cfgs: List[Dict[str, Any]],
                             weights: List[float],
                             asset: str,
                             start: str = "2020-10-01",
                             end: str = "2026-04-13",
                             tx_cost: float = 0.0004,
                             leverage: float = 1.0,
                             bars_funding=None) -> Dict[str, Any]:
    """k 멤버 → true single-account EW ensemble metrics."""
    from unified_backtest import run as bt_run, load_data
    if bars_funding is None:
        ivs = sorted(set(c.get("iv", "D") for c in member_cfgs))
        bars_funding = {iv: load_data(iv) for iv in ivs}

    # 1) 각 멤버 trace 추출
    target_dfs = []
    prices_ref = None
    for cfg in member_cfgs:
        iv = cfg.get("iv", "D")
        bars, funding = bars_funding[iv]
        trace = []
        try:
            bt_run(
                bars, funding, interval=iv, asset_type=asset,
                leverage=cfg.get("lev", leverage),
                universe_size=3, cap=1 / 3, tx_cost=tx_cost,
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
                start_date=start, end_date=end,
                _trace=trace,
            )
        except Exception as e:
            return {"status": "error", "error": f"member BT: {str(e)[:160]}"}
        if not trace:
            return {"status": "error", "error": "empty trace"}
        target_dfs.append(_build_target_series(trace))
        if prices_ref is None:
            # bars는 dict {asset: DataFrame}; price ref 만들기 위해 close 모음
            prices_ref = pd.DataFrame({a: df["close"] for a, df in bars.items() if "close" in df.columns})

    # 2) merge
    merged = _merge_targets(target_dfs, weights)

    # 3) portfolio sim
    pv_df = _simulate_portfolio(prices_ref, merged, leverage=leverage, tx_cost=tx_cost)
    if pv_df.empty:
        return {"status": "error", "error": "empty pv"}
    m = _metrics_from_pv(pv_df["PV"])
    m["status"] = "ok"
    return m


if __name__ == "__main__":
    print("redesign_ensemble_engine: import-only module")
