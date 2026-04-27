"""주식 엔진 adapter — cfg dict + phase → metrics dict.

V17 실운영 universe 기준 SP 생성 후 stock_engine_snap.run_snapshot_ensemble 호출.
Metrics: equity DataFrame → Cal/CAGR/MDD/Sh 변환.

cfg 필드:
  snap (bars=days), canary_sma, canary_hyst, canary_type, select,
  def_mom, health

TODO (엔진 수정 필요):
  - monthly_anchor_mode 고정 False (phase_offset 적용 되는 calendar mode)
  - tx_cost = cfg tx_cost → SP.tx_cost 반영
  - tranche_days 는 V21 스타일 (1,) 고정
"""
from __future__ import annotations
import os
import sys
from dataclasses import replace
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, REPO)
sys.path.insert(0, CAP)

# V17 실운영 universe
STOCK_OFFENSE = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
STOCK_DEFENSE = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')
STOCK_CANARY = ('EEM',)

_INITIALIZED = False


def _init_once():
    global _INITIALIZED
    if _INITIALIZED:
        return
    from stock_engine import load_prices, precompute, _init, ALL_TICKERS
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    _INITIALIZED = True


def run_stock_from_cfg(
    cfg: Dict[str, Any],
    *,
    phase_offset: int = 0,
    tx_cost: float = 0.0025,
    start: str = '2017-04-01',
    end: str = '2025-12-31',
    year_only: Optional[int] = None,
    execution_delay_bars: int = 0,
    exclude_assets=None,
    with_trace: bool = False,
) -> Dict[str, Any]:
    """cfg → metrics dict (Cal/CAGR/MDD/Sh/rebal).
    execution_delay_bars/exclude_assets 는 stock_engine_snap 로 pass-through.
    """
    _init_once()
    from stock_engine import SP
    from stock_engine_snap import run_snapshot_ensemble
    import stock_engine as tsi

    if year_only is not None:
        start = f"{year_only}-01-01"
        end = f"{year_only}-12-31"

    n_pick = int(cfg.get("n_pick", 3))
    params = SP(
        offensive=STOCK_OFFENSE,
        defensive=STOCK_DEFENSE,
        canary_assets=STOCK_CANARY,
        canary_sma=int(cfg["canary_sma"]),
        canary_hyst=float(cfg["canary_hyst"]),
        canary_type=str(cfg.get("canary_type", "sma")),
        health=str(cfg.get("health", "none")),
        defense="top2",
        def_mom_period=int(cfg.get("def_mom", 252)),
        select=str(cfg.get("select", "mom3_sh3")),
        n_mom=n_pick, n_sh=n_pick,
        # grid v3 에서 sweep 되면 cfg 에 실제 값 있음. 없으면 V17 기본 (252)
        sharpe_lookback=int(cfg.get("sharpe_lookback", 252)),
        mom_style=str(cfg.get("mom_style", "default")),
        crash="none",
        weight="ew",
        tranche_days=(1,),
        tx_cost=tx_cost,
        start=start, end=end,
        capital=10000.0,
    )

    # 엔진은 모듈 전역 _g_prices/_g_ind 에 의존 (_init 로 주입)
    prices = tsi._g_prices
    ind = tsi._g_ind
    trace = [] if with_trace else None
    try:
        eq_df = run_snapshot_ensemble(
            prices, ind, params,
            snap_days=int(cfg["snap"]),
            n_snap=3,
            monthly_anchor_mode=False,
            phase_offset=int(phase_offset),
            execution_delay_bars=int(execution_delay_bars),
            exclude_assets=exclude_assets,
            _trace=trace,
        )
    except Exception as e:
        return {"status": "error", "error": f"engine: {str(e)[:200]}"}

    if eq_df is None or (hasattr(eq_df, "empty") and eq_df.empty):
        return {"status": "error", "error": "empty equity"}

    # eq_df 에 'PV' 또는 'Value' 컬럼 기대
    for col in ("PV", "Value", "value", "pv"):
        if col in eq_df.columns:
            eq = eq_df[col]
            break
    else:
        return {"status": "error", "error": f"no equity col in {list(eq_df.columns)[:5]}"}
    rebal = 0
    try:
        rebal = int(eq_df.attrs.get("rebal_count", 0))
    except Exception:
        rebal = 0

    yrs = (eq.index[-1] - eq.index[0]).days / 365.25 if len(eq) > 1 else 0
    if yrs <= 0 or eq.iloc[-1] <= 0:
        return {"status": "error", "error": "invalid equity"}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {
        "status": "ok",
        "Cal": float(cal), "CAGR": float(cagr),
        "MDD": float(mdd), "Sh": float(sh),
        "rebal": rebal,
        "_equity": eq,
        "_trace": trace,
    }


if __name__ == "__main__":
    # Smoke test
    cfg = {
        "snap": 30, "canary_sma": 200, "canary_hyst": 0.005,
        "canary_type": "sma", "select": "mom3_sh3",
        "def_mom": 252, "health": "none",
    }
    print(run_stock_from_cfg(cfg))
