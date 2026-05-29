"""Cap Defend V25 — Stock 전략 (Pure functions).

executor 가 trade time 에 직접 호출. recommend 도 display 용으로 동일 함수 호출.
signal_state.json 의존 없음. 입력 = 가격 dict + 이전 risk_on, 출력 = (picks, weights, risk_on, meta).

V25 spec
- Universe: R7 = SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ
- Defense: IEF, BIL, BNDX, GLD, PDBC
- EEM canary SMA200 + 0.5% hyst
- Offense: Z-score (Mom12M + Sharpe126) 랭킹 → 3-mom (30/72/230) 필터 → cap=1/3+Cash
- Defense: 6M return top3 (positive only)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

CASH_ASSET = 'Cash'
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
STOCK_CANARY_TICKER = 'EEM'
STOCK_CANARY_MA_PERIOD = 200
STOCK_CANARY_HYST = 0.005   # 0.5%

# 3-mom periods (V25 best per window rank-sum BT)
MOM_SHORT = 30
MOM_MID = 72
MOM_LONG = 230

# Sharpe lookback (Z-score component)
SHARPE_LB = 126
MOM12M_DAYS = 252


def calc_ret(s: pd.Series, d: int) -> float:
    if len(s) < d + 1: return float('nan')
    v0 = s.iloc[-1-d]
    if v0 == 0: return float('nan')
    return float(s.iloc[-1] / v0 - 1)


def calc_sharpe(s: pd.Series, d: int) -> float:
    if len(s) < d + 1: return 0.0
    ret = s.pct_change().iloc[-d:]
    std = ret.std()
    return float((ret.mean() / std) * np.sqrt(252)) if std and std > 0 else 0.0


def calc_weighted_mom(s: pd.Series) -> float:
    """V15: Pure 12-month momentum."""
    if len(s) < MOM12M_DAYS + 1: return float('-inf')
    return calc_ret(s, MOM12M_DAYS)


def eem_canary(eem: Optional[pd.Series], prev_risk_on: Optional[bool],
               hyst: float = STOCK_CANARY_HYST,
               sma_period: int = STOCK_CANARY_MA_PERIOD) -> Tuple[bool, dict]:
    """EEM SMA200 + hysteresis canary.

    EEM 데이터 결측·부족 → prev_risk_on 그대로 유지 (자동 risk-off 안 함).
    prev_risk_on 도 없으면 보수적으로 risk-off.
    """
    if eem is None or len(eem) < sma_period:
        fallback = prev_risk_on if prev_risk_on is not None else False
        return bool(fallback), {'canary': 'data_missing', 'canary_risk_on': bool(fallback)}
    eem_sma = eem.rolling(sma_period).mean().iloc[-1]
    eem_cur = eem.iloc[-1]
    if pd.isna(eem_sma) or pd.isna(eem_cur) or float(eem_sma) <= 0:
        fallback = prev_risk_on if prev_risk_on is not None else False
        return bool(fallback), {'canary': 'data_invalid', 'canary_risk_on': bool(fallback)}
    dist = float(eem_cur / eem_sma - 1)
    meta = {
        'canary_eem_cur': float(eem_cur),
        'canary_eem_sma': float(eem_sma),
        'canary_dist': dist,
        'canary_sma_period': sma_period,
        'canary_hyst': hyst,
    }
    if dist > hyst:
        risk_on = True
    elif dist < -hyst:
        risk_on = False
    elif prev_risk_on is not None:
        risk_on = bool(prev_risk_on)  # dead zone → 이전 상태 (False 명시 보존)
    else:
        risk_on = bool(eem_cur > eem_sma)
    meta['canary_risk_on'] = risk_on
    return risk_on, meta


def compute_offense(all_prices: Dict[str, pd.Series]) -> Tuple[List[str], Dict[str, float], dict]:
    """Z-score rank → 3-mom filter → cap=1/3+Cash."""
    rows = []
    for t in OFFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(t)
        if p is None or len(p) < MOM12M_DAYS + 1:
            continue
        rows.append({
            'Ticker': t,
            'Mom12M': calc_weighted_mom(p),
            'Sharpe126': calc_sharpe(p, SHARPE_LB),
            'Mom30': calc_ret(p, MOM_SHORT),
            'Mom72': calc_ret(p, MOM_MID),
            'Mom230': calc_ret(p, MOM_LONG),
        })
    if not rows:
        return [], {CASH_ASSET: 1.0}, {'reason': 'no_data'}
    df = pd.DataFrame(rows).set_index('Ticker')
    m_std = df['Mom12M'].std()
    s_std = df['Sharpe126'].std()
    df['Z_Mom'] = (df['Mom12M'] - df['Mom12M'].mean()) / m_std if m_std and m_std > 0 else 0
    df['Z_Sh'] = (df['Sharpe126'] - df['Sharpe126'].mean()) / s_std if s_std and s_std > 0 else 0
    df['ZScore'] = df['Z_Mom'] + df['Z_Sh']
    df['mom_pass'] = (
        df['Mom30'].notna() & (df['Mom30'] > 0) &
        df['Mom72'].notna() & (df['Mom72'] > 0) &
        df['Mom230'].notna() & (df['Mom230'] > 0)
    )
    ranked = df.sort_values('ZScore', ascending=False).index.tolist()
    picks: List[str] = []
    excluded: List[str] = []
    for t in ranked:
        if bool(df.at[t, 'mom_pass']):
            picks.append(t)
            if len(picks) >= 3:
                break
        else:
            excluded.append(t)
    per_pick = 1.0 / 3
    weights: Dict[str, float] = {t: per_pick for t in picks}
    cash_slot = 1.0 - per_pick * len(picks)
    if cash_slot > 1e-9:
        weights[CASH_ASSET] = cash_slot
    meta = {
        'df': df,
        'picks': picks,
        'excluded_by_mom': excluded,
        'cash_slot': cash_slot,
    }
    return picks, weights, meta


def compute_defense(all_prices: Dict[str, pd.Series]) -> Tuple[List[str], Dict[str, float], dict]:
    """Top 3 by 6M return (positive only)."""
    rows = []
    for t in DEFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(t)
        if p is None:
            continue
        r = calc_ret(p, 126)
        if pd.notna(r):
            rows.append({'Ticker': t, '6m_Ret': r})
    if not rows:
        return [], {CASH_ASSET: 1.0}, {'reason': 'no_data'}
    rows.sort(key=lambda x: -x['6m_Ret'])
    picks = [r['Ticker'] for r in rows[:3] if r['6m_Ret'] > 0]
    if not picks:
        return [], {CASH_ASSET: 1.0}, {'reason': 'all_negative', 'rows': rows}
    per_pick = 1.0 / len(picks)
    weights: Dict[str, float] = {t: per_pick for t in picks}
    return picks, weights, {'rows': rows, 'picks': picks}


def compute_strategy(all_prices: Dict[str, pd.Series],
                     prev_risk_on: Optional[bool] = None) -> dict:
    """전략 단일 진입점.

    Returns:
        {
          'risk_on': bool,
          'picks': List[str],
          'weights': Dict[str, float],   # sums ≤ 1.0 (Cash slot 가능)
          'mode': 'offense'|'defense'|'no_data',
          'meta': dict,
          'canary_meta': dict,
        }
    """
    eem = all_prices.get(STOCK_CANARY_TICKER)
    risk_on, canary_meta = eem_canary(eem, prev_risk_on)
    if risk_on:
        picks, weights, meta = compute_offense(all_prices)
        mode = 'offense' if picks else 'offense_cash'
    else:
        picks, weights, meta = compute_defense(all_prices)
        mode = 'defense' if picks else 'defense_cash'
    return {
        'risk_on': risk_on,
        'picks': picks,
        'weights': weights,
        'mode': mode,
        'meta': meta,
        'canary_meta': canary_meta,
    }
