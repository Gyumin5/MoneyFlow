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
from datetime import timedelta
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
    """V25 weighted rank momentum: 0.5*ret63 + 0.3*ret126 + 0.2*ret252.

    채택 BT(bt_stock_window_rank → bt_stock_coin_v3.precompute) 의 Z-rank 모멘텀 정의와 일치.
    (2026-06-06 정정: 이전엔 순수 252일 — V15 잔재 — 으로 들어가 라이브가 채택 BT 와
    종목 선정 16.8% 날 어긋났음. replay-diff 로 발견, 가중 전환 시 100% 정합 검증.)
    """
    if len(s) < MOM12M_DAYS + 1: return float('-inf')
    r63 = calc_ret(s, 63)
    r126 = calc_ret(s, 126)
    r252 = calc_ret(s, MOM12M_DAYS)
    if any(np.isnan(x) for x in (r63, r126, r252)):
        return float('-inf')
    return 0.5 * r63 + 0.3 * r126 + 0.2 * r252


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


STOCK_ALL_TICKERS = sorted(set(OFFENSIVE_STOCK_UNIVERSE)
                           | set(DEFENSIVE_STOCK_UNIVERSE)
                           | {STOCK_CANARY_TICKER})


def stock_basis(all_prices: Dict[str, pd.Series],
                now_kst: Optional[pd.Timestamp] = None) -> dict:
    """이 신호가 실제로 쓴 주식 종가의 날짜. 리포트에 그대로 보여주기 위한 값이다.

    왜 따로 재는가. signal_state.json 의 meta.signal_date 는 비트코인 시계열의 마지막
    날짜다 — 주식이 쓴 종가의 날짜가 아니다. 2026-09-29 실측에서 표기는 09-28 인데
    주식 계산에 실제로 들어간 종가는 09-25 였다. 그 날짜를 주식 목표 옆에 보여주려면
    가격 dict 에서 직접 재는 수밖에 없다.

    반환:
      per_ticker  {티커: 'YYYY-MM-DD'}  — 전략이 실제로 보는 종목만
      basis       전 종목이 같을 때 그 날짜, 다르면 None
      mixed       종목마다 날짜가 다른가
      oldest      가장 오래된 날짜 (실제 계산을 지배하는 쪽)
      expected    직전 평일 (아래 한계 참조)
      lag_bd      oldest 와 expected 사이 영업일 수. 못 정하면 None
      warn        운영자에게 보일 한 줄. 이상 없으면 빈 문자열

    한계 하나를 그대로 안고 간다 — 미국 휴장일 달력이 이 기계에 없다(pandas_market_calendars
    미설치). 그래서 expected 는 달력상 직전 평일이고, 추수감사절처럼 평일 휴장이면
    하루 지연으로 보인다. 단정하지 않고 "휴장일이면 정상" 을 문장에 같이 적는다.
    executor 가 쓰는 판정(_resolve_common_date + DATE_ALIGN_TOLERANCE_BD)도 같은
    평일 기준이라 기준을 새로 만들지 않는다.
    """
    per = {}
    for t in STOCK_ALL_TICKERS:
        s = all_prices.get(t)
        if s is None or len(s) == 0:
            continue
        try:
            per[t] = pd.Timestamp(s.index[-1]).date()
        except Exception:
            continue
    if not per:
        return {'per_ticker': {}, 'basis': None, 'mixed': False, 'oldest': None,
                'expected': None, 'lag_bd': None,
                'warn': '주식 가격을 한 종목도 못 받았다 — 기준일 확인 불가'}

    oldest, newest = min(per.values()), max(per.values())
    mixed = oldest != newest

    if now_kst is None:
        now_kst = pd.Timestamp.now(tz='Asia/Seoul')
    # executor 와 같은 자름: KST 어제까지의 종가만 쓴다(장중 미완성 봉 차단).
    max_close = (now_kst - pd.Timedelta(days=1)).date()
    expected = max_close
    while expected.weekday() >= 5:            # 토·일은 세션이 아니다
        expected -= timedelta(days=1)
    if expected > oldest:
        lag_bd = int(np.busday_count(oldest + timedelta(days=1),
                                     expected + timedelta(days=1)))
    else:
        lag_bd = 0

    warn = ''
    if mixed:
        warn = (f'종목마다 종가 날짜가 다르다 ({oldest} ~ {newest}) — '
                f'계산은 가장 오래된 {oldest} 기준으로 섞여 들어간다')
    elif lag_bd >= 1:
        warn = (f'주식 종가가 직전 평일 {expected} 보다 {lag_bd}영업일 이전이다 '
                f'(미국 휴장일이었으면 정상). 오늘 밤 체결 종목과 다를 수 있다')

    return {
        'per_ticker': {t: str(d) for t, d in sorted(per.items())},
        'basis': None if mixed else oldest,
        'mixed': mixed,
        'oldest': oldest,
        'expected': expected,
        'lag_bd': lag_bd,
        'warn': warn,
    }


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
