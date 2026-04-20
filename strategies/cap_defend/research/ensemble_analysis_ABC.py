#!/usr/bin/env python3
"""Strategy C equity + V21 현물 equity 상관 + 앙상블 효과 측정.

C: Mean reversion (1h_dip24_thr-0.15_tp0.08_ts24_lev1.0) — Sharpe 1.88, MDD -14%
V21 현물: 기존 백테스트 재활용 또는 run_current_coin_v20_backtest.py 출력.

측정:
- C equity vs V21 현물 equity 상관계수
- V21 카나리 OFF 구간 C 수익
- 앙상블 (V21 현물 + C) 성과 vs V21 현물 단독
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'


def build_C_equity():
    """Strategy C 대표 파라미터 equity 재생성"""
    sys.path.insert(0, HERE)
    from backtest_mean_reversion import load_btc, run_strategy, metrics
    df = load_btc('1h')
    # 1h_dip24_thr-0.15_tp0.08_ts24_lev1.0
    eq = run_strategy(df, dip_bars=24, dip_threshold=-0.15,
                      take_profit=0.08, time_stop_bars=24, lev=1.0)
    # Daily resample (V21과 시간축 맞춤)
    eq_daily = eq.resample('D').last().ffill()
    return eq_daily, metrics(eq, '1h')


def build_V21_spot_equity():
    """V21 현물 equity. 기존 run_current_coin_v20_backtest.py의 출력 재사용 또는 재계산.
    여기선 간단히 BTC buy&hold를 proxy로 쓰는 대신, phase3_10x의 k3_4b270476 앙상블을 build_ensemble_full_equity로 생성"""
    sys.path.insert(0, HERE)
    from phase4_3asset import build_ensemble_full_equity
    spot_top = pd.read_csv(os.path.join(HERE, 'phase3_10x', 'spot_top.csv'))
    ens = spot_top[spot_top['ensemble_tag'] == 'ENS_spot_k3_4b270476'].iloc[0]
    eq = build_ensemble_full_equity(ens)
    if isinstance(eq.index, pd.DatetimeIndex) and eq.index.tz is not None:
        eq = eq.copy(); eq.index = eq.index.tz_localize(None)
    return eq


def metrics_daily(eq, label=''):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'label': label, 'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0}
    bpy = 252
    sharpe = (rets.mean() * bpy) / (rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'label': label, 'Sharpe': round(sharpe, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3)}


def main():
    print('[Ensemble Analysis: V21 현물 + Strategy C]')
    print('Building C equity...')
    c_eq_daily, c_intra = build_C_equity()
    print(f'  C intraday metrics: {c_intra}')
    print(f'  C daily bars: {len(c_eq_daily)} ({c_eq_daily.index[0].date()} ~ {c_eq_daily.index[-1].date()})')

    print('\nBuilding V21 spot equity...')
    v21_eq = build_V21_spot_equity()
    print(f'  V21 spot bars: {len(v21_eq)} ({v21_eq.index[0].date()} ~ {v21_eq.index[-1].date()})')

    # Align
    common_idx = c_eq_daily.index.intersection(v21_eq.index)
    c = c_eq_daily.loc[common_idx]
    v = v21_eq.loc[common_idx]
    print(f'\nCommon period: {common_idx[0].date()} ~ {common_idx[-1].date()}  ({len(common_idx)} days)')

    # Normalize both to start=1
    c = c / c.iloc[0]
    v = v / v.iloc[0]

    # Daily returns
    c_rets = c.pct_change().dropna()
    v_rets = v.pct_change().dropna()
    common2 = c_rets.index.intersection(v_rets.index)
    c_rets = c_rets.loc[common2]; v_rets = v_rets.loc[common2]

    # 상관
    corr = c_rets.corr(v_rets)
    print(f'\nCorrelation (daily returns): {corr:.4f}')

    # V21 단독 metrics
    m_v = metrics_daily(v, 'V21_spot')
    m_c = metrics_daily(c, 'C_daily')

    # 앙상블: 여러 비중
    print(f'\nV21 single: Sharpe={m_v["Sharpe"]} CAGR={m_v["CAGR"]:.2%} MDD={m_v["MDD"]:.2%} Cal={m_v["Cal"]}')
    print(f'C single:   Sharpe={m_c["Sharpe"]} CAGR={m_c["CAGR"]:.2%} MDD={m_c["MDD"]:.2%} Cal={m_c["Cal"]}')

    print(f'\n=== 앙상블 (V21 * w + C * (1-w)) ===')
    for w in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]:
        # 포트폴리오 수익: rebalance daily, weighted return
        port_rets = w * v_rets + (1 - w) * c_rets
        port_eq = (1 + port_rets).cumprod()
        m = metrics_daily(port_eq, f'{w:.0%}V21+{1-w:.0%}C')
        print(f'  V21 {w:.0%} + C {1-w:.0%}: Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} MDD={m["MDD"]:.2%} Cal={m["Cal"]}')


if __name__ == '__main__':
    main()
