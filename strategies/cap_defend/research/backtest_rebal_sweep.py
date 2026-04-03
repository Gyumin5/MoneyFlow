#!/usr/bin/env python3
"""통합 포트폴리오 리밸런싱 전략 sweep.

V17 코인+주식 NAV를 사전 계산한 뒤,
다양한 비중/주기/밴드/동적 규칙을 Sleeve 합성으로 비교.

NAV는 한 번만 계산, 합성만 반복 → 매우 빠름.
"""

import sys, os, time
import numpy as np, pandas as pd
from dataclasses import replace

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
from backtest_integrated_60_40 import (
    get_coin_nav, get_stock_nav,
    COIN_VERSIONS, STOCK_VERSIONS,
    check_crash_vt,
)
from coin_engine import (
    load_universe, load_all_prices, filter_universe, calc_metrics,
)
from stock_engine import (
    load_prices as load_stock_prices, precompute as stock_precompute,
    _init as stock_init, ALL_TICKERS,
)
import stock_engine as tsi


def simulate_sleeve_flex(stock_nav, coin_nav, config):
    """유연한 Sleeve 합성.

    config keys:
        stock_ratio: float (default 0.588)
        coin_ratio: float (default 0.392)
        rebal_freq: 'M'=매월, 'Q'=분기, 'S'=반기, 'N'=없음 (default 'M')
        band: float (0=무조건 리밸, 0.05=±5%p 밴드) (default 0)
        turnover_min: float (이 이하 턴오버는 무시) (default 0)
        tx_cost: float (리밸 편도 비용) (default 0.002)
    """
    stock_ratio = config.get('stock_ratio', 0.588)
    coin_ratio = config.get('coin_ratio', 0.392)
    rebal_freq = config.get('rebal_freq', 'M')
    band = config.get('band', 0)
    turnover_min = config.get('turnover_min', 0)
    tx_cost = config.get('tx_cost', 0.002)

    common = stock_nav.index.intersection(coin_nav.index)
    if len(common) < 100:
        return None

    s = stock_nav.loc[common] / stock_nav.loc[common].iloc[0]
    c = coin_nav.loc[common] / coin_nav.loc[common].iloc[0]

    cash_ratio = max(1.0 - stock_ratio - coin_ratio, 0)
    s_units = stock_ratio
    c_units = coin_ratio
    cash = cash_ratio
    prev_month = None
    prev_quarter = None
    prev_half = None
    values = []
    rebal_count = 0

    for date in common:
        s_val = s_units * s.loc[date]
        c_val = c_units * c.loc[date]
        total = s_val + c_val + cash

        cur_month = date.strftime('%Y-%m')
        cur_quarter = f"{date.year}-Q{(date.month-1)//3+1}"
        cur_half = f"{date.year}-H{1 if date.month <= 6 else 2}"

        # 리밸런싱 시점 판단
        do_rebal = False
        if rebal_freq == 'M' and prev_month is not None and cur_month != prev_month:
            do_rebal = True
        elif rebal_freq == 'Q' and prev_quarter is not None and cur_quarter != prev_quarter:
            do_rebal = True
        elif rebal_freq == 'S' and prev_half is not None and cur_half != prev_half:
            do_rebal = True

        if do_rebal and total > 0:
            cur_s_pct = s_val / total
            drift = abs(cur_s_pct - stock_ratio)

            # 밴드 체크
            if band > 0 and drift < band:
                do_rebal = False

            # 턴오버 최소 체크
            if do_rebal:
                cur_c_pct = c_val / total
                turnover = abs(cur_s_pct - stock_ratio) + abs(cur_c_pct - coin_ratio)
                if turnover_min > 0 and turnover < turnover_min:
                    do_rebal = False

            if do_rebal:
                cost = total * turnover * tx_cost / 2 if 'turnover' in dir() else 0
                total -= cost
                s_units = (total * stock_ratio) / s.loc[date]
                c_units = (total * coin_ratio) / c.loc[date]
                cash = total * cash_ratio
                rebal_count += 1

        s_val = s_units * s.loc[date]
        c_val = c_units * c.loc[date]
        total = s_val + c_val + cash
        values.append({'Date': date, 'Value': total})
        prev_month = cur_month
        prev_quarter = cur_quarter
        prev_half = cur_half

    df = pd.DataFrame(values).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    return df


def main():
    t0 = time.time()
    print("데이터 로딩...")

    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    coin_prices = load_all_prices(list(all_t))

    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)

    print(f"  완료 ({time.time()-t0:.1f}s)")

    # V17 NAV 사전 계산 (한 번만)
    start, end = '2018-01-01', '2025-06-30'
    print(f"\nV17 NAV 계산 ({start}~{end})...")

    cfg = COIN_VERSIONS['V17']
    cfg['params']['start_date'] = start
    cfg['params']['end_date'] = end
    coin_nav = get_coin_nav(coin_prices, coin_um[40], cfg['params'],
                             cfg['dd_lookback'], cfg['dd_threshold'],
                             cfg['bl_drop'], cfg['bl_days'],
                             cfg['drift_threshold'], cfg['post_flip_delay'])

    stock_nav = get_stock_nav(stock_prices, stock_ind,
                               STOCK_VERSIONS['V17'], start, end)

    print(f"  코인 NAV: {len(coin_nav)} days, 주식 NAV: {len(stock_nav)} days")

    # ═══ 전략 Sweep ═══
    strategies = {}

    # A. 비중
    for sp in [80, 70, 60, 50, 40, 30, 20]:
        cp = 98 - sp
        strategies[f'A: S{sp}:C{cp} 매월'] = dict(stock_ratio=sp/100, coin_ratio=cp/100)

    # B. 리밸런싱 주기 (60:40 고정)
    strategies['B: 60:40 매월'] = dict(rebal_freq='M')
    strategies['B: 60:40 분기'] = dict(rebal_freq='Q')
    strategies['B: 60:40 반기'] = dict(rebal_freq='S')
    strategies['B: 60:40 안함'] = dict(rebal_freq='N')

    # C. 밴드 (60:40, 매월 체크)
    strategies['C: 밴드 없음 (매월)'] = dict(band=0)
    strategies['C: ±3%p 밴드'] = dict(band=0.03)
    strategies['C: ±5%p 밴드'] = dict(band=0.05)
    strategies['C: ±10%p 밴드'] = dict(band=0.10)

    # D. 턴오버 최소 (60:40, 매월)
    strategies['D: 턴오버 제한 없음'] = dict(turnover_min=0)
    strategies['D: 턴오버 >5%'] = dict(turnover_min=0.05)
    strategies['D: 턴오버 >10%'] = dict(turnover_min=0.10)
    strategies['D: 턴오버 >20%'] = dict(turnover_min=0.20)

    # E. 리밸런싱 비용 민감도
    strategies['E: tx 0%'] = dict(tx_cost=0)
    strategies['E: tx 0.1%'] = dict(tx_cost=0.001)
    strategies['E: tx 0.2% (기본)'] = dict(tx_cost=0.002)
    strategies['E: tx 0.5%'] = dict(tx_cost=0.005)

    print(f"\n전략 sweep: {len(strategies)}개")
    print(f"{'='*95}")
    print(f"{'전략':<28s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'Rebal':>6s}")
    print(f"{'-'*70}")

    for name, cfg in strategies.items():
        full_cfg = dict(stock_ratio=0.588, coin_ratio=0.392, rebal_freq='M',
                        band=0, turnover_min=0, tx_cost=0.002)
        full_cfg.update(cfg)

        df = simulate_sleeve_flex(stock_nav, coin_nav, full_cfg)
        if df is None:
            print(f"  {name:<26s}  실패")
            continue

        m = calc_metrics(df)
        s = m.get('Sharpe', 0)
        c = m.get('CAGR', 0)
        mdd = m.get('MDD', 0)
        cal = c / abs(mdd) if mdd != 0 else 0
        rebal = df.attrs.get('rebal_count', 0)
        print(f"  {name:<26s} {s:>7.3f} {c:>+8.1%} {mdd:>+8.1%} {cal:>7.2f} {rebal:>6d}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
