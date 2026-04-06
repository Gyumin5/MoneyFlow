#!/usr/bin/env python3
"""V12~V19 포트폴리오 백테스트 — 밴드 리밸런싱 포함.

V12~V18: 주식60 / 코인40, 밴드 0~12% 스윕
V19: 주식60 / 코인25 / 선물15, 밴드 0~12% 스윕
"""

import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, filter_universe, load_all_prices
from coin_helpers import B
from backtest_official import (
    run_coin_backtest, COIN_VERSIONS, STOCK_VERSIONS, DEFENSE_TICKERS,
    check_crash_vt,
)
from stock_engine import (
    SP, load_prices as load_stock_prices,
    precompute as stock_precompute,
    _init as stock_init, run_bt, ALL_TICKERS,
)
import stock_engine as tsi
from dataclasses import replace


def band_rebalance(asset_returns, target_weights, band_pp, tx_cost=0.001):
    """밴드 리밸런싱 포트폴리오 시뮬레이션.

    Args:
        asset_returns: dict of {name: pd.Series} daily returns (pct_change)
        target_weights: dict of {name: float}
        band_pp: 리밸런싱 밴드 (0.08 = 8%p). 0이면 매일 리밸런싱.
        tx_cost: 리밸런싱 시 거래비용 (편도)

    Returns:
        equity: pd.Series, rebal_count: int
    """
    names = list(target_weights.keys())
    tw = np.array([target_weights[n] for n in names])

    # Align all return series to common dates
    common_idx = asset_returns[names[0]].index
    for n in names[1:]:
        common_idx = common_idx.intersection(asset_returns[n].index)
    common_idx = common_idx.sort_values()

    if len(common_idx) < 10:
        return None, 0

    # Initialize
    portfolio_value = 1.0
    holdings = tw * portfolio_value  # dollar amount per asset
    equity = []
    rebal_count = 0

    for i, date in enumerate(common_idx):
        if i == 0:
            equity.append(portfolio_value)
            continue

        # Apply daily returns
        for j, n in enumerate(names):
            r = asset_returns[n].loc[date]
            if not np.isnan(r):
                holdings[j] *= (1 + r)

        portfolio_value = holdings.sum()
        if portfolio_value <= 0:
            equity.append(0)
            continue

        # Check drift
        current_weights = holdings / portfolio_value
        drift = np.abs(current_weights - tw)
        max_drift = drift.max()

        if band_pp == 0 or max_drift >= band_pp:
            # Rebalance: calculate turnover
            new_holdings = tw * portfolio_value
            turnover = np.abs(new_holdings - holdings).sum() / 2  # one-way
            cost = turnover * tx_cost
            portfolio_value -= cost
            holdings = tw * portfolio_value
            if i > 0:  # don't count initial allocation
                rebal_count += 1

        equity.append(portfolio_value)

    eq = pd.Series(equity, index=common_idx)
    return eq, rebal_count


def full_metrics(eq):
    """확장 지표."""
    if eq is None or len(eq) < 10:
        return None
    days = (eq.index[-1] - eq.index[0]).days
    yrs = days / 365.25
    if yrs <= 0:
        return None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax()
    dd = eq / peak - 1
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    dr = eq.pct_change().dropna()
    ann = np.sqrt(365)
    sharpe = dr.mean() / dr.std() * ann if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = dr.mean() / down.std() * ann if len(down) > 1 and down.std() > 0 else sharpe
    yearly = eq.resample('A').last().pct_change().dropna()
    worst_yr = yearly.min() if len(yearly) > 0 else 0
    return {
        'CAGR': cagr, 'MDD': mdd, 'Sharpe': sharpe, 'Sortino': sortino,
        'Calmar': calmar, 'WorstYr': worst_yr,
    }


def get_coin_equity(ver, coin_prices, coin_um, start, end):
    if ver not in COIN_VERSIONS:
        return None
    cfg = COIN_VERSIONS[ver]
    top_n = cfg['params'].get('top_n', 40)
    um = coin_um.get(top_n, coin_um[40])
    p = B(**cfg['params'])
    p.start_date = start
    p.end_date = end
    r = run_coin_backtest(coin_prices, um, (1, 10, 19),
                          dd_lookback=cfg['dd_lookback'], dd_threshold=cfg['dd_threshold'],
                          bl_drop=cfg['bl_drop'], bl_days=cfg['bl_days'],
                          drift_threshold=cfg['drift_threshold'],
                          post_flip_delay=cfg['post_flip_delay'],
                          params_base=p, defense=cfg.get('defense', False))
    return r.get('equity_curve')


def get_stock_equity(ver, start, end):
    if ver not in STOCK_VERSIONS:
        return None
    from stock_engine import _g_prices, _g_ind
    base = STOCK_VERSIONS[ver]
    sp = replace(base, start=start, end=end)

    # 11-anchor 평균
    equities = []
    for a in range(1, 12):
        tp = replace(sp, _anchor=a, _n_tranches=1)
        df = run_bt(_g_prices, _g_ind, tp)
        if df is not None:
            equities.append(df['Value'])

    if hasattr(sp, 'tranche_days') and sp.tranche_days:
        equities = []
        for a in range(1, 12):
            tp = replace(sp, _anchor=a)
            dfs = []
            for day in tp.tranche_days:
                ttp = replace(tp, _anchor=day, _n_tranches=1)
                ttp.capital = tp.capital / len(tp.tranche_days)
                df = run_bt(_g_prices, _g_ind, ttp)
                if df is not None:
                    dfs.append(df)
            if dfs:
                merged = dfs[0]['Value'].copy()
                for df in dfs[1:]:
                    common = merged.index.intersection(df.index)
                    merged.loc[common] += df.loc[common, 'Value']
                equities.append(merged)

    if not equities:
        return None
    common_idx = equities[0].index
    for eq in equities[1:]:
        common_idx = common_idx.intersection(eq.index)
    if len(common_idx) < 10:
        return None
    avg = pd.Series(0.0, index=common_idx)
    for eq in equities:
        avg += eq.loc[common_idx]
    avg /= len(equities)
    return avg


def get_futures_equity():
    from backtest_futures_full import load_data as load_futures_data, run as frun
    from futures_live_config import CURRENT_STRATEGIES, CURRENT_LIVE_COMBO
    from futures_live_config import START as F_START, END as F_END
    from futures_ensemble_engine import SingleAccountEngine, combine_targets

    fdata = {iv: load_futures_data(iv) for iv in ["4h", "2h", "1h"]}
    bars_1h, funding_1h = fdata["1h"]
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= F_START) & (bars_1h["BTC"].index <= F_END)
    ]
    traces = {}
    for name, cfg in CURRENT_STRATEGIES.items():
        rc = dict(cfg)
        iv = rc.pop("interval")
        bars, funding = fdata[iv]
        trace = []
        frun(bars, funding, interval=iv, leverage=1.0,
             start_date=F_START, end_date=F_END, _trace=trace, **rc)
        traces[name] = trace
    combined = combine_targets(
        {k: traces[k] for k in CURRENT_LIVE_COMBO}, CURRENT_LIVE_COMBO, all_dates
    )
    engine = SingleAccountEngine(
        bars_1h, funding_1h, leverage=5.0,
        stop_kind="prev_close_pct", stop_pct=0.15,
        stop_gate="cash_guard", stop_gate_cash_threshold=0.34,
        per_coin_leverage_mode="cap_mom_blend_543_cash",
        leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
        leverage_cash_threshold=0.34, leverage_partial_cash_threshold=0.0,
        leverage_count_floor_max=2, leverage_count_mid_max=4,
        leverage_canary_floor_gap=0.015, leverage_canary_mid_gap=0.04,
        leverage_canary_high_gap=0.08, leverage_canary_sma_bars=1200,
        leverage_mom_lookback_bars=24*30, leverage_vol_lookback_bars=24*90,
    )
    m = engine.run(combined)
    return m.get('_equity')


def eq_to_daily(eq):
    """Equity curve → 일봉 리샘플링 (시간봉이면 일봉으로 변환)."""
    if eq is None or len(eq) < 2:
        return eq
    # 시간봉 인덱스인지 확인 (하루에 여러 행)
    if len(eq) > (eq.index[-1] - eq.index[0]).days * 1.5:
        # 일봉보다 빈번 → 일봉 마지막 값으로 리샘플
        daily = eq.resample('D').last().dropna()
        return daily
    return eq


def eq_to_returns(eq):
    """Equity curve → daily returns."""
    daily = eq_to_daily(eq)
    return daily.pct_change().fillna(0)


def print_row(label, m, rebal):
    if m is None:
        print(f"  {label:<28s}  {'N/A':>50s}")
        return
    print(f"  {label:<28s} {m['Sharpe']:>6.2f} {m['Sortino']:>7.2f} {m['CAGR']:>+7.1%} "
          f"{m['MDD']:>+7.1%} {m['Calmar']:>6.2f} {m['WorstYr']:>+7.1%} {rebal:>5d}")


def main():
    t0 = time.time()
    print("데이터 로딩...")

    # Coin
    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values():
            all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    all_t.update(DEFENSE_TICKERS)
    coin_prices = load_all_prices(list(all_t))

    # Stock
    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt

    print(f"  완료 ({time.time()-t0:.1f}s)")

    COIN_START, COIN_END = '2018-01-01', '2025-06-30'
    STOCK_START, STOCK_END = '2018-01-01', '2025-12-31'

    BANDS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 1.0]  # 1.0 = never rebal (B&H)

    header = (f"  {'전략':<28s} {'Sharpe':>6s} {'Sortino':>7s} {'CAGR':>7s} "
              f"{'MDD':>7s} {'Calmar':>6s} {'최악년':>7s} {'리밸':>5s}")
    sep = "  " + "-" * 80

    # ═══ V12~V18: 주식60/코인40 밴드 스윕 ═══
    versions_6040 = ['V14', 'V17', 'V18']  # 주요 버전만

    print("\n" + "=" * 90)
    print("V14/V17/V18 — 주식60/코인40 밴드 스윕 [2018~2025]")
    print("=" * 90)

    for ver in versions_6040:
        print(f"\n  ── {ver} ──")
        print(header)
        print(sep)

        coin_eq = get_coin_equity(ver, coin_prices, coin_um, COIN_START, COIN_END)
        stock_eq = get_stock_equity(ver, STOCK_START, STOCK_END)

        if coin_eq is None or stock_eq is None:
            print(f"  {ver}: 데이터 없음")
            continue

        coin_ret = eq_to_returns(coin_eq)
        stock_ret = eq_to_returns(stock_eq)

        for band in BANDS:
            asset_rets = {'stock': stock_ret, 'coin': coin_ret}
            target_w = {'stock': 0.60, 'coin': 0.40}
            tx = 0.003  # 주식0.1% + 코인0.4% 가중평균 ≈ 0.3%
            eq, rc = band_rebalance(asset_rets, target_w, band, tx_cost=tx)
            m = full_metrics(eq)
            label = f"band {band:.0%}" if band < 1.0 else "B&H (리밸없음)"
            print_row(label, m, rc)

    # ═══ V19: 주식60/코인25/선물15 밴드 스윕 ═══
    print("\n\n" + "=" * 90)
    print("V19 — 주식60/코인25/선물15 밴드 스윕 [2020.10~2025.06]")
    print("  (선물 데이터가 2020.10부터이므로 공통 구간 사용)")
    print("=" * 90)
    print(header)
    print(sep)

    coin_eq_v19 = get_coin_equity('V19', coin_prices, coin_um, '2020-10-01', '2025-06-30')
    stock_eq_v19 = get_stock_equity('V19', '2020-10-01', '2025-12-31')
    futures_eq = get_futures_equity()

    if coin_eq_v19 is not None and stock_eq_v19 is not None and futures_eq is not None:
        coin_ret = eq_to_returns(coin_eq_v19)
        stock_ret = eq_to_returns(stock_eq_v19)
        futures_ret = eq_to_returns(futures_eq)

        for band in BANDS:
            asset_rets = {'stock': stock_ret, 'coin': coin_ret, 'futures': futures_ret}
            target_w = {'stock': 0.60, 'coin': 0.25, 'futures': 0.15}
            tx = 0.002  # 가중평균
            eq, rc = band_rebalance(asset_rets, target_w, band, tx_cost=tx)
            m = full_metrics(eq)
            label = f"band {band:.0%}" if band < 1.0 else "B&H (리밸없음)"
            print_row(label, m, rc)

        # 60:40 비교용
        print(f"\n  ── V19 60:40 (선물 제외) 비교 ──")
        print(header)
        print(sep)
        for band in [0.0, 0.08, 1.0]:
            asset_rets = {'stock': stock_ret, 'coin': coin_ret}
            target_w = {'stock': 0.60, 'coin': 0.40}
            eq, rc = band_rebalance(asset_rets, target_w, band, tx_cost=0.003)
            m = full_metrics(eq)
            label = f"60/40 band {band:.0%}" if band < 1.0 else "60/40 B&H"
            print_row(label, m, rc)

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
