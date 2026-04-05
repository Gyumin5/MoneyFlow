#!/usr/bin/env python3
"""자산배분 평가: 주식 60% / 현물 30% / 선물 10% 월간 리밸런싱.

각 자산군의 equity curve를 구한 뒤, 60/30/10 비중으로 월초 리밸런싱.
비교:
  1. 주식 100%
  2. 현물 100%
  3. 선물 100%
  4. 주식60 + 현물40 (현행 V18)
  5. 주식60 + 현물30 + 선물10 (신규)
  6. 주식50 + 현물30 + 선물20
"""

import os, sys, time
import numpy as np, pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# === Imports ===
from coin_engine import (
    load_universe, load_all_prices, filter_universe, calc_metrics,
    DEFENSE_TICKERS,
)
from coin_helpers import B
from backtest_official import run_coin_backtest, COIN_VERSIONS, STOCK_VERSIONS
from stock_engine import (
    SP, load_prices as load_stock_prices, precompute as stock_precompute,
    _init as stock_init, _run_one, run_bt, ALL_TICKERS,
)
import stock_engine as tsi
from backtest_official import check_crash_vt
from backtest_futures_full import load_data
from futures_live_config import CURRENT_STRATEGIES, START as FUT_START, END as FUT_END
from futures_ensemble_engine import SingleAccountEngine, combine_targets


# 선물 엔진 공통 파라미터
ENGINE_KWARGS = dict(
    leverage=5.0,
    stop_kind="prev_close_pct",
    stop_pct=0.15,
    stop_gate="cash_guard",
    stop_gate_cash_threshold=0.34,
    per_coin_leverage_mode="cap_mom_blend_543_cash",
    leverage_floor=3.0,
    leverage_mid=4.0,
    leverage_ceiling=5.0,
    leverage_cash_threshold=0.34,
    leverage_partial_cash_threshold=0.0,
    leverage_count_floor_max=2,
    leverage_count_mid_max=4,
    leverage_canary_floor_gap=0.015,
    leverage_canary_mid_gap=0.04,
    leverage_canary_high_gap=0.08,
    leverage_canary_sma_bars=1200,
    leverage_mom_lookback_bars=24 * 30,
    leverage_vol_lookback_bars=24 * 90,
)

# 현행 실거래 전략 (4h1+4h2+1h1)
LIVE_COMBO = {
    "live_1h1": 1/3,
    "live_4h1": 1/3,
    "live_4h2": 1/3,
}

# d0.03 제외 추천 후보 (4전략)
FIXED_BASE = dict(
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    n_snapshots=3,
)

CANDIDATE_STRATEGIES = {
    "4h_d005": dict(
        interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        snap_interval_bars=60, **FIXED_BASE),
    "2h_b60_S240": dict(
        interval="2h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=120, **FIXED_BASE),
    "2h_b60_S120": dict(
        interval="2h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=120, **FIXED_BASE),
    "4h_b60_M20_120": dict(
        interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=21, **FIXED_BASE),
}

CANDIDATE_COMBO = {k: 0.25 for k in CANDIDATE_STRATEGIES}


def generate_trace(data, cfg, start, end):
    from backtest_futures_full import run
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=start, end_date=end, _trace=trace, **run_cfg)
    return trace


def get_coin_equity(prices, um, start, end):
    """V18 코인 equity curve (daily)."""
    cfg = COIN_VERSIONS['V18']
    top_n = cfg['params'].get('top_n', 40)
    p = B(**cfg['params'])
    p.start_date = start
    p.end_date = end
    r = run_coin_backtest(prices, um.get(top_n, um[40]), (1, 10, 19),
                          dd_lookback=cfg['dd_lookback'], dd_threshold=cfg['dd_threshold'],
                          bl_drop=cfg['bl_drop'], bl_days=cfg['bl_days'],
                          drift_threshold=cfg['drift_threshold'],
                          post_flip_delay=cfg['post_flip_delay'],
                          params_base=p, defense=cfg.get('defense', False))
    return r['equity_curve']


def get_stock_equity(start, end):
    """V17 주식 equity curve (daily, 4-tranche)."""
    base = STOCK_VERSIONS['V17']
    sp = replace(base, start=start, end=end,
                 _n_tranches=4, tranche_days=(1, 8, 15, 22))
    # Run each tranche
    from stock_engine import run_bt as srun_bt
    dfs = []
    for day in sp.tranche_days:
        tp = SP(**{k: v for k, v in sp.__dict__.items()})
        tp._anchor = day
        tp._n_tranches = 1
        tp.capital = sp.capital / len(sp.tranche_days)
        df = srun_bt(tsi._g_prices, tsi._g_ind, tp)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    merged = dfs[0].copy()
    for df in dfs[1:]:
        common = merged.index.intersection(df.index)
        merged.loc[common, 'Value'] += df.loc[common, 'Value']
    return merged['Value']


def get_futures_equity(data, strategies, combo, start, end):
    """선물 앙상블 equity curve (hourly → daily resample)."""
    intervals = set()
    for cfg in strategies.values():
        intervals.add(cfg['interval'])

    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[
        (bars_1h['BTC'].index >= start) & (bars_1h['BTC'].index <= end)]

    traces = {name: generate_trace(data, cfg, start, end)
              for name, cfg in strategies.items()}
    combined = combine_targets({k: traces[k] for k in combo}, combo, all_dates)

    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    m = engine.run(combined)
    eq = m['_equity']
    # daily resample
    eq_daily = eq.resample('D').last().dropna()
    return eq_daily, m


def monthly_rebalance(equity_curves, weights, start, end):
    """월간 리밸런싱 포트폴리오.

    equity_curves: dict of {name: pd.Series (daily, start=capital)}
    weights: dict of {name: float} (합 = 1.0)
    """
    # Normalize all to daily returns
    returns = {}
    for name, eq in equity_curves.items():
        eq = eq.sort_index()
        # daily returns
        ret = eq.pct_change().fillna(0)
        returns[name] = ret

    # Common dates
    common = None
    for name, ret in returns.items():
        if common is None:
            common = ret.index
        else:
            common = common.intersection(ret.index)
    common = common.sort_values()
    common = common[(common >= start) & (common <= end)]

    if len(common) < 2:
        return None, None

    # Initialize
    capital = 10000.0
    allocations = {name: capital * w for name, w in weights.items()}
    portfolio_values = []
    prev_month = None

    for date in common:
        cur_month = date.strftime('%Y-%m')

        # Monthly rebalance at month start
        if prev_month is not None and cur_month != prev_month:
            total = sum(allocations.values())
            allocations = {name: total * w for name, w in weights.items()}

        # Apply daily return
        for name in allocations:
            r = returns[name].get(date, 0)
            allocations[name] *= (1 + r)

        total = sum(allocations.values())
        portfolio_values.append({'Date': date, 'Value': total})
        prev_month = cur_month

    df = pd.DataFrame(portfolio_values).set_index('Date')
    eq = df['Value']
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[-1] <= 0:
        return eq, None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return eq, {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Calmar': cal}


def main():
    t0 = time.time()

    # === Common period ===
    # 선물 데이터 2020-10-01 시작 → 공통 구간
    START = '2020-10-01'
    END = '2026-03-28'

    print("=" * 80)
    print("자산배분 평가: 주식 / 현물코인 / 선물 앙상블")
    print(f"기간: {START} ~ {END}")
    print("=" * 80)

    # === Load data ===
    print("\n데이터 로딩...")

    # Coin
    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    all_t.update(DEFENSE_TICKERS)
    coin_prices = load_all_prices(list(all_t))

    # Stock
    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt

    # Futures
    fut_intervals = set()
    for cfg in CURRENT_STRATEGIES.values():
        fut_intervals.add(cfg['interval'])
    for cfg in CANDIDATE_STRATEGIES.values():
        fut_intervals.add(cfg['interval'])
    fut_data = {iv: load_data(iv) for iv in fut_intervals}

    print(f"  데이터 로딩 완료 ({time.time()-t0:.1f}s)")

    # === Get equity curves ===
    print("\n개별 equity curve 생성...")

    # Coin (V18)
    coin_eq = get_coin_equity(coin_prices, coin_um, START, END)
    coin_ret = coin_eq.pct_change().dropna()
    coin_cagr = (coin_eq.iloc[-1] / coin_eq.iloc[0]) ** (
        1 / ((coin_eq.index[-1] - coin_eq.index[0]).days / 365.25)) - 1
    coin_mdd = (coin_eq / coin_eq.cummax() - 1).min()
    coin_sh = coin_ret.mean() / coin_ret.std() * np.sqrt(365) if coin_ret.std() > 0 else 0
    print(f"  코인(V18):   Sharpe={coin_sh:.3f}  CAGR={coin_cagr:+.1%}  MDD={coin_mdd:+.1%}")

    # Stock (V17, 4-tranche)
    stock_eq = get_stock_equity(START, END)
    stock_ret = stock_eq.pct_change().dropna()
    stock_cagr = (stock_eq.iloc[-1] / stock_eq.iloc[0]) ** (
        1 / ((stock_eq.index[-1] - stock_eq.index[0]).days / 365.25)) - 1
    stock_mdd = (stock_eq / stock_eq.cummax() - 1).min()
    stock_sh = stock_ret.mean() / stock_ret.std() * np.sqrt(252) if stock_ret.std() > 0 else 0
    print(f"  주식(V17):   Sharpe={stock_sh:.3f}  CAGR={stock_cagr:+.1%}  MDD={stock_mdd:+.1%}")

    # Futures - current live
    fut_live_eq, fut_live_m = get_futures_equity(
        fut_data, CURRENT_STRATEGIES, LIVE_COMBO, START, END)
    print(f"  선물(현행):  Sharpe={fut_live_m['Sharpe']:.3f}  CAGR={fut_live_m['CAGR']:+.1%}  "
          f"MDD={fut_live_m['MDD']:+.1%}  Liq={fut_live_m['Liq']}")

    # Futures - candidate
    fut_cand_eq, fut_cand_m = get_futures_equity(
        fut_data, CANDIDATE_STRATEGIES, CANDIDATE_COMBO, START, END)
    print(f"  선물(후보):  Sharpe={fut_cand_m['Sharpe']:.3f}  CAGR={fut_cand_m['CAGR']:+.1%}  "
          f"MDD={fut_cand_m['MDD']:+.1%}  Liq={fut_cand_m['Liq']}")

    print(f"\n  equity curve 생성 완료 ({time.time()-t0:.1f}s)")

    # === Portfolio combinations ===
    print("\n" + "=" * 80)
    print("포트폴리오 비교 (월간 리밸런싱)")
    print("=" * 80)

    curves_live = {'stock': stock_eq, 'coin': coin_eq, 'futures': fut_live_eq}
    curves_cand = {'stock': stock_eq, 'coin': coin_eq, 'futures': fut_cand_eq}

    combos = [
        ("주식 100%",                  {'stock': 1.0},                         curves_live),
        ("현물 100%",                  {'coin': 1.0},                          curves_live),
        ("선물 100% (현행)",           {'futures': 1.0},                       curves_live),
        ("선물 100% (후보)",           {'futures': 1.0},                       curves_cand),
        ("주식60+현물40 (현행배분)",    {'stock': 0.6, 'coin': 0.4},           curves_live),
        ("주식60+현물30+선물10(현행)", {'stock': 0.6, 'coin': 0.3, 'futures': 0.1}, curves_live),
        ("주식60+현물30+선물10(후보)", {'stock': 0.6, 'coin': 0.3, 'futures': 0.1}, curves_cand),
        ("주식50+현물30+선물20(현행)", {'stock': 0.5, 'coin': 0.3, 'futures': 0.2}, curves_live),
        ("주식50+현물30+선물20(후보)", {'stock': 0.5, 'coin': 0.3, 'futures': 0.2}, curves_cand),
        ("주식40+현물30+선물30(현행)", {'stock': 0.4, 'coin': 0.3, 'futures': 0.3}, curves_live),
        ("주식40+현물30+선물30(후보)", {'stock': 0.4, 'coin': 0.3, 'futures': 0.3}, curves_cand),
    ]

    print(f"\n  {'포트폴리오':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
    print(f"  {'-'*70}")

    for name, wt, curves in combos:
        _, m = monthly_rebalance(curves, wt, START, END)
        if m:
            print(f"  {name:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f}")
        else:
            print(f"  {name:<35s}   --- 계산 실패 ---")

    # === Sub-period analysis ===
    print("\n" + "=" * 80)
    print("서브기간 분석")
    print("=" * 80)

    mid = '2023-07-01'
    periods = [
        (f"전반 ({START}~{mid})", START, mid),
        (f"후반 ({mid}~{END})", mid, END),
    ]

    key_combos = [
        ("주식60+현물40 (현행)",       {'stock': 0.6, 'coin': 0.4},           curves_live),
        ("주식60+현물30+선물10(현행)", {'stock': 0.6, 'coin': 0.3, 'futures': 0.1}, curves_live),
        ("주식60+현물30+선물10(후보)", {'stock': 0.6, 'coin': 0.3, 'futures': 0.1}, curves_cand),
        ("주식50+현물30+선물20(후보)", {'stock': 0.5, 'coin': 0.3, 'futures': 0.2}, curves_cand),
    ]

    for period_name, ps, pe in periods:
        print(f"\n  [{period_name}]")
        print(f"  {'포트폴리오':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
        print(f"  {'-'*70}")
        for name, wt, curves in key_combos:
            _, m = monthly_rebalance(curves, wt, ps, pe)
            if m:
                print(f"  {name:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f}")

    # === Correlation analysis ===
    print("\n" + "=" * 80)
    print("일간 수익률 상관관계")
    print("=" * 80)

    common = stock_eq.index.intersection(coin_eq.index).intersection(fut_live_eq.index)
    common = common.sort_values()
    rets = pd.DataFrame({
        'Stock': stock_eq.reindex(common).pct_change(),
        'Coin': coin_eq.reindex(common).pct_change(),
        'Futures(live)': fut_live_eq.reindex(common).pct_change(),
        'Futures(cand)': fut_cand_eq.reindex(common).pct_change(),
    }).dropna()

    print(f"\n{rets.corr().round(3).to_string()}")

    # === Rebalancing benefit ===
    print("\n" + "=" * 80)
    print("리밸런싱 효과 (60/30/10 후보)")
    print("=" * 80)

    # Without rebalancing (buy-and-hold)
    common_all = stock_eq.index.intersection(coin_eq.index).intersection(fut_cand_eq.index)
    common_all = common_all.sort_values()
    common_all = common_all[(common_all >= START) & (common_all <= END)]

    if len(common_all) > 2:
        s_norm = stock_eq.reindex(common_all) / stock_eq.reindex(common_all).iloc[0]
        c_norm = coin_eq.reindex(common_all) / coin_eq.reindex(common_all).iloc[0]
        f_norm = fut_cand_eq.reindex(common_all) / fut_cand_eq.reindex(common_all).iloc[0]
        bnh = (s_norm * 0.6 + c_norm * 0.3 + f_norm * 0.1) * 10000
        bnh_yrs = (bnh.index[-1] - bnh.index[0]).days / 365.25
        bnh_cagr = (bnh.iloc[-1] / bnh.iloc[0]) ** (1/bnh_yrs) - 1
        bnh_mdd = (bnh / bnh.cummax() - 1).min()
        bnh_dr = bnh.pct_change().dropna()
        bnh_sh = bnh_dr.mean() / bnh_dr.std() * np.sqrt(365) if bnh_dr.std() > 0 else 0

        _, rebal_m = monthly_rebalance(curves_cand, {'stock': 0.6, 'coin': 0.3, 'futures': 0.1}, START, END)

        print(f"\n  {'방식':<25s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
        print(f"  {'-'*55}")
        print(f"  {'BnH (리밸런싱 없음)':<25s} {bnh_sh:>7.3f} {bnh_cagr:>+8.1%} {bnh_mdd:>+8.1%} {bnh_cagr/abs(bnh_mdd) if bnh_mdd != 0 else 0:>7.2f}")
        if rebal_m:
            print(f"  {'월간 리밸런싱':<25s} {rebal_m['Sharpe']:>7.3f} {rebal_m['CAGR']:>+8.1%} {rebal_m['MDD']:>+8.1%} {rebal_m['Calmar']:>7.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
