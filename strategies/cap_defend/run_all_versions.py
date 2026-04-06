#!/usr/bin/env python3
"""V12~V19 전 버전 비교 + 통합 포트폴리오 성과.

V12~V18: 코인/주식 단독 + 60:40 합산
V19: 코인/주식/선물 단독 + 60:25:15 합산
"""

import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import calc_metrics, load_universe, filter_universe, load_all_prices
from coin_helpers import B
from backtest_official import (
    run_coin_backtest, COIN_VERSIONS, STOCK_VERSIONS, DEFENSE_TICKERS,
    check_crash_vt,
)
from stock_engine import (
    SP, load_prices as load_stock_prices,
    precompute as stock_precompute,
    _init as stock_init, _run_one, run_bt, ALL_TICKERS,
)
import stock_engine as tsi
from dataclasses import replace


def full_metrics(eq):
    """확장 지표 계산 (일별 equity Series)."""
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

    # Max drawdown duration (days)
    in_dd = dd < 0
    dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
    dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)
    max_dd_dur = 0
    starts = dd_starts[dd_starts].index
    ends = dd_ends[dd_ends].index
    for s in starts:
        e_candidates = ends[ends > s]
        if len(e_candidates) > 0:
            dur = (e_candidates[0] - s).days
            max_dd_dur = max(max_dd_dur, dur)
        else:
            dur = (eq.index[-1] - s).days
            max_dd_dur = max(max_dd_dur, dur)

    # Win rate (daily)
    win_rate = (dr > 0).sum() / len(dr) if len(dr) > 0 else 0

    # Worst year return
    yearly = eq.resample('A').last().pct_change().dropna()
    worst_yr = yearly.min() if len(yearly) > 0 else 0

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1

    return {
        'CAGR': cagr, 'MDD': mdd, 'Sharpe': sharpe, 'Sortino': sortino,
        'Calmar': calmar, 'TotalRet': total_ret,
        'MaxDD_Days': max_dd_dur, 'WinRate': win_rate, 'WorstYr': worst_yr,
    }


def run_coin_version(ver, coin_prices, coin_um, start, end):
    """코인 백테스트 → equity curve 반환."""
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


def run_stock_version(ver, start, end):
    """주식 백테스트 (11-anchor 평균) → equity curve 반환."""
    if ver not in STOCK_VERSIONS:
        return None
    base = STOCK_VERSIONS[ver]
    sp = replace(base, start=start, end=end)

    # 11-anchor 평균 equity curve
    all_eq = []
    for a in range(1, 12):
        tp = replace(sp, _anchor=a)
        df_result = _run_one(tp)
        if df_result is None:
            continue
        # _run_one returns metrics dict, need equity curve
        # Re-run to get equity curve directly
        pass

    # _run_one returns metrics, not equity. Run via run_bt instead.
    from stock_engine import _g_prices, _g_ind
    equities = []
    for a in range(1, 12):
        tp = replace(sp, _anchor=a)
        tp._n_tranches = 1
        df = run_bt(_g_prices, _g_ind, tp)
        if df is not None:
            equities.append(df['Value'])

    if not equities:
        return None

    # 4-tranche version
    if hasattr(sp, 'tranche_days') and sp.tranche_days:
        equities = []
        for a in range(1, 12):
            tp = replace(sp, _anchor=a)
            # Run tranche version manually
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

    # Align and average
    common_idx = equities[0].index
    for eq in equities[1:]:
        common_idx = common_idx.intersection(eq.index)
    if len(common_idx) < 10:
        return None

    avg_eq = pd.Series(0.0, index=common_idx)
    for eq in equities:
        avg_eq += eq.loc[common_idx]
    avg_eq /= len(equities)
    return avg_eq


def run_futures_v19():
    """선물 V19 백테스트 → equity curve 반환."""
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
    eq = m.get('_equity')
    return eq


def combine_portfolio(equities, weights):
    """여러 equity curve를 비중대로 합산."""
    # Normalize all to start at 1.0, then weight
    normed = {}
    common_idx = None
    for name, eq in equities.items():
        if eq is None:
            continue
        n = eq / eq.iloc[0]
        normed[name] = n
        if common_idx is None:
            common_idx = n.index
        else:
            common_idx = common_idx.intersection(n.index)

    if common_idx is None or len(common_idx) < 10:
        return None

    port = pd.Series(0.0, index=common_idx)
    for name, n in normed.items():
        w = weights.get(name, 0)
        port += n.loc[common_idx] * w
    return port


def print_row(label, m):
    if m is None:
        print(f"  {label:<22s}  {'N/A':>60s}")
        return
    print(f"  {label:<22s} {m['Sharpe']:>6.2f} {m['Sortino']:>7.2f} {m['CAGR']:>+7.1%} "
          f"{m['MDD']:>+7.1%} {m['Calmar']:>6.2f} {m['TotalRet']:>+8.0%} "
          f"{m['MaxDD_Days']:>5.0f}  {m['WinRate']:>5.1%} {m['WorstYr']:>+7.1%}")


def main():
    t0 = time.time()
    print("데이터 로딩...")

    # Coin data
    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values():
            all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    all_t.update(DEFENSE_TICKERS)
    coin_prices = load_all_prices(list(all_t))

    # Stock data
    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt

    print(f"  완료 ({time.time()-t0:.1f}s)")

    # Period: 공통 비교 구간
    COIN_START, COIN_END = '2018-01-01', '2025-06-30'
    STOCK_START, STOCK_END = '2018-01-01', '2025-12-31'

    versions = ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19']

    header = (f"  {'전략':<22s} {'Sharpe':>6s} {'Sortino':>7s} {'CAGR':>7s} "
              f"{'MDD':>7s} {'Calmar':>6s} {'총수익':>8s} "
              f"{'DD일':>5s}  {'승률':>5s} {'최악년':>7s}")
    sep = "  " + "-" * 95

    # ═══ 1. 코인 단독 ═══
    print("\n" + "=" * 100)
    print(f"코인 전략 단독 [{COIN_START} ~ {COIN_END}]")
    print("=" * 100)
    print(header)
    print(sep)

    coin_equities = {}
    for ver in versions:
        coin_ver = ver if ver != 'V19' else 'V19'
        eq = run_coin_version(coin_ver, coin_prices, coin_um, COIN_START, COIN_END)
        coin_equities[ver] = eq
        m = full_metrics(eq) if eq is not None else None
        print_row(f"{ver} 코인", m)

    # ═══ 2. 주식 단독 ═══
    print("\n" + "=" * 100)
    print(f"주식 전략 단독 [{STOCK_START} ~ {STOCK_END}]")
    print("=" * 100)
    print(header)
    print(sep)

    stock_equities = {}
    for ver in versions:
        eq = run_stock_version(ver, STOCK_START, STOCK_END)
        stock_equities[ver] = eq
        m = full_metrics(eq) if eq is not None else None
        print_row(f"{ver} 주식", m)

    # ═══ 3. 선물 단독 (V19만) ═══
    print("\n" + "=" * 100)
    print("선물 전략 단독 [2020-10-01 ~ 2026-03-28] (V19)")
    print("=" * 100)
    print(header)
    print(sep)

    futures_eq = run_futures_v19()
    futures_m = full_metrics(futures_eq) if futures_eq is not None else None
    print_row("V19 선물(d005,5x)", futures_m)

    # ═══ 4. 통합 포트폴리오 ═══
    print("\n" + "=" * 100)
    print("통합 포트폴리오")
    print("=" * 100)
    print(header)
    print(sep)

    # V12~V18: 60:40 (주식:코인)
    for ver in ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18']:
        port_eq = combine_portfolio(
            {'stock': stock_equities.get(ver), 'coin': coin_equities.get(ver)},
            {'stock': 0.60, 'coin': 0.40}
        )
        m = full_metrics(port_eq) if port_eq is not None else None
        print_row(f"{ver} (주식60/코인40)", m)

    # V19: 60:25:15 (주식:코인:선물)
    port_eq_v19 = combine_portfolio(
        {'stock': stock_equities.get('V19'), 'coin': coin_equities.get('V19'),
         'futures': futures_eq},
        {'stock': 0.60, 'coin': 0.25, 'futures': 0.15}
    )
    m_v19 = full_metrics(port_eq_v19) if port_eq_v19 is not None else None
    print_row("V19 (60/25/15)", m_v19)

    # V19 with 60:40 for comparison
    port_eq_v19_6040 = combine_portfolio(
        {'stock': stock_equities.get('V19'), 'coin': coin_equities.get('V19')},
        {'stock': 0.60, 'coin': 0.40}
    )
    m_v19_6040 = full_metrics(port_eq_v19_6040) if port_eq_v19_6040 is not None else None
    print_row("V19 (주식60/코인40)", m_v19_6040)

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
