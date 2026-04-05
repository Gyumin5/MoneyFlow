#!/usr/bin/env python3
"""InvVol+카나리: 리밸런싱 주기 비교 (매일 vs 매주 vs 매월)."""

import os, sys, time
import numpy as np, pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, load_all_prices, filter_universe, DEFENSE_TICKERS
from coin_helpers import B
from backtest_official import run_coin_backtest, COIN_VERSIONS, STOCK_VERSIONS
from stock_engine import (
    SP, load_prices as load_stock_prices, precompute as stock_precompute,
    _init as stock_init, ALL_TICKERS,
)
import stock_engine as tsi
from backtest_official import check_crash_vt
from backtest_futures_full import load_data
from futures_ensemble_engine import SingleAccountEngine, combine_targets

ENGINE_KWARGS = dict(
    leverage=5.0, stop_kind="prev_close_pct", stop_pct=0.15,
    stop_gate="cash_guard", stop_gate_cash_threshold=0.34,
    per_coin_leverage_mode="cap_mom_blend_543_cash",
    leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
    leverage_cash_threshold=0.34, leverage_partial_cash_threshold=0.0,
    leverage_count_floor_max=2, leverage_count_mid_max=4,
    leverage_canary_floor_gap=0.015, leverage_canary_mid_gap=0.04,
    leverage_canary_high_gap=0.08, leverage_canary_sma_bars=1200,
    leverage_mom_lookback_bars=24*30, leverage_vol_lookback_bars=24*90,
)

FIXED_BASE = dict(
    canary_hyst=0.015, drift_threshold=0.0,
    dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0, n_snapshots=3,
)

CAND4 = {
    "4h_d005": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                    health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
                    snap_interval_bars=60, **FIXED_BASE),
    "2h_b60_S240": dict(interval="2h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                        snap_interval_bars=120, **FIXED_BASE),
    "2h_b60_S120": dict(interval="2h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                        snap_interval_bars=120, **FIXED_BASE),
    "4h_b60_M20_120": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                           health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                           snap_interval_bars=21, **FIXED_BASE),
}

START = '2020-10-01'
END = '2026-03-28'


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
    cfg = COIN_VERSIONS['V18']
    p = B(**cfg['params'])
    p.start_date = start; p.end_date = end
    r = run_coin_backtest(prices, um.get(cfg['params'].get('top_n', 40), um[40]), (1, 10, 19),
                          dd_lookback=cfg['dd_lookback'], dd_threshold=cfg['dd_threshold'],
                          bl_drop=cfg['bl_drop'], bl_days=cfg['bl_days'],
                          drift_threshold=cfg['drift_threshold'],
                          post_flip_delay=cfg['post_flip_delay'],
                          params_base=p, defense=cfg.get('defense', False))
    return r['equity_curve'], r.get('canary_history', [])


def get_stock_equity(start, end):
    base = STOCK_VERSIONS['V17']
    sp = replace(base, start=start, end=end, _n_tranches=4, tranche_days=(1, 8, 15, 22))
    from stock_engine import run_bt as srun_bt
    dfs = []
    for day in sp.tranche_days:
        tp = SP(**{k: v for k, v in sp.__dict__.items()})
        tp._anchor = day; tp._n_tranches = 1
        tp.capital = sp.capital / len(sp.tranche_days)
        df = srun_bt(tsi._g_prices, tsi._g_ind, tp)
        if df is not None: dfs.append(df)
    if not dfs: return None
    merged = dfs[0].copy()
    for df in dfs[1:]:
        common = merged.index.intersection(df.index)
        merged.loc[common, 'Value'] += df.loc[common, 'Value']
    return merged['Value']


def get_futures_equity(data, strategies, start, end):
    combo = {k: 1.0/len(strategies) for k in strategies}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[
        (bars_1h['BTC'].index >= start) & (bars_1h['BTC'].index <= end)]
    traces = {name: generate_trace(data, cfg, start, end)
              for name, cfg in strategies.items()}
    combined = combine_targets({k: traces[k] for k in combo}, combo, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    m = engine.run(combined)
    return m['_equity'].resample('D').last().dropna()


def calc_metrics(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[-1] <= 0: return None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    vol = dr.std() * np.sqrt(365)
    return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Calmar': cal, 'Vol': vol}


def _get_common(returns, start, end):
    common = None
    for ret in returns.values():
        common = ret.index if common is None else common.intersection(ret.index)
    common = common.sort_values()
    return common[(common >= start) & (common <= end)]


def _apply_constraints(weights, floor, ceiling):
    w = dict(weights)
    for _ in range(10):
        clamped = False
        for n in w:
            if w[n] < floor: w[n] = floor; clamped = True
            elif w[n] > ceiling: w[n] = ceiling; clamped = True
        if not clamped: break
        total = sum(w.values())
        w = {n: v/total for n, v in w.items()}
    return w


def run_method(returns, canary_series, start, end, base_on, base_off,
               rebal_freq='monthly', lookback=60, floor=0.05, ceiling=0.50,
               blend_ratio=0.5, method='invvol_canary', band=0.0,
               min_rebal_turnover=0.0):
    """통합 리밸런싱 엔진.

    rebal_freq: 'daily', 'weekly', 'biweekly', 'monthly'
    method: 'static', 'invvol', 'canary', 'invvol_canary'
    band: > 0이면 drift가 band 미만이면 리밸 건너뜀
    min_rebal_turnover: > 0이면 계산된 비중 변화가 이 값 미만이면 리밸 건너뜀
    """
    common = _get_common(returns, start, end)
    capital = 10000.0
    names = list(base_on.keys())
    alloc = {n: capital * base_on[n] for n in names}
    pv_list = []
    ret_history = {n: [] for n in names}
    prev_regime = True
    prev_rebal_date = None
    rebal_count = 0

    for date in common:
        for n in names:
            r = returns[n].get(date, 0)
            ret_history[n].append(r)
            alloc[n] *= (1 + r)

        total = sum(alloc.values())
        cur_weights = {n: alloc[n]/total for n in names}

        # Determine if rebalancing happens today
        do_rebal = False
        if prev_rebal_date is None:
            do_rebal = False  # skip first day
        elif rebal_freq == 'daily':
            do_rebal = True
        elif rebal_freq == 'weekly':
            do_rebal = (date - prev_rebal_date).days >= 7
        elif rebal_freq == 'biweekly':
            do_rebal = (date - prev_rebal_date).days >= 14
        elif rebal_freq == 'monthly':
            do_rebal = date.strftime('%Y-%m') != prev_rebal_date.strftime('%Y-%m')

        if prev_rebal_date is None:
            prev_rebal_date = date

        if do_rebal:
            canary_on = canary_series.get(date, prev_regime)

            if method == 'static':
                target = base_on
            elif method == 'canary':
                target = base_on if canary_on else base_off
            elif method == 'invvol':
                vols = {}
                for n in names:
                    hist = ret_history[n][-lookback:]
                    vols[n] = np.std(hist) * np.sqrt(365) if len(hist) >= 20 else 1.0
                if all(v > 0 for v in vols.values()):
                    inv_vols = {n: 1.0/v for n, v in vols.items()}
                    total_iv = sum(inv_vols.values())
                    iv_w = {n: iv/total_iv for n, iv in inv_vols.items()}
                    target = {n: base_on[n]*(1-blend_ratio) + iv_w[n]*blend_ratio for n in names}
                    tw = sum(target.values())
                    target = {n: w/tw for n, w in target.items()}
                    target = _apply_constraints(target, floor, ceiling)
                else:
                    target = base_on
            elif method == 'invvol_canary':
                base = base_on if canary_on else base_off
                vols = {}
                for n in names:
                    hist = ret_history[n][-lookback:]
                    vols[n] = np.std(hist) * np.sqrt(365) if len(hist) >= 20 else 1.0
                if all(v > 0 for v in vols.values()):
                    inv_vols = {n: 1.0/v for n, v in vols.items()}
                    total_iv = sum(inv_vols.values())
                    iv_w = {n: iv/total_iv for n, iv in inv_vols.items()}
                    target = {n: base[n]*(1-blend_ratio) + iv_w[n]*blend_ratio for n in names}
                    tw = sum(target.values())
                    target = {n: w/tw for n, w in target.items()}
                    target = _apply_constraints(target, floor, ceiling)
                else:
                    target = base
            else:
                target = base_on

            prev_regime = canary_on

            # Band check: skip if drift too small
            if band > 0:
                max_drift = max(abs(cur_weights[n] - target[n]) for n in names)
                if max_drift < band:
                    pv_list.append(total)
                    continue

            # Min turnover check
            if min_rebal_turnover > 0:
                turnover = sum(abs(cur_weights[n] - target[n]) for n in names) / 2
                if turnover < min_rebal_turnover:
                    pv_list.append(total)
                    continue

            alloc = {n: total * target[n] for n in names}
            rebal_count += 1
            prev_rebal_date = date

        pv_list.append(sum(alloc.values()))

    eq = pd.Series(pv_list, index=common)
    return eq, rebal_count


def main():
    t0 = time.time()
    print("=" * 100)
    print("리밸런싱 주기 비교: 매일 vs 매주 vs 격주 vs 매월")
    print(f"기간: {START} ~ {END}")
    print("=" * 100)

    # Load
    print("\n데이터 로딩...")
    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD']); all_t.update(DEFENSE_TICKERS)
    coin_prices = load_all_prices(list(all_t))

    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt

    fut_data = {iv: load_data(iv) for iv in ["4h", "2h", "1h"]}
    print(f"  완료 ({time.time()-t0:.1f}s)")

    print("\nEquity curves 생성...")
    coin_eq, canary_hist = get_coin_equity(coin_prices, coin_um, START, END)
    stock_eq = get_stock_equity(START, END)
    fut_eq = get_futures_equity(fut_data, CAND4, START, END)
    print(f"  완료 ({time.time()-t0:.1f}s)")

    canary_series = {e['Date']: e['canary_on'] for e in canary_hist}
    returns = {
        'stock': stock_eq.pct_change().fillna(0),
        'coin': coin_eq.pct_change().fillna(0),
        'futures': fut_eq.pct_change().fillna(0),
    }

    base_on = {'stock': 0.60, 'coin': 0.25, 'futures': 0.15}
    base_off = {'stock': 0.85, 'coin': 0.00, 'futures': 0.15}
    mid = '2023-07-01'

    # === Phase 1: 주기별 비교 (모든 방법론) ===
    print("\n" + "=" * 100)
    print("Phase 1: 방법론 x 리밸런싱 주기")
    print("=" * 100)

    methods = ['static', 'invvol', 'canary', 'invvol_canary']
    freqs = ['daily', 'weekly', 'biweekly', 'monthly']

    for method in methods:
        method_name = {'static': '정적', 'invvol': 'InvVol', 'canary': '카나리', 'invvol_canary': 'InvVol+카나리'}[method]
        print(f"\n  [{method_name}]")
        print(f"  {'주기':<12s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s} {'Rebal':>6s}")
        print(f"  {'-'*60}")

        for freq in freqs:
            eq, rc = run_method(returns, canary_series, START, END,
                                base_on, base_off, rebal_freq=freq,
                                lookback=60, method=method)
            m = calc_metrics(eq)
            freq_name = {'daily': '매일', 'weekly': '매주', 'biweekly': '격주', 'monthly': '매월'}[freq]
            print(f"  {freq_name:<12s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {rc:>6d}")

    # === Phase 2: InvVol+카나리 상세 (lookback x 주기) ===
    print("\n" + "=" * 100)
    print("Phase 2: InvVol+카나리 — lookback x 주기 상세")
    print("=" * 100)

    print(f"\n  {'lookback':<10s} {'주기':<10s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s} {'Rebal':>6s}")
    print(f"  {'-'*65}")

    for lb in [30, 60, 90, 120]:
        for freq in freqs:
            eq, rc = run_method(returns, canary_series, START, END,
                                base_on, base_off, rebal_freq=freq,
                                lookback=lb, method='invvol_canary')
            m = calc_metrics(eq)
            freq_name = {'daily': '매일', 'weekly': '매주', 'biweekly': '격주', 'monthly': '매월'}[freq]
            label_lb = f"{lb}d"
            print(f"  {label_lb:<10s} {freq_name:<10s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {rc:>6d}")

    # === Phase 3: 매일 리밸 + 밴드/최소턴오버 필터 ===
    print("\n" + "=" * 100)
    print("Phase 3: 매일 InvVol+카나리 + 거래 필터 (불필요한 리밸 제거)")
    print("=" * 100)

    print(f"\n  {'필터':<30s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s} {'Rebal':>6s}")
    print(f"  {'-'*75}")

    # No filter
    eq, rc = run_method(returns, canary_series, START, END,
                        base_on, base_off, rebal_freq='daily',
                        lookback=60, method='invvol_canary')
    m = calc_metrics(eq)
    print(f"  {'필터 없음 (매일)':<30s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {rc:>6d}")

    # Band filters
    for band in [0.01, 0.02, 0.03, 0.05, 0.08]:
        eq, rc = run_method(returns, canary_series, START, END,
                            base_on, base_off, rebal_freq='daily',
                            lookback=60, method='invvol_canary', band=band)
        m = calc_metrics(eq)
        label = f"밴드 {band:.0%}"
        print(f"  {label:<30s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {rc:>6d}")

    # Min turnover filters
    for mt in [0.01, 0.02, 0.03, 0.05]:
        eq, rc = run_method(returns, canary_series, START, END,
                            base_on, base_off, rebal_freq='daily',
                            lookback=60, method='invvol_canary',
                            min_rebal_turnover=mt)
        m = calc_metrics(eq)
        label = f"최소턴오버 {mt:.0%}"
        print(f"  {label:<30s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {rc:>6d}")

    # === Phase 4: 블렌드 비율 (InvVol 비중) ===
    print("\n" + "=" * 100)
    print("Phase 4: InvVol 블렌드 비율 (매일 vs 매월)")
    print("=" * 100)

    print(f"\n  {'블렌드':<15s} {'주기':<8s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s}")
    print(f"  {'-'*60}")

    for blend in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
        for freq in ['daily', 'monthly']:
            eq, rc = run_method(returns, canary_series, START, END,
                                base_on, base_off, rebal_freq=freq,
                                lookback=60, method='invvol_canary',
                                blend_ratio=blend)
            m = calc_metrics(eq)
            freq_name = {'daily': '매일', 'monthly': '매월'}[freq]
            label = f"InvVol {blend:.0%}"
            print(f"  {label:<15s} {freq_name:<8s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    # === Phase 5: 서브기간 + 다양한 비율 기준 ===
    print("\n" + "=" * 100)
    print("Phase 5: 서브기간 안정성 (주요 설정)")
    print("=" * 100)

    configs = [
        ("정적 월간", 'monthly', 'static', 60, 0.5),
        ("정적 매일", 'daily', 'static', 60, 0.5),
        ("InvVol+카나리 월간 60d", 'monthly', 'invvol_canary', 60, 0.5),
        ("InvVol+카나리 매일 60d", 'daily', 'invvol_canary', 60, 0.5),
        ("InvVol+카나리 매일 90d", 'daily', 'invvol_canary', 90, 0.5),
        ("InvVol+카나리 매주 60d", 'weekly', 'invvol_canary', 60, 0.5),
        ("InvVol 매일 60d", 'daily', 'invvol', 60, 0.5),
        ("InvVol 매주 60d", 'weekly', 'invvol', 60, 0.5),
    ]

    base_ratios = [
        ('60/25/15', {'stock': 0.60, 'coin': 0.25, 'futures': 0.15},
                     {'stock': 0.85, 'coin': 0.00, 'futures': 0.15}),
        ('60/30/10', {'stock': 0.60, 'coin': 0.30, 'futures': 0.10},
                     {'stock': 0.90, 'coin': 0.00, 'futures': 0.10}),
        ('50/30/20', {'stock': 0.50, 'coin': 0.30, 'futures': 0.20},
                     {'stock': 0.80, 'coin': 0.00, 'futures': 0.20}),
    ]

    for ratio_name, bon, boff in base_ratios:
        print(f"\n  [{ratio_name}]")
        print(f"  {'설정':<30s} | {'전체Cal':>8s} {'전체Sh':>7s} | {'전반Cal':>8s} | {'후반Cal':>8s} | {'비율':>6s} | {'Rebal':>6s}")
        print(f"  {'-'*90}")

        for name, freq, method, lb, blend in configs:
            eq_full, rc_full = run_method(returns, canary_series, START, END,
                                          bon, boff, rebal_freq=freq,
                                          lookback=lb, method=method, blend_ratio=blend)
            eq_front, _ = run_method(returns, canary_series, START, mid,
                                     bon, boff, rebal_freq=freq,
                                     lookback=lb, method=method, blend_ratio=blend)
            eq_back, _ = run_method(returns, canary_series, mid, END,
                                    bon, boff, rebal_freq=freq,
                                    lookback=lb, method=method, blend_ratio=blend)
            mf = calc_metrics(eq_full)
            mfr = calc_metrics(eq_front)
            mb = calc_metrics(eq_back)
            if mf and mfr and mb:
                r = max(mfr['Calmar'], mb['Calmar']) / min(mfr['Calmar'], mb['Calmar']) if min(mfr['Calmar'], mb['Calmar']) > 0 else 99
                print(f"  {name:<30s} | {mf['Calmar']:>8.2f} {mf['Sharpe']:>7.3f} | {mfr['Calmar']:>8.2f} | {mb['Calmar']:>8.2f} | {r:>6.2f}x | {rc_full:>6d}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
