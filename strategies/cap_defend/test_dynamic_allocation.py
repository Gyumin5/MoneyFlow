#!/usr/bin/env python3
"""동적 자산배분 연구: 정적 vs 다양한 동적 방법론.

기준: 주식/현물/선물 + d005 4전략 앙상블
비교 방법론:
  1. 정적 월간 (baseline)
  2. 밴드 리밸런싱 (drift-based)
  3. 역변동성 가중 (inverse vol)
  4. 모멘텀 틸트 (recent performance tilt)
  5. 변동성 타겟 (vol targeting)
  6. 카나리 기반 (canary regime switch)
  7. 복합: 역변동성 + 카나리
"""

import os, sys, time
import numpy as np, pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import (
    load_universe, load_all_prices, filter_universe, DEFENSE_TICKERS,
)
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
    leverage=5.0,
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


# ═══ Data loaders (reused from test_allocation_sweep) ═══

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
    top_n = cfg['params'].get('top_n', 40)
    p = B(**cfg['params'])
    p.start_date = start; p.end_date = end
    r = run_coin_backtest(prices, um.get(top_n, um[40]), (1, 10, 19),
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
    eq_daily = m['_equity'].resample('D').last().dropna()
    return eq_daily


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


# ═══ Allocation Methods ═══

def static_monthly(returns, base_weights, start, end):
    """정적 월간 리밸런싱."""
    common = _get_common(returns, start, end)
    capital = 10000.0
    alloc = {n: capital * w for n, w in base_weights.items()}
    pv_list = []
    prev_month = None
    for date in common:
        cur_month = date.strftime('%Y-%m')
        if prev_month is not None and cur_month != prev_month:
            total = sum(alloc.values())
            alloc = {n: total * w for n, w in base_weights.items()}
        for n in alloc:
            alloc[n] *= (1 + returns[n].get(date, 0))
        pv_list.append(sum(alloc.values()))
        prev_month = cur_month
    return pd.Series(pv_list, index=common)


def band_rebalance(returns, base_weights, start, end, band=0.05):
    """밴드 리밸런싱: drift가 band 초과 시에만 리밸런싱."""
    common = _get_common(returns, start, end)
    capital = 10000.0
    alloc = {n: capital * w for n, w in base_weights.items()}
    pv_list = []
    rebal_count = 0
    for date in common:
        for n in alloc:
            alloc[n] *= (1 + returns[n].get(date, 0))
        total = sum(alloc.values())
        # Check drift
        max_drift = max(abs(alloc[n]/total - base_weights[n]) for n in alloc)
        if max_drift > band:
            alloc = {n: total * w for n, w in base_weights.items()}
            rebal_count += 1
        pv_list.append(total)
    eq = pd.Series(pv_list, index=common)
    return eq, rebal_count


def inverse_vol(returns, base_weights, start, end, lookback=60, floor=0.05, ceiling=0.50):
    """역변동성 가중: 변동성 낮은 자산에 더 배분.
    base_weights의 비율 구조는 유지하되 vol 역수로 틸트."""
    common = _get_common(returns, start, end)
    capital = 10000.0
    names = list(base_weights.keys())
    alloc = {n: capital * base_weights[n] for n in names}
    pv_list = []
    prev_month = None

    ret_history = {n: [] for n in names}

    for date in common:
        cur_month = date.strftime('%Y-%m')
        for n in names:
            r = returns[n].get(date, 0)
            ret_history[n].append(r)
            alloc[n] *= (1 + r)

        if prev_month is not None and cur_month != prev_month:
            total = sum(alloc.values())
            # Compute inverse vol weights
            vols = {}
            for n in names:
                hist = ret_history[n][-lookback:]
                if len(hist) >= 20:
                    vols[n] = np.std(hist) * np.sqrt(365)
                else:
                    vols[n] = 1.0

            if all(v > 0 for v in vols.values()):
                inv_vols = {n: 1.0/v for n, v in vols.items()}
                total_iv = sum(inv_vols.values())
                raw_weights = {n: iv/total_iv for n, iv in inv_vols.items()}
                # Apply floor/ceiling constraints
                weights = _apply_constraints(raw_weights, floor, ceiling)
            else:
                weights = base_weights

            alloc = {n: total * weights[n] for n in names}

        pv_list.append(sum(alloc.values()))
        prev_month = cur_month
    return pd.Series(pv_list, index=common)


def momentum_tilt(returns, base_weights, start, end, lookback=60, tilt_strength=0.5,
                  floor=0.05, ceiling=0.50):
    """모멘텀 틸트: 최근 수익률 기반으로 기본 비중에서 틸트.
    tilt_strength: 0=기본비중, 1=전부 모멘텀 비중."""
    common = _get_common(returns, start, end)
    capital = 10000.0
    names = list(base_weights.keys())
    alloc = {n: capital * base_weights[n] for n in names}
    pv_list = []
    prev_month = None
    ret_history = {n: [] for n in names}

    for date in common:
        cur_month = date.strftime('%Y-%m')
        for n in names:
            r = returns[n].get(date, 0)
            ret_history[n].append(r)
            alloc[n] *= (1 + r)

        if prev_month is not None and cur_month != prev_month:
            total = sum(alloc.values())
            # Compute momentum scores (cumulative return over lookback)
            scores = {}
            for n in names:
                hist = ret_history[n][-lookback:]
                if len(hist) >= 20:
                    cum = np.prod([1+r for r in hist]) - 1
                    scores[n] = max(cum, 0.001)  # floor to avoid negative weights
                else:
                    scores[n] = 1.0

            total_score = sum(scores.values())
            mom_weights = {n: s/total_score for n, s in scores.items()}

            # Blend: base*(1-tilt) + momentum*tilt
            blended = {}
            for n in names:
                blended[n] = base_weights[n]*(1-tilt_strength) + mom_weights[n]*tilt_strength
            total_w = sum(blended.values())
            blended = {n: w/total_w for n, w in blended.items()}
            blended = _apply_constraints(blended, floor, ceiling)
            alloc = {n: total * blended[n] for n in names}

        pv_list.append(sum(alloc.values()))
        prev_month = cur_month
    return pd.Series(pv_list, index=common)


def vol_target(returns, base_weights, start, end, target_vol=0.20, lookback=60,
               min_exposure=0.5, max_exposure=1.5):
    """변동성 타겟: 포트폴리오 변동성을 목표치로 조정.
    exposure < 1이면 일부 현금, > 1이면 레버리지 (cap at max_exposure)."""
    common = _get_common(returns, start, end)
    capital = 10000.0
    names = list(base_weights.keys())
    alloc = {n: capital * base_weights[n] for n in names}
    cash = 0.0
    pv_list = []
    prev_month = None
    port_ret_history = []

    for date in common:
        cur_month = date.strftime('%Y-%m')
        total_before = sum(alloc.values()) + cash
        # Apply returns to allocated portion
        for n in names:
            alloc[n] *= (1 + returns[n].get(date, 0))
        total_after = sum(alloc.values()) + cash
        if total_before > 0:
            port_ret = total_after / total_before - 1
        else:
            port_ret = 0
        port_ret_history.append(port_ret)

        if prev_month is not None and cur_month != prev_month:
            total = sum(alloc.values()) + cash
            # Estimate portfolio vol
            hist = port_ret_history[-lookback:]
            if len(hist) >= 20:
                realized_vol = np.std(hist) * np.sqrt(365)
                if realized_vol > 0:
                    exposure = target_vol / realized_vol
                    exposure = max(min_exposure, min(max_exposure, exposure))
                else:
                    exposure = 1.0
            else:
                exposure = 1.0

            invested = total * exposure
            cash = total - invested
            if cash < 0: cash = 0  # can't leverage beyond max
            alloc = {n: invested * base_weights[n] for n in names}

        pv_list.append(sum(alloc.values()) + cash)
        prev_month = cur_month
    return pd.Series(pv_list, index=common)


def canary_regime(returns, canary_series, start, end,
                  risk_on_weights=None, risk_off_weights=None):
    """카나리 기반 레짐 전환: 코인 카나리 Off시 현물→주식 이전.
    risk_on: normal weights
    risk_off: crypto 축소, stock 확대"""
    common = _get_common(returns, start, end)
    capital = 10000.0
    names = list(risk_on_weights.keys())
    alloc = {n: capital * risk_on_weights[n] for n in names}
    pv_list = []
    prev_month = None
    prev_regime = True  # start risk-on

    for date in common:
        cur_month = date.strftime('%Y-%m')
        for n in names:
            alloc[n] *= (1 + returns[n].get(date, 0))

        # Check canary
        canary_on = canary_series.get(date, prev_regime)

        if prev_month is not None and cur_month != prev_month:
            total = sum(alloc.values())
            if canary_on:
                weights = risk_on_weights
            else:
                weights = risk_off_weights
            alloc = {n: total * weights[n] for n in names}

        pv_list.append(sum(alloc.values()))
        prev_month = cur_month
        prev_regime = canary_on
    return pd.Series(pv_list, index=common)


def combined_invvol_canary(returns, canary_series, start, end,
                           base_on_weights, base_off_weights,
                           lookback=60, floor=0.05, ceiling=0.50):
    """복합: 역변동성 + 카나리 레짐.
    카나리 On: invvol 기반 (base_on 구조)
    카나리 Off: 현물 축소 + invvol"""
    common = _get_common(returns, start, end)
    capital = 10000.0
    names = list(base_on_weights.keys())
    alloc = {n: capital * base_on_weights[n] for n in names}
    pv_list = []
    prev_month = None
    ret_history = {n: [] for n in names}
    prev_regime = True

    for date in common:
        cur_month = date.strftime('%Y-%m')
        for n in names:
            r = returns[n].get(date, 0)
            ret_history[n].append(r)
            alloc[n] *= (1 + r)

        canary_on = canary_series.get(date, prev_regime)

        if prev_month is not None and cur_month != prev_month:
            total = sum(alloc.values())
            base = base_on_weights if canary_on else base_off_weights

            vols = {}
            for n in names:
                hist = ret_history[n][-lookback:]
                if len(hist) >= 20:
                    vols[n] = np.std(hist) * np.sqrt(365)
                else:
                    vols[n] = 1.0

            if all(v > 0 for v in vols.values()):
                inv_vols = {n: 1.0/v for n, v in vols.items()}
                total_iv = sum(inv_vols.values())
                iv_weights = {n: iv/total_iv for n, iv in inv_vols.items()}
                # Blend invvol with base: 50/50
                blended = {n: (base[n] + iv_weights[n]) / 2 for n in names}
                total_w = sum(blended.values())
                blended = {n: w/total_w for n, w in blended.items()}
                blended = _apply_constraints(blended, floor, ceiling)
            else:
                blended = base
            alloc = {n: total * blended[n] for n in names}

        pv_list.append(sum(alloc.values()))
        prev_month = cur_month
        prev_regime = canary_on
    return pd.Series(pv_list, index=common)


# ═══ Helpers ═══

def _get_common(returns, start, end):
    common = None
    for ret in returns.values():
        idx = ret.index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    return common[(common >= start) & (common <= end)]


def _apply_constraints(weights, floor, ceiling):
    """Apply floor/ceiling and renormalize."""
    names = list(weights.keys())
    w = dict(weights)
    for _ in range(10):  # iterate to converge
        clamped = False
        for n in names:
            if w[n] < floor:
                w[n] = floor; clamped = True
            elif w[n] > ceiling:
                w[n] = ceiling; clamped = True
        if not clamped:
            break
        total = sum(w.values())
        w = {n: v/total for n, v in w.items()}
    return w


def main():
    t0 = time.time()
    print("=" * 100)
    print("동적 자산배분 연구")
    print(f"기간: {START} ~ {END}")
    print("=" * 100)

    # === Load ===
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

    # === Equity curves ===
    print("\nEquity curves 생성...")
    coin_eq, canary_hist = get_coin_equity(coin_prices, coin_um, START, END)
    stock_eq = get_stock_equity(START, END)
    fut_eq = get_futures_equity(fut_data, CAND4, START, END)
    print(f"  완료 ({time.time()-t0:.1f}s)")

    # Build canary series (date → bool)
    canary_series = {}
    for entry in canary_hist:
        canary_series[entry['Date']] = entry['canary_on']

    # Daily returns
    returns = {
        'stock': stock_eq.pct_change().fillna(0),
        'coin': coin_eq.pct_change().fillna(0),
        'futures': fut_eq.pct_change().fillna(0),
    }

    # === Test configurations ===
    BASE_RATIOS = [
        ('60/30/10', {'stock': 0.60, 'coin': 0.30, 'futures': 0.10}),
        ('60/25/15', {'stock': 0.60, 'coin': 0.25, 'futures': 0.15}),
        ('50/30/20', {'stock': 0.50, 'coin': 0.30, 'futures': 0.20}),
    ]

    print("\n" + "=" * 100)
    print("Phase 1: 정적 월간 vs 밴드 리밸런싱")
    print("=" * 100)

    for ratio_name, base_w in BASE_RATIOS:
        print(f"\n  [{ratio_name}]")
        print(f"  {'방법':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s} {'Rebal':>6s}")
        print(f"  {'-'*80}")

        # Static monthly
        eq = static_monthly(returns, base_w, START, END)
        m = calc_metrics(eq)
        # Count monthly rebals
        months = len(set(d.strftime('%Y-%m') for d in eq.index)) - 1
        print(f"  {'정적 월간':<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {months:>6d}")

        # Band rebalancing
        for band in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            eq, rc = band_rebalance(returns, base_w, START, END, band=band)
            m = calc_metrics(eq)
            label = f"밴드 {band:.0%}"
            print(f"  {label:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%} {rc:>6d}")

    print("\n" + "=" * 100)
    print("Phase 2: 역변동성 가중 (Inverse Volatility)")
    print("=" * 100)

    for ratio_name, base_w in BASE_RATIOS:
        print(f"\n  [{ratio_name} 기준]")
        print(f"  {'방법':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s}")
        print(f"  {'-'*75}")

        eq = static_monthly(returns, base_w, START, END)
        m = calc_metrics(eq)
        print(f"  {'정적 월간 (baseline)':<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

        for lb in [30, 60, 90, 120]:
            eq = inverse_vol(returns, base_w, START, END, lookback=lb)
            m = calc_metrics(eq)
            label = f"InvVol lb={lb}d"
            print(f"  {label:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

        # Different floor/ceiling
        for fl, cl in [(0.05, 0.60), (0.10, 0.50), (0.10, 0.40), (0.15, 0.40)]:
            eq = inverse_vol(returns, base_w, START, END, lookback=60, floor=fl, ceiling=cl)
            m = calc_metrics(eq)
            label = f"InvVol fl={fl:.0%}/cl={cl:.0%}"
            print(f"  {label:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    print("\n" + "=" * 100)
    print("Phase 3: 모멘텀 틸트")
    print("=" * 100)

    for ratio_name, base_w in BASE_RATIOS:
        print(f"\n  [{ratio_name} 기준]")
        print(f"  {'방법':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s}")
        print(f"  {'-'*75}")

        eq = static_monthly(returns, base_w, START, END)
        m = calc_metrics(eq)
        print(f"  {'정적 월간 (baseline)':<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

        for lb in [30, 60, 90, 120]:
            for tilt in [0.3, 0.5, 0.7]:
                eq = momentum_tilt(returns, base_w, START, END, lookback=lb, tilt_strength=tilt)
                m = calc_metrics(eq)
                label = f"Mom lb={lb}d tilt={tilt:.0%}"
                print(f"  {label:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    print("\n" + "=" * 100)
    print("Phase 4: 변동성 타겟")
    print("=" * 100)

    for ratio_name, base_w in BASE_RATIOS:
        print(f"\n  [{ratio_name} 기준]")
        print(f"  {'방법':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s}")
        print(f"  {'-'*75}")

        eq = static_monthly(returns, base_w, START, END)
        m = calc_metrics(eq)
        print(f"  {'정적 월간 (baseline)':<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

        for tv in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for lb in [60, 90]:
                eq = vol_target(returns, base_w, START, END, target_vol=tv, lookback=lb)
                m = calc_metrics(eq)
                label = f"VolTarget {tv:.0%} lb={lb}d"
                print(f"  {label:<35s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    print("\n" + "=" * 100)
    print("Phase 5: 카나리 레짐 전환")
    print("=" * 100)

    # Risk-off: reduce coin, increase stock
    regime_configs = [
        ("Off: coin→0, stock+coin분", {'stock': 0.85, 'coin': 0.00, 'futures': 0.15}),
        ("Off: coin반감",             {'stock': 0.72, 'coin': 0.13, 'futures': 0.15}),
        ("Off: coin→stock",           {'stock': 0.75, 'coin': 0.10, 'futures': 0.15}),
        ("Off: 전부축소→현금50%",       {'stock': 0.30, 'coin': 0.05, 'futures': 0.15}),
    ]

    base_on = {'stock': 0.60, 'coin': 0.25, 'futures': 0.15}
    print(f"\n  [Risk-On: 60/25/15 기준]")
    print(f"  {'방법':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s}")
    print(f"  {'-'*75}")

    eq = static_monthly(returns, base_on, START, END)
    m = calc_metrics(eq)
    print(f"  {'정적 60/25/15 (baseline)':<40s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    for desc, off_w in regime_configs:
        # Need to handle the "cash" portion for risk-off
        total_w = sum(off_w.values())
        if total_w < 1.0:
            # Remaining goes to "cash" — simulate by not investing
            # For simplicity, put remainder into stock (safest)
            adj_w = dict(off_w)
            adj_w['stock'] = adj_w.get('stock', 0) + (1.0 - total_w)
        else:
            adj_w = off_w
        eq = canary_regime(returns, canary_series, START, END,
                           risk_on_weights=base_on, risk_off_weights=adj_w)
        m = calc_metrics(eq)
        print(f"  {desc:<40s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    print("\n" + "=" * 100)
    print("Phase 6: 복합 (InvVol + 카나리)")
    print("=" * 100)

    base_off_configs = [
        ("Off: coin→0",    {'stock': 0.85, 'coin': 0.00, 'futures': 0.15}),
        ("Off: coin반감",  {'stock': 0.72, 'coin': 0.13, 'futures': 0.15}),
    ]

    print(f"\n  [Risk-On: 60/25/15 + InvVol 블렌딩]")
    print(f"  {'방법':<45s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Vol':>7s}")
    print(f"  {'-'*80}")

    for desc, off_w in base_off_configs:
        for lb in [60, 90]:
            eq = combined_invvol_canary(returns, canary_series, START, END,
                                        base_on, off_w, lookback=lb)
            m = calc_metrics(eq)
            label = f"InvVol(lb={lb}) + {desc}"
            print(f"  {label:<45s} {m['Sharpe']:>7.3f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Calmar']:>7.2f} {m['Vol']:>7.1%}")

    # === Phase 7: 서브기간 Top 5 ===
    print("\n" + "=" * 100)
    print("Phase 7: 주요 방법론 서브기간 분석")
    print("=" * 100)

    mid = '2023-07-01'
    base_w = {'stock': 0.60, 'coin': 0.25, 'futures': 0.15}

    test_methods = [
        ("정적 월간", lambda s, e: static_monthly(returns, base_w, s, e)),
        ("밴드 5%", lambda s, e: band_rebalance(returns, base_w, s, e, band=0.05)[0]),
        ("밴드 10%", lambda s, e: band_rebalance(returns, base_w, s, e, band=0.10)[0]),
        ("InvVol 60d", lambda s, e: inverse_vol(returns, base_w, s, e, lookback=60)),
        ("InvVol 90d", lambda s, e: inverse_vol(returns, base_w, s, e, lookback=90)),
        ("Mom 60d/50%", lambda s, e: momentum_tilt(returns, base_w, s, e, lookback=60, tilt_strength=0.5)),
        ("VolTarget 20%", lambda s, e: vol_target(returns, base_w, s, e, target_vol=0.20)),
        ("카나리(coin→0)", lambda s, e: canary_regime(returns, canary_series, s, e,
            risk_on_weights=base_w, risk_off_weights={'stock': 0.85, 'coin': 0.00, 'futures': 0.15})),
        ("InvVol+카나리", lambda s, e: combined_invvol_canary(returns, canary_series, s, e,
            base_w, {'stock': 0.85, 'coin': 0.00, 'futures': 0.15}, lookback=60)),
    ]

    print(f"\n  {'방법':<25s} | {'전체Cal':>8s} {'전체Sh':>7s} | {'전반Cal':>8s} | {'후반Cal':>8s} | {'비율':>6s}")
    print(f"  {'-'*85}")

    for name, fn in test_methods:
        eq_full = fn(START, END)
        eq_front = fn(START, mid)
        eq_back = fn(mid, END)
        m_full = calc_metrics(eq_full)
        m_front = calc_metrics(eq_front)
        m_back = calc_metrics(eq_back)
        if m_full and m_front and m_back:
            r = max(m_front['Calmar'], m_back['Calmar']) / min(m_front['Calmar'], m_back['Calmar']) if min(m_front['Calmar'], m_back['Calmar']) > 0 else 99
            print(f"  {name:<25s} | {m_full['Calmar']:>8.2f} {m_full['Sharpe']:>7.3f} | {m_front['Calmar']:>8.2f} | {m_back['Calmar']:>8.2f} | {r:>6.2f}x")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
