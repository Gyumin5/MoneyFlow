#!/usr/bin/env python3
"""Drawdown health filter test (parallelized).
Base: T50 + BL15% + Drift10% + PFD5 (Sharpe 1.671)

Tests drawdown from N-day high as health filter:
  - Mode A: Current health + drawdown (additional gate)
  - Mode B: Drawdown + Vol only (replace Mom with drawdown)
  - Mode C: Drawdown only (no mom, no vol)
"""

import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    load_all_prices, filter_universe, load_universe,
    get_universe_for_date, resolve_canary, get_healthy_coins,
    select_coins, compute_weights,
    execute_rebalance, _port_val, _close_to,
    calc_metrics, get_vol,
)
from test_matrix import (
    B, merge_snapshots, calc_current_weights, calc_half_turnover,
    check_blacklist, update_blacklist, ANCHOR_DAYS, _empty,
)

# Global data (shared via fork)
G_PRICES = None
G_UNIVERSE_MAP = None


def check_drawdown(ticker, prices, date, lookback, threshold):
    close = _close_to(ticker, prices, date)
    if len(close) < lookback:
        return False
    recent = close.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0:
        return False
    dd = close.iloc[-1] / peak - 1
    return dd > threshold


def compute_signal_weights_dd(prices, universe_map, date, params, state,
                               blacklist, dd_lookback=0, dd_threshold=0,
                               dd_mode='add'):
    canary_on = state.get('canary_on', False)
    if not canary_on:
        return {'CASH': 1.0}

    universe = get_universe_for_date(universe_map, date)
    if blacklist:
        universe = [t for t in universe if t not in blacklist]

    if dd_lookback > 0 and dd_mode == 'add':
        healthy = get_healthy_coins(prices, universe, date, params, state)
        if dd_threshold < 0:
            healthy = [t for t in healthy
                       if check_drawdown(t, prices, date, dd_lookback, dd_threshold)]

    elif dd_lookback > 0 and dd_mode == 'replace':
        healthy = []
        for t in universe:
            close = _close_to(t, prices, date)
            if len(close) < 90:
                continue
            vol = get_vol(close, 90)
            if vol > params.vol_cap:
                continue
            if not check_drawdown(t, prices, date, dd_lookback, dd_threshold):
                continue
            healthy.append(t)

    elif dd_lookback > 0 and dd_mode == 'only':
        healthy = []
        for t in universe:
            close = _close_to(t, prices, date)
            if len(close) < dd_lookback:
                continue
            if not check_drawdown(t, prices, date, dd_lookback, dd_threshold):
                continue
            healthy.append(t)

    else:
        healthy = get_healthy_coins(prices, universe, date, params, state)

    state['healthy_count'] = len(healthy)
    state['current_healthy_set'] = set(healthy)
    if not healthy:
        return {'CASH': 1.0}
    picks = select_coins(healthy, prices, date, params, state)
    if not picks:
        return {'CASH': 1.0}
    weights = compute_weights(picks, prices, date, params, state)
    return weights


def run_dd_backtest(prices, universe_map, snapshot_days,
                     dd_lookback=0, dd_threshold=0, dd_mode='add',
                     bl_drop=-0.15, bl_days=7,
                     drift_threshold=0.10,
                     post_flip_delay=5, params_base=None):
    if params_base is None:
        params_base = B()

    btc = prices.get('BTC-USD')
    if btc is None:
        return _empty()

    all_dates = btc.index[(btc.index >= params_base.start_date) &
                          (btc.index <= params_base.end_date)]
    if len(all_dates) == 0:
        return _empty()

    holdings = {}
    cash = params_base.initial_capital
    state = {
        'prev_canary': False, 'canary_off_days': 0,
        'health_fail_streak': {}, 'prev_picks': [],
        'scaled_months': 2, 'month_start_value': params_base.initial_capital,
        'high_watermark': params_base.initial_capital,
        'crash_cooldown': 0, 'coin_cooldowns': {},
        'recent_port_vals': [], 'prev_month': None,
        'catastrophic_triggered': False, 'risk_force_rebal': False,
        'canary_on_date': None, 'post_flip_refreshed': False,
    }

    n_snap = len(snapshot_days)
    snapshots = [{'CASH': 1.0} for _ in range(n_snap)]
    snap_done = {}
    blacklist = {}

    portfolio_values = []
    rebal_count = 0
    drift_rebal_count = 0

    def _compute(date):
        return compute_signal_weights_dd(
            prices, universe_map, date, params_base, state, blacklist,
            dd_lookback, dd_threshold, dd_mode)

    for i, date in enumerate(all_dates):
        cur_month = date.strftime('%Y-%m')
        imc = (state['prev_month'] is not None and cur_month != state['prev_month'])

        pv = _port_val(holdings, cash, prices, date)
        state['current_port_val'] = pv
        state['high_watermark'] = max(state['high_watermark'], pv)
        state['recent_port_vals'].append(pv)
        if len(state['recent_port_vals']) > 60:
            state['recent_port_vals'] = state['recent_port_vals'][-60:]

        if imc:
            state['month_start_value'] = pv
            state['catastrophic_triggered'] = False

        canary_on = resolve_canary(prices, date, params_base, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['canary_on'] = canary_on
        state['is_month_change'] = imc

        update_blacklist(blacklist)

        newly_bl = []
        if bl_drop < 0 and canary_on and holdings:
            newly_bl = check_blacklist(holdings, prices, date, bl_drop, blacklist, bl_days)
            if newly_bl:
                for si in range(n_snap):
                    snap = snapshots[si]
                    removed = 0
                    for coin in newly_bl:
                        if coin in snap:
                            removed += snap.pop(coin)
                    if removed > 0:
                        snap['CASH'] = snap.get('CASH', 0) + removed
                        total = sum(snap.values())
                        if total > 0:
                            snapshots[si] = {t: w/total for t, w in snap.items()}

        state['_blacklist'] = set(blacklist.keys())

        need_rebal = False

        if i == 0:
            for si in range(n_snap):
                snapshots[si] = _compute(date)
            need_rebal = True

        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = _compute(date)
            else:
                for si in range(n_snap):
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True

        elif post_flip_delay > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= post_flip_delay:
                    state['post_flip_refreshed'] = True
                    for si in range(n_snap):
                        snapshots[si] = _compute(date)
                    need_rebal = True

        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = _compute(date)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        if bl_drop < 0 and newly_bl:
            need_rebal = True

        combined = merge_snapshots(snapshots)

        if not need_rebal and canary_on and drift_threshold > 0:
            current_w = calc_current_weights(holdings, cash, prices, date)
            ht = calc_half_turnover(current_w, combined)
            if ht >= drift_threshold:
                need_rebal = True
                drift_rebal_count += 1

        if need_rebal:
            holdings, cash = execute_rebalance(holdings, cash, combined, prices,
                                               date, params_base.tx_cost)
            rebal_count += 1

        pv = _port_val(holdings, cash, prices, date)
        portfolio_values.append({'Date': date, 'Value': pv})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month

    if not portfolio_values:
        return _empty()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    return {
        'metrics': m,
        'rebal_count': rebal_count,
        'drift_rebal_count': drift_rebal_count,
    }


def _worker(args):
    """Worker for one (anchor, config) combination."""
    base_d, label, lb, th, mode = args
    snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
    r = run_dd_backtest(
        G_PRICES, G_UNIVERSE_MAP, snap_days,
        dd_lookback=lb, dd_threshold=th, dd_mode=mode,
        bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10,
        post_flip_delay=5, params_base=B()
    )
    return (base_d, label, r['metrics'], r['rebal_count'])


def main():
    global G_PRICES, G_UNIVERSE_MAP

    print("Loading data...")
    um_raw = load_universe()
    fm = filter_universe(um_raw, 50)

    all_tickers = set()
    for ts in fm.values():
        all_tickers.update(ts)
    all_tickers.update(['BTC-USD', 'ETH-USD'])

    G_PRICES = load_all_prices(all_tickers)
    G_UNIVERSE_MAP = fm
    print(f"  {len(G_PRICES)} tickers loaded")

    t0 = time.time()

    lookbacks = [14, 21, 30, 60]
    thresholds = [-0.15, -0.20, -0.25, -0.30, -0.40]

    configs = []
    configs.append(('기준 (Mom21+Mom90+Vol5%)', 0, 0, 'add'))

    for lb in lookbacks:
        for th in thresholds:
            label = f"H+DD({lb}d,{int(abs(th)*100)}%)"
            configs.append((label, lb, th, 'add'))

    for lb in lookbacks:
        for th in [-0.20, -0.25, -0.30]:
            label = f"DD({lb}d,{int(abs(th)*100)}%)+Vol"
            configs.append((label, lb, th, 'replace'))

    for lb in [21, 30, 60]:
        for th in [-0.20, -0.25, -0.30]:
            label = f"DD({lb}d,{int(abs(th)*100)}%)only"
            configs.append((label, lb, th, 'only'))

    # Build all jobs
    jobs = []
    for base_d in ANCHOR_DAYS:
        for label, lb, th, mode in configs:
            jobs.append((base_d, label, lb, th, mode))

    n_workers = min(cpu_count(), len(jobs))
    print(f"  {len(configs)} configs × {len(ANCHOR_DAYS)} anchors = {len(jobs)} backtests")
    print(f"  Using {n_workers} workers")

    all_results = {c[0]: [] for c in configs}
    all_rebals = {c[0]: [] for c in configs}

    with Pool(n_workers) as pool:
        results = pool.map(_worker, jobs)

    # Organize results by anchor order
    by_anchor_label = {}
    for base_d, label, metrics, rebal in results:
        by_anchor_label[(base_d, label)] = (metrics, rebal)

    for base_d in ANCHOR_DAYS:
        for label, lb, th, mode in configs:
            m, r = by_anchor_label[(base_d, label)]
            all_results[label].append(m)
            all_rebals[label].append(r)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    base_sharpe = avg_metrics(all_results['기준 (Mom21+Mom90+Vol5%)'])[0]['Sharpe']

    # ── Section 1: Mode A — Health + Drawdown ──
    print(f"\n{'=' * 100}")
    print(f"  1. Mode A: 기존 헬스 + 드로다운 필터 (추가 게이트)")
    print(f"     기존: Mom(21)>0 AND Mom(90)>0 AND Vol(90)≤5% + DD(N일 고점 대비 하락률)")
    print(f"{'=' * 100}")

    print(f"\n  {'':>8}", end="")
    for th in thresholds:
        print(f" {int(abs(th)*100):>5}%", end="")
    print()
    print(f"  {'─' * (8 + 7 * len(thresholds))}")

    for lb in lookbacks:
        print(f"  {lb:>3}일  ", end="")
        for th in thresholds:
            label = f"H+DD({lb}d,{int(abs(th)*100)}%)"
            avg, _ = avg_metrics(all_results[label])
            delta = avg['Sharpe'] - base_sharpe
            marker = '★' if delta > 0.03 else '↑' if delta > 0.005 else ' '
            print(f" {avg['Sharpe']:>5.3f}{marker}", end="")
        print()

    print(f"\n  기준 Sharpe: {base_sharpe:.3f}")

    print(f"\n  {'전략':>22} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5}")
    print(f"  {'─' * 75}")

    mode_a = [c for c in configs if c[3] == 'add']
    for label, lb, th, mode in mode_a:
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(all_rebals[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0.005 else ''
        print(f"  {label:>22} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f}{marker}")

    # ── Section 2: Mode B ──
    print(f"\n{'=' * 100}")
    print(f"  2. Mode B: 드로다운 + Vol만 (Mom 대체)")
    print(f"     조건: DD(N일 고점 대비) > -X% AND Vol(90) ≤ 5%")
    print(f"{'=' * 100}")

    mode_b = [c for c in configs if c[3] == 'replace']
    print(f"\n  {'전략':>22} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5}")
    print(f"  {'─' * 75}")
    for label, lb, th, mode in mode_b:
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(all_rebals[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0.005 else ''
        print(f"  {label:>22} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f}{marker}")

    # ── Section 3: Mode C ──
    print(f"\n{'=' * 100}")
    print(f"  3. Mode C: 드로다운만 (Mom/Vol 없음)")
    print(f"     조건: DD(N일 고점 대비) > -X% 만 적용")
    print(f"{'=' * 100}")

    mode_c = [c for c in configs if c[3] == 'only']
    print(f"\n  {'전략':>22} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5}")
    print(f"  {'─' * 75}")
    for label, lb, th, mode in mode_c:
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(all_rebals[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0.005 else ''
        print(f"  {label:>22} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f}{marker}")

    # ── Section 4: Rankings ──
    print(f"\n{'=' * 100}")
    print(f"  4. Sharpe 순위 Top 15")
    print(f"{'=' * 100}")

    ranked = sorted(all_results.items(),
                    key=lambda x: avg_metrics(x[1])[0]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'전략':>22} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 70}")
    for i, (label, results) in enumerate(ranked[:15], 1):
        avg, std = avg_metrics(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {i:>3}. {label:>22} {avg['Sharpe']:>7.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Section 5: Calmar Top 10 ──
    print(f"\n  Calmar 순위 Top 10")
    ranked_c = sorted(all_results.items(),
                      key=lambda x: avg_metrics(x[1])[0]['CAGR'] / abs(avg_metrics(x[1])[0]['MDD'])
                      if avg_metrics(x[1])[0]['MDD'] != 0 else 0, reverse=True)

    print(f"\n  {'순위':>3} {'전략':>22} {'Calmar':>7} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7}")
    print(f"  {'─' * 60}")
    for i, (label, results) in enumerate(ranked_c[:10], 1):
        avg, std = avg_metrics(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {i:>3}. {label:>22} {calmar:>7.2f} {avg['Sharpe']:>7.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%}")

    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
