#!/usr/bin/env python3
"""Daily drawdown EXIT test (parallelized).
Base: T50 + BL15% + Drift10% + PFD5 (Sharpe 1.671)

Tests DD as a DAILY EXIT mechanism:
  - Every day, check held coins' drawdown from N-day high
  - If DD exceeds threshold, sell that coin
  - Two exit actions: cash (hold until rebal) or rebalance (redistribute)
"""

import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    load_all_prices, filter_universe, load_universe,
    resolve_canary,
    execute_rebalance, _port_val, get_price, _close_to,
    calc_metrics,
)
from test_matrix import (
    B, merge_snapshots, calc_current_weights, calc_half_turnover,
    compute_signal_weights_filtered,
    check_blacklist, update_blacklist, ANCHOR_DAYS, _empty,
)

# Global data (shared via fork)
G_PRICES = None
G_UNIVERSE_MAP = None


def check_coin_drawdown(ticker, prices, date, lookback, threshold):
    """Returns True if coin should be EXITED (drawdown exceeded)."""
    close = _close_to(ticker, prices, date)
    if len(close) < lookback:
        return False
    recent = close.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0:
        return False
    dd = close.iloc[-1] / peak - 1
    return dd <= threshold


def run_dd_exit_backtest(prices, universe_map, snapshot_days,
                          dd_lookback=30, dd_threshold=-0.25,
                          exit_action='cash',
                          bl_drop=-0.15, bl_days=7,
                          drift_threshold=0.10,
                          post_flip_delay=5, params_base=None):
    if params_base is None:
        params_base = B()

    btc = prices.get('BTC-USD')
    if btc is None:
        return _empty_ext()

    all_dates = btc.index[(btc.index >= params_base.start_date) &
                          (btc.index <= params_base.end_date)]
    if len(all_dates) == 0:
        return _empty_ext()

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
    dd_exit_count = 0

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

        # Blacklist check
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

        # ── Daily DD exit check ──
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped:
            exited_coins = []
            for ticker in list(holdings.keys()):
                if holdings[ticker] <= 0:
                    continue
                if check_coin_drawdown(ticker, prices, date, dd_lookback, dd_threshold):
                    exited_coins.append(ticker)

            if exited_coins:
                dd_exit_count += len(exited_coins)

                if exit_action == 'cash':
                    for coin in exited_coins:
                        p = get_price(coin, prices, date)
                        cash += holdings[coin] * p * (1 - params_base.tx_cost)
                        del holdings[coin]
                        for si in range(n_snap):
                            if coin in snapshots[si]:
                                removed_w = snapshots[si].pop(coin)
                                snapshots[si]['CASH'] = snapshots[si].get('CASH', 0) + removed_w
                                total = sum(snapshots[si].values())
                                if total > 0:
                                    snapshots[si] = {t: w/total for t, w in snapshots[si].items()}

                elif exit_action == 'rebalance':
                    for coin in exited_coins:
                        for si in range(n_snap):
                            if coin in snapshots[si]:
                                del snapshots[si][coin]
                                total = sum(snapshots[si].values())
                                if total > 0:
                                    snapshots[si] = {t: w/total for t, w in snapshots[si].items()}
                                else:
                                    snapshots[si] = {'CASH': 1.0}
                    need_rebal = True

        # ── Standard snapshot logic ──
        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights_filtered(
                    prices, universe_map, date, params_base, state, blacklist)
            need_rebal = True

        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights_filtered(
                        prices, universe_map, date, params_base, state, blacklist)
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
                        snapshots[si] = compute_signal_weights_filtered(
                            prices, universe_map, date, params_base, state, blacklist)
                    need_rebal = True

        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights_filtered(
                        prices, universe_map, date, params_base, state, blacklist)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        if bl_drop < 0 and newly_bl:
            need_rebal = True

        combined = merge_snapshots(snapshots)

        # Drift check
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
        return _empty_ext()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    return {
        'metrics': m,
        'rebal_count': rebal_count,
        'drift_rebal_count': drift_rebal_count,
        'dd_exit_count': dd_exit_count,
    }


def _empty_ext():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'rebal_count': 0, 'drift_rebal_count': 0, 'dd_exit_count': 0}


def _worker(args):
    """Worker for one (anchor, config) combination."""
    base_d, label, lb, th, action = args
    snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
    r = run_dd_exit_backtest(
        G_PRICES, G_UNIVERSE_MAP, snap_days,
        dd_lookback=lb, dd_threshold=th,
        exit_action=action,
        bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10,
        post_flip_delay=5, params_base=B()
    )
    return (base_d, label, r['metrics'], r['rebal_count'], r['dd_exit_count'])


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
    configs.append(('기준 (DD탈출 없음)', 0, 0, 'cash'))

    # Cash exit
    for lb in lookbacks:
        for th in thresholds:
            label = f"탈출→현금 DD({lb}d,{int(abs(th)*100)}%)"
            configs.append((label, lb, th, 'cash'))

    # Rebalance exit
    for lb in lookbacks:
        for th in [-0.20, -0.25, -0.30]:
            label = f"탈출→재분배 DD({lb}d,{int(abs(th)*100)}%)"
            configs.append((label, lb, th, 'rebalance'))

    # Build all jobs
    jobs = []
    for base_d in ANCHOR_DAYS:
        for label, lb, th, action in configs:
            jobs.append((base_d, label, lb, th, action))

    n_workers = min(cpu_count(), len(jobs))
    print(f"  {len(configs)} configs × {len(ANCHOR_DAYS)} anchors = {len(jobs)} backtests")
    print(f"  Using {n_workers} workers")

    all_results = {c[0]: [] for c in configs}
    all_rebals = {c[0]: [] for c in configs}
    all_exits = {c[0]: [] for c in configs}

    with Pool(n_workers) as pool:
        results = pool.map(_worker, jobs)

    # Organize results
    by_anchor_label = {}
    for base_d, label, metrics, rebal, exits in results:
        by_anchor_label[(base_d, label)] = (metrics, rebal, exits)

    for base_d in ANCHOR_DAYS:
        for label, lb, th, action in configs:
            m, r, e = by_anchor_label[(base_d, label)]
            all_results[label].append(m)
            all_rebals[label].append(r)
            all_exits[label].append(e)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    base_sharpe = avg_metrics(all_results['기준 (DD탈출 없음)'])[0]['Sharpe']

    # ── Section 1: Cash exit heatmap ──
    print(f"\n{'=' * 100}")
    print(f"  1. 탈출 → 현금 (DD 초과 시 매도, 다음 리밸런싱까지 현금 보유)")
    print(f"{'=' * 100}")

    print(f"\n  Sharpe 히트맵 (기준: {base_sharpe:.3f})")
    print(f"\n  {'':>8}", end="")
    for th in thresholds:
        print(f" {int(abs(th)*100):>6}%", end="")
    print()
    print(f"  {'─' * (8 + 8 * len(thresholds))}")

    for lb in lookbacks:
        print(f"  {lb:>3}일  ", end="")
        for th in thresholds:
            label = f"탈출→현금 DD({lb}d,{int(abs(th)*100)}%)"
            avg, _ = avg_metrics(all_results[label])
            delta = avg['Sharpe'] - base_sharpe
            marker = '★' if delta > 0.03 else '↑' if delta > 0.005 else '↓' if delta < -0.03 else ' '
            print(f" {avg['Sharpe']:>5.3f}{marker}", end="")
        print()

    # Detail table
    print(f"\n  {'전략':>28} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'탈출':>5}")
    print(f"  {'─' * 80}")

    cash_configs = [c for c in configs if c[3] == 'cash']
    for label, lb, th, action in cash_configs:
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_exit = np.mean(all_exits[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0.005 else ''
        print(f"  {label:>28} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_exit:>5.0f}{marker}")

    # ── Section 2: Rebalance exit ──
    print(f"\n{'=' * 100}")
    print(f"  2. 탈출 → 재분배 (DD 초과 시 매도, 나머지 코인에 즉시 재분배)")
    print(f"{'=' * 100}")

    rebal_configs = [c for c in configs if c[3] == 'rebalance']
    print(f"\n  {'전략':>28} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'탈출':>5}")
    print(f"  {'─' * 80}")
    for label, lb, th, action in rebal_configs:
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_exit = np.mean(all_exits[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0.005 else ''
        print(f"  {label:>28} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_exit:>5.0f}{marker}")

    # ── Section 3: Rankings ──
    print(f"\n{'=' * 100}")
    print(f"  3. Sharpe 순위 Top 15")
    print(f"{'=' * 100}")

    ranked = sorted(all_results.items(),
                    key=lambda x: avg_metrics(x[1])[0]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'전략':>28} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'탈출':>5}")
    print(f"  {'─' * 80}")
    for i, (label, results) in enumerate(ranked[:15], 1):
        avg, std = avg_metrics(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        avg_exit = np.mean(all_exits[label])
        print(f"  {i:>3}. {label:>28} {avg['Sharpe']:>7.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_exit:>5.0f}")

    # ── Section 4: MDD comparison ──
    print(f"\n{'=' * 100}")
    print(f"  4. MDD 순위 Top 10 (낮을수록 좋음)")
    print(f"{'=' * 100}")

    ranked_mdd = sorted(all_results.items(),
                        key=lambda x: avg_metrics(x[1])[0]['MDD'], reverse=True)

    print(f"\n  {'순위':>3} {'전략':>28} {'MDD':>7} {'Sharpe':>7} {'CAGR':>8} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for i, (label, results) in enumerate(ranked_mdd[:10], 1):
        avg, std = avg_metrics(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {i:>3}. {label:>28} {avg['MDD']:>6.1%} {avg['Sharpe']:>7.3f}"
              f" {avg['CAGR']:>+7.1%} {calmar:>7.2f}")

    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
