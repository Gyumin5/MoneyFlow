#!/usr/bin/env python3
"""T40 + DD Exit sweep (parallelized).
Verify DD exit parameters are optimal on T40 universe."""

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

G_PRICES = None
G_UNIVERSE_MAPS = None


def check_coin_drawdown(ticker, prices, date, lookback, threshold):
    close = _close_to(ticker, prices, date)
    if len(close) < lookback:
        return False
    recent = close.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0:
        return False
    dd = close.iloc[-1] / peak - 1
    return dd <= threshold


def run_backtest(prices, universe_map, snapshot_days,
                  dd_lookback=0, dd_threshold=0,
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

        # Daily DD exit
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped:
            exited = []
            for ticker in list(holdings.keys()):
                if holdings[ticker] <= 0:
                    continue
                if check_coin_drawdown(ticker, prices, date, dd_lookback, dd_threshold):
                    exited.append(ticker)
            if exited:
                dd_exit_count += len(exited)
                for coin in exited:
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
        'dd_exit_count': dd_exit_count,
    }


def _empty_ext():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'rebal_count': 0, 'dd_exit_count': 0}


def _worker(args):
    base_d, label, top_n, lb, th = args
    snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
    r = run_backtest(
        G_PRICES, G_UNIVERSE_MAPS[top_n], snap_days,
        dd_lookback=lb, dd_threshold=th,
        bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10,
        post_flip_delay=5, params_base=B()
    )
    return (base_d, label, r['metrics'], r['rebal_count'], r['dd_exit_count'])


def main():
    global G_PRICES, G_UNIVERSE_MAPS

    print("Loading data...")
    um_raw = load_universe()

    G_UNIVERSE_MAPS = {}
    for top_n in [20, 30, 40, 50]:
        G_UNIVERSE_MAPS[top_n] = filter_universe(um_raw, top_n)

    all_tickers = set()
    for fm in G_UNIVERSE_MAPS.values():
        for ts in fm.values():
            all_tickers.update(ts)
    all_tickers.update(['BTC-USD', 'ETH-USD'])

    G_PRICES = load_all_prices(all_tickers)
    print(f"  {len(G_PRICES)} tickers loaded")

    t0 = time.time()

    # DD exit sweep for both T40 and T50
    lookbacks = [21, 30, 45, 60, 90]
    thresholds = [-0.15, -0.20, -0.25, -0.30, -0.40]

    configs = []
    for top_n in [20, 30, 40, 50]:
        # Baseline (no DD)
        configs.append((f'T{top_n}+BL (기준)', top_n, 0, 0))
        # DD sweep
        for lb in lookbacks:
            for th in thresholds:
                label = f"T{top_n}+DD({lb}d,{int(abs(th)*100)}%)"
                configs.append((label, top_n, lb, th))

    jobs = []
    for base_d in ANCHOR_DAYS:
        for label, top_n, lb, th in configs:
            jobs.append((base_d, label, top_n, lb, th))

    n_workers = min(cpu_count(), len(jobs))
    print(f"  {len(configs)} configs × {len(ANCHOR_DAYS)} anchors = {len(jobs)} backtests")
    print(f"  Using {n_workers} workers")

    all_results = {c[0]: [] for c in configs}
    all_rebals = {c[0]: [] for c in configs}
    all_exits = {c[0]: [] for c in configs}

    with Pool(n_workers) as pool:
        results = pool.map(_worker, jobs)

    by_key = {}
    for base_d, label, metrics, rebal, exits in results:
        by_key[(base_d, label)] = (metrics, rebal, exits)

    for base_d in ANCHOR_DAYS:
        for label, top_n, lb, th in configs:
            m, r, e = by_key[(base_d, label)]
            all_results[label].append(m)
            all_rebals[label].append(r)
            all_exits[label].append(e)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    def avg_m(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    # ── T40 results ──
    for top_n in [20, 30, 40, 50]:
        base_label = f'T{top_n}+BL (기준)'
        base_sharpe = avg_m(all_results[base_label])[0]['Sharpe']

        print(f"\n{'=' * 100}")
        print(f"  T{top_n} + BL15% + DD Exit → 현금")
        print(f"{'=' * 100}")

        # Heatmap
        print(f"\n  Sharpe 히트맵 (기준: {base_sharpe:.3f})")
        print(f"\n  {'':>8}", end="")
        for th in thresholds:
            print(f" {int(abs(th)*100):>6}%", end="")
        print()
        print(f"  {'─' * (8 + 8 * len(thresholds))}")

        for lb in lookbacks:
            print(f"  {lb:>3}일  ", end="")
            for th in thresholds:
                label = f"T{top_n}+DD({lb}d,{int(abs(th)*100)}%)"
                avg, _ = avg_m(all_results[label])
                delta = avg['Sharpe'] - base_sharpe
                marker = '★' if delta > 0.03 else '↑' if delta > 0.005 else '↓' if delta < -0.03 else ' '
                print(f" {avg['Sharpe']:>5.3f}{marker}", end="")
            print()

        # Detail
        print(f"\n  {'전략':>24} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'탈출':>5}")
        print(f"  {'─' * 80}")

        # Baseline first
        avg, std = avg_m(all_results[base_label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        avg_exit = np.mean(all_exits[base_label])
        print(f"  {base_label:>24} {avg['Sharpe']:>7.3f} {'+0.000':>7} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_exit:>5.0f}")

        # Then sorted by Sharpe
        tn_configs = [(l, tn, lb, th) for l, tn, lb, th in configs
                      if tn == top_n and lb > 0]
        ranked = sorted(tn_configs,
                        key=lambda x: avg_m(all_results[x[0]])[0]['Sharpe'], reverse=True)

        for label, tn, lb, th in ranked:
            avg, std = avg_m(all_results[label])
            calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
            delta = avg['Sharpe'] - base_sharpe
            avg_exit = np.mean(all_exits[label])
            marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0.005 else ''
            print(f"  {label:>24} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
                  f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
                  f" {avg_exit:>5.0f}{marker}")

    # ── Cross comparison ──
    print(f"\n{'=' * 100}")
    print(f"  전체 유니버스 최종 비교 (Top 15)")
    print(f"{'=' * 100}")

    all_ranked = sorted(all_results.items(),
                        key=lambda x: avg_m(x[1])[0]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'전략':>24} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'탈출':>5}")
    print(f"  {'─' * 80}")
    for i, (label, results) in enumerate(all_ranked[:15], 1):
        avg, std = avg_m(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        avg_exit = np.mean(all_exits[label])
        print(f"  {i:>3}. {label:>24} {avg['Sharpe']:>7.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_exit:>5.0f}")

    # MDD ranking
    print(f"\n  MDD 순위 Top 10")
    ranked_mdd = sorted(all_results.items(),
                        key=lambda x: avg_m(x[1])[0]['MDD'], reverse=True)
    print(f"\n  {'순위':>3} {'전략':>24} {'MDD':>7} {'Sharpe':>7} {'CAGR':>8} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for i, (label, results) in enumerate(ranked_mdd[:10], 1):
        avg, std = avg_m(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {i:>3}. {label:>24} {avg['MDD']:>6.1%} {avg['Sharpe']:>7.3f}"
              f" {avg['CAGR']:>+7.1%} {calmar:>7.2f}")

    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
