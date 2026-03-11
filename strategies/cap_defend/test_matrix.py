#!/usr/bin/env python3
"""Matrix test: Universe size × Market Breadth × Blacklist.
Base strategy: Snapshot B + PFD5 + Drift 10%.
All tests use 10-date averaging."""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, filter_universe, load_universe,
    get_universe_for_date, resolve_canary, get_healthy_coins,
    select_coins, compute_weights,
    execute_rebalance, _port_val, get_price,
    calc_metrics, calc_yearly_metrics
)

ANCHOR_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]


def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
    )
    base.update(kw)
    return Params(**base)


def compute_signal_weights(prices, universe_map, date, params, state):
    canary_on = state.get('canary_on', False)
    if not canary_on:
        return {'CASH': 1.0}
    universe = get_universe_for_date(universe_map, date)
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


def merge_snapshots(snapshots):
    if not snapshots:
        return {'CASH': 1.0}
    combined = {}
    n = len(snapshots)
    for snap in snapshots:
        for t, w in snap.items():
            combined[t] = combined.get(t, 0) + w / n
    total = sum(combined.values())
    if total > 0:
        return {t: w / total for t, w in combined.items()}
    return {'CASH': 1.0}


def calc_current_weights(holdings, cash, prices, date):
    pv = _port_val(holdings, cash, prices, date)
    if pv <= 0:
        return {'CASH': 1.0}
    weights = {}
    for t, units in holdings.items():
        p = get_price(t, prices, date)
        val = units * p
        if val > 0:
            weights[t] = val / pv
    cash_w = cash / pv
    if cash_w > 0.001:
        weights['CASH'] = cash_w
    return weights


def calc_half_turnover(current_w, target_w):
    all_keys = set(current_w.keys()) | set(target_w.keys())
    l1 = sum(abs(target_w.get(k, 0) - current_w.get(k, 0)) for k in all_keys)
    return l1 / 2


# ═══════════════════════════════════════════════════════════════════
# Market Breadth: % of universe coins above SMA(20)
# ═══════════════════════════════════════════════════════════════════

def calc_breadth(prices, universe, date, sma_period=20):
    """Calculate % of universe coins with price > SMA(N)."""
    above = 0
    total = 0
    for ticker in universe:
        df = prices.get(ticker)
        if df is None or date not in df.index:
            continue
        loc = df.index.get_loc(date)
        if loc < sma_period:
            continue
        close = df['Close'].values
        sma = np.mean(close[loc - sma_period + 1:loc + 1])
        total += 1
        if close[loc] > sma:
            above += 1
    if total == 0:
        return 0.5
    return above / total


# ═══════════════════════════════════════════════════════════════════
# Blacklist: coins with extreme daily drop
# ═══════════════════════════════════════════════════════════════════

def check_blacklist(holdings, prices, date, threshold, blacklist, bl_days):
    """Check held coins for extreme daily drops. Add to blacklist."""
    newly_blacklisted = []
    for ticker in list(holdings.keys()):
        if holdings[ticker] <= 0:
            continue
        df = prices.get(ticker)
        if df is None or date not in df.index:
            continue
        loc = df.index.get_loc(date)
        if loc < 1:
            continue
        close = df['Close'].values
        daily_ret = close[loc] / close[loc - 1] - 1
        if daily_ret <= threshold:  # e.g., -0.15
            blacklist[ticker] = bl_days  # blacklist for N days
            newly_blacklisted.append(ticker)
    return newly_blacklisted


def update_blacklist(blacklist):
    """Decrement blacklist counters, remove expired."""
    expired = []
    for ticker in list(blacklist.keys()):
        blacklist[ticker] -= 1
        if blacklist[ticker] <= 0:
            expired.append(ticker)
    for t in expired:
        del blacklist[t]


# ═══════════════════════════════════════════════════════════════════
# Main backtest with all features
# ═══════════════════════════════════════════════════════════════════

def run_matrix_backtest(prices, universe_map, snapshot_days,
                         breadth_threshold=0.0,
                         bl_drop=0.0, bl_days=0,
                         drift_threshold=0.10,
                         post_flip_delay=5, params_base=None):
    """
    Args:
        breadth_threshold: if breadth < this, scale to 50% cash (0 = disabled)
        bl_drop: blacklist coins dropping more than this (e.g., -0.15) (0 = disabled)
        bl_days: how many days to blacklist
        drift_threshold: drift rebalancing threshold
    """
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
    blacklist = {}  # ticker -> remaining days

    portfolio_values = []
    rebal_count = 0
    drift_rebal_count = 0
    breadth_trigger_count = 0
    bl_trigger_count = 0

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

        # Update blacklist counters daily
        update_blacklist(blacklist)

        # Check for newly blacklisted coins
        newly_bl = []
        if bl_drop < 0 and canary_on and holdings:
            newly_bl = check_blacklist(holdings, prices, date, bl_drop, blacklist, bl_days)
            if newly_bl:
                bl_trigger_count += len(newly_bl)
                # Remove blacklisted coins from all snapshots
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

        # Also filter blacklisted coins from signal computation
        state['_blacklist'] = set(blacklist.keys())

        need_rebal = False
        force_rebal = False

        # First day
        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights_filtered(
                    prices, universe_map, date, params_base, state, blacklist)
            need_rebal = True
            force_rebal = True

        # Canary flip
        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights_filtered(
                        prices, universe_map, date, params_base, state, blacklist)
            else:
                for si in range(n_snap):
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True
            force_rebal = True

        # PFD refresh
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
                    force_rebal = True

        # Regular monthly snapshots
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

        # Blacklist triggered rebal (newly blacklisted coins need to be sold)
        if bl_drop < 0 and 'newly_bl' in locals() and newly_bl:
            need_rebal = True
            force_rebal = True
            newly_bl = []  # reset for next iteration

        combined = merge_snapshots(snapshots)

        # Market breadth scaling
        if breadth_threshold > 0 and canary_on:
            universe = get_universe_for_date(universe_map, date)
            breadth = calc_breadth(prices, universe, date)
            if breadth < breadth_threshold:
                # Scale to 50% cash
                scaled = {}
                cash_add = 0
                for t, w in combined.items():
                    if t == 'CASH':
                        scaled[t] = w
                    else:
                        scaled[t] = w * 0.5
                        cash_add += w * 0.5
                scaled['CASH'] = scaled.get('CASH', 0) + cash_add
                if combined != scaled:
                    combined = scaled
                    need_rebal = True
                    breadth_trigger_count += 1

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
        return _empty()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)

    return {
        'metrics': m,
        'rebal_count': rebal_count,
        'drift_rebal_count': drift_rebal_count,
        'breadth_trigger_count': breadth_trigger_count,
        'bl_trigger_count': bl_trigger_count,
        'pv': pvdf,
    }


def compute_signal_weights_filtered(prices, universe_map, date, params, state, blacklist):
    """Compute weights excluding blacklisted coins."""
    canary_on = state.get('canary_on', False)
    if not canary_on:
        return {'CASH': 1.0}
    universe = get_universe_for_date(universe_map, date)
    # Filter out blacklisted coins from universe
    if blacklist:
        universe = [t for t in universe if t not in blacklist]
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


def _empty():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'rebal_count': 0, 'drift_rebal_count': 0,
            'breadth_trigger_count': 0, 'bl_trigger_count': 0,
            'pv': pd.DataFrame()}


def main():
    print("Loading data...")

    # Load all data for largest universe (Top 50)
    # We'll filter down for smaller universes
    um_raw = load_universe()

    # Pre-build universe maps for each top_n
    universe_sizes = [20, 30, 40, 50]
    universe_maps = {}
    all_prices = {}

    for top_n in universe_sizes:
        fm = filter_universe(um_raw, top_n)
        universe_maps[top_n] = fm

    # Load prices for all tickers across all universes
    all_tickers = set()
    for fm in universe_maps.values():
        for ts in fm.values():
            all_tickers.update(ts)
    all_tickers.update(['BTC-USD', 'ETH-USD'])

    from strategy_engine import load_all_prices
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()

    # Matrix dimensions
    breadth_options = [0, 0.25, 0.35]       # OFF, <25%, <35%
    bl_options = [(0, 0), (-0.15, 7), (-0.20, 7)]  # OFF, -15%/7d, -20%/7d

    # Build all configs
    configs = []
    for top_n in universe_sizes:
        for br in breadth_options:
            for bl_drop, bl_days in bl_options:
                label = f"T{top_n}"
                if br > 0:
                    label += f"+BR{int(br*100)}%"
                if bl_drop < 0:
                    label += f"+BL{int(abs(bl_drop)*100)}%"
                configs.append((label, top_n, br, bl_drop, bl_days))

    print(f"  {len(configs)} configs × {len(ANCHOR_DAYS)} anchors = {len(configs)*len(ANCHOR_DAYS)} backtests")

    # Results storage
    all_results = {c[0]: [] for c in configs}
    all_rebals = {c[0]: [] for c in configs}

    for base_d in ANCHOR_DAYS:
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for label, top_n, br, bl_drop, bl_days in configs:
            snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
            r = run_matrix_backtest(
                prices, universe_maps[top_n], snap_days,
                breadth_threshold=br,
                bl_drop=bl_drop, bl_days=bl_days,
                drift_threshold=0.10,
                post_flip_delay=5, params_base=B()
            )
            all_results[label].append(r['metrics'])
            all_rebals[label].append(r['rebal_count'])

        # Show a few samples
        s50 = all_results['T50'][-1]['Sharpe']
        s30 = all_results['T30'][-1]['Sharpe']
        s20 = all_results['T20'][-1]['Sharpe']
        print(f"T50={s50:.3f} T30={s30:.3f} T20={s20:.3f}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    base_sharpe = avg_metrics(all_results['T50'])[0]['Sharpe']

    # ── Section 1: Universe size comparison (baseline only) ──
    print(f"\n{'=' * 100}")
    print(f"  1. 유니버스 크기별 비교 (브레드스/블랙리스트 OFF)")
    print(f"{'=' * 100}")

    print(f"\n  {'전략':>12} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 60}")
    for top_n in universe_sizes:
        label = f"T{top_n}"
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0 else ''
        print(f"  {'Top'+str(top_n):>12} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}{marker}")

    # ── Section 2: Full matrix ──
    print(f"\n{'=' * 100}")
    print(f"  2. 전체 매트릭스 (유니버스 × 브레드스 × 블랙리스트)")
    print(f"{'=' * 100}")

    print(f"\n  {'전략':>24} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5}")
    print(f"  {'─' * 85}")

    for top_n in universe_sizes:
        for label, tn, br, bl_drop, bl_days in configs:
            if tn != top_n:
                continue
            avg, std = avg_metrics(all_results[label])
            calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
            delta = avg['Sharpe'] - base_sharpe
            avg_rebal = np.mean(all_rebals[label])
            marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0 else ''
            print(f"  {label:>24} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
                  f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
                  f" {avg_rebal:>5.0f}{marker}")
        print()  # separator between universe sizes

    # ── Section 3: Top 10 ranking ──
    print(f"{'=' * 100}")
    print(f"  3. Sharpe 순위 Top 10")
    print(f"{'=' * 100}")

    ranked = sorted(all_results.items(),
                    key=lambda x: avg_metrics(x[1])[0]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'전략':>24} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 70}")
    for i, (label, results) in enumerate(ranked[:10], 1):
        avg, std = avg_metrics(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {i:>3}. {label:>24} {avg['Sharpe']:>7.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Section 4: Robustness (lowest σ) ──
    print(f"\n  σ(Sharpe) 순위 Top 10 (낮을수록 강건)")
    ranked_s = sorted(all_results.items(),
                      key=lambda x: avg_metrics(x[1])[1])

    print(f"\n  {'순위':>3} {'전략':>24} {'σ(S)':>6} {'Sharpe':>7}")
    print(f"  {'─' * 45}")
    for i, (label, results) in enumerate(ranked_s[:10], 1):
        avg, std = avg_metrics(results)
        print(f"  {i:>3}. {label:>24} {std:>6.3f} {avg['Sharpe']:>7.3f}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
