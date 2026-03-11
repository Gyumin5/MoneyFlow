#!/usr/bin/env python3
"""Test snapshot-averaging (Method B) vs pure tranche (Method A).
Method B: Single portfolio, 3 frozen weight snapshots averaged.
All tests use 10-date averaging."""

import os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, init_pool, run_single,
    get_universe_for_date, resolve_canary, get_healthy_coins,
    select_coins, compute_weights, apply_risk, should_rebalance,
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


# ═══════════════════════════════════════════════════════════════════
# Method B: Snapshot Averaging Backtest
# ═══════════════════════════════════════════════════════════════════

def compute_signal_weights(prices, universe_map, date, params, state):
    """Compute target weights for a given date using full signal logic."""
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
    """Average N weight snapshots into combined target."""
    if not snapshots:
        return {'CASH': 1.0}

    combined = {}
    n = len(snapshots)
    for snap in snapshots:
        for t, w in snap.items():
            combined[t] = combined.get(t, 0) + w / n

    # Normalize (should sum to ~1.0 but clean up)
    total = sum(combined.values())
    if total > 0:
        return {t: w / total for t, w in combined.items()}
    return {'CASH': 1.0}


def run_snapshot_backtest(prices, universe_map, snapshot_days, post_flip_delay=5, params_base=None):
    """Run backtest with snapshot averaging method.

    Args:
        snapshot_days: list of anchor days, e.g. [1, 10, 19]
        post_flip_delay: days after OFF→ON flip for refresh
        params_base: base Params for signal computation
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

    # Snapshot storage: one weight dict per snapshot slot
    n_snap = len(snapshot_days)
    snapshots = [{'CASH': 1.0} for _ in range(n_snap)]
    snap_done = {}  # 'YYYY-MM_snapN' → True (prevent double trigger)

    portfolio_values = []
    rebal_count = 0

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

        # 1. Canary
        canary_on = resolve_canary(prices, date, params_base, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['canary_on'] = canary_on
        state['is_month_change'] = imc

        # 2. Determine if any snapshot needs updating
        need_rebal = False

        # First day
        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
            need_rebal = True

        # Canary flip → update ALL snapshots
        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
            else:
                for si in range(n_snap):
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True

        # PFD: post-flip refresh
        elif post_flip_delay > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= post_flip_delay:
                    state['post_flip_refreshed'] = True
                    for si in range(n_snap):
                        snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
                    need_rebal = True

        # Regular monthly snapshots (each on its anchor day)
        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights(prices, universe_map, date, params_base, state)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        # 3. Compute combined target
        combined = merge_snapshots(snapshots)

        # 4. Rebalance if needed
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
    ym = calc_yearly_metrics(pvdf)

    return {
        'metrics': m,
        'yearly': ym,
        'rebal_count': rebal_count,
        'pv': pvdf,
    }


def _empty():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'yearly': {}, 'rebal_count': 0, 'pv': pd.DataFrame()}


# ═══════════════════════════════════════════════════════════════════
# Pure tranche (Method A) — for comparison
# ═══════════════════════════════════════════════════════════════════

def run_pure_tranche(prices, universe_map, base_anchor, n_tranches=3, spacing=9, pfd=5):
    """Run pure tranche: N independent backtests, sum equity curves."""
    tranche_pvs = []
    total_rebal = 0
    for ti in range(n_tranches):
        anchor = (base_anchor - 1 + ti * spacing) % 28 + 1
        p = B(rebalancing=f'RX{anchor}', post_flip_delay=pfd,
              initial_capital=10000.0 / n_tranches)
        from strategy_engine import run_backtest
        r = run_backtest(prices, universe_map, p)
        if r['pv'] is not None and len(r['pv']) > 0:
            tranche_pvs.append(r['pv'])
            total_rebal += r['rebal_count']

    if not tranche_pvs:
        return _empty()

    combined = tranche_pvs[0].copy()
    for tv in tranche_pvs[1:]:
        combined = combined.add(tv, fill_value=0)

    m = calc_metrics(combined)
    ym = calc_yearly_metrics(combined)
    return {'metrics': m, 'yearly': ym, 'rebal_count': total_rebal, 'pv': combined}


# ═══════════════════════════════════════════════════════════════════
# Date-averaged comparison
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()

    # Test across 10 anchor offsets
    results_A = []  # Pure tranche
    results_B = []  # Snapshot averaging
    results_base = []  # Single monthly (baseline)

    for di, base_d in enumerate(ANCHOR_DAYS):
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        # Baseline: single portfolio, single anchor
        p = B(rebalancing=f'RX{base_d}', post_flip_delay=5)
        from strategy_engine import run_backtest
        r_base = run_backtest(prices, universe, p)
        results_base.append(r_base['metrics'])

        # Method A: Pure tranche (3 independent)
        r_a = run_pure_tranche(prices, universe, base_d)
        results_A.append(r_a['metrics'])

        # Method B: Snapshot averaging
        snap_days = [(base_d - 1 + i * 9) % 28 + 1 for i in range(3)]
        r_b = run_snapshot_backtest(prices, universe, snap_days, post_flip_delay=5, params_base=B())
        results_B.append(r_b['metrics'])

        print(f"Base={r_base['metrics']['Sharpe']:.3f}  "
              f"A={r_a['metrics']['Sharpe']:.3f}  "
              f"B={r_b['metrics']['Sharpe']:.3f}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Average metrics
    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    base_avg, base_std = avg_metrics(results_base)
    a_avg, a_std = avg_metrics(results_A)
    b_avg, b_std = avg_metrics(results_B)

    print(f"\n{'=' * 90}")
    print(f"  날짜 평균 비교: 베이스라인 vs 순수 트랜치(A) vs 스냅샷 평균(B)")
    print(f"{'=' * 90}")
    print(f"\n  {'전략':<28} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 70}")

    for name, avg, std in [
        ('PFD5 단일 (베이스라인)', base_avg, base_std),
        ('A: 순수 트랜치 3분할+PFD5', a_avg, a_std),
        ('B: 스냅샷 평균+PFD5', b_avg, b_std),
    ]:
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {name:<28} {avg['Sharpe']:>7.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}")

    # Per-anchor comparison
    print(f"\n  앵커별 Sharpe:")
    print(f"  {'앵커':>5} {'Base':>7} {'A:순수':>7} {'B:스냅샷':>7}")
    print(f"  {'─' * 30}")
    for i, d in enumerate(ANCHOR_DAYS):
        print(f"  {d:>5} {results_base[i]['Sharpe']:>7.3f}"
              f" {results_A[i]['Sharpe']:>7.3f}"
              f" {results_B[i]['Sharpe']:>7.3f}")

    # Also test snapshot with different number of snapshots
    print(f"\n{'=' * 90}")
    print(f"  스냅샷 개수별 비교")
    print(f"{'=' * 90}")

    for n_snap in [2, 3, 4]:
        spacing = 28 // n_snap
        snap_results = []
        for base_d in ANCHOR_DAYS:
            snap_days = [(base_d - 1 + i * spacing) % 28 + 1 for i in range(n_snap)]
            r = run_snapshot_backtest(prices, universe, snap_days, post_flip_delay=5, params_base=B())
            snap_results.append(r['metrics'])
        avg, std = avg_metrics(snap_results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  B{n_snap}: 스냅샷 {n_snap}개 (간격 {spacing}일)"
              f"  Sharpe {avg['Sharpe']:.3f} σ={std:.3f}"
              f"  CAGR {avg['CAGR']:+.1%} MDD {avg['MDD']:.1%} Cal {calmar:.2f}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
