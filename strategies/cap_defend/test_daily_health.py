#!/usr/bin/env python3
"""Test daily health check for held coins.
When a held coin fails health filter, exit immediately.
Two exit modes: cash (hold cash until next anchor) or replace (buy next best coin).
All tests use 10-date averaging."""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data,
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


def check_coin_health(ticker, prices, date, params):
    """Check if a single coin passes health filter (Mom21+Mom90+Vol)."""
    df = prices.get(ticker)
    if df is None or date not in df.index:
        return False
    loc = df.index.get_loc(date)
    if loc < 90:
        return False
    close = df['Close'].values

    # Mom short (21)
    if loc >= params.health_mom_short:
        mom_s = close[loc] / close[loc - params.health_mom_short] - 1
        if mom_s <= 0:
            return False

    # Mom long (90)
    if loc >= params.health_mom_long:
        mom_l = close[loc] / close[loc - params.health_mom_long] - 1
        if mom_l <= 0:
            return False

    # Volatility cap
    if loc >= 90:
        rets = np.diff(close[loc-90:loc+1]) / close[loc-90:loc]
        vol = np.std(rets) * np.sqrt(365)
        if vol > params.vol_cap:
            return False

    return True


def check_coin_sma(ticker, prices, date, sma_period):
    """Check if coin price > SMA(N). Faster trend filter."""
    df = prices.get(ticker)
    if df is None or date not in df.index:
        return False
    loc = df.index.get_loc(date)
    if loc < sma_period:
        return False
    close = df['Close'].values
    sma = np.mean(close[loc - sma_period + 1:loc + 1])
    return close[loc] > sma


def run_daily_health_backtest(prices, universe_map, snapshot_days,
                               daily_health=False, exit_mode='cash',
                               exit_filter='health',
                               drift_threshold=0.10,
                               post_flip_delay=5, params_base=None):
    """Run snapshot backtest with optional daily health check.

    Args:
        daily_health: if True, check health of held coins daily
        exit_mode: 'cash' = sell to cash, 'replace' = buy next best coin
        exit_filter: 'health' = full health filter (Mom21+Mom90+Vol)
                     'sma10' = Price < SMA(10) → exit
                     'sma20' = Price < SMA(20) → exit
                     'health+sma20' = health OR SMA(20) fail → exit
        drift_threshold: drift rebalancing threshold (0 = disabled)
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
    portfolio_values = []
    rebal_count = 0
    health_exit_count = 0
    drift_rebal_count = 0

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

        need_rebal = False
        force_rebal = False

        # First day
        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
            need_rebal = True
            force_rebal = True

        # Canary flip
        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
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
                        snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
                    need_rebal = True
                    force_rebal = True

        # Regular monthly snapshots
        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights(prices, universe_map, date, params_base, state)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        # ── Daily health check for held coins ──
        if daily_health and canary_on and not canary_flipped and not force_rebal:
            held_coins = [t for t in holdings if holdings[t] > 0]
            failed_coins = []
            for coin in held_coins:
                failed = False
                if exit_filter == 'health':
                    failed = not check_coin_health(coin, prices, date, params_base)
                elif exit_filter == 'sma10':
                    failed = not check_coin_sma(coin, prices, date, 10)
                elif exit_filter == 'sma20':
                    failed = not check_coin_sma(coin, prices, date, 20)
                elif exit_filter == 'health+sma20':
                    failed = (not check_coin_health(coin, prices, date, params_base)
                              or not check_coin_sma(coin, prices, date, 20))
                if failed:
                    failed_coins.append(coin)

            if failed_coins:
                health_exit_count += len(failed_coins)

                if exit_mode == 'cash':
                    # Remove failed coins from all snapshots → replace with CASH
                    for si in range(n_snap):
                        snap = snapshots[si]
                        removed_weight = 0
                        for coin in failed_coins:
                            if coin in snap:
                                removed_weight += snap.pop(coin)
                        if removed_weight > 0:
                            snap['CASH'] = snap.get('CASH', 0) + removed_weight
                        # Normalize
                        total = sum(snap.values())
                        if total > 0:
                            snapshots[si] = {t: w/total for t, w in snap.items()}

                elif exit_mode == 'replace':
                    # Remove failed coins, recompute with fresh picks
                    universe = get_universe_for_date(universe_map, date)
                    healthy = get_healthy_coins(prices, universe, date, params_base, state)
                    for si in range(n_snap):
                        snap = snapshots[si]
                        had_failed = any(coin in snap for coin in failed_coins)
                        if had_failed:
                            # Recompute this snapshot entirely
                            if healthy:
                                picks = select_coins(healthy, prices, date, params_base, state)
                                if picks:
                                    snapshots[si] = compute_weights(picks, prices, date, params_base, state)
                                else:
                                    snapshots[si] = {'CASH': 1.0}
                            else:
                                snapshots[si] = {'CASH': 1.0}

                need_rebal = True
                force_rebal = True  # health exit is forced

        combined = merge_snapshots(snapshots)

        # Drift check on non-rebal days
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
    ym = calc_yearly_metrics(pvdf)

    return {
        'metrics': m, 'yearly': ym,
        'rebal_count': rebal_count,
        'health_exit_count': health_exit_count,
        'drift_rebal_count': drift_rebal_count,
        'pv': pvdf,
    }


def _empty():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'yearly': {}, 'rebal_count': 0, 'health_exit_count': 0,
            'drift_rebal_count': 0, 'pv': pd.DataFrame()}


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()

    # Test configurations: (label, daily_health, exit_mode, exit_filter, drift_threshold)
    configs = [
        # Baseline: no daily health check
        ('기준 (D10%)',              False, 'cash',    'health',      0.10),
        ('기준 (D15%)',              False, 'cash',    'health',      0.15),
        ('기준 (D20%)',              False, 'cash',    'health',      0.20),
        # Daily health filter → cash
        ('헬스+현금 (D10%)',         True,  'cash',    'health',      0.10),
        ('헬스+현금 (D15%)',         True,  'cash',    'health',      0.15),
        ('헬스+현금 (D20%)',         True,  'cash',    'health',      0.20),
        # Daily health filter → replace
        ('헬스+대체 (D10%)',         True,  'replace', 'health',      0.10),
        ('헬스+대체 (D15%)',         True,  'replace', 'health',      0.15),
        ('헬스+대체 (D20%)',         True,  'replace', 'health',      0.20),
        # SMA(10) filter → cash
        ('SMA10+현금 (D10%)',        True,  'cash',    'sma10',       0.10),
        # SMA(20) filter → cash / replace
        ('SMA20+현금 (D10%)',        True,  'cash',    'sma20',       0.10),
        ('SMA20+대체 (D10%)',        True,  'replace', 'sma20',       0.10),
        # Health + SMA(20) combined → cash / replace
        ('헬스+SMA20+현금 (D10%)',   True,  'cash',    'health+sma20', 0.10),
        ('헬스+SMA20+대체 (D10%)',   True,  'replace', 'health+sma20', 0.10),
        # No drift, health only
        ('헬스+현금 (드리프트없음)',    True,  'cash',    'health',      0.0),
        ('헬스+대체 (드리프트없음)',    True,  'replace', 'health',      0.0),
        # No PFD5, health only
        ('헬스+현금 (PFD0,D10%)',    True,  'cash',    'health',      0.10),
    ]

    all_results = {c[0]: [] for c in configs}
    all_rebals = {c[0]: [] for c in configs}
    all_health_exits = {c[0]: [] for c in configs}
    all_drift_rebals = {c[0]: [] for c in configs}

    for base_d in ANCHOR_DAYS:
        snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for label, dh, em, ef, dt in configs:
            pfd = 0 if 'PFD0' in label else 5
            r = run_daily_health_backtest(
                prices, universe, snap_days,
                daily_health=dh, exit_mode=em, exit_filter=ef,
                drift_threshold=dt,
                post_flip_delay=pfd, params_base=B()
            )
            all_results[label].append(r['metrics'])
            all_rebals[label].append(r['rebal_count'])
            all_health_exits[label].append(r.get('health_exit_count', 0))
            all_drift_rebals[label].append(r.get('drift_rebal_count', 0))

        # Show key comparison
        s_base = all_results['기준 (D10%)'][-1]['Sharpe']
        s_cash = all_results['헬스+현금 (D10%)'][-1]['Sharpe']
        s_repl = all_results['헬스+대체 (D10%)'][-1]['Sharpe']
        print(f"기준={s_base:.3f}  현금={s_cash:.3f}  대체={s_repl:.3f}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    base_sharpe = avg_metrics(all_results['기준 (D10%)'])[0]['Sharpe']

    print(f"\n{'=' * 110}")
    print(f"  매일 헬스체크 테스트 — 스냅샷B + PFD5, 날짜 평균 (10 앵커)")
    print(f"{'=' * 110}")

    print(f"\n  {'전략':<28} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7}"
          f" {'Calmar':>7} {'리밸':>5} {'퇴출':>5} {'드리프트':>7}")
    print(f"  {'─' * 100}")

    for label, _, _, _, _ in configs:
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(all_rebals[label])
        avg_health = np.mean(all_health_exits[label])
        avg_drift = np.mean(all_drift_rebals[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0 else ''
        print(f"  {label:<28} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f} {avg_health:>5.0f} {avg_drift:>7.0f}{marker}")

    # Per-anchor for key configs
    print(f"\n  앵커별 Sharpe:")
    key_labels = ['기준 (D10%)', '헬스+현금 (D10%)', '헬스+대체 (D10%)']
    header = f"  {'앵커':>5}"
    for lb in key_labels:
        short = lb[:12]
        header += f" {short:>12}"
    print(header)
    print(f"  {'─' * 45}")
    for idx, d in enumerate(ANCHOR_DAYS):
        row = f"  {d:>5}"
        for lb in key_labels:
            row += f" {all_results[lb][idx]['Sharpe']:>12.3f}"
        print(row)

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
