#!/usr/bin/env python3
"""Final validation: 4 strategies × stress tests (parallelized).

Strategies:
  A: T40+BL15% (no DD)
  B: T40+BL15%+DD(60d,25%)→cash
  C: T50+BL15%+DD(60d,25%)→cash
  D: T50+BL15%+DD(30d,20%)→cash

Tests:
  1. Asymmetric tx cost: normal rebal 0.4%, emergency exit 0.4~2.0%
  2. Yearly breakdown: per-year CAGR, MDD, Sharpe
  3. Remove top N days: zero out best 5/10 days
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
    check_blacklist, update_blacklist, ANCHOR_DAYS,
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


def run_validation_backtest(prices, universe_map, snapshot_days,
                             dd_lookback=0, dd_threshold=0,
                             bl_drop=-0.15, bl_days=7,
                             drift_threshold=0.10,
                             post_flip_delay=5, params_base=None,
                             emergency_tx_cost=None):
    """
    emergency_tx_cost: if set, use this cost for DD exit/BL/crash trades
                       instead of normal tx_cost
    """
    if params_base is None:
        params_base = B()

    normal_tx = params_base.tx_cost
    emerg_tx = emergency_tx_cost if emergency_tx_cost else normal_tx

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
    dd_exit_count = 0
    is_emergency = False

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
        is_emergency = False

        # DD exit
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped:
            exited = []
            for ticker in list(holdings.keys()):
                if holdings[ticker] <= 0:
                    continue
                if check_coin_drawdown(ticker, prices, date, dd_lookback, dd_threshold):
                    exited.append(ticker)
            if exited:
                dd_exit_count += len(exited)
                is_emergency = True
                for coin in exited:
                    p = get_price(coin, prices, date)
                    cash += holdings[coin] * p * (1 - emerg_tx)
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
            is_emergency = True
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
            is_emergency = True

        combined = merge_snapshots(snapshots)

        if not need_rebal and canary_on and drift_threshold > 0:
            current_w = calc_current_weights(holdings, cash, prices, date)
            ht = calc_half_turnover(current_w, combined)
            if ht >= drift_threshold:
                need_rebal = True

        if need_rebal:
            tx = emerg_tx if is_emergency else normal_tx
            holdings, cash = execute_rebalance(holdings, cash, combined, prices,
                                               date, tx)
            rebal_count += 1

        pv = _port_val(holdings, cash, prices, date)
        portfolio_values.append({'Date': date, 'Value': pv})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month

    if not portfolio_values:
        return _empty()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)

    # Yearly breakdown
    yearly = {}
    pvdf['Return'] = pvdf['Value'].pct_change().fillna(0)
    for year in sorted(pvdf.index.year.unique()):
        ydf = pvdf[pvdf.index.year == year]
        if len(ydf) < 20:
            continue
        ym = calc_metrics(ydf[['Value']])
        yearly[year] = ym

    return {
        'metrics': m,
        'rebal_count': rebal_count,
        'dd_exit_count': dd_exit_count,
        'pv': pvdf,
        'yearly': yearly,
    }


def _empty():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'rebal_count': 0, 'dd_exit_count': 0,
            'pv': pd.DataFrame(), 'yearly': {}}


# Strategy definitions
STRATEGIES = {
    'A: T40 (기본)': (40, 0, 0),
    'B: T40+DD60/25': (40, 60, -0.25),
    'C: T50+DD60/25': (50, 60, -0.25),
    'D: T50+DD30/20': (50, 30, -0.20),
}

# Emergency cost levels
EMERG_COSTS = [0.004, 0.006, 0.008, 0.010, 0.015, 0.020]


def _worker(args):
    base_d, strat_name, top_n, dd_lb, dd_th, emerg_cost = args
    snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
    r = run_validation_backtest(
        G_PRICES, G_UNIVERSE_MAPS[top_n], snap_days,
        dd_lookback=dd_lb, dd_threshold=dd_th,
        bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10,
        post_flip_delay=5, params_base=B(),
        emergency_tx_cost=emerg_cost
    )
    return (base_d, strat_name, emerg_cost, r['metrics'], r['rebal_count'],
            r['dd_exit_count'], r['yearly'])


def remove_top_days(pvdf, n_remove):
    """Remove top N return days (set to 0% return)."""
    if pvdf.empty:
        return pvdf
    pv = pvdf['Value'].copy()
    rets = pv.pct_change().fillna(0)
    top_idx = rets.nlargest(n_remove).index
    for idx in top_idx:
        rets.loc[idx] = 0
    # Reconstruct values
    new_val = [pv.iloc[0]]
    for i in range(1, len(pv)):
        new_val.append(new_val[-1] * (1 + rets.iloc[i]))
    result = pvdf.copy()
    result['Value'] = new_val
    return result


def main():
    global G_PRICES, G_UNIVERSE_MAPS

    print("Loading data...")
    um_raw = load_universe()

    G_UNIVERSE_MAPS = {}
    for top_n in [40, 50]:
        G_UNIVERSE_MAPS[top_n] = filter_universe(um_raw, top_n)

    all_tickers = set()
    for fm in G_UNIVERSE_MAPS.values():
        for ts in fm.values():
            all_tickers.update(ts)
    all_tickers.update(['BTC-USD', 'ETH-USD'])

    G_PRICES = load_all_prices(all_tickers)
    print(f"  {len(G_PRICES)} tickers loaded")

    t0 = time.time()

    # Build jobs: 4 strategies × 6 cost levels × 10 anchors
    jobs = []
    for base_d in ANCHOR_DAYS:
        for strat_name, (top_n, dd_lb, dd_th) in STRATEGIES.items():
            for ec in EMERG_COSTS:
                jobs.append((base_d, strat_name, top_n, dd_lb, dd_th, ec))

    n_workers = min(cpu_count(), len(jobs))
    print(f"  {len(STRATEGIES)} strategies × {len(EMERG_COSTS)} costs × {len(ANCHOR_DAYS)} anchors = {len(jobs)} backtests")
    print(f"  Using {n_workers} workers")

    with Pool(n_workers) as pool:
        results = pool.map(_worker, jobs)

    # Organize results
    # Key: (strat_name, emerg_cost) -> list of (metrics, rebal, exits, yearly) per anchor
    organized = {}
    for base_d, sn, ec, metrics, rebal, exits, yearly in results:
        key = (sn, ec)
        if key not in organized:
            organized[key] = {'metrics': [], 'rebals': [], 'exits': [], 'yearlys': [], 'pvs': []}
        organized[key]['metrics'].append(metrics)
        organized[key]['rebals'].append(rebal)
        organized[key]['exits'].append(exits)
        organized[key]['yearlys'].append(yearly)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    def avg_m(metrics_list):
        avg = {}
        for key in metrics_list[0]:
            avg[key] = np.mean([r[key] for r in metrics_list])
        std_s = np.std([r['Sharpe'] for r in metrics_list])
        return avg, std_s

    # ══════════════════════════════════════════════════════════════════
    # Section 1: Asymmetric tx cost sensitivity
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  1. 비대칭 거래비용 민감도 (정기=0.4%, 긴급탈출=변동)")
    print(f"     긴급탈출: DD Exit, 블랙리스트, 카나리아 플립, 크래시 브레이커")
    print(f"{'=' * 110}")

    # Sharpe table
    print(f"\n  Sharpe:")
    print(f"  {'전략':>16}", end="")
    for ec in EMERG_COSTS:
        print(f" {ec*100:>5.1f}%", end="")
    print()
    print(f"  {'─' * (16 + 7 * len(EMERG_COSTS))}")

    for sn in STRATEGIES:
        print(f"  {sn:>16}", end="")
        for ec in EMERG_COSTS:
            avg, _ = avg_m(organized[(sn, ec)]['metrics'])
            print(f" {avg['Sharpe']:>5.3f}", end="")
        print()

    # CAGR table
    print(f"\n  CAGR:")
    print(f"  {'전략':>16}", end="")
    for ec in EMERG_COSTS:
        print(f" {ec*100:>5.1f}%", end="")
    print()
    print(f"  {'─' * (16 + 7 * len(EMERG_COSTS))}")

    for sn in STRATEGIES:
        print(f"  {sn:>16}", end="")
        for ec in EMERG_COSTS:
            avg, _ = avg_m(organized[(sn, ec)]['metrics'])
            print(f" {avg['CAGR']:>+4.0%}", end="")
        print()

    # MDD table
    print(f"\n  MDD:")
    print(f"  {'전략':>16}", end="")
    for ec in EMERG_COSTS:
        print(f" {ec*100:>5.1f}%", end="")
    print()
    print(f"  {'─' * (16 + 7 * len(EMERG_COSTS))}")

    for sn in STRATEGIES:
        print(f"  {sn:>16}", end="")
        for ec in EMERG_COSTS:
            avg, _ = avg_m(organized[(sn, ec)]['metrics'])
            print(f" {avg['MDD']:>5.1%}", end="")
        print()

    # Calmar table
    print(f"\n  Calmar:")
    print(f"  {'전략':>16}", end="")
    for ec in EMERG_COSTS:
        print(f" {ec*100:>5.1f}%", end="")
    print()
    print(f"  {'─' * (16 + 7 * len(EMERG_COSTS))}")

    for sn in STRATEGIES:
        print(f"  {sn:>16}", end="")
        for ec in EMERG_COSTS:
            avg, _ = avg_m(organized[(sn, ec)]['metrics'])
            calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
            print(f" {calmar:>5.2f}", end="")
        print()

    # Detail at key cost levels
    for ec in [0.004, 0.010, 0.020]:
        print(f"\n  긴급비용 {ec*100:.1f}% 상세:")
        print(f"  {'전략':>16} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5} {'탈출':>5}")
        print(f"  {'─' * 70}")
        for sn in STRATEGIES:
            avg, std = avg_m(organized[(sn, ec)]['metrics'])
            calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
            avg_rebal = np.mean(organized[(sn, ec)]['rebals'])
            avg_exit = np.mean(organized[(sn, ec)]['exits'])
            print(f"  {sn:>16} {avg['Sharpe']:>7.3f} {std:>6.3f}"
                  f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
                  f" {avg_rebal:>5.0f} {avg_exit:>5.0f}")

    # ══════════════════════════════════════════════════════════════════
    # Section 2: Yearly breakdown (at normal cost 0.4%)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  2. 연도별 성과 분해 (긴급비용 0.4%)")
    print(f"{'=' * 110}")

    ec_normal = 0.004
    for sn in STRATEGIES:
        print(f"\n  {sn}:")
        yearlys = organized[(sn, ec_normal)]['yearlys']

        # Collect all years
        all_years = set()
        for y in yearlys:
            all_years.update(y.keys())
        all_years = sorted(all_years)

        print(f"  {'연도':>6} {'CAGR':>8} {'MDD':>7} {'Sharpe':>7}")
        print(f"  {'─' * 32}")

        for year in all_years:
            cagrs = []
            mdds = []
            sharpes = []
            for y in yearlys:
                if year in y:
                    cagrs.append(y[year]['CAGR'])
                    mdds.append(y[year]['MDD'])
                    sharpes.append(y[year]['Sharpe'])
            if cagrs:
                print(f"  {year:>6} {np.mean(cagrs):>+7.1%} {np.mean(mdds):>6.1%} {np.mean(sharpes):>7.2f}")

        # Overall
        avg, std = avg_m(organized[(sn, ec_normal)]['metrics'])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {'전체':>6} {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {avg['Sharpe']:>7.2f}  (Calmar {calmar:.2f})")

    # ══════════════════════════════════════════════════════════════════
    # Section 3: Remove top N days (at normal cost, using anchor 1)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print(f"  3. 최고 수익일 제거 테스트 (Top 5/10일 수익률 → 0%)")
    print(f"{'=' * 110}")

    # Run fresh backtests for anchor=1 to get pv series
    from test_matrix import _empty as matrix_empty
    print(f"\n  앵커 1 기준으로 계산 중...", flush=True)

    for sn, (top_n, dd_lb, dd_th) in STRATEGIES.items():
        snap_days = [1, 10, 19]
        r = run_validation_backtest(
            G_PRICES, G_UNIVERSE_MAPS[top_n], snap_days,
            dd_lookback=dd_lb, dd_threshold=dd_th,
            bl_drop=-0.15, bl_days=7,
            drift_threshold=0.10,
            post_flip_delay=5, params_base=B(),
            emergency_tx_cost=0.004
        )
        pv = r['pv']
        if pv.empty:
            continue

        m_base = calc_metrics(pv)

        print(f"\n  {sn}:")
        print(f"  {'조건':>12} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
        print(f"  {'─' * 45}")

        calmar = m_base['CAGR'] / abs(m_base['MDD']) if m_base['MDD'] != 0 else 0
        print(f"  {'원본':>12} {m_base['Sharpe']:>7.3f} {m_base['CAGR']:>+7.1%} {m_base['MDD']:>6.1%} {calmar:>7.2f}")

        for n_remove in [5, 10, 20]:
            pv_mod = remove_top_days(pv, n_remove)
            m_mod = calc_metrics(pv_mod)
            calmar_m = m_mod['CAGR'] / abs(m_mod['MDD']) if m_mod['MDD'] != 0 else 0
            print(f"  {'Top'+str(n_remove)+'일 제거':>12} {m_mod['Sharpe']:>7.3f} {m_mod['CAGR']:>+7.1%} {m_mod['MDD']:>6.1%} {calmar_m:>7.2f}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
