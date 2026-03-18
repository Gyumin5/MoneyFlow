#!/usr/bin/env python3
"""V15 코인 전략 개선안 백테스트 — 1라운드 (방어 강화).

Baseline: V14/V15 3트랜치, Drift OFF
개선안:
  I1: 보유 코인 일간 급락 청산 (-12%, -15%, -20%)
  I2: HC<3 현금 비중 (soft cap 35%, min3종, 고정20%)
  I3: DD Exit 강화 (30d-20%, 60d-20%, 2단계 -18%/-25%)
  I4: Drift 재도입 (20%, 25%, 30%)
  I5: Crash 민감화 (-7%, -8%, 다단계)

각 개선안을 개별 테스트 + 유망 조합 테스트.
"""

import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine_v15 import (
    V15Params, Tranche, load_data,
    check_crash, check_health,
    execute_rebalance, sell_coin_from_tranche, sell_all_from_tranche,
    check_drift, compute_target_weights,
    get_universe_for_date, _close_to, calc_ret, get_vol,
    get_price, calc_metrics, calc_yearly_metrics,
    _get_daily_returns, _bl_cache,
)


def run_improved_backtest(prices, universe_map, improvements=None,
                          tranche_days=(1, 10, 19),
                          start='2018-01-01', end='2025-06-30',
                          tx_cost=0.004, capital=10000.0):
    """Run V15 3-tranche backtest with configurable improvements.

    improvements dict keys:
      coin_stop: float or None — individual coin daily stop-loss (e.g., -0.12)
      coin_stop_ban: int — days to ban after stop (default 0)
      hc_min: int or None — minimum healthy count to invest (e.g., 3)
      soft_cap: float or None — max per-coin weight (e.g., 0.35)
      dd_lookback: int — DD exit lookback (default 60)
      dd_threshold: float — DD exit threshold (default -0.25)
      dd_partial: float or None — partial exit threshold (e.g., -0.18 for 50% exit)
      drift_threshold: float — 0 = disabled (default 0)
      crash_threshold: float — (default -0.10)
      crash_cool: int — (default 3)
      crash_tier2: tuple or None — (threshold, cool) for tier 2 (e.g., (-0.05, 1))
    """
    if improvements is None:
        improvements = {}

    params = V15Params()
    coin_stop = improvements.get('coin_stop', None)
    coin_stop_ban = improvements.get('coin_stop_ban', 0)
    hc_min = improvements.get('hc_min', None)
    soft_cap = improvements.get('soft_cap', None)
    dd_lookback = improvements.get('dd_lookback', 60)
    dd_threshold = improvements.get('dd_threshold', -0.25)
    dd_partial = improvements.get('dd_partial', None)
    drift_thresh = improvements.get('drift_threshold', 0.0)
    crash_thresh = improvements.get('crash_threshold', -0.10)
    crash_cool = improvements.get('crash_cool', 3)
    crash_tier2 = improvements.get('crash_tier2', None)

    n_tr = len(tranche_days)
    capital_per = capital / n_tr
    tranches = [Tranche(i, tranche_days[i], capital_per) for i in range(n_tr)]

    prev_canary = False
    canary_on = False
    canary_on_date = None
    crash_cooldown = 0
    prev_month = None
    coin_ban_until = {}  # {ticker: date} — stop-loss ban

    rebal_count = 0
    dd_exit_count = 0
    stop_count = 0
    crash_count = 0

    btc = prices.get('BTC-USD')
    if btc is None:
        return None
    all_dates = btc.index[(btc.index >= start) & (btc.index <= end)]
    if len(all_dates) < 2:
        return None

    portfolio_values = []

    for date in all_dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        if is_month_change:
            for tr in tranches:
                tr.anchor_done_this_month = False

        # Crash Breaker
        crash_just_ended = False

        # Tier 2 crash (optional)
        if crash_tier2 and crash_cooldown == 0 and not is_first:
            btc_s = _close_to('BTC-USD', prices, date)
            if len(btc_s) >= 2:
                btc_ret = btc_s.iloc[-1] / btc_s.iloc[-2] - 1
                if btc_ret <= crash_tier2[0] and btc_ret > crash_thresh:
                    for tr in tranches:
                        sell_all_from_tranche(tr, prices, date, tx_cost)
                    crash_cooldown = crash_tier2[1]
                    crash_count += 1

        # Main crash
        if crash_cooldown > 0:
            crash_cooldown -= 1
            if crash_cooldown == 0:
                btc_s = _close_to('BTC-USD', prices, date)
                if len(btc_s) >= 2 and (btc_s.iloc[-1] / btc_s.iloc[-2] - 1) <= crash_thresh:
                    crash_cooldown = crash_cool
                else:
                    crash_just_ended = True
        elif not is_first:
            btc_s = _close_to('BTC-USD', prices, date)
            if len(btc_s) >= 2 and (btc_s.iloc[-1] / btc_s.iloc[-2] - 1) <= crash_thresh:
                for tr in tranches:
                    sell_all_from_tranche(tr, prices, date, tx_cost)
                crash_cooldown = crash_cool
                crash_count += 1

        # Canary (always)
        btc_s = _close_to('BTC-USD', prices, date)
        if len(btc_s) >= params.canary_sma:
            price = btc_s.iloc[-1]
            sma = btc_s.rolling(params.canary_sma).mean().iloc[-1]
            if not np.isnan(sma):
                hyst = params.canary_hyst
                if prev_canary:
                    canary_on = not (price < sma * (1 - hyst))
                else:
                    canary_on = price > sma * (1 + hyst)

        signal_flipped = (prev_canary is not None and canary_on != prev_canary)
        flip_on = signal_flipped and canary_on
        if flip_on:
            canary_on_date = date
            for tr in tranches:
                tr.post_flip_refreshed = False

        if crash_cooldown > 0 and not crash_just_ended:
            pv = sum(tr.value(prices, date) for tr in tranches)
            portfolio_values.append({'Date': date, 'Value': pv})
            prev_canary = canary_on
            prev_month = cur_month
            continue

        # Blacklist (stateless)
        universe = get_universe_for_date(universe_map, date)
        blacklist = set()
        for t in universe:
            rets = _get_daily_returns(t, prices)
            if len(rets) == 0:
                continue
            mask = rets.index <= date
            if mask.sum() < 7:
                continue
            recent = rets.loc[mask].iloc[-7:]
            if (recent <= -0.15).any():
                blacklist.add(t)

        # Add stop-loss banned coins to blacklist
        for t, ban_date in list(coin_ban_until.items()):
            if date <= ban_date:
                blacklist.add(t)
            else:
                del coin_ban_until[t]

        # Per-tranche
        for tr in tranches:
            tr.dd_exited_today = set()

            if canary_on and not crash_just_ended:
                # I1: Individual coin daily stop-loss
                if coin_stop is not None:
                    for t in list(tr.holdings.keys()):
                        if tr.holdings.get(t, 0) <= 0:
                            continue
                        rets = _get_daily_returns(t, prices)
                        if len(rets) == 0:
                            continue
                        m = rets.index <= date
                        if m.sum() < 1:
                            continue
                        daily_ret = rets.loc[m].iloc[-1]
                        if daily_ret <= coin_stop:
                            sell_coin_from_tranche(tr, t, prices, date, tx_cost)
                            tr.dd_exited_today.add(t)
                            stop_count += 1
                            if coin_stop_ban > 0:
                                coin_ban_until[t] = date + pd.Timedelta(days=coin_stop_ban)

                # DD Exit (with optional partial)
                for t in list(tr.holdings.keys()):
                    if t in tr.dd_exited_today:
                        continue
                    if tr.holdings.get(t, 0) <= 0:
                        continue
                    s = _close_to(t, prices, date)
                    if len(s) < dd_lookback:
                        continue
                    peak = s.iloc[-dd_lookback:].max()
                    if peak <= 0:
                        continue
                    dd = s.iloc[-1] / peak - 1

                    if dd <= dd_threshold:
                        sell_coin_from_tranche(tr, t, prices, date, tx_cost)
                        tr.dd_exited_today.add(t)
                        dd_exit_count += 1
                    elif dd_partial is not None and dd <= dd_partial:
                        # Partial exit: sell 50%
                        units = tr.holdings.get(t, 0)
                        if units > 0:
                            half = units * 0.5
                            p = get_price(t, prices, date)
                            if p > 0:
                                tx = half * p * tx_cost
                                tr.holdings[t] -= half
                                tr.cash += half * p - tx
                                dd_exit_count += 1

            # Rebalance triggers
            do_rebal = False
            reason = 'none'

            if is_first:
                do_rebal = True; reason = 'init'
            if not do_rebal and signal_flipped:
                do_rebal = True; reason = 'flip'
            if not do_rebal and crash_just_ended:
                do_rebal = True; reason = 'crash_end'
            if (not do_rebal and canary_on and not tr.post_flip_refreshed
                    and canary_on_date and (date - canary_on_date).days >= 5):
                do_rebal = True; reason = 'pfd5'
                tr.post_flip_refreshed = True
            if (not do_rebal and not tr.anchor_done_this_month
                    and date.day >= tr.anchor_day and not is_first):
                do_rebal = True; reason = 'monthly'

            # Drift
            if (not do_rebal and drift_thresh > 0 and canary_on and tr.holdings):
                target_w = compute_target_weights(
                    prices, universe_map, date, params, blacklist, tr.dd_exited_today)
                if target_w and check_drift(tr, target_w, prices, date, drift_thresh):
                    do_rebal = True; reason = 'drift'

            if do_rebal:
                if canary_on:
                    # Get healthy coins
                    candidates = [t for t in universe
                                  if t not in blacklist and t not in tr.dd_exited_today]
                    healthy = []
                    for t in candidates:
                        if check_health(t, prices, date, params):
                            healthy.append(t)
                            if len(healthy) >= 5:
                                break

                    # I2: minimum healthy count
                    if hc_min and len(healthy) < hc_min:
                        # Not enough → invest only what we have, rest cash
                        if healthy:
                            if soft_cap:
                                w = min(1.0 / len(healthy), soft_cap)
                            else:
                                w = 1.0 / len(healthy)
                            target_w = {t: w for t in healthy}
                        else:
                            target_w = {}
                    elif soft_cap and healthy:
                        w = min(1.0 / len(healthy), soft_cap)
                        target_w = {t: w for t in healthy}
                    elif healthy:
                        target_w = {t: 1.0 / len(healthy) for t in healthy}
                    else:
                        target_w = {}

                    if target_w:
                        execute_rebalance(tr, target_w, prices, date, tx_cost)
                    else:
                        sell_all_from_tranche(tr, prices, date, tx_cost)
                else:
                    sell_all_from_tranche(tr, prices, date, tx_cost)

                if reason == 'monthly':
                    tr.anchor_done_this_month = True
                rebal_count += 1

        pv = sum(tr.value(prices, date) for tr in tranches)
        portfolio_values.append({'Date': date, 'Value': pv})
        prev_canary = canary_on
        prev_month = cur_month

    if not portfolio_values:
        return None
    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    ym = calc_yearly_metrics(pvdf)
    return {
        'metrics': m, 'yearly': ym, 'pv': pvdf,
        'rebal_count': rebal_count, 'dd_exit_count': dd_exit_count,
        'stop_count': stop_count, 'crash_count': crash_count,
    }


def main():
    print('V15 코인 개선안 백테스트 — 1라운드 (방어 강화)')
    print('=' * 120)

    print('Loading data...')
    prices, universe = load_data()
    print(f'  {len(prices)} tickers loaded\n')

    CONFIGS = [
        # Baseline
        ('Baseline', {}),

        # I1: Coin daily stop-loss
        ('I1: Stop -12%', {'coin_stop': -0.12}),
        ('I1: Stop -15%', {'coin_stop': -0.15}),
        ('I1: Stop -20%', {'coin_stop': -0.20}),
        ('I1: Stop -15% +14d ban', {'coin_stop': -0.15, 'coin_stop_ban': 14}),

        # I2: HC minimum / soft cap
        ('I2: SoftCap 35%', {'soft_cap': 0.35}),
        ('I2: SoftCap 50%', {'soft_cap': 0.50}),
        ('I2: Min3 + Cap35%', {'hc_min': 3, 'soft_cap': 0.35}),

        # I3: DD Exit variants
        ('I3: DD 30d-20%', {'dd_lookback': 30, 'dd_threshold': -0.20}),
        ('I3: DD 60d-20%', {'dd_lookback': 60, 'dd_threshold': -0.20}),
        ('I3: DD 2step -18/-25', {'dd_partial': -0.18}),

        # I4: Drift re-enable
        ('I4: Drift 20%', {'drift_threshold': 0.20}),
        ('I4: Drift 25%', {'drift_threshold': 0.25}),
        ('I4: Drift 30%', {'drift_threshold': 0.30}),

        # I5: Crash sensitivity
        ('I5: Crash -7%', {'crash_threshold': -0.07}),
        ('I5: Crash -8%', {'crash_threshold': -0.08}),
        ('I5: 2tier -5%/1d+-10%/3d', {'crash_tier2': (-0.05, 1)}),

        # Promising combos
        ('Combo1: Stop-15%+DD30d-20%', {'coin_stop': -0.15, 'dd_lookback': 30, 'dd_threshold': -0.20}),
        ('Combo2: Stop-15%+Drift25%', {'coin_stop': -0.15, 'drift_threshold': 0.25}),
        ('Combo3: Cap35%+Stop-15%+Drift25%', {'soft_cap': 0.35, 'coin_stop': -0.15, 'drift_threshold': 0.25}),
    ]

    N_OFFSETS = 10
    results = {}

    for name, imp in CONFIGS:
        _bl_cache.clear()
        print(f'  {name}...')
        sharpes, cagrs, mdds, sortinos, calmars = [], [], [], [], []

        for offset in range(N_OFFSETS):
            _bl_cache.clear()
            days = tuple((d - 1 + offset) % 28 + 1 for d in (1, 10, 19))
            r = run_improved_backtest(prices, universe, imp, tranche_days=days)
            if r:
                m = r['metrics']
                sharpes.append(m['Sharpe'])
                cagrs.append(m['CAGR'])
                mdds.append(m['MDD'])
                sortinos.append(m.get('Sortino', 0))
                cal = abs(m['CAGR'] / m['MDD']) if m['MDD'] != 0 else 0
                calmars.append(cal)

        if sharpes:
            results[name] = {
                'Sharpe': np.mean(sharpes), 'σ': np.std(sharpes),
                'CAGR': np.mean(cagrs), 'MDD': np.mean(mdds),
                'Sortino': np.mean(sortinos), 'Calmar': np.mean(calmars),
                'rebal': r['rebal_count'] if r else 0,
                'dd_exits': r['dd_exit_count'] if r else 0,
                'stops': r.get('stop_count', 0) if r else 0,
                'crashes': r.get('crash_count', 0) if r else 0,
            }
            # Yearly from last run
            if r and 'yearly' in r:
                results[name]['yearly'] = {yr: ym['CAGR'] for yr, ym in r['yearly'].items()}

    # Print results
    print(f'\n{"=" * 140}')
    print(f'  V15 개선안 비교 (10-anchor 평균, 3트랜치)')
    print(f'{"=" * 140}')
    print(f'{"Config":<30} {"Sharpe":>7} {"σ(S)":>5} {"CAGR":>7} {"MDD":>7} {"Sortino":>8} {"Calmar":>7} {"Rebal":>6} {"DD":>4} {"Stop":>5} {"Crsh":>5}  {"2022":>6} {"2024":>6}')
    print('-' * 140)

    base_sh = results.get('Baseline', {}).get('Sharpe', 0)

    for name, _ in CONFIGS:
        r = results.get(name)
        if not r:
            continue
        yr = r.get('yearly', {})
        marker = ' ★' if r['Sharpe'] > base_sh + 0.01 and name != 'Baseline' else ''
        print(f'{name:<30} {r["Sharpe"]:>7.3f} {r["σ"]:>4.3f} {r["CAGR"]:>+6.1%} {r["MDD"]:>6.1%} '
              f'{r["Sortino"]:>8.3f} {r["Calmar"]:>7.2f} {r["rebal"]:>6} {r["dd_exits"]:>4} '
              f'{r["stops"]:>5} {r["crashes"]:>5}  '
              f'{yr.get(2022, 0):>+5.1%} {yr.get(2024, 0):>+5.1%}{marker}')

    # Delta table
    print(f'\n{"=" * 80}')
    print(f'  Baseline 대비 변화 (Sharpe 기준 상위 5)')
    print(f'{"=" * 80}')
    ranked = sorted([(n, r) for n, r in results.items() if n != 'Baseline'],
                    key=lambda x: -x[1]['Sharpe'])
    print(f'{"Config":<30} {"ΔSharpe":>9} {"ΔCAGR":>8} {"ΔMDD":>8}')
    print('-' * 60)
    base = results['Baseline']
    for name, r in ranked[:7]:
        ds = r['Sharpe'] - base['Sharpe']
        dc = r['CAGR'] - base['CAGR']
        dm = r['MDD'] - base['MDD']
        verdict = '✓' if ds > 0 else '✗'
        print(f'{name:<30} {ds:>+9.3f} {dc:>+7.1%} {dm:>+7.1%} {verdict}')


if __name__ == '__main__':
    main()
