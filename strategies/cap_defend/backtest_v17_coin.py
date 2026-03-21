#!/usr/bin/env python3
"""V17 코인 공식 백테스트 (look-ahead 수정).

시그널: t-1일 종가 기준 (실매매와 동일: 09:20에 전일 종가 판단)
체결: t일 가격
엔진: 구 엔진 (test_t40_dd 기반, 3-snapshot 합성)

V17 코인 전략:
- 카나리: BTC > SMA(60), 1% hysteresis
- 헬스: Mom(21)>0 AND Mom(90)>0 AND Vol(90)<=5%
- 선정: 시총순 Top 5, Equal Weight
- DD Exit: 60d peak -25%
- Blacklist: -15% daily → 7d
- PFD5: 플립 후 5일 리프레시
- Drift: 10% 반턴오버
- 3-snapshot: Day 1/10/19
"""

import sys, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_engine import (
    load_universe, load_all_prices, filter_universe, calc_metrics,
    resolve_canary, get_price, _close_to, _port_val,
    execute_rebalance, Params,
)
from test_matrix import (
    B, merge_snapshots, calc_current_weights, calc_half_turnover,
    compute_signal_weights_filtered, check_blacklist, update_blacklist,
)
from test_t40_dd import check_coin_drawdown, _empty_ext


def run_backtest_v17(prices, universe_map, snapshot_days=(1, 10, 19),
                     dd_lookback=60, dd_threshold=-0.25,
                     bl_drop=-0.15, bl_days=7,
                     drift_threshold=0.10, post_flip_delay=5,
                     params_base=None):
    """V17 코인 백테스트 — look-ahead 수정 버전."""
    if params_base is None:
        params_base = B(selection='mcap', n_picks=5, weighting='ew', top_n=40)

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
    dd_exit_count = 0
    prev_date = None

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

        # ★ 시그널은 전일(prev_date) 종가 기준
        sig_date = prev_date if prev_date is not None else date

        canary_on = resolve_canary(prices, sig_date, params_base, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['canary_on'] = canary_on
        state['is_month_change'] = imc
        update_blacklist(blacklist)

        # ★ Blacklist: 전일 기준 판단
        newly_bl = []
        if bl_drop < 0 and canary_on and holdings:
            newly_bl = check_blacklist(holdings, prices, sig_date, bl_drop, blacklist, bl_days)
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
                            snapshots[si] = {t: w / total for t, w in snap.items()}

        state['_blacklist'] = set(blacklist.keys())
        need_rebal = False

        # ★ DD Exit: 전일 기준 판단, 당일 체결
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped:
            exited = []
            for ticker in list(holdings.keys()):
                if holdings[ticker] <= 0:
                    continue
                if check_coin_drawdown(ticker, prices, sig_date, dd_lookback, dd_threshold):
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
                                snapshots[si] = {t: w / total for t, w in snapshots[si].items()}

        # ★ 종목 선정: 전일 기준
        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights_filtered(
                    prices, universe_map, sig_date, params_base, state, blacklist)
            need_rebal = True
        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights_filtered(
                        prices, universe_map, sig_date, params_base, state, blacklist)
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
                            prices, universe_map, sig_date, params_base, state, blacklist)
                    need_rebal = True

        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights_filtered(
                        prices, universe_map, sig_date, params_base, state, blacklist)
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

        # ★ 체결: 당일 가격
        if need_rebal:
            holdings, cash = execute_rebalance(holdings, cash, combined, prices,
                                               date, params_base.tx_cost)
            rebal_count += 1

        pv = _port_val(holdings, cash, prices, date)
        portfolio_values.append({'Date': date, 'Value': pv})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month
        prev_date = date

    if not portfolio_values:
        return _empty_ext()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    return {'metrics': m, 'rebal_count': rebal_count, 'dd_exit_count': dd_exit_count, 'pv': pvdf}


if __name__ == '__main__':
    print("V17 코인 백테스트 (LA 수정)")
    print("=" * 70)

    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values():
        all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))

    params = B(selection='mcap', n_picks=5, weighting='ew', top_n=40)

    for start, end in [('2018-01-01', '2025-06-30'), ('2019-01-01', '2025-06-30'),
                       ('2021-01-01', '2025-06-30')]:
        params.start_date = start
        params.end_date = end
        r = run_backtest_v17(prices, um40, params_base=params)
        m = r['metrics']
        s, c, mdd = m['Sharpe'], m['CAGR'], m['MDD']
        cal = c / abs(mdd) if mdd != 0 else 0
        print(f"  {start}~{end}: Sharpe {s:.3f}, CAGR {c:+.1%}, MDD {mdd:+.1%}, Calmar {cal:.2f}")
