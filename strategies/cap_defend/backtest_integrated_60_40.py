#!/usr/bin/env python3
"""V12~V17 통합 포트폴리오 백테스트 — 주식:코인 60:40, 월간 리밸런싱.

각 전략을 독립 실행하여 NAV를 구한 뒤, Sleeve 합성 방식으로 월간 60:40 리밸런싱.
코인 NAV: 구 엔진 (3-snapshot 합성, LA 수정)
주식 NAV: stock_engine (LA 수정)

시그널: t-1 종가, 체결: t일 — 양쪽 모두 동일.
"""

import sys, os, time, argparse
import numpy as np, pandas as pd
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Coin
from coin_engine import (
    load_universe, load_all_prices, filter_universe, calc_metrics, _close_to,
    resolve_canary, get_price, _port_val, execute_rebalance,
)
from coin_helpers import (
    B, merge_snapshots, calc_current_weights, calc_half_turnover,
    compute_signal_weights_filtered, check_blacklist, update_blacklist,
)
from coin_dd_exit import check_coin_drawdown, _empty_ext

# Stock
from stock_engine import (
    SP, load_prices as load_stock_prices,
    precompute as stock_precompute,
    _init as stock_init, _run_one, run_bt, get_val, ALL_TICKERS,
    metrics as stock_metrics,
)
import stock_engine as tsi


def check_crash_vt(params, ind, date):
    if params.crash == 'vt':
        r = get_val(ind, 'VT', date, 'ret')
        return not np.isnan(r) and r <= -params.crash_thresh
    return False


# ═══ Coin NAV (from backtest_official.py run_coin_backtest) ═══

def get_coin_nav(prices, universe_map, params_kw, dd_lookback, dd_threshold,
                 bl_drop, bl_days, drift_threshold, post_flip_delay,
                 snapshot_days=(1, 10, 19)):
    """코인 NAV 계산 (LA 수정, crash breaker 포함)."""
    params_base = B(**params_kw)

    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= params_base.start_date) &
                          (btc.index <= params_base.end_date)]
    if len(all_dates) == 0: return None

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

        sig_date = prev_date if prev_date is not None else date

        # Crash Breaker (G5)
        crash_cooldown = state.get('crash_cooldown', 0)
        crash_just_ended = False
        if crash_cooldown > 0:
            crash_cooldown -= 1
            if i > 0 and params_base.risk == 'G5':
                btc_close = _close_to('BTC-USD', prices, sig_date)
                if btc_close is not None and len(btc_close) >= 2:
                    if btc_close.iloc[-1] / btc_close.iloc[-2] - 1 <= -0.10:
                        crash_cooldown = 3
            if crash_cooldown == 0:
                crash_just_ended = True
            state['crash_cooldown'] = crash_cooldown
        elif i > 0 and params_base.risk == 'G5':
            btc_close = _close_to('BTC-USD', prices, sig_date)
            if btc_close is not None and len(btc_close) >= 2:
                if btc_close.iloc[-1] / btc_close.iloc[-2] - 1 <= -0.10:
                    for t in list(holdings.keys()):
                        cash += holdings[t] * get_price(t, prices, date) * (1 - params_base.tx_cost)
                    holdings = {}
                    for si in range(n_snap):
                        snapshots[si] = {'CASH': 1.0}
                    state['crash_cooldown'] = 3

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

        newly_bl = []
        if bl_drop < 0 and canary_on and holdings:
            newly_bl = check_blacklist(holdings, prices, sig_date, bl_drop, blacklist, bl_days)
            if newly_bl:
                for si in range(n_snap):
                    snap = snapshots[si]
                    removed = sum(snap.pop(c, 0) for c in newly_bl)
                    if removed > 0:
                        snap['CASH'] = snap.get('CASH', 0) + removed
                        total = sum(snap.values())
                        if total > 0:
                            snapshots[si] = {t: w / total for t, w in snap.items()}
        state['_blacklist'] = set(blacklist.keys())

        need_rebal = False
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped:
            exited = [t for t in holdings if holdings[t] > 0
                      and check_coin_drawdown(t, prices, sig_date, dd_lookback, dd_threshold)]
            if exited:
                for coin in exited:
                    cash += holdings.pop(coin) * get_price(coin, prices, date) * (1 - params_base.tx_cost)
                    for si in range(n_snap):
                        if coin in snapshots[si]:
                            w = snapshots[si].pop(coin)
                            snapshots[si]['CASH'] = snapshots[si].get('CASH', 0) + w
                            total = sum(snapshots[si].values())
                            if total > 0:
                                snapshots[si] = {t: v / total for t, v in snapshots[si].items()}

        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights_filtered(prices, universe_map, sig_date, params_base, state, blacklist)
            need_rebal = True
        elif canary_flipped:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights_filtered(prices, universe_map, sig_date, params_base, state, blacklist) if canary_on else {'CASH': 1.0}
            need_rebal = True
        elif post_flip_delay > 0 and canary_on:
            fd = state.get('canary_on_date')
            if fd and not state.get('post_flip_refreshed') and (date - fd).days >= post_flip_delay:
                state['post_flip_refreshed'] = True
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights_filtered(prices, universe_map, sig_date, params_base, state, blacklist)
                need_rebal = True

        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights_filtered(prices, universe_map, sig_date, params_base, state, blacklist)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        if bl_drop < 0 and newly_bl:
            need_rebal = True

        combined = merge_snapshots(snapshots)
        if not need_rebal and canary_on and drift_threshold > 0:
            if calc_half_turnover(calc_current_weights(holdings, cash, prices, date), combined) >= drift_threshold:
                need_rebal = True

        if state.get('crash_cooldown', 0) > 0:
            need_rebal = False

        if crash_just_ended and not holdings:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights_filtered(prices, universe_map, sig_date, params_base, state, blacklist) if canary_on else {'CASH': 1.0}
            combined = merge_snapshots(snapshots)
            need_rebal = True

        if need_rebal:
            holdings, cash = execute_rebalance(holdings, cash, combined, prices, date, params_base.tx_cost)

        portfolio_values.append({'Date': date, 'Value': _port_val(holdings, cash, prices, date)})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month
        prev_date = date

    if not portfolio_values: return None
    return pd.DataFrame(portfolio_values).set_index('Date')['Value']


# ═══ Stock NAV ═══

def get_stock_nav(stock_prices, stock_ind, sp_base, start, end):
    """주식 NAV (11-anchor 평균)."""
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt
    dfs = []
    for anchor in range(1, 12):
        p = replace(sp_base, start=start, end=end, _anchor=anchor)
        df = run_bt(stock_prices, stock_ind, p)
        if df is not None:
            dfs.append(df['Value'])
    if not dfs: return None
    combined = pd.concat(dfs, axis=1)
    return combined.mean(axis=1)


# ═══ Sleeve 합성 ═══

def simulate_sleeve(stock_nav, coin_nav, stock_ratio=0.588, coin_ratio=0.392,
                    tx_cost_rebal=0.002):
    """두 NAV를 월간 60:40 리밸런싱."""
    common = stock_nav.index.intersection(coin_nav.index)
    if len(common) < 100: return None

    s = stock_nav.loc[common] / stock_nav.loc[common].iloc[0]
    c = coin_nav.loc[common] / coin_nav.loc[common].iloc[0]

    cash_ratio = 1.0 - stock_ratio - coin_ratio
    s_units = stock_ratio
    c_units = coin_ratio
    cash = cash_ratio
    prev_month = None
    values = []

    for date in common:
        s_val = s_units * s.loc[date]
        c_val = c_units * c.loc[date]
        total = s_val + c_val + cash

        cur_month = date.strftime('%Y-%m')
        is_month_change = prev_month is not None and cur_month != prev_month

        if is_month_change and total > 0:
            cur_s_pct = s_val / total
            turnover = abs(cur_s_pct - stock_ratio) + abs(1 - cur_s_pct - cash / total - coin_ratio)
            cost = total * turnover * tx_cost_rebal / 2
            total -= cost
            s_units = (total * stock_ratio) / s.loc[date]
            c_units = (total * coin_ratio) / c.loc[date]
            cash = total * cash_ratio

        s_val = s_units * s.loc[date]
        c_val = c_units * c.loc[date]
        total = s_val + c_val + cash
        values.append({'Date': date, 'Value': total})
        prev_month = cur_month

    return pd.DataFrame(values).set_index('Date')


# ═══ Version Definitions ═══

COIN_VERSIONS = {
    'V12': dict(
        params=dict(canary='K1', sma_period=50, vote_smas=(), vote_threshold=1,
                    health='H1', health_sma=30, health_mom_short=21, vol_cap=0.10,
                    selection='S6', n_picks=5, weighting='W2', top_n=50),
        dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0,
        drift_threshold=0, post_flip_delay=0),
    'V14': dict(
        params=dict(sma_period=60, canary_band=1.0, health_sma=0,
                    selection='baseline', n_picks=5, weighting='WC', top_n=40, risk='G5'),
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5),
    'V17': dict(
        params=dict(sma_period=60, canary_band=1.0, health_sma=0, health_mom_short=30,
                    selection='baseline', n_picks=5, weighting='WC', top_n=40, risk='G5'),
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5),
}

OFF_R7 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
OFF_R6 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC')
OFF_12 = ('SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC', 'QUAL', 'MTUM', 'IQLT', 'IMTM')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

STOCK_VERSIONS = {
    'V12': SP(offensive=OFF_12, defensive=DEF, canary_assets=('VT', 'EEM'),
              canary_sma=200, canary_hyst=0.0, select='mom3_sh3', weight='ew',
              defense='top1', def_mom_period=126, health='none', tx_cost=0.001),
    'V14': SP(offensive=OFF_R6, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='mom3_sh3', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.001),
    'V17': SP(offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='zscore3', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.001,
              crash='vt', crash_thresh=0.03, crash_cool=3, sharpe_lookback=252),
}


def main():
    t0 = time.time()
    print("데이터 로딩...")

    # Coin
    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    coin_prices = load_all_prices(list(all_t))

    # Stock
    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)

    print(f"  완료 ({time.time()-t0:.1f}s)")

    VERSIONS = ['V12', 'V14', 'V17']
    PERIODS = [('2018-01-01', '2025-06-30'), ('2019-01-01', '2025-06-30'), ('2021-01-01', '2025-06-30')]

    print(f"\n{'='*95}")
    print("통합 60:40 포트폴리오 (Sleeve 합성, 월간 리밸런싱)")
    print(f"{'='*95}")

    for start, end in PERIODS:
        print(f"\n  [{start} ~ {end}]")
        print(f"  {'버전':<8s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}  |  {'코인S':>6s} {'코인C':>7s} {'주식S':>6s} {'주식C':>7s}")
        print(f"  {'-'*80}")

        for ver in VERSIONS:
            # Coin NAV
            cfg = COIN_VERSIONS[ver]
            top_n = cfg['params'].get('top_n', 40)
            um = coin_um.get(top_n, coin_um[40])
            cfg['params']['start_date'] = start
            cfg['params']['end_date'] = end
            coin_nav = get_coin_nav(coin_prices, um, cfg['params'],
                                     cfg['dd_lookback'], cfg['dd_threshold'],
                                     cfg['bl_drop'], cfg['bl_days'],
                                     cfg['drift_threshold'], cfg['post_flip_delay'])

            # Stock NAV
            stock_nav = get_stock_nav(stock_prices, stock_ind,
                                      STOCK_VERSIONS[ver], start, end)

            if coin_nav is None or stock_nav is None:
                print(f"  {ver:<8s}  데이터 부족")
                continue

            # Individual metrics
            cm = calc_metrics(pd.DataFrame({'Value': coin_nav}))
            sm = calc_metrics(pd.DataFrame({'Value': stock_nav}))

            # Combined 60:40
            combo = simulate_sleeve(stock_nav, coin_nav)
            if combo is None:
                print(f"  {ver:<8s}  합성 실패")
                continue

            m = calc_metrics(combo)
            s = m.get('Sharpe', 0)
            c = m.get('CAGR', 0)
            mdd = m.get('MDD', 0)
            cal = c / abs(mdd) if mdd != 0 else 0

            print(f"  {ver:<8s} {s:>7.3f} {c:>+8.1%} {mdd:>+8.1%} {cal:>7.2f}"
                  f"  |  {cm['Sharpe']:>6.2f} {cm['CAGR']:>+7.1%} {sm['Sharpe']:>6.2f} {sm['CAGR']:>+7.1%}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
