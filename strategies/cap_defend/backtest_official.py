#!/usr/bin/env python3
"""V12~V17 공식 백테스트 — 코인 + 주식.

코인: 구 엔진 (3-snapshot 합성, LA 수정: t-1 시그널 → t 체결)
주식: test_stock_improve.py 엔진 (이미 LA 수정됨)

Usage:
  python3 backtest_official.py              # 전 버전 비교
  python3 backtest_official.py --version v17  # V17만
  python3 backtest_official.py --coin-only   # 코인만
  python3 backtest_official.py --stock-only  # 주식만
"""

import sys, os, time, argparse
import numpy as np, pandas as pd
from dataclasses import replace
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# === Coin imports ===
from coin_engine import (
    load_universe, load_all_prices, filter_universe, calc_metrics,
    resolve_canary, get_price, _port_val, execute_rebalance,
)
from coin_helpers import (
    B, merge_snapshots, calc_current_weights, calc_half_turnover,
    compute_signal_weights_filtered, check_blacklist, update_blacklist,
)
from coin_dd_exit import check_coin_drawdown, _empty_ext

# === Stock imports ===
from stock_engine import (
    SP, load_prices as load_stock_prices,
    precompute as stock_precompute,
    _init as stock_init, _run_one, get_val, ALL_TICKERS,
)
import stock_engine as tsi


# ═══════════════════════════════════════════════════════════════════
# COIN BACKTEST (LA 수정)
# ═══════════════════════════════════════════════════════════════════

def run_coin_backtest(prices, universe_map, snapshot_days=(1, 10, 19),
                      dd_lookback=0, dd_threshold=0,
                      bl_drop=0, bl_days=7,
                      drift_threshold=0, post_flip_delay=0,
                      params_base=None):
    """코인 백테스트 — t-1 시그널, t 체결."""
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
                dd_exit_count += len(exited)
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

        if need_rebal:
            holdings, cash = execute_rebalance(holdings, cash, combined, prices, date, params_base.tx_cost)
            rebal_count += 1

        portfolio_values.append({'Date': date, 'Value': _port_val(holdings, cash, prices, date)})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month
        prev_date = date

    if not portfolio_values:
        return _empty_ext()
    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    return {'metrics': calc_metrics(pvdf), 'rebal_count': rebal_count, 'dd_exit_count': dd_exit_count}


# ═══════════════════════════════════════════════════════════════════
# VERSION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

COIN_VERSIONS = {
    # V12: SMA50, H1(SMA30+Mom21+Vol10%), Sharpe(S6) Top5, InvVol(W2), Top50
    'V12': dict(
        params=dict(canary='K1', vote_smas=(), vote_threshold=1,
                    health='H1', health_sma=30, health_mom_short=21, vol_cap=0.10,
                    selection='S6', n_picks=5, weighting='W2', top_n=50),
        dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0,
        drift_threshold=0, post_flip_delay=0),
    # V13: ≈ V12 (MultiBonus → S6로 근사, 엔진에 정확한 키 없음)
    'V13': dict(
        params=dict(canary='K1', vote_smas=(), vote_threshold=1,
                    health='H1', health_sma=30, health_mom_short=21, vol_cap=0.10,
                    selection='S6', n_picks=5, weighting='W2', top_n=50),
        dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0,
        drift_threshold=0, post_flip_delay=0),
    # V14: K8(SMA60+1%hyst), HK(Mom21+Mom90+Vol5%), 시총Top5(baseline), WC(EW+20%Cap), G5(Crash), T40
    'V14': dict(
        params=dict(canary_band=1.0, health_sma=0,
                    selection='baseline', n_picks=5, weighting='WC', top_n=40,
                    risk='G5'),
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5),
    # V15: = V14 (Mom21 동일)
    'V15': dict(
        params=dict(canary_band=1.0, health_sma=0,
                    health_mom_short=21,
                    selection='baseline', n_picks=5, weighting='WC', top_n=40,
                    risk='G5'),
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5),
    # V16: Mom30 (악화)
    'V16': dict(
        params=dict(canary_band=1.0, health_sma=0,
                    health_mom_short=30,
                    selection='baseline', n_picks=5, weighting='WC', top_n=40,
                    risk='G5'),
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5),
    # V17: Mom30 (V16과 동일, 정확한 키 기준 최적)
    'V17': dict(
        params=dict(canary_band=1.0, health_sma=0,
                    health_mom_short=30,
                    selection='baseline', n_picks=5, weighting='WC', top_n=40,
                    risk='G5'),
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5),
}

OFF_R7 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
OFF_R6 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC')
OFF_12 = ('SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC', 'QUAL', 'MTUM', 'IQLT', 'IMTM')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

def check_crash_vt(params, ind, date):
    if params.crash == 'vt':
        r = get_val(ind, 'VT', date, 'ret')
        return not np.isnan(r) and r <= -params.crash_thresh
    return False

STOCK_VERSIONS = {
    # V12: 12종, VT&EEM 카나리 (hyst 없음), Mom3+Sh3 union, Top1 방어
    'V12': SP(offensive=OFF_12, defensive=DEF, canary_assets=('VT', 'EEM'),
              canary_sma=200, canary_hyst=0.0, select='mom3_sh3', weight='ew',
              defense='top1', def_mom_period=126, health='none', tx_cost=0.001),
    # V13: 12종, VT&EEM 카나리 + 1% hyst, Mom3+Sh3
    'V13': SP(offensive=OFF_12, defensive=DEF, canary_assets=('VT', 'EEM'),
              canary_sma=200, canary_hyst=0.01, select='mom3_sh3', weight='ew',
              defense='top1', def_mom_period=126, health='none', tx_cost=0.001),
    # V14: R6, EEM only + 0.5% hyst, Mom3+Sh3, Top3 방어
    'V14': SP(offensive=OFF_R6, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='mom3_sh3', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.001),
    # V15: R7(+VNQ), Zscore4 Sh63
    'V15': SP(offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='zscore4', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.001,
              sharpe_lookback=63),
    # V16: = V15 (주식 변경 없음)
    'V16': SP(offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='zscore4', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.001,
              sharpe_lookback=63),
    # V17: Zscore3 Sh252d + VT Crash -3%/3d
    'V17': SP(offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='zscore3', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.001,
              crash='vt', crash_thresh=0.03, crash_cool=3, sharpe_lookback=252),
}


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='all', help='v12,v14,v15,v17 or all')
    parser.add_argument('--coin-only', action='store_true')
    parser.add_argument('--stock-only', action='store_true')
    args = parser.parse_args()

    versions = args.version.upper().split(',') if args.version != 'all' else ['V12', 'V13', 'V14', 'V15', 'V16', 'V17']

    t0 = time.time()
    print("데이터 로딩...")

    # Coin data
    coin_prices, coin_um = None, None
    if not args.stock_only:
        um_raw = load_universe()
        coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
        all_t = set()
        for fm in coin_um.values():
            for ts in fm.values(): all_t.update(ts)
        all_t.update(['BTC-USD', 'ETH-USD'])
        coin_prices = load_all_prices(list(all_t))

    # Stock data
    if not args.coin_only:
        stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
        stock_ind = stock_precompute(stock_prices)
        stock_init(stock_prices, stock_ind)
        tsi.check_crash = check_crash_vt

    print(f"  완료 ({time.time()-t0:.1f}s)")

    PERIODS_COIN = [('2018-01-01', '2025-06-30'), ('2019-01-01', '2025-06-30'), ('2021-01-01', '2025-06-30')]
    PERIODS_STOCK = [('2017-01-01', '2025-12-31'), ('2018-01-01', '2025-12-31'), ('2021-01-01', '2025-12-31')]
    SNAP = (1, 10, 19)

    # ═══ COIN ═══
    if not args.stock_only:
        print("\n" + "=" * 85)
        print("코인 전략 (LA 수정, 단일 앵커 1/10/19)")
        print("=" * 85)

        for start, end in PERIODS_COIN:
            print(f"\n  [{start} ~ {end}]")
            print(f"  {'버전':<8s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'DD':>4s} {'Rebal':>5s}")
            print(f"  {'-'*50}")

            for ver in versions:
                if ver not in COIN_VERSIONS: continue
                cfg = COIN_VERSIONS[ver]
                top_n = cfg['params'].get('top_n', 40)
                um = coin_um.get(top_n, coin_um[40])
                p = B(**cfg['params'])
                p.start_date = start; p.end_date = end

                r = run_coin_backtest(coin_prices, um, SNAP,
                                      dd_lookback=cfg['dd_lookback'], dd_threshold=cfg['dd_threshold'],
                                      bl_drop=cfg['bl_drop'], bl_days=cfg['bl_days'],
                                      drift_threshold=cfg['drift_threshold'],
                                      post_flip_delay=cfg['post_flip_delay'], params_base=p)
                m = r['metrics']
                s, c, mdd = m['Sharpe'], m['CAGR'], m['MDD']
                cal = c / abs(mdd) if mdd != 0 else 0
                print(f"  {ver:<8s} {s:>7.3f} {c:>+8.1%} {mdd:>+8.1%} {cal:>7.2f} {r['dd_exit_count']:>4d} {r['rebal_count']:>5d}")

    # ═══ STOCK ═══
    if not args.coin_only:
        print("\n\n" + "=" * 85)
        print("주식 전략 (LA 수정, 11-anchor 평균)")
        print("=" * 85)

        for start, end in PERIODS_STOCK:
            print(f"\n  [{start} ~ {end}]")
            print(f"  {'버전':<8s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
            print(f"  {'-'*40}")

            for ver in versions:
                if ver not in STOCK_VERSIONS: continue
                base = STOCK_VERSIONS[ver]
                sp = replace(base, start=start, end=end)
                rs = [_run_one(replace(sp, _anchor=a)) for a in range(1, 12)]
                rs = [r for r in rs if r]
                if rs:
                    s = np.mean([r['Sharpe'] for r in rs])
                    c = np.mean([r['CAGR'] for r in rs])
                    mdd = np.mean([r['MDD'] for r in rs])
                    cal = np.mean([r.get('Calmar', 0) for r in rs])
                    print(f"  {ver:<8s} {s:>7.3f} {c:>+8.1%} {mdd:>+8.1%} {cal:>7.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
