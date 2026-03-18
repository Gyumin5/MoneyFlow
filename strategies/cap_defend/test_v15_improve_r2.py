#!/usr/bin/env python3
"""V15 코인 개선안 2라운드 — 구조 변경.

1라운드 우승: Baseline + Stop -15% (Sharpe 1.550)
이 위에 구조 변경을 추가 테스트:
  I6: 다단계 카나리아 (SMA20/60/120 → 비중 스케일링)
  I7: 복합 카나리아 (BTC SMA60 + ETH/BTC SMA30, 2-of-2)
  I8: 역변동성 가중 (1/Vol)
  I9: Health 파라미터 그리드
  I10: 시총+Mom 혼합 랭킹
"""

import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine_v15 import (
    V15Params, Tranche, load_data,
    check_health, execute_rebalance,
    sell_coin_from_tranche, sell_all_from_tranche,
    compute_target_weights, get_universe_for_date,
    _close_to, calc_ret, get_vol, get_sma,
    get_price, calc_metrics, calc_yearly_metrics,
    _get_daily_returns, _bl_cache,
)
from strategy_engine import load_all_prices


def run_r2_backtest(prices, universe_map, config,
                    tranche_days=(1, 10, 19),
                    start='2018-01-01', end='2025-06-30',
                    tx_cost=0.004, capital=10000.0):
    """2라운드 백테스트 — 구조 변경 포함."""

    canary_mode = config.get('canary_mode', 'standard')  # standard/graded/composite
    weight_mode = config.get('weight_mode', 'ew')  # ew/inv_vol
    health_mom_s = config.get('health_mom_s', 21)
    health_mom_l = config.get('health_mom_l', 90)
    health_vol_w = config.get('health_vol_w', 90)
    health_vol_cap = config.get('health_vol_cap', 0.05)
    selection_mode = config.get('selection', 'mcap')  # mcap/hybrid
    coin_stop = config.get('coin_stop', -0.15)  # 1라운드 우승자 기본 포함
    soft_cap = config.get('soft_cap', None)  # max per-coin weight (e.g., 0.35)

    params = V15Params()
    n_tr = len(tranche_days)
    capital_per = capital / n_tr
    tranches = [Tranche(i, tranche_days[i], capital_per) for i in range(n_tr)]

    prev_canary = False
    canary_on = False
    canary_on_date = None
    crash_cooldown = 0
    prev_month = None

    rebal_count = 0
    dd_exit_count = 0
    stop_count = 0

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

        # Crash Breaker (standard -10%)
        crash_just_ended = False
        if crash_cooldown > 0:
            crash_cooldown -= 1
            if crash_cooldown == 0:
                btc_s = _close_to('BTC-USD', prices, date)
                if len(btc_s) >= 2 and (btc_s.iloc[-1] / btc_s.iloc[-2] - 1) <= -0.10:
                    crash_cooldown = 3
                else:
                    crash_just_ended = True
        elif not is_first:
            btc_s = _close_to('BTC-USD', prices, date)
            if len(btc_s) >= 2 and (btc_s.iloc[-1] / btc_s.iloc[-2] - 1) <= -0.10:
                for tr in tranches:
                    sell_all_from_tranche(tr, prices, date, tx_cost)
                crash_cooldown = 3

        # ── Canary ──
        btc_s = _close_to('BTC-USD', prices, date)
        exposure_scale = 1.0  # for graded canary

        if canary_mode == 'graded':
            # I6: SMA20/60/120 → count above → 33/66/100%
            count = 0
            for sma_p in [20, 60, 120]:
                if len(btc_s) >= sma_p:
                    sma_val = btc_s.rolling(sma_p).mean().iloc[-1]
                    if not np.isnan(sma_val) and btc_s.iloc[-1] > sma_val:
                        count += 1
            canary_on = count > 0
            exposure_scale = count / 3.0  # 0.33, 0.66, 1.0

        elif canary_mode == 'composite':
            # I7: BTC SMA60 + ETH/BTC SMA30, both must pass
            btc_ok = False
            if len(btc_s) >= 60:
                sma60 = btc_s.rolling(60).mean().iloc[-1]
                hyst = 0.01
                if prev_canary:
                    btc_ok = not (btc_s.iloc[-1] < sma60 * (1 - hyst))
                else:
                    btc_ok = btc_s.iloc[-1] > sma60 * (1 + hyst)

            eth_btc_ok = False
            eth_btc = _close_to('ETH-BTC', prices, date)
            if len(eth_btc) >= 30:
                sma30 = eth_btc.rolling(30).mean().iloc[-1]
                if not np.isnan(sma30):
                    eth_btc_ok = eth_btc.iloc[-1] > sma30

            canary_on = btc_ok and eth_btc_ok

        else:  # standard
            if len(btc_s) >= 60:
                sma = btc_s.rolling(60).mean().iloc[-1]
                if not np.isnan(sma):
                    hyst = 0.01
                    if prev_canary:
                        canary_on = not (btc_s.iloc[-1] < sma * (1 - hyst))
                    else:
                        canary_on = btc_s.iloc[-1] > sma * (1 + hyst)

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

        # Blacklist
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

        for tr in tranches:
            tr.dd_exited_today = set()

            if canary_on and not crash_just_ended:
                # Coin stop-loss (-15%)
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
                        if rets.loc[m].iloc[-1] <= coin_stop:
                            sell_coin_from_tranche(tr, t, prices, date, tx_cost)
                            tr.dd_exited_today.add(t)
                            stop_count += 1

                # DD Exit (60d -25%)
                for t in list(tr.holdings.keys()):
                    if t in tr.dd_exited_today:
                        continue
                    if tr.holdings.get(t, 0) <= 0:
                        continue
                    s = _close_to(t, prices, date)
                    if len(s) < 60:
                        continue
                    peak = s.iloc[-60:].max()
                    if peak > 0 and (s.iloc[-1] / peak - 1) <= -0.25:
                        sell_coin_from_tranche(tr, t, prices, date, tx_cost)
                        tr.dd_exited_today.add(t)
                        dd_exit_count += 1

            # Rebalance triggers
            do_rebal = False
            if is_first:
                do_rebal = True
            if not do_rebal and signal_flipped:
                do_rebal = True
            if not do_rebal and crash_just_ended:
                do_rebal = True
            if (not do_rebal and canary_on and not tr.post_flip_refreshed
                    and canary_on_date and (date - canary_on_date).days >= 5):
                do_rebal = True
                tr.post_flip_refreshed = True
            if (not do_rebal and not tr.anchor_done_this_month
                    and date.day >= tr.anchor_day and not is_first):
                do_rebal = True
                tr.anchor_done_this_month = True

            if do_rebal:
                if canary_on:
                    candidates = [t for t in universe
                                  if t not in blacklist and t not in tr.dd_exited_today]

                    # Health filter (configurable)
                    healthy = []
                    for t in candidates:
                        s = _close_to(t, prices, date)
                        if len(s) < health_mom_l + 1:
                            continue
                        mom_s = calc_ret(s, health_mom_s)
                        mom_l = calc_ret(s, health_mom_l)
                        vol = get_vol(s, health_vol_w)
                        if mom_s > 0 and mom_l > 0 and vol <= health_vol_cap:
                            healthy.append(t)
                            if len(healthy) >= 5:
                                break

                    # I10: Hybrid selection (market cap + momentum rank)
                    if selection_mode == 'hybrid' and len(healthy) > 5:
                        scored = []
                        for i, t in enumerate(healthy):
                            s = _close_to(t, prices, date)
                            mom90 = calc_ret(s, 90) if len(s) >= 91 else 0
                            vol90 = get_vol(s, 90) if len(s) >= 91 else 1
                            # Lower rank = better. Market cap rank = position in universe
                            mcap_rank = i
                            scored.append((t, mcap_rank, mom90, vol90))
                        # Composite: 50% mcap + 30% mom + 20% low vol
                        for j, (t, mcap_r, mom, vol) in enumerate(scored):
                            pass
                        # Simple: sort by composite score
                        from scipy.stats import rankdata
                        mcap_ranks = rankdata([s[1] for s in scored])
                        mom_ranks = rankdata([-s[2] for s in scored])  # higher mom = better
                        vol_ranks = rankdata([s[3] for s in scored])   # lower vol = better
                        composite = 0.5 * mcap_ranks + 0.3 * mom_ranks + 0.2 * vol_ranks
                        order = np.argsort(composite)
                        healthy = [scored[i][0] for i in order[:5]]

                    picks = healthy[:5]

                    if picks:
                        # Weighting
                        if weight_mode == 'inv_vol':
                            vols = {}
                            for t in picks:
                                s = _close_to(t, prices, date)
                                v = get_vol(s, 90) if len(s) >= 91 else 0.01
                                vols[t] = max(v, 0.001)
                            inv = {t: 1.0 / v for t, v in vols.items()}
                            total = sum(inv.values())
                            target_w = {t: w / total for t, w in inv.items()}
                        else:
                            target_w = {t: 1.0 / len(picks) for t in picks}

                        # Soft cap: limit per-coin weight
                        if soft_cap:
                            target_w = {t: min(w, soft_cap) for t, w in target_w.items()}

                        # Graded canary: scale down exposure
                        if canary_mode == 'graded' and exposure_scale < 1.0:
                            target_w = {t: w * exposure_scale for t, w in target_w.items()}

                        execute_rebalance(tr, target_w, prices, date, tx_cost)
                    else:
                        sell_all_from_tranche(tr, prices, date, tx_cost)
                else:
                    sell_all_from_tranche(tr, prices, date, tx_cost)
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
    return {'metrics': m, 'yearly': ym, 'pv': pvdf,
            'rebal_count': rebal_count, 'dd_exit_count': dd_exit_count,
            'stop_count': stop_count}


def main():
    print('V15 개선안 2라운드 — 구조 변경 (기반: Stop -15%)')
    print('=' * 120)

    print('Loading data...')
    prices, universe = load_data()
    # Load ETH-BTC for composite canary
    extra = load_all_prices({'ETH-BTC'})
    prices.update(extra)
    print(f'  {len(prices)} tickers loaded\n')

    CONFIGS = [
        ('R1 Winner (Stop-15%)', {}),  # baseline = standard canary + stop -15%
        ('I6: Graded Canary', {'canary_mode': 'graded'}),
        ('I7: Composite (BTC+ETH/BTC)', {'canary_mode': 'composite'}),
        ('I8: InvVol Weight', {'weight_mode': 'inv_vol'}),
        ('I9a: Mom14+Mom60+Vol5%', {'health_mom_s': 14, 'health_mom_l': 60}),
        ('I9b: Mom30+Mom90+Vol5%', {'health_mom_s': 30}),
        ('I9c: Mom21+Mom90+Vol7%', {'health_vol_cap': 0.07}),
        ('I9d: Mom21+Mom60+Vol5%', {'health_mom_l': 60}),
        ('I10: Hybrid Selection', {'selection': 'hybrid'}),
        # Combos
        ('Best: Graded+InvVol', {'canary_mode': 'graded', 'weight_mode': 'inv_vol'}),
        ('Best: Composite+InvVol', {'canary_mode': 'composite', 'weight_mode': 'inv_vol'}),
    ]

    N_OFFSETS = 10
    results = {}

    for name, cfg in CONFIGS:
        _bl_cache.clear()
        print(f'  {name}...')
        sharpes, cagrs, mdds, sortinos = [], [], [], []

        for offset in range(N_OFFSETS):
            _bl_cache.clear()
            days = tuple((d - 1 + offset) % 28 + 1 for d in (1, 10, 19))
            r = run_r2_backtest(prices, universe, cfg, tranche_days=days)
            if r:
                m = r['metrics']
                sharpes.append(m['Sharpe'])
                cagrs.append(m['CAGR'])
                mdds.append(m['MDD'])
                sortinos.append(m.get('Sortino', 0))

        if sharpes:
            results[name] = {
                'Sharpe': np.mean(sharpes), 'σ': np.std(sharpes),
                'CAGR': np.mean(cagrs), 'MDD': np.mean(mdds),
                'Sortino': np.mean(sortinos),
                'Calmar': abs(np.mean(cagrs) / np.mean(mdds)) if np.mean(mdds) != 0 else 0,
            }
            if r and 'yearly' in r:
                results[name]['yearly'] = {yr: ym['CAGR'] for yr, ym in r['yearly'].items()}
                results[name]['rebal'] = r['rebal_count']

    # Print
    print(f'\n{"=" * 130}')
    print(f'  2라운드 결과 (10-anchor 평균, 기반: Stop -15%)')
    print(f'{"=" * 130}')
    print(f'{"Config":<32} {"Sharpe":>7} {"σ":>5} {"CAGR":>7} {"MDD":>7} {"Sortino":>8} {"Calmar":>7}  {"2022":>6} {"2024":>6}')
    print('-' * 130)

    base_sh = results.get('R1 Winner (Stop-15%)', {}).get('Sharpe', 0)
    for name, _ in CONFIGS:
        r = results.get(name)
        if not r:
            continue
        yr = r.get('yearly', {})
        marker = ' ★' if r['Sharpe'] > base_sh + 0.01 and name != 'R1 Winner (Stop-15%)' else ''
        print(f'{name:<32} {r["Sharpe"]:>7.3f} {r["σ"]:>4.3f} {r["CAGR"]:>+6.1%} {r["MDD"]:>6.1%} '
              f'{r["Sortino"]:>8.3f} {r["Calmar"]:>7.2f}  '
              f'{yr.get(2022, 0):>+5.1%} {yr.get(2024, 0):>+5.1%}{marker}')

    # Delta
    print(f'\n{"Config":<32} {"ΔSharpe":>9} {"ΔCAGR":>8} {"ΔMDD":>8}')
    print('-' * 60)
    base = results.get('R1 Winner (Stop-15%)', {})
    for name, _ in CONFIGS:
        if name == 'R1 Winner (Stop-15%)':
            continue
        r = results.get(name, {})
        if not r:
            continue
        ds = r['Sharpe'] - base.get('Sharpe', 0)
        dc = r['CAGR'] - base.get('CAGR', 0)
        dm = r['MDD'] - base.get('MDD', 0)
        v = '✓' if ds > 0 else '✗'
        print(f'{name:<32} {ds:>+9.3f} {dc:>+7.1%} {dm:>+7.1%} {v}')


if __name__ == '__main__':
    main()
