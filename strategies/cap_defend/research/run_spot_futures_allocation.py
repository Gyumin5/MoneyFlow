#!/usr/bin/env python3
"""현물 V18 + 선물 최종전략 상위 자본배분 백테스트.

목표:
- 업비트 현물 V18 sleeve
- 바이낸스 선물 최종전략 sleeve
- 두 sleeve를 다양한 비중과 리밸런싱 규칙으로 합성

리밸런싱 규칙:
- monthly: 월초 리밸런싱
- band: 목표 비중에서 band 이상 이탈 시 리밸런싱
- monthly_or_band: 월초 또는 band 이탈 시 리밸런싱
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_official import run_coin_backtest
from coin_engine import load_universe, load_all_prices, filter_universe, calc_metrics
from coin_helpers import B

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets
from run_stoploss_test import START as FUT_START, END as FUT_END

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "spot_futures_allocation_results.csv")


SPOT_START = FUT_START
SPOT_END = FUT_END

SPOT_CFG = dict(
    params=dict(
        start_date=SPOT_START,
        end_date=SPOT_END,
        sma_period=50,
        canary_band=1.5,
        vote_smas=(50,),
        health_sma=0,
        health_mom_short=30,
        selection='SG',
        n_picks=5,
        weighting='WG',
        top_n=40,
        risk='G5',
    ),
    dd_lookback=60,
    dd_threshold=-0.25,
    bl_drop=-0.15,
    bl_days=7,
    drift_threshold=0.10,
    post_flip_delay=5,
)

FUT_STRATEGIES = {
    '1h_09': dict(
        interval='1h', sma_bars=168, mom_short_bars=36, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
        vol_threshold=0.80, n_snapshots=3, snap_interval_bars=27,
    ),
    '4h_01': dict(
        interval='4h', sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom1vol', vol_mode='daily',
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120,
    ),
    '4h_09': dict(
        interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
        vol_threshold=0.60, n_snapshots=3, snap_interval_bars=21,
    ),
}
FUT_WEIGHTS = {'1h_09': 1 / 3, '4h_01': 1 / 3, '4h_09': 1 / 3}

RATIOS = [
    ('80_20', 0.80, 0.20),
    ('70_30', 0.70, 0.30),
    ('60_40', 0.60, 0.40),
    ('50_50', 0.50, 0.50),
]

RULES = [
    ('monthly', None),
    ('band_10', 0.10),
    ('monthly_or_band_10', 0.10),
]


def get_spot_nav():
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values():
        all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))

    cfg = SPOT_CFG
    r = run_coin_backtest(
        prices,
        um40,
        snapshot_days=(1, 10, 19),
        dd_lookback=cfg['dd_lookback'],
        dd_threshold=cfg['dd_threshold'],
        bl_drop=cfg['bl_drop'],
        bl_days=cfg['bl_days'],
        drift_threshold=cfg['drift_threshold'],
        post_flip_delay=cfg['post_flip_delay'],
        params_base=B(**cfg['params']),
    )
    return r['equity_curve'].copy()


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(
        bars, funding,
        interval=interval,
        leverage=1.0,
        start_date=FUT_START,
        end_date=FUT_END,
        _trace=trace,
        **run_cfg,
    )
    return trace


def get_futures_nav():
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    traces = {name: generate_trace(data, cfg) for name, cfg in FUT_STRATEGIES.items()}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= FUT_START) & (bars_1h['BTC'].index <= FUT_END)]
    combined = combine_targets(traces, FUT_WEIGHTS, all_dates)
    engine = SingleAccountEngine(
        bars_1h,
        funding_1h,
        leverage=5.0,
        stop_kind='prev_close_pct',
        stop_pct=0.15,
        stop_gate='cash_guard',
        stop_gate_cash_threshold=0.34,
        per_coin_leverage_mode='cap_mom_blend_543_cash',
        leverage_floor=3.0,
        leverage_mid=4.0,
        leverage_ceiling=5.0,
        leverage_cash_threshold=0.34,
        leverage_partial_cash_threshold=0.0,
        leverage_count_floor_max=2,
        leverage_count_mid_max=4,
        leverage_canary_floor_gap=0.015,
        leverage_canary_mid_gap=0.04,
        leverage_canary_high_gap=0.08,
        leverage_canary_sma_bars=1200,
        leverage_mom_lookback_bars=24 * 30,
        leverage_vol_lookback_bars=24 * 90,
    )
    m = engine.run(combined)
    return m['_equity'].copy()


def simulate_sleeves(spot_nav, fut_nav, spot_ratio, fut_ratio, rule_name, band=None, tx_cost_rebal=0.002):
    common = spot_nav.index.intersection(fut_nav.index)
    if len(common) < 100:
        return None

    s = spot_nav.loc[common] / spot_nav.loc[common].iloc[0]
    f = fut_nav.loc[common] / fut_nav.loc[common].iloc[0]

    spot_units = spot_ratio
    fut_units = fut_ratio
    cash = 1.0 - spot_ratio - fut_ratio
    prev_month = None
    values = []
    rebal_count = 0

    for date in common:
        s_val = spot_units * s.loc[date]
        f_val = fut_units * f.loc[date]
        total = s_val + f_val + cash

        if total <= 0:
            values.append({'Date': date, 'Value': 0.0})
            continue

        cur_month = date.strftime('%Y-%m')
        is_month_change = prev_month is not None and cur_month != prev_month
        fut_pct = f_val / total
        drift = abs(fut_pct - fut_ratio)

        do_rebal = False
        if rule_name == 'monthly':
            do_rebal = is_month_change
        elif rule_name == 'band':
            do_rebal = band is not None and drift >= band
        elif rule_name == 'monthly_or_band':
            do_rebal = is_month_change or (band is not None and drift >= band)

        if do_rebal and total > 0:
            cur_spot_pct = s_val / total
            cur_fut_pct = f_val / total
            turnover = abs(cur_spot_pct - spot_ratio) + abs(cur_fut_pct - fut_ratio)
            cost = total * turnover * tx_cost_rebal / 2
            total -= cost
            if total < 0:
                total = 0
            if total > 0:
                spot_units = (total * spot_ratio) / s.loc[date]
                fut_units = (total * fut_ratio) / f.loc[date]
                cash = total * (1.0 - spot_ratio - fut_ratio)
            else:
                spot_units = 0
                fut_units = 0
                cash = 0
            rebal_count += 1

        s_val = spot_units * s.loc[date]
        f_val = fut_units * f.loc[date]
        total = s_val + f_val + cash
        values.append({'Date': date, 'Value': total})
        prev_month = cur_month

    df = pd.DataFrame(values).set_index('Date')
    return df['Value'], rebal_count


def main():
    parser = argparse.ArgumentParser(description='현물 V18 + 선물 최종전략 자본배분 비교')
    parser.add_argument('--spot-only', action='store_true')
    parser.add_argument('--futures-only', action='store_true')
    args = parser.parse_args()

    t0 = time.time()
    print('Loading spot V18 NAV...')
    spot_nav = get_spot_nav().resample('D').last().dropna()
    print(f'  spot days: {len(spot_nav)}')

    print('Loading futures final NAV...')
    fut_nav = get_futures_nav().resample('D').last().dropna()
    print(f'  futures days: {len(fut_nav)}')

    if args.spot_only:
        m = calc_metrics(pd.DataFrame({'Value': spot_nav}))
        print('spot_only', m)
        return
    if args.futures_only:
        m = calc_metrics(pd.DataFrame({'Value': fut_nav}))
        print('futures_only', m)
        return

    rows = []
    base_common = spot_nav.index.intersection(fut_nav.index)
    spot_base = calc_metrics(pd.DataFrame({'Value': spot_nav.loc[base_common]}))
    fut_base = calc_metrics(pd.DataFrame({'Value': fut_nav.loc[base_common]}))

    for ratio_name, spot_ratio, fut_ratio in RATIOS:
        for rule_name, band in RULES:
            sim_rule = rule_name
            if rule_name == 'band_10':
                sim_rule = 'band'
            elif rule_name == 'monthly_or_band_10':
                sim_rule = 'monthly_or_band'
            series, rebal_count = simulate_sleeves(
                spot_nav, fut_nav, spot_ratio, fut_ratio,
                sim_rule,
                band=band,
            )
            if series is None:
                continue
            m = calc_metrics(pd.DataFrame({'Value': series}))
            row = {
                'ratio': ratio_name,
                'spot_ratio': spot_ratio,
                'futures_ratio': fut_ratio,
                'rule': rule_name,
                'band': band if band is not None else 0.0,
                'Sharpe': m.get('Sharpe', 0.0),
                'CAGR': m.get('CAGR', 0.0),
                'MDD': m.get('MDD', 0.0),
                'Cal': (m.get('CAGR', 0.0) / abs(m.get('MDD', 0.0))) if m.get('MDD', 0.0) != 0 else 0.0,
                'Rebal': rebal_count,
                'spot_CAGR': spot_base.get('CAGR', 0.0),
                'futures_CAGR': fut_base.get('CAGR', 0.0),
            }
            rows.append(row)
            print(
                f"{ratio_name:<6} {rule_name:<18} "
                f"Sh={row['Sharpe']:.2f} CAGR={row['CAGR']:+.1%} "
                f"MDD={row['MDD']:+.1%} Cal={row['Cal']:.2f} Rebal={rebal_count}"
            )

    rows.sort(key=lambda r: (-r['Cal'], r['ratio'], r['rule']))
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['ratio', 'spot_ratio', 'futures_ratio', 'rule', 'band', 'Sharpe', 'CAGR', 'MDD', 'Cal', 'Rebal', 'spot_CAGR', 'futures_CAGR']
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUT_CSV}")
    print("\nTop 10")
    for row in rows[:10]:
        print(
            f"{row['ratio']:<6} {row['rule']:<18} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%} Rebal={row['Rebal']}"
        )
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
