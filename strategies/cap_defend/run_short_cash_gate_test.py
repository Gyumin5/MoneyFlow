#!/usr/bin/env python3
"""합산 CASH 100% 구간에서만 작동하는 BTC 숏 sleeve 테스트.

아이디어:
- 기존 롱 전략은 그대로 둔다
- 합산 목표가 CASH 100%일 때만 숏 sleeve 진입 가능
- 단, BTC 약세 필터를 추가로 만족할 때만 숏

비교 대상:
- baseline: CASH 100%면 그냥 현금
- BTC short 50% / 100%
- 약세 필터 조합별 비교
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run, SLIPPAGE_MAP
from run_ensemble import SingleAccountEngine, combine_targets
from run_final_signal_compare import STRATEGIES
from run_stoploss_test import START, END


LONG_COMBO = {'new_1h_09': 1 / 3, 'new_4h_01': 1 / 3, 'new_4h_09': 1 / 3}

SHORT_CASES = [
    dict(
        name='baseline_cash_only',
        desc='합산 CASH 100%면 현금 유지',
        short_weight=0.0,
        require_below_sma=False,
        require_mom_short_neg=False,
        require_mom_long_neg=False,
    ),
    dict(
        name='cash100_btc_short50_sma',
        desc='CASH100 + BTC<SMA면 BTC short 50%',
        short_weight=0.5,
        require_below_sma=True,
        require_mom_short_neg=False,
        require_mom_long_neg=False,
    ),
    dict(
        name='cash100_btc_short100_sma',
        desc='CASH100 + BTC<SMA면 BTC short 100%',
        short_weight=1.0,
        require_below_sma=True,
        require_mom_short_neg=False,
        require_mom_long_neg=False,
    ),
    dict(
        name='cash100_btc_short50_sma_m36',
        desc='CASH100 + BTC<SMA + Mom36<0면 BTC short 50%',
        short_weight=0.5,
        require_below_sma=True,
        require_mom_short_neg=True,
        require_mom_long_neg=False,
    ),
    dict(
        name='cash100_btc_short100_sma_m36',
        desc='CASH100 + BTC<SMA + Mom36<0면 BTC short 100%',
        short_weight=1.0,
        require_below_sma=True,
        require_mom_short_neg=True,
        require_mom_long_neg=False,
    ),
    dict(
        name='cash100_btc_short50_sma_m36_m720',
        desc='CASH100 + BTC<SMA + Mom36<0 + Mom720<0면 BTC short 50%',
        short_weight=0.5,
        require_below_sma=True,
        require_mom_short_neg=True,
        require_mom_long_neg=True,
    ),
    dict(
        name='cash100_btc_short100_sma_m36_m720',
        desc='CASH100 + BTC<SMA + Mom36<0 + Mom720<0면 BTC short 100%',
        short_weight=1.0,
        require_below_sma=True,
        require_mom_short_neg=True,
        require_mom_long_neg=True,
    ),
]


class LongShortSingleAccountEngine(SingleAccountEngine):
    """롱/숏 공존 target을 순노출 하나로 실행하는 단순 엔진.

    target 예:
    - BTC: +0.20  -> 롱 20%
    - BTC: -0.50  -> 숏 50%
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_holdings = {}
        self.short_entry_prices = {}
        self.short_entry_bar_index = {}
        self.short_margins = {}

    def run(self, target_series):
        self.capital = self.initial_capital
        self.holdings = {}
        self.entry_prices = {}
        self.entry_bar_index = {}
        self.margins = {}
        self.short_holdings = {}
        self.short_entry_prices = {}
        self.short_entry_bar_index = {}
        self.short_margins = {}
        self.reentry_cooldown = {}
        liq_count = 0
        stop_count = 0
        rebal_count = 0
        pv_list = []

        norm_funding = {}
        for coin, fr in self.funding.items():
            norm_funding[coin] = fr.copy()
            norm_funding[coin].index = norm_funding[coin].index.floor('h')

        prev_target = {}

        for date, target in target_series:
            for coin in list(self.reentry_cooldown.keys()):
                self.reentry_cooldown[coin] -= 1
                if self.reentry_cooldown[coin] <= 0:
                    del self.reentry_cooldown[coin]

            # 롱 청산/스탑
            for coin in list(self.holdings.keys()):
                df = self.bars.get(coin)
                if df is None:
                    continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < 0:
                    continue
                low = float(df['Low'].iloc[ci])
                if low <= 0:
                    continue

                liq_price = self._get_liq_price(coin)
                stop_price = self._get_stop_price(coin, date, target) if self.stop_kind != 'none' else None
                hit_liq = liq_price is not None and low <= liq_price
                hit_stop = stop_price is not None and low <= stop_price

                if hit_stop and (not hit_liq or stop_price > liq_price):
                    if self._execute_stop_exit(coin, date, stop_price):
                        stop_count += 1
                    continue

                if hit_liq:
                    pnl = self.holdings[coin] * (low - self.entry_prices[coin])
                    eq = self.margins[coin] + pnl
                    returned = max(eq - max(eq, 0) * 0.015, 0)
                    self.capital += returned
                    del self.holdings[coin]
                    del self.entry_prices[coin]
                    del self.margins[coin]
                    self.entry_bar_index.pop(coin, None)
                    liq_count += 1

            # 숏 청산
            for coin in list(self.short_holdings.keys()):
                df = self.bars.get(coin)
                if df is None:
                    continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < 0:
                    continue
                high = float(df['High'].iloc[ci])
                liq_price = self._get_short_liq_price(coin)
                if liq_price is not None and high >= liq_price:
                    eq = self.short_margins[coin] + self.short_holdings[coin] * (self.short_entry_prices[coin] - liq_price)
                    returned = max(eq - max(eq, 0) * 0.015, 0)
                    self.capital += returned
                    del self.short_holdings[coin]
                    del self.short_entry_prices[coin]
                    del self.short_margins[coin]
                    self.short_entry_bar_index.pop(coin, None)
                    liq_count += 1

            # funding
            for coin in list(self.holdings.keys()):
                fr_series = norm_funding.get(coin)
                if fr_series is None:
                    continue
                if date in fr_series.index:
                    fr = float(fr_series.loc[date])
                    cur = self._get_price(coin, date)
                    if cur > 0 and fr != 0 and not np.isnan(fr):
                        self.capital -= self.holdings[coin] * cur * fr

            for coin in list(self.short_holdings.keys()):
                fr_series = norm_funding.get(coin)
                if fr_series is None:
                    continue
                if date in fr_series.index:
                    fr = float(fr_series.loc[date])
                    cur = self._get_price(coin, date)
                    if cur > 0 and fr != 0 and not np.isnan(fr):
                        self.capital += self.short_holdings[coin] * cur * fr

            self.capital = max(self.capital, 0)

            need_rebal = (target != prev_target)
            if need_rebal and target is not None:
                self._execute_rebalance(target, date)
                rebal_count += 1

            pv = self.capital
            for coin in self.holdings:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.margins[coin] + self.holdings[coin] * (cur - self.entry_prices[coin])
            for coin in self.short_holdings:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.short_margins[coin] + self.short_holdings[coin] * (self.short_entry_prices[coin] - cur)
            pv_list.append({'Date': date, 'Value': max(pv, 0)})
            prev_target = dict(target)

        if not pv_list:
            return {}
        pvdf = pd.DataFrame(pv_list).set_index('Date')
        eq = pvdf['Value']
        eq_daily = eq.resample('D').last().dropna()
        yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
        if eq_daily.iloc[-1] <= 0 or yrs <= 0:
            return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0,
                    'Liq': liq_count, 'Stops': stop_count, 'Rebal': rebal_count, '_equity': eq}
        cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
        dr = eq_daily.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (eq / eq.cummax() - 1).min()
        cal = cagr / abs(mdd) if mdd != 0 else 0
        return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal,
                'Liq': liq_count, 'Stops': stop_count, 'Rebal': rebal_count, '_equity': eq}

    def _get_short_liq_price(self, coin):
        qty = self.short_holdings.get(coin, 0)
        entry = self.short_entry_prices.get(coin, 0)
        margin = self.short_margins.get(coin, 0)
        if qty <= 0 or entry <= 0:
            return None
        denom = qty * (1.0 + self.maint_rate)
        if denom <= 0:
            return None
        liq_price = (margin + qty * entry) / denom
        return liq_price if liq_price > 0 else None

    def _execute_rebalance(self, target, date):
        self._last_target_weights = {k: abs(v) for k, v in target.items() if k != 'CASH'}
        pv = self.capital
        for coin in self.holdings:
            cur = self._get_price(coin, date)
            if cur > 0:
                pv += self.margins[coin] + self.holdings[coin] * (cur - self.entry_prices[coin])
        for coin in self.short_holdings:
            cur = self._get_price(coin, date)
            if cur > 0:
                pv += self.short_margins[coin] + self.short_holdings[coin] * (self.short_entry_prices[coin] - cur)
        if pv <= 0:
            return

        target_qty = {}
        target_margin = {}
        target_lev = self._get_coin_leverage_map(date, {k: abs(v) for k, v in target.items()})
        for coin, w in target.items():
            if coin == 'CASH' or abs(w) <= 0:
                continue
            cur = self._get_price(coin, date)
            if cur <= 0:
                continue
            coin_lev = target_lev.get(coin, self.leverage)
            tmgn = pv * abs(w) * 0.95
            tnotional = tmgn * coin_lev
            target_qty[coin] = (tnotional / cur) * (1 if w > 0 else -1)
            target_margin[coin] = tmgn

        all_coins = set(self.holdings) | set(self.short_holdings) | {c for c, w in target.items() if c != 'CASH' and abs(w) > 0}
        for coin in all_coins:
            cur = self._get_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            cur_qty = self.holdings.get(coin, 0.0) - self.short_holdings.get(coin, 0.0)
            tgt_qty = target_qty.get(coin, 0.0)

            # 롱 -> 숏, 숏 -> 롱 전환은 먼저 청산
            if cur_qty > 0 and tgt_qty <= 0:
                pnl = self.holdings[coin] * (cur * (1 - slip) - self.entry_prices[coin])
                tx = self.holdings[coin] * cur * self.tx_cost
                self.capital += self.margins[coin] + pnl - tx
                del self.holdings[coin]
                del self.entry_prices[coin]
                del self.margins[coin]
                self.entry_bar_index.pop(coin, None)
                cur_qty = 0.0
            elif cur_qty < 0 and tgt_qty >= 0:
                sqty = self.short_holdings[coin]
                pnl = sqty * (self.short_entry_prices[coin] - cur * (1 + slip))
                tx = sqty * cur * self.tx_cost
                self.capital += self.short_margins[coin] + pnl - tx
                del self.short_holdings[coin]
                del self.short_entry_prices[coin]
                del self.short_margins[coin]
                self.short_entry_bar_index.pop(coin, None)
                cur_qty = 0.0

            # 동일 방향 조정
            if tgt_qty == 0:
                continue
            if tgt_qty > 0:
                cur_long = self.holdings.get(coin, 0.0)
                delta = tgt_qty - cur_long
                if delta <= 0:
                    if cur_long > 0 and delta < -cur_long * 0.05:
                        sell_qty = -delta
                        sell_frac = sell_qty / cur_long
                        sell_margin = self.margins[coin] * sell_frac
                        pnl = sell_qty * (cur * (1 - slip) - self.entry_prices[coin])
                        tx = sell_qty * cur * self.tx_cost
                        self.capital += sell_margin + pnl - tx
                        self.holdings[coin] -= sell_qty
                        self.margins[coin] -= sell_margin
                else:
                    entry_p = cur * (1 + slip)
                    if coin not in self.holdings:
                        margin = target_margin[coin]
                        coin_lev = target_lev.get(coin, self.leverage)
                        notional = margin * coin_lev
                        qty = notional / entry_p
                        tx = notional * self.tx_cost
                        if self.capital >= margin + tx:
                            self.capital -= margin + tx
                            self.holdings[coin] = qty
                            self.entry_prices[coin] = entry_p
                            self.entry_bar_index[coin] = self._get_bar_index(coin, date)
                            self.margins[coin] = margin
                    elif delta > self.holdings[coin] * 0.05:
                        add_notional = delta * entry_p
                        coin_lev = target_lev.get(coin, self.leverage)
                        add_margin = add_notional / coin_lev
                        tx = add_notional * self.tx_cost
                        if self.capital >= add_margin + tx:
                            self.capital -= add_margin + tx
                            total = self.holdings[coin] + delta
                            self.entry_prices[coin] = (
                                self.entry_prices[coin] * self.holdings[coin] + entry_p * delta
                            ) / total
                            self.holdings[coin] = total
                            self.margins[coin] += add_margin
            else:
                tgt_short = -tgt_qty
                cur_short = self.short_holdings.get(coin, 0.0)
                delta = tgt_short - cur_short
                if delta <= 0:
                    if cur_short > 0 and delta < -cur_short * 0.05:
                        buyback_qty = -delta
                        cover_frac = buyback_qty / cur_short
                        cover_margin = self.short_margins[coin] * cover_frac
                        pnl = buyback_qty * (self.short_entry_prices[coin] - cur * (1 + slip))
                        tx = buyback_qty * cur * self.tx_cost
                        self.capital += cover_margin + pnl - tx
                        self.short_holdings[coin] -= buyback_qty
                        self.short_margins[coin] -= cover_margin
                else:
                    entry_p = cur * (1 - slip)
                    if coin not in self.short_holdings:
                        margin = target_margin[coin]
                        coin_lev = target_lev.get(coin, self.leverage)
                        notional = margin * coin_lev
                        qty = notional / entry_p
                        tx = notional * self.tx_cost
                        if self.capital >= margin + tx:
                            self.capital -= margin + tx
                            self.short_holdings[coin] = qty
                            self.short_entry_prices[coin] = entry_p
                            self.short_entry_bar_index[coin] = self._get_bar_index(coin, date)
                            self.short_margins[coin] = margin
                    elif delta > self.short_holdings[coin] * 0.05:
                        add_notional = delta * entry_p
                        coin_lev = target_lev.get(coin, self.leverage)
                        add_margin = add_notional / coin_lev
                        tx = add_notional * self.tx_cost
                        if self.capital >= add_margin + tx:
                            self.capital -= add_margin + tx
                            total = self.short_holdings[coin] + delta
                            self.short_entry_prices[coin] = (
                                self.short_entry_prices[coin] * self.short_holdings[coin] + entry_p * delta
                            ) / total
                            self.short_holdings[coin] = total
                            self.short_margins[coin] += add_margin


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(
        bars, funding,
        interval=interval,
        leverage=1.0,
        start_date=START,
        end_date=END,
        _trace=trace,
        **run_cfg,
    )
    return trace


def build_long_combined(data):
    traces = {name: generate_trace(data, STRATEGIES[name]) for name in LONG_COMBO}
    bars_1h, _ = data['1h']
    all_dates_1h = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]
    combined = combine_targets(traces, LONG_COMBO, all_dates_1h)
    return combined, all_dates_1h


def build_btc_filter_series(bars_1h):
    btc = bars_1h['BTC']
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc['Close'].values
    out = {}
    for date in dates:
        ci = btc.index.get_loc(date)
        sma = None
        if ci >= 168:
            sma = float(np.mean(close[ci - 168 + 1:ci + 1]))
        mom36 = None
        if ci >= 36:
            mom36 = float(close[ci] / close[ci - 36] - 1.0)
        mom720 = None
        if ci >= 720:
            mom720 = float(close[ci] / close[ci - 720] - 1.0)
        out[date] = dict(
            price=float(close[ci]),
            sma=sma,
            below_sma=(sma is not None and close[ci] < sma),
            mom36=mom36,
            mom36_neg=(mom36 is not None and mom36 < 0),
            mom720=mom720,
            mom720_neg=(mom720 is not None and mom720 < 0),
        )
    return out


def build_case_target_series(long_combined, btc_filter, case):
    series = []
    for date, target in long_combined:
        # 기본은 기존 롱 목표 그대로
        new_target = dict(target)
        cash_only = set(new_target.keys()) <= {'CASH'} and abs(new_target.get('CASH', 0.0) - 1.0) < 1e-9
        if cash_only and case['short_weight'] > 0:
            info = btc_filter.get(date, {})
            cond = True
            if case['require_below_sma']:
                cond = cond and info.get('below_sma', False)
            if case['require_mom_short_neg']:
                cond = cond and info.get('mom36_neg', False)
            if case['require_mom_long_neg']:
                cond = cond and info.get('mom720_neg', False)
            if cond:
                sw = case['short_weight']
                new_target = {'BTC': -sw, 'CASH': 1.0 - sw}
        series.append((date, new_target))
    return series


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']

    print('Building long combined baseline...')
    long_combined, _ = build_long_combined(data)
    btc_filter = build_btc_filter_series(bars_1h)

    print('\nResults')
    selected_cases = [
        c for c in SHORT_CASES
        if c['name'] in {
            'baseline_cash_only',
            'cash100_btc_short50_sma',
            'cash100_btc_short100_sma',
            'cash100_btc_short50_sma_m36',
            'cash100_btc_short100_sma_m36',
            'cash100_btc_short50_sma_m36_m720',
            'cash100_btc_short100_sma_m36_m720',
        }
    ]

    rows = []
    for case in selected_cases:
        target_series = build_case_target_series(long_combined, btc_filter, case)
        engine = LongShortSingleAccountEngine(
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
        m = engine.run(target_series)
        row = {
            'name': case['name'],
            'desc': case['desc'],
            'Cal': m.get('Cal', 0),
            'CAGR': m.get('CAGR', 0),
            'MDD': m.get('MDD', 0),
            'Sharpe': m.get('Sharpe', 0),
            'Liq': m.get('Liq', 0),
            'Stops': m.get('Stops', 0),
            'Rebal': m.get('Rebal', 0),
        }
        rows.append(row)
        print(
            f"{row['name']:<30} "
            f"Cal={row['Cal']:.2f} "
            f"CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} "
            f"Liq={row['Liq']} Stops={row['Stops']} Rebal={row['Rebal']}"
        )

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['name']))
    print('\nTop candidates')
    for row in rows[:5]:
        print(
            f"- {row['name']}: Cal={row['Cal']:.2f}, "
            f"CAGR={row['CAGR']:+.1%}, MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
