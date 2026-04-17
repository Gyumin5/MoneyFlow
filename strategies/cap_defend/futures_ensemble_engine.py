#!/usr/bin/env python3
"""선물 앙상블 엔진 — SingleAccountEngine + combine_targets.

공식 백테스트(run_current_futures_backtest.py)에서 사용하는 핵심 클래스/함수.
원본: research/run_ensemble.py에서 추출.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest_futures_full import SLIPPAGE_MAP, get_mcap


class SingleAccountEngine:
    """합산 비중 시계열을 받아 단일 계정 선물 실행."""

    def __init__(self, bars_1h, funding, leverage=2.0, tx_cost=0.0004,
                 maint_rate=0.004, initial_capital=10000.0,
                 stop_kind='none', stop_pct=0.0, stop_lookback_bars=0,
                 reentry_cooldown_bars=0,
                 stop_gate='always',
                 stop_gate_cash_threshold=0.0,
                 stop_atr_lookback_bars=0,
                 stop_atr_mult=0.0,
                 stop_liq_buffer_pct=0.05,
                 leverage_mode='fixed',
                 per_coin_leverage_mode='none',
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
                 leverage_mom_lookback_bars=720,
                 leverage_vol_lookback_bars=2160,
                 fill_mode='open'):
        self.bars = bars_1h
        self.fill_mode = fill_mode
        self.funding = funding
        self.leverage = leverage
        self.tx_cost = tx_cost
        self.maint_rate = maint_rate
        self.initial_capital = initial_capital
        self.stop_kind = stop_kind
        self.stop_pct = stop_pct
        self.stop_lookback_bars = stop_lookback_bars
        self.reentry_cooldown_bars = reentry_cooldown_bars
        self.stop_gate = stop_gate
        self.stop_gate_cash_threshold = stop_gate_cash_threshold
        self.stop_atr_lookback_bars = stop_atr_lookback_bars
        self.stop_atr_mult = stop_atr_mult
        self.stop_liq_buffer_pct = stop_liq_buffer_pct
        self.leverage_mode = leverage_mode
        self.per_coin_leverage_mode = per_coin_leverage_mode
        self.leverage_floor = leverage_floor
        self.leverage_mid = leverage_mid
        self.leverage_ceiling = leverage_ceiling
        self.leverage_cash_threshold = leverage_cash_threshold
        self.leverage_partial_cash_threshold = leverage_partial_cash_threshold
        self.leverage_count_floor_max = leverage_count_floor_max
        self.leverage_count_mid_max = leverage_count_mid_max
        self.leverage_canary_floor_gap = leverage_canary_floor_gap
        self.leverage_canary_mid_gap = leverage_canary_mid_gap
        self.leverage_canary_high_gap = leverage_canary_high_gap
        self.leverage_canary_sma_bars = leverage_canary_sma_bars
        self.leverage_mom_lookback_bars = leverage_mom_lookback_bars
        self.leverage_vol_lookback_bars = leverage_vol_lookback_bars
        self._last_target_weights = {}

    def run(self, target_series):
        """target_series: list of (date, {coin: weight, 'CASH': weight})
        -> equity curve + metrics."""
        self.capital = self.initial_capital
        self.holdings = {}
        self.entry_prices = {}
        self.entry_bar_index = {}
        self.margins = {}
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
                    del self.holdings[coin]; del self.entry_prices[coin]; del self.margins[coin]
                    self.entry_bar_index.pop(coin, None)
                    liq_count += 1

            for coin in list(self.holdings.keys()):
                fr_series = norm_funding.get(coin)
                if fr_series is None: continue
                if date in fr_series.index:
                    fr = float(fr_series.loc[date])
                    if fr != 0 and not np.isnan(fr):
                        cur = self._get_price(coin, date)
                        if cur > 0:
                            self.capital -= self.holdings[coin] * cur * fr
            self.capital = max(self.capital, 0)

            need_rebal = (target != prev_target)
            if need_rebal and target:
                self._execute_rebalance(target, date)
                rebal_count += 1

            pv = self.capital
            for coin in self.holdings:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.margins[coin] + self.holdings[coin] * (cur - self.entry_prices[coin])
            pv_list.append({'Date': date, 'Value': max(pv, 0)})
            prev_target = target

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
        mdd_m_list = []
        for offset in range(0, 30, 3):
            sampled = eq_daily.iloc[offset::30]
            if len(sampled) >= 2:
                mdd_m_list.append(float((sampled / sampled.cummax() - 1).min()))
        if mdd_m_list:
            mdd_m_avg = float(np.mean(mdd_m_list))
            mdd_m_min = float(np.min(mdd_m_list))
            mdd_m_max = float(np.max(mdd_m_list))
        else:
            mdd_m_avg = mdd_m_min = mdd_m_max = mdd
        cal_m = cagr / abs(mdd_m_avg) if mdd_m_avg != 0 else 0
        return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal,
                'MDD_m_avg': mdd_m_avg, 'MDD_m_min': mdd_m_min, 'MDD_m_max': mdd_m_max,
                'Cal_m': cal_m,
                'Liq': liq_count, 'Stops': stop_count, 'Rebal': rebal_count, '_equity': eq}

    def _get_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None: return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        return float(df['Close'].iloc[ci]) if ci >= 0 else 0

    def _get_fill_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None:
            return 0
        if self.fill_mode == 'close':
            return self._get_price(coin, date)
        try:
            ci = df.index.get_loc(date)
        except KeyError:
            return 0
        return float(df['Open'].iloc[ci])

    def _get_bar_index(self, coin, date):
        df = self.bars.get(coin)
        if df is None:
            return -1
        return df.index.get_indexer([date], method='ffill')[0]

    def _stop_enabled(self, coin, target):
        if self.stop_gate == 'always':
            return True
        cash_w = target.get('CASH', 0.0) if target else 0.0
        if self.stop_gate == 'target_exit_only':
            target_w = target.get(coin, 0.0) if target else 0.0
            return target_w <= 0
        if self.stop_gate == 'cash_guard':
            return cash_w >= self.stop_gate_cash_threshold
        if self.stop_gate == 'target_exit_or_cash_guard':
            target_w = target.get(coin, 0.0) if target else 0.0
            return target_w <= 0 or cash_w >= self.stop_gate_cash_threshold
        return True

    def _get_canary_gap(self, date):
        btc = self.bars.get('BTC')
        if btc is None:
            return None
        ci = self._get_bar_index('BTC', date)
        if ci < self.leverage_canary_sma_bars:
            return None
        close_arr = btc['Close'].values[:ci + 1]
        sma = float(np.mean(close_arr[-self.leverage_canary_sma_bars:]))
        if sma <= 0:
            return None
        return close_arr[-1] / sma - 1.0

    def _get_target_coins(self, target):
        if not target:
            return []
        return [coin for coin, w in target.items() if coin != 'CASH' and w > 0]

    def _get_account_leverage(self, date, target):
        if self.leverage_mode == 'fixed':
            return self.leverage

        cash_w = target.get('CASH', 0.0) if target else 0.0
        coin_count = len(self._get_target_coins(target))
        canary_gap = self._get_canary_gap(date)

        if self.leverage_mode == 'cash_based_543':
            if cash_w >= self.leverage_cash_threshold:
                return self.leverage_floor
            if cash_w > self.leverage_partial_cash_threshold:
                return self.leverage_mid
            return self.leverage_ceiling

        if self.leverage_mode == 'count_based_543':
            if coin_count <= self.leverage_count_floor_max:
                return self.leverage_floor
            if coin_count <= self.leverage_count_mid_max:
                return self.leverage_mid
            return self.leverage_ceiling

        if self.leverage_mode == 'canary_based_543':
            if canary_gap is None or canary_gap < self.leverage_canary_mid_gap:
                return self.leverage_floor
            if canary_gap < self.leverage_canary_high_gap:
                return self.leverage_mid
            return self.leverage_ceiling

        if self.leverage_mode == 'mixed_score_543':
            score = 0
            if cash_w >= self.leverage_cash_threshold:
                score -= 2
            elif cash_w > self.leverage_partial_cash_threshold:
                score -= 1
            if coin_count >= 5:
                score += 1
            elif coin_count <= 2:
                score -= 1
            if canary_gap is not None:
                if canary_gap >= self.leverage_canary_high_gap:
                    score += 2
                elif canary_gap >= self.leverage_canary_mid_gap:
                    score += 1
                elif canary_gap < self.leverage_canary_floor_gap:
                    score -= 1
            if score >= 2:
                return self.leverage_ceiling
            if score >= 0:
                return self.leverage_mid
            return self.leverage_floor

        return self.leverage

    def _apply_cash_degrade(self, lev, cash_w):
        if cash_w < self.leverage_cash_threshold:
            return lev
        if lev >= self.leverage_ceiling:
            return self.leverage_mid
        if lev >= self.leverage_mid:
            return self.leverage_floor
        return self.leverage_floor

    def _rank_coins_by_signal(self, coins, date, mode):
        if mode == 'rank_543_cash':
            ranked = sorted(coins, key=lambda c: (-self._last_target_weights.get(c, 0.0), c))
            return ranked

        scored = []
        for coin in coins:
            ci = self._get_bar_index(coin, date)
            df = self.bars.get(coin)
            if df is None or ci <= 0:
                continue
            close = df['Close'].values
            if mode == 'momrank_543_cash':
                if ci < self.leverage_mom_lookback_bars:
                    score = -999.0
                else:
                    score = close[ci] / close[ci - self.leverage_mom_lookback_bars] - 1.0
                scored.append((coin, score))
            elif mode == 'lowvol_543_cash':
                if ci < self.leverage_vol_lookback_bars:
                    score = -999.0
                else:
                    rets = np.diff(np.log(close[ci - self.leverage_vol_lookback_bars:ci + 1]))
                    vol = float(np.std(rets)) if len(rets) > 0 else 999.0
                    score = -vol
                scored.append((coin, score))
            elif mode == 'cap_mom_blend_543_cash':
                if ci < self.leverage_mom_lookback_bars:
                    mom = -999.0
                else:
                    mom = close[ci] / close[ci - self.leverage_mom_lookback_bars] - 1.0
                mcap = get_mcap(date)
                cap_rank = mcap.index(coin) if coin in mcap else len(mcap)
                score = mom - cap_rank * 1e-4
                scored.append((coin, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [coin for coin, _ in scored] + [coin for coin in coins if coin not in {c for c, _ in scored}]

    def _get_coin_leverage_map(self, date, target):
        coins = self._get_target_coins(target)
        cash_w = target.get('CASH', 0.0) if target else 0.0
        if not coins:
            return {}

        if self.per_coin_leverage_mode == 'none':
            lev = self._get_account_leverage(date, target)
            return {coin: lev for coin in coins}

        ranked = self._rank_coins_by_signal(coins, date, self.per_coin_leverage_mode)
        lev_map = {}
        for idx, coin in enumerate(ranked):
            if idx == 0:
                lev = self.leverage_ceiling
            elif idx <= 2:
                lev = self.leverage_mid
            else:
                lev = self.leverage_floor
            lev_map[coin] = self._apply_cash_degrade(lev, cash_w)
        return lev_map

    def _calc_atr(self, coin, ci, lookback):
        df = self.bars.get(coin)
        if df is None or lookback <= 0 or ci < lookback + 1:
            return None
        high = df['High'].iloc[ci - lookback:ci].values
        low = df['Low'].iloc[ci - lookback:ci].values
        prev_close = df['Close'].iloc[ci - lookback - 1:ci - 1].values
        if len(high) != lookback or len(prev_close) != lookback:
            return None
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ])
        atr = float(np.mean(tr))
        return atr if atr > 0 else None

    def _get_stop_price(self, coin, date, target=None):
        df = self.bars.get(coin)
        if df is None or self.stop_kind == 'none':
            return None
        if not self._stop_enabled(coin, target):
            return None
        ci = self._get_bar_index(coin, date)
        if ci <= 0:
            return None

        if self.stop_kind == 'prev_close_pct':
            if self.stop_pct <= 0:
                return None
            ref = float(df['Close'].iloc[ci - 1])
        elif self.stop_kind == 'highest_close_since_entry_pct':
            if self.stop_pct <= 0:
                return None
            start_ci = self.entry_bar_index.get(coin, -1)
            if start_ci < 0:
                return None
            ref = float(np.max(df['Close'].iloc[start_ci:ci]))
        elif self.stop_kind == 'highest_high_since_entry_pct':
            if self.stop_pct <= 0:
                return None
            start_ci = self.entry_bar_index.get(coin, -1)
            if start_ci < 0:
                return None
            ref = float(np.max(df['High'].iloc[start_ci:ci]))
        elif self.stop_kind == 'rolling_high_close_pct':
            if self.stop_pct <= 0:
                return None
            if self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars:
                return None
            ref = float(np.max(df['Close'].iloc[ci - self.stop_lookback_bars:ci]))
        elif self.stop_kind == 'rolling_high_high_pct':
            if self.stop_pct <= 0:
                return None
            if self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars:
                return None
            ref = float(np.max(df['High'].iloc[ci - self.stop_lookback_bars:ci]))
        elif self.stop_kind == 'atr_highest_high_since_entry':
            start_ci = self.entry_bar_index.get(coin, -1)
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if start_ci < 0 or atr is None or self.stop_atr_mult <= 0:
                return None
            ref = float(np.max(df['High'].iloc[start_ci:ci]))
            return self._floor_above_liq(coin, max(ref - self.stop_atr_mult * atr, 0.0))
        elif self.stop_kind == 'atr_rolling_high_high':
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars or atr is None or self.stop_atr_mult <= 0:
                return None
            ref = float(np.max(df['High'].iloc[ci - self.stop_lookback_bars:ci]))
            return self._floor_above_liq(coin, max(ref - self.stop_atr_mult * atr, 0.0))
        else:
            return None

        if ref <= 0:
            return None
        stop_price = ref * (1.0 - self.stop_pct)
        return self._floor_above_liq(coin, stop_price)

    def _floor_above_liq(self, coin, stop_price):
        if stop_price is None or self.stop_liq_buffer_pct <= 0:
            return stop_price
        liq_price = self._get_liq_price(coin)
        if liq_price is None or liq_price <= 0:
            return stop_price
        floor = liq_price * (1.0 + self.stop_liq_buffer_pct)
        return max(stop_price, floor)

    def _execute_stop_exit(self, coin, date, stop_price):
        ci = self._get_bar_index(coin, date)
        if ci < 0:
            return False
        low = float(self.bars[coin]['Low'].iloc[ci])
        if low <= 0 or low > stop_price:
            return False
        slip = SLIPPAGE_MAP.get(coin, 0.0005)
        cur_open = float(self.bars[coin]['Open'].iloc[ci])
        exit_p = min(cur_open, stop_price) * (1 - slip)
        pnl = self.holdings[coin] * (exit_p - self.entry_prices[coin])
        tx = self.holdings[coin] * exit_p * self.tx_cost
        self.capital += self.margins[coin] + pnl - tx
        del self.holdings[coin]; del self.entry_prices[coin]; del self.margins[coin]
        self.entry_bar_index.pop(coin, None)
        if self.reentry_cooldown_bars > 0:
            self.reentry_cooldown[coin] = self.reentry_cooldown_bars
        return True

    def _get_liq_price(self, coin):
        qty = self.holdings.get(coin, 0)
        entry = self.entry_prices.get(coin, 0)
        margin = self.margins.get(coin, 0)
        if qty <= 0 or entry <= 0:
            return None
        denom = qty * (1.0 - self.maint_rate)
        if denom <= 0:
            return None
        liq_price = (qty * entry - margin) / denom
        if liq_price <= 0:
            return None
        return liq_price

    def _execute_rebalance(self, target, date):
        """delta rebalancing. Modifies self.capital/holdings/margins/entry_prices."""
        self._last_target_weights = dict(target)
        pv = self.capital
        for coin in self.holdings:
            cur = self._get_price(coin, date)
            if cur > 0:
                pv += self.margins[coin] + self.holdings[coin] * (cur - self.entry_prices[coin])
        if pv <= 0:
            return

        target_qty = {}
        target_margin = {}
        target_lev = self._get_coin_leverage_map(date, target)
        for coin, w in target.items():
            if coin == 'CASH' or w <= 0: continue
            cur = self._get_fill_price(coin, date)
            if cur <= 0: continue
            coin_lev = target_lev.get(coin, self.leverage)
            tmgn = pv * w * 0.95
            tnotional = tmgn * coin_lev
            target_qty[coin] = tnotional / cur
            target_margin[coin] = tmgn

        for coin in list(self.holdings.keys()):
            cur = self._get_fill_price(coin, date)
            if cur <= 0: continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in target_qty:
                pnl = self.holdings[coin] * (cur * (1 - slip) - self.entry_prices[coin])
                tx = self.holdings[coin] * cur * self.tx_cost
                self.capital += self.margins[coin] + pnl - tx
                del self.holdings[coin]; del self.entry_prices[coin]; del self.margins[coin]
                self.entry_bar_index.pop(coin, None)
            else:
                delta = target_qty[coin] - self.holdings[coin]
                if delta < -self.holdings[coin] * 0.05:
                    sell_qty = -delta
                    sell_frac = sell_qty / self.holdings[coin]
                    sell_margin = self.margins[coin] * sell_frac
                    pnl = sell_qty * (cur * (1 - slip) - self.entry_prices[coin])
                    tx = sell_qty * cur * self.tx_cost
                    self.capital += sell_margin + pnl - tx
                    self.holdings[coin] -= sell_qty
                    self.margins[coin] -= sell_margin

        for coin, tqty in target_qty.items():
            cur = self._get_fill_price(coin, date)
            if cur <= 0: continue
            if coin in self.reentry_cooldown and coin not in self.holdings:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in self.holdings:
                entry_p = cur * (1 + slip)
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
            else:
                delta = tqty - self.holdings[coin]
                if delta > self.holdings[coin] * 0.05:
                    entry_p = cur * (1 + slip)
                    add_notional = delta * entry_p
                    coin_lev = target_lev.get(coin, self.leverage)
                    add_margin = add_notional / coin_lev
                    tx = add_notional * self.tx_cost
                    if self.capital >= add_margin + tx:
                        self.capital -= add_margin + tx
                        total = self.holdings[coin] + delta
                        self.entry_prices[coin] = (self.entry_prices[coin] * self.holdings[coin] + entry_p * delta) / total
                        self.holdings[coin] = total
                        self.margins[coin] += add_margin


def combine_targets(traces, weights, all_dates_1h):
    """여러 전략의 trace를 1h 기준으로 합산."""
    sorted_traces = {}
    for key, trace in traces.items():
        entries = [(entry['date'], entry['target']) for entry in trace]
        entries.sort(key=lambda x: x[0])
        sorted_traces[key] = entries

    trace_idx = {key: 0 for key in traces}
    combined_series = []
    latest_targets = {key: {'CASH': 1.0} for key in traces}

    for date in all_dates_1h:
        for key in traces:
            entries = sorted_traces[key]
            idx = trace_idx[key]
            while idx < len(entries) and entries[idx][0] <= date:
                latest_targets[key] = entries[idx][1]
                idx += 1
            trace_idx[key] = idx

        merged = {}
        for key, w in weights.items():
            for coin, cw in latest_targets[key].items():
                merged[coin] = merged.get(coin, 0) + cw * w
        combined_series.append((date, merged))

    return combined_series
