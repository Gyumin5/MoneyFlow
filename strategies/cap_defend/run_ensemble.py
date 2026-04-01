#!/usr/bin/env python3
"""단일 계정 앙상블 백테스트.

각 전략이 독립적으로 목표 비중을 생성 → 가중 합산 → 하나의 포트폴리오로 실행.
단일 계정: 증거금 공유, 겹치는 코인 비중 합산, 거래비용 정확 반영.

방법:
1. 각 전략을 _trace 모드로 실행 → 매 봉 목표 비중 시계열 생성
2. 1h (가장 세밀한 간격) 기준으로 시간 축 통일
3. 각 봉에서: 각 전략의 최신 목표 비중을 가중 평균으로 합산
4. 합산 비중으로 단일 실행 엔진 구동 (delta 리밸런싱, 비용, 펀딩, 청산)
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from backtest_futures_full import load_data, run, SLIPPAGE_MAP, get_mcap

START = '2020-10-01'
END = '2026-03-28'


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
                 leverage_vol_lookback_bars=2160):
        self.bars = bars_1h
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
        → equity curve + metrics."""
        self.capital = self.initial_capital  # 인스턴스 변수로 승격 (mutable scope)
        self.holdings = {}
        self.entry_prices = {}
        self.entry_bar_index = {}
        self.margins = {}
        self.reentry_cooldown = {}
        liq_count = 0
        stop_count = 0
        rebal_count = 0
        pv_list = []

        # 펀딩 인덱스 정규화 (밀리초 오차 제거)
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

            # ── 청산/스탑 체크 (롱 전용) ──
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

            # ── 펀딩비 (정규화된 인덱스) ──
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

            # ── 리밸런싱 (target 변경 시) ──
            need_rebal = (target != prev_target)
            if need_rebal and target:
                self._execute_rebalance(target, date)
                rebal_count += 1

            # ── PV 기록 ──
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
        # 일봉으로 리샘플 (메트릭 계산용)
        eq_daily = eq.resample('D').last().dropna()
        yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
        if eq_daily.iloc[-1] <= 0 or yrs <= 0:
            return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0,
                    'Liq': liq_count, 'Stops': stop_count, 'Rebal': rebal_count, '_equity': eq}
        cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
        dr = eq_daily.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        # MDD는 원본 해상도(1h)로 계산
        mdd = (eq / eq.cummax() - 1).min()
        cal = cagr / abs(mdd) if mdd != 0 else 0
        return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal,
                'Liq': liq_count, 'Stops': stop_count, 'Rebal': rebal_count, '_equity': eq}

    def _get_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None: return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        return float(df['Close'].iloc[ci]) if ci >= 0 else 0

    def _get_bar_index(self, coin, date):
        df = self.bars.get(coin)
        if df is None:
            return -1
        return df.index.get_indexer([date], method='ffill')[0]

    def _stop_enabled(self, coin, target):
        if self.stop_gate == 'always':
            return True
        target_w = target.get(coin, 0.0) if target else 0.0
        cash_w = target.get('CASH', 0.0) if target else 0.0
        if self.stop_gate == 'target_exit_only':
            return target_w <= 0
        if self.stop_gate == 'cash_guard':
            return cash_w >= self.stop_gate_cash_threshold
        if self.stop_gate == 'target_exit_or_cash_guard':
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
            return max(ref - self.stop_atr_mult * atr, 0.0)
        elif self.stop_kind == 'atr_rolling_high_high':
            atr = self._calc_atr(coin, ci, self.stop_atr_lookback_bars)
            if self.stop_lookback_bars <= 0 or ci < self.stop_lookback_bars or atr is None or self.stop_atr_mult <= 0:
                return None
            ref = float(np.max(df['High'].iloc[ci - self.stop_lookback_bars:ci]))
            return max(ref - self.stop_atr_mult * atr, 0.0)
        else:
            return None

        if ref <= 0:
            return None
        return ref * (1.0 - self.stop_pct)

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
        """delta 리밸런싱. self.capital/holdings/margins/entry_prices를 직접 변경."""
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
            cur = self._get_price(coin, date)
            if cur <= 0: continue
            coin_lev = target_lev.get(coin, self.leverage)
            tmgn = pv * w * 0.95
            tnotional = tmgn * coin_lev
            target_qty[coin] = tnotional / cur
            target_margin[coin] = tmgn

        # 매도
        for coin in list(self.holdings.keys()):
            cur = self._get_price(coin, date)
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

        # 매수
        for coin, tqty in target_qty.items():
            cur = self._get_price(coin, date)
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


def generate_traces(strategies, data):
    """각 전략의 목표 비중 시계열을 생성."""
    traces = {}
    for key, spec in strategies.items():
        iv = spec['interval']
        bars, funding = data[iv]
        trace = []
        run(bars, funding, interval=iv, leverage=1.0,  # leverage는 실행 엔진에서 적용
            start_date=START, end_date=END, _trace=trace, **spec['config'])
        traces[key] = trace
        print(f"  {key}: {len(trace)} bars traced")
    return traces


def combine_targets(traces, weights, all_dates_1h):
    """여러 전략의 trace를 1h 기준으로 합산.
    각 봉에서 각 전략의 최신 target을 가중 평균.
    D/4h의 타임스탬프는 1h와 정확히 안 맞으므로 <= date인 최신 target 사용."""
    # 각 전략의 (date, target) 리스트를 정렬된 형태로 저장
    sorted_traces = {}
    for key, trace in traces.items():
        entries = [(entry['date'], entry['target']) for entry in trace]
        entries.sort(key=lambda x: x[0])
        sorted_traces[key] = entries

    # 각 전략별 현재 인덱스 (이분탐색 대신 순차 전진)
    trace_idx = {key: 0 for key in traces}

    combined_series = []
    latest_targets = {key: {'CASH': 1.0} for key in traces}

    for date in all_dates_1h:
        # 각 전략: date 이하의 최신 target으로 업데이트
        for key in traces:
            entries = sorted_traces[key]
            idx = trace_idx[key]
            while idx < len(entries) and entries[idx][0] <= date:
                latest_targets[key] = entries[idx][1]
                idx += 1
            trace_idx[key] = idx

        # 가중 합산
        merged = {}
        for key, w in weights.items():
            for coin, cw in latest_targets[key].items():
                merged[coin] = merged.get(coin, 0) + cw * w
        combined_series.append((date, merged))

    return combined_series


# ─── 전략 정의 ───
STRATEGIES = {
    'D1': {'interval': 'D', 'config': dict(
        sma_bars=50, mom_short_bars=100, mom_long_bars=300,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=20)},
    'D2': {'interval': 'D', 'config': dict(
        sma_bars=40, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10)},
    'D3': {'interval': 'D', 'config': dict(
        sma_bars=40, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10)},
    '4h1': {'interval': '4h', 'config': dict(
        sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom1vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120)},
    '4h2': {'interval': '4h', 'config': dict(
        sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=18)},
    '4h3': {'interval': '4h', 'config': dict(
        sma_bars=240, mom_short_bars=200, mom_long_bars=1200,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=18)},
    '1h1': {'interval': '1h', 'config': dict(
        sma_bars=200, mom_short_bars=200, mom_long_bars=1200,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=336)},
    '1h2': {'interval': '1h', 'config': dict(
        sma_bars=200, mom_short_bars=100, mom_long_bars=600,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=168)},
    '1h3': {'interval': '1h', 'config': dict(
        sma_bars=200, mom_short_bars=50, mom_long_bars=300,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=72)},
}

ENSEMBLES = [
    # 단일
    ('D1', {'D1': 1.0}),
    ('4h2', {'4h2': 1.0}),
    ('1h1', {'1h1': 1.0}),
    # 2개 조합
    ('D1+4h2', {'D1': 0.5, '4h2': 0.5}),
    ('D1+1h1', {'D1': 0.5, '1h1': 0.5}),
    ('4h2+1h1', {'4h2': 0.5, '1h1': 0.5}),
    ('4h1+1h1', {'4h1': 0.5, '1h1': 0.5}),
    ('4h2+1h1 7:3', {'4h2': 0.7, '1h1': 0.3}),
    # 3개 조합
    ('D1+4h2+1h1', {'D1': 1/3, '4h2': 1/3, '1h1': 1/3}),
    ('D2+4h1+1h2', {'D2': 1/3, '4h1': 1/3, '1h2': 1/3}),
    ('D1+4h2+1h3', {'D1': 1/3, '4h2': 1/3, '1h3': 1/3}),
]


if __name__ == '__main__':
    t0 = time.time()

    # 데이터 로드
    data = {}
    for iv in ['D', '4h', '1h']:
        print(f"Loading {iv}...")
        data[iv] = load_data(iv)

    # 1h 기준 시간 축
    btc_1h = data['1h'][0]['BTC']
    all_dates_1h = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]
    print(f"1h dates: {len(all_dates_1h)}")

    # 각 전략의 trace 생성
    print("\nGenerating traces...")
    needed = set()
    for name, weights in ENSEMBLES:
        needed.update(weights.keys())
    strats = {k: v for k, v in STRATEGIES.items() if k in needed}
    traces = generate_traces(strats, data)

    # 앙상블 실행
    bars_1h, funding_1h = data['1h']

    for lev in [1.5, 2.0, 3.0]:
        print(f"\n{'='*75}")
        print(f"  Leverage {lev}x | Single Account Ensemble")
        print(f"{'='*75}")
        print(f"  {'Name':<18s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Rb':>5s}")
        print(f"  {'-'*55}")

        for name, weights in ENSEMBLES:
            # 관련 traces만 선택
            sel_traces = {k: traces[k] for k in weights if k in traces}
            if not sel_traces:
                continue

            # 합산 target 시계열 생성
            combined = combine_targets(sel_traces, weights, all_dates_1h)

            # 단일 계정 엔진 실행
            engine = SingleAccountEngine(bars_1h, funding_1h, leverage=lev)
            m = engine.run(combined)
            if not m:
                print(f"  {name:<18s} FAILED")
                continue

            liq = f"💀{m['Liq']}" if m.get('Liq', 0) > 0 else ""
            print(f"  {name:<18s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%}"
                  f" {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {liq:>4s} {m['Rebal']:>5d}")

        sys.stdout.flush()

    print(f"\n총 소요: {time.time()-t0:.0f}s")
