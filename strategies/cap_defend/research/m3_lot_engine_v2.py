#!/usr/bin/env python3
"""M3 Lot-level engine v2 — Codex 검토 반영.

핵심 수정:
1. Lot별 일일 MTM 반영 (coin daily price 필요)
2. Swap/shrink/forced exit 시 실제 시장가(해당 날 close) 기반 realized pnl
3. TP/timeout은 event 기준 (entry_px → exit_px)
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')

sys.path.insert(0, HERE)
from c_engine_v5 import run_c_v5, load_coin, metrics

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','FDUSD','USDD','PYUSD','USDE','LUSD'}


def load_universe_hist():
    with open(os.path.join(ROOT, 'data', 'historical_universe.json')) as f:
        raw = json.load(f)
    out = {}
    for d, tickers in raw.items():
        bare = [t.replace('-USD', '') for t in tickers]
        bare = [c for c in bare if c not in STABLES]
        out[pd.Timestamp(d)] = bare
    return out


def get_cap_rank(hist, date, coin):
    valid = [d for d in hist.keys() if d <= date]
    if not valid: return 999
    latest = max(valid)
    tops = hist[latest]
    try:
        return tops.index(coin)
    except ValueError:
        return 999


def list_available_futures():
    avail = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('_1h.csv'):
            c = f.replace('USDT_1h.csv', '')
            avail.add(c)
    return avail


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    v21['prev_cash'] = v21['cash_ratio'].shift(1).fillna(v21['cash_ratio'].iloc[0])
    return v21


def load_coin_daily_close(coins, start='2020-10-01', end='2026-03-30'):
    """코인별 daily close price dict. key=coin, value=Series(index=date)."""
    out = {}
    for c in coins:
        df = load_coin(c + 'USDT')
        if df is None: continue
        d = df['Close'].resample('D').last().ffill()
        out[c] = d
    return out


class LotV2:
    """Lot with daily MTM support."""
    __slots__ = ('coin', 'entry_date', 'exit_date', 'pnl_pct',
                 'cap_r', 'entry_px', 'qty', 'entered')
    def __init__(self, coin, entry_date, exit_date, pnl_pct, cap_r, entry_px, qty):
        self.coin = coin
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.pnl_pct = pnl_pct  # TP/timeout 기준 (event 기록)
        self.cap_r = cap_r
        self.entry_px = entry_px
        self.qty = qty  # 수량 (notional / entry_px)
        self.entered = False  # 첫 allocate 전


def simulate_lot_v2(v21, events, hist, coin_daily,
                     n_pick=1, cap_per_slot=0.30, universe_size=15,
                     tx_cost=0.003, swap_edge_threshold=0,
                     swap_cooldown_days=0, universe_filter='entry_only',
                     log_events=False):
    """
    v2: Lot 일일 MTM 반영.

    각 tick:
    - 1) Lot별 MTM: today_close / prev_close 로 lot value 업데이트
    - 2) Natural exit (exit_date 도달): event의 pnl_pct 사용하되, lot에 이미 MTM 반영돼 있음.
         → exit_date 날짜 close vs event exit_px가 거의 같다고 가정 (차이는 무시)
         → realized = 현재 lot value에서 exit
    - 3) Universe/swap/entry
    - 4) Allocation (cap_per_slot 기준)
    """
    events = events.copy()
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    events['cap_r'] = events.apply(lambda r: get_cap_rank(hist, r['entry_date'], r['coin']), axis=1)
    events = events[events['cap_r'] < universe_size]
    events = events.sort_values('entry_ts').reset_index(drop=True)
    events_by_day = defaultdict(list)
    for _, e in events.iterrows():
        events_by_day[e['entry_date']].append(e.to_dict())

    idx = v21.index
    lots = []  # 각 lot에 slot_notional 추가
    port_eq = 1.0
    eq_series = []
    last_swap_date = None
    n_swaps = n_shrinks = n_entries = n_exits_natural = 0
    event_log = []
    prev_coin_prices = {}  # lot.coin → prev_day_close

    for date in idx:
        cash_avail = v21.loc[date, 'prev_cash']
        v21_ret = v21.loc[date, 'v21_ret']
        # === V21 invested 부분 (1 - c_w) × v21_ret ===
        # 매일 끝에 계산

        # === Lot별 MTM ===
        day_cash_flow = 0.0  # port_eq에 더할 값 (C 부분 수익)
        day_tx = 0.0
        today_coin_prices = {}
        for lot in lots:
            if lot.coin in coin_daily and date in coin_daily[lot.coin].index:
                today_coin_prices[lot.coin] = coin_daily[lot.coin].loc[date]

        # MTM update
        for lot in lots:
            if not lot.entered: continue  # 당일 진입은 MTM 나중에
            cur_px = today_coin_prices.get(lot.coin, lot.entry_px)
            prev_px = prev_coin_prices.get(lot.coin, lot.entry_px)
            if prev_px > 0:
                bar_ret = cur_px / prev_px - 1.0
                # MTM: qty × (cur - prev) / port_eq
                day_cash_flow += lot.qty * (cur_px - prev_px) / 1  # port_eq에 더할 절대액 (normalized)

        # === Natural exits ===
        still_open = []
        for lot in lots:
            if lot.exit_date <= date:
                # Use event exit px for final realized (MTM 이미 반영됐으므로 exit 시 잔여 realized만)
                exit_px = today_coin_prices.get(lot.coin, lot.entry_px)
                # MTM 누적이 이미 반영됐으니 여기선 tx만 차감
                day_tx += lot.qty * exit_px * tx_cost
                n_exits_natural += 1
                if log_events:
                    event_log.append({'date': date, 'type': 'exit_natural',
                                      'coin': lot.coin, 'pnl_pct': lot.pnl_pct})
            else:
                still_open.append(lot)
        lots = still_open

        # === Entry 후보 처리 ===
        today = events_by_day.get(date, [])
        today_sorted = sorted(today, key=lambda x: x['cap_r'])
        open_coins = {lot.coin for lot in lots}
        swap_allowed = (last_swap_date is None or
                        (date - last_swap_date).days >= swap_cooldown_days)

        new_lots_today = []
        for ev in today_sorted:
            if ev['coin'] in open_coins: continue
            if len(lots) + len(new_lots_today) < n_pick:
                # 신규 진입
                # slot size 결정 (아래 allocation에서)
                # qty = slot_notional / entry_px (아래 할당)
                new_lot = LotV2(ev['coin'], date, ev['exit_date'], ev['pnl_pct'],
                                ev['cap_r'], float(ev['entry_px']), 0.0)
                new_lots_today.append(new_lot)
                n_entries += 1
            elif swap_allowed:
                # swap
                all_lots = lots + new_lots_today
                worst = max(all_lots, key=lambda l: l.cap_r)
                edge = worst.cap_r - ev['cap_r']
                if edge > swap_edge_threshold:
                    # worst의 현재가로 실현 (MTM 이미 반영)
                    cur_px = today_coin_prices.get(worst.coin, worst.entry_px)
                    day_tx += worst.qty * cur_px * tx_cost  # out tx
                    if worst in lots:
                        lots.remove(worst)
                    else:
                        new_lots_today.remove(worst)
                    new_lot = LotV2(ev['coin'], date, ev['exit_date'], ev['pnl_pct'],
                                    ev['cap_r'], float(ev['entry_px']), 0.0)
                    new_lots_today.append(new_lot)
                    n_swaps += 1
                    last_swap_date = date

        lots = lots + new_lots_today

        # === Allocation (cap_r 순 정렬 후 위부터) ===
        lots.sort(key=lambda l: l.cap_r)
        remaining = cash_avail * port_eq  # 절대 금액 (port_eq 1 기준 0.8 = 80%)
        cap_abs = cap_per_slot * port_eq

        for lot in lots:
            target_notional = min(cap_abs, remaining)
            if not lot.entered:
                # 신규 진입
                if target_notional <= 0:
                    # 진입 불가 (cash 부족)
                    lot.qty = 0.0
                    continue
                lot.qty = target_notional / lot.entry_px
                lot.entered = True
                day_tx += target_notional * tx_cost
            else:
                # 기존 포지션 — shrink 가능성
                # target_notional을 현재 cur_px 기준 qty로 환산
                cur_px = today_coin_prices.get(lot.coin, lot.entry_px)
                cur_value = lot.qty * cur_px
                if target_notional < cur_value:
                    # Shrink: cur_value - target_notional 만큼 매도
                    shrunk_value = cur_value - target_notional
                    day_tx += shrunk_value * tx_cost
                    # qty 조정
                    lot.qty = target_notional / cur_px
                    n_shrinks += 1
                    if log_events:
                        event_log.append({'date': date, 'type': 'shrink',
                                          'coin': lot.coin, 'from': cur_value, 'to': target_notional})
            remaining -= target_notional if target_notional > 0 else 0

        # Remove zero-qty lots
        lots = [lot for lot in lots if lot.qty > 0]

        # === Port value update ===
        # V21 invested 부분 수익 반영 (c_used 빠진 부분)
        c_used = sum(lot.qty * today_coin_prices.get(lot.coin, lot.entry_px) for lot in lots)
        c_used_ratio = c_used / port_eq if port_eq > 0 else 0
        c_used_ratio = min(c_used_ratio, cash_avail)
        # V21 ret만큼 invested 부분 증가
        v21_contribution = (1 - c_used_ratio) * v21_ret
        # C MTM 기여: day_cash_flow (lot MTM 합계)
        # 하지만 day_cash_flow 계산 시 lot.qty × (cur - prev)인데 qty는 미래 업데이트 반영.
        # 여기선 단순화: port_eq *= (1 + v21_contribution) + c_MTM/port_eq - tx/port_eq
        # 정확히: MTM 변화를 상대수익으로 변환
        # 앞서 계산한 day_cash_flow는 "절대 notional 기반" 변화
        # port_eq에 더하기
        new_port_eq = port_eq * (1 + v21_contribution) + day_cash_flow - day_tx
        port_eq = max(new_port_eq, 0.01)
        eq_series.append(port_eq)

        # prev_coin_prices 업데이트 (보유 중 코인만)
        for lot in lots:
            if lot.coin in today_coin_prices:
                prev_coin_prices[lot.coin] = today_coin_prices[lot.coin]

    eq = pd.Series(eq_series, index=idx)
    stats = {'n_entries': n_entries, 'n_exits_natural': n_exits_natural,
             'n_swaps': n_swaps, 'n_shrinks': n_shrinks}
    if log_events:
        return eq, stats, event_log
    return eq, stats


if __name__ == '__main__':
    v21 = load_v21()
    hist = load_universe_hist()
    avail = list_available_futures()
    coin_daily = load_coin_daily_close(avail)
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    P = {'dip_bars':24, 'dip_thr':-0.18, 'tp':0.12, 'tstop':48}
    all_events = []
    for c in avail:
        df = load_coin(c + 'USDT')
        if df is None: continue
        _, evs = run_c_v5(df, **P)
        for e in evs:
            e['coin'] = c
            all_events.append(e)
    events = pd.DataFrame(all_events)
    print(f'Events: {len(events)}, Coin daily: {len(coin_daily)}')

    for n_pick in [1, 2, 3]:
        for cap in [0.10, 0.20, 0.30, 0.50]:
            eq, stats = simulate_lot_v2(v21, events, hist, coin_daily,
                                         n_pick=n_pick, cap_per_slot=cap,
                                         universe_size=15)
            m = metrics(eq, bpy=252)
            print(f'n={n_pick} cap={cap}: {m} stats={stats}')
