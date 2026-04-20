#!/usr/bin/env python3
"""M3 Lot-level engine — 사용자 규칙 + Codex 검토 반영.

핵심 기능:
1. V21 100% 우선, C는 남은 cash에서 순서대로
2. Lot-level 회계 (slot 축소 시 realized, 잔여 lot TP/timeout 기준 유지)
3. 3시점 재검산 (C 신규 진입 직전 / V21 target 변경 / 일마감)
4. Swap: 시총 상위 dip 시 기존 교체, 최소우위 조건 + cooldown
5. Universe 이탈 시 유지 (entry 필터만)
6. 전일 확정 V21 cash 사용 (look-ahead 방지)
7. Forced shrink/swap 별도 로그
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')

sys.path.insert(0, HERE)
from c_engine_v5 import run_c_v5, load_coin, metrics

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','FDUSD','USDD','PYUSD','USDE','LUSD'}


# === Historical Universe ===
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
    """date 시점 시총 순위 (낮을수록 상위). 없으면 999."""
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
    for f in os.listdir(os.path.join(ROOT, 'data', 'futures')):
        if f.endswith('_1h.csv'):
            c = f.replace('USDT_1h.csv', '')
            avail.add(c)
    return avail


# === V21 daily state ===
def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    # 전일 cash (look-ahead 방지)
    v21['prev_cash'] = v21['cash_ratio'].shift(1).fillna(v21['cash_ratio'].iloc[0])
    return v21


def load_v21_no2021():
    v21 = load_v21()
    v21 = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    v21['prev_cash'] = v21['cash_ratio'].shift(1).fillna(v21['cash_ratio'].iloc[0])
    return v21


# === Lot 단위 M3 simulator ===
class Lot:
    """C 포지션 단위. slot 축소 시 일부 realized."""
    __slots__ = ('coin', 'entry_date', 'exit_date', 'pnl_pct', 'cap_r',
                 'original_alloc', 'current_alloc', 'entry_ts')
    def __init__(self, coin, entry_date, exit_date, pnl_pct, cap_r, alloc, entry_ts=None):
        self.coin = coin
        self.entry_date = entry_date
        self.exit_date = exit_date
        self.pnl_pct = pnl_pct
        self.cap_r = cap_r
        self.original_alloc = alloc
        self.current_alloc = alloc
        self.entry_ts = entry_ts or entry_date


def simulate_lot(v21, events, hist,
                 n_pick=3, cap_per_slot=0.10, universe_size=15,
                 tx_cost=0.003, swap_edge_threshold=0.0,
                 swap_cooldown_days=0, universe_filter='entry_only',
                 log_events=False):
    """
    Args:
        n_pick: 최대 동시 보유 lot 수
        cap_per_slot: 각 slot 최대 cap 비율 (총 eq 대비)
        universe_size: historical universe Top N
        tx_cost: 편도 TX
        swap_edge_threshold: swap 시 최소 cap_rank 차이 (예: 1 = 1계단 상위일 때만)
        swap_cooldown_days: 마지막 swap 후 N일 이후만 재swap
        universe_filter: 'entry_only' (진입 시에만 필터) | 'persistent' (보유 중 이탈 시 청산)
        log_events: True면 event log 반환

    Returns:
        equity Series, stats dict, event_log (옵션)
    """
    events = events.copy()
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    # entry date 기준 cap_rank
    events['cap_r'] = events.apply(lambda r: get_cap_rank(hist, r['entry_date'], r['coin']), axis=1)
    events = events[events['cap_r'] < universe_size]
    events = events.sort_values('entry_ts').reset_index(drop=True)

    events_by_day = defaultdict(list)
    for _, e in events.iterrows():
        events_by_day[e['entry_date']].append(e.to_dict())

    idx = v21.index
    lots: list[Lot] = []
    port_rets = []
    event_log = []
    last_swap_date = None
    n_swaps = 0
    n_shrinks = 0
    n_entries = 0
    n_exits = 0

    for date in idx:
        cash_avail = v21.loc[date, 'prev_cash']  # 전일 확정 cash (look-ahead 방지)
        v21_ret = v21.loc[date, 'v21_ret']
        day_realized_pnl = 0.0  # port_ret 에 더함 (이미 slot * realized 반영)
        day_tx = 0.0

        # 1) 자연 exit (TP/timeout)
        still_open = []
        for lot in lots:
            if lot.exit_date <= date:
                # realized pnl
                realized = lot.pnl_pct / 100.0
                day_realized_pnl += lot.current_alloc * realized
                day_tx += lot.current_alloc * tx_cost
                n_exits += 1
                if log_events:
                    event_log.append({'date': date, 'type': 'exit_natural',
                                      'coin': lot.coin, 'alloc': lot.current_alloc,
                                      'pnl_pct': lot.pnl_pct,
                                      'reason': 'TP_timeout'})
            else:
                still_open.append(lot)
        lots = still_open

        # 2) Universe 이탈 처리 (persistent 모드만)
        if universe_filter == 'persistent':
            new_lots = []
            for lot in lots:
                cur_rank = get_cap_rank(hist, date, lot.coin)
                if cur_rank >= universe_size:
                    # 강제 청산 (unrealized 0 가정)
                    day_tx += lot.current_alloc * tx_cost
                    if log_events:
                        event_log.append({'date': date, 'type': 'universe_exit',
                                          'coin': lot.coin, 'alloc': lot.current_alloc,
                                          'reason': 'universe_out'})
                else:
                    new_lots.append(lot)
            lots = new_lots

        # 3) Entry 후보 처리
        today = events_by_day.get(date, [])
        today_sorted = sorted(today, key=lambda x: x['cap_r'])
        open_coins = {lot.coin for lot in lots}
        swap_allowed = (last_swap_date is None or
                        (date - last_swap_date).days >= swap_cooldown_days)

        for ev in today_sorted:
            if ev['coin'] in open_coins: continue

            if len(lots) < n_pick:
                # 신규 진입
                lots.append(Lot(ev['coin'], date, ev['exit_date'], ev['pnl_pct'],
                                ev['cap_r'], 0.0, entry_ts=ev['entry_ts']))
                n_entries += 1
                # Entry tx는 아래 allocation 이후 처리
                if log_events:
                    event_log.append({'date': date, 'type': 'entry',
                                      'coin': ev['coin'], 'cap_r': ev['cap_r']})
                open_coins.add(ev['coin'])
            elif swap_allowed:
                # Swap 후보
                worst = max(lots, key=lambda l: l.cap_r)
                edge = worst.cap_r - ev['cap_r']
                if edge > swap_edge_threshold:
                    # Swap: worst 청산 + 신규 진입
                    # worst의 unrealized는 그대로 0 가정 (보수적)
                    day_tx += worst.current_alloc * tx_cost  # out
                    if log_events:
                        event_log.append({'date': date, 'type': 'swap_out',
                                          'coin': worst.coin, 'alloc': worst.current_alloc})
                    lots.remove(worst)
                    lots.append(Lot(ev['coin'], date, ev['exit_date'], ev['pnl_pct'],
                                    ev['cap_r'], 0.0, entry_ts=ev['entry_ts']))
                    n_swaps += 1
                    last_swap_date = date
                    if log_events:
                        event_log.append({'date': date, 'type': 'swap_in',
                                          'coin': ev['coin'], 'cap_r': ev['cap_r']})
                    open_coins.add(ev['coin'])

        # 4) Allocation: 3시점 재검산 중 "일마감"에 해당
        # Lots를 cap_r 순 정렬 후 위부터 cap_per_slot 할당
        lots.sort(key=lambda l: l.cap_r)
        remaining = cash_avail
        new_lots_after_shrink = []
        for lot in lots:
            target_alloc = min(cap_per_slot, remaining)
            prev_alloc = lot.current_alloc
            # Entry bar에 allocation 0 → target 으로 증가 시 tx
            if prev_alloc == 0.0 and target_alloc > 0:
                day_tx += target_alloc * tx_cost  # 새 진입 tx
            elif target_alloc < prev_alloc:
                # Shrink 발생 (부분 realized)
                shrunk = prev_alloc - target_alloc
                # TP 도달 전이라 현재는 pnl_pct 를 모름. 보수적: realized 0 + tx.
                day_tx += shrunk * tx_cost
                n_shrinks += 1
                if log_events:
                    event_log.append({'date': date, 'type': 'shrink',
                                      'coin': lot.coin, 'before': prev_alloc, 'after': target_alloc})
                if target_alloc == 0.0:
                    # 전량 shrink = 청산
                    continue
            lot.current_alloc = target_alloc
            remaining -= target_alloc
            if remaining < 0: remaining = 0
            new_lots_after_shrink.append(lot)
        lots = new_lots_after_shrink

        # 5) Port ret
        c_w = sum(lot.current_alloc for lot in lots)
        c_w = min(c_w, cash_avail)
        port_ret = (1 - c_w) * v21_ret + day_realized_pnl - day_tx
        port_rets.append(port_ret)

    port_eq = (1 + pd.Series(port_rets, index=idx)).cumprod()
    stats = {'n_entries': n_entries, 'n_exits': n_exits,
             'n_swaps': n_swaps, 'n_shrinks': n_shrinks}
    if log_events:
        return port_eq, stats, event_log
    return port_eq, stats


if __name__ == '__main__':
    v21 = load_v21()
    hist = load_universe_hist()
    avail = list_available_futures()
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    # Test run (best config 예상)
    from c_engine_v5 import run_c_v5
    P = {'dip_bars':24, 'dip_thr':-0.18, 'tp':0.08, 'tstop':48}
    all_events = []
    for c in avail:
        df = load_coin(c + 'USDT')
        if df is None: continue
        _, evs = run_c_v5(df, **P)
        for e in evs:
            e['coin'] = c
            all_events.append(e)
    events = pd.DataFrame(all_events)
    print(f'Events: {len(events)}')

    # Quick test
    for n in [1, 2, 3]:
        for cap in [0.10, 0.30]:
            port_eq, stats = simulate_lot(v21, events, hist,
                                           n_pick=n, cap_per_slot=cap,
                                           universe_size=15,
                                           swap_edge_threshold=0,
                                           swap_cooldown_days=0)
            m = metrics(port_eq, bpy=252)
            print(f'n={n} cap_per_slot={cap}: {m} stats={stats}')
