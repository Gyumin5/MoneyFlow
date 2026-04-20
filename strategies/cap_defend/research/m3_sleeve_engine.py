#!/usr/bin/env python3
"""M3 Sleeve Engine v3 — Codex 조언 반영.

V21 sleeve + C sleeve 분리.
port_eq[t] = v21_alone_eq[t] + c_sleeve_cumulative[t]

C sleeve = sum_{lots active} (qty × cur_px) + c_cash
          (initial 0, V21 cash에서 빌려온 부분)

모든 cash flow:
- Entry: V21_borrowed += target_notional, lot.qty = target/entry_px, c_sleeve -= target×tx (tx만 순손실)
- Exit (TP/timeout): c_sleeve 는 MTM으로 이미 반영. realized pnl = 0 추가. tx -= qty×px×tx
- Swap out: worst lot 청산 at cur_px. c_sleeve 변화 = 0 (MTM이미), tx -= qty×cur×tx
- Shrink: lot qty 감소 + 차액 반납. tx -= shrunk_value × tx
- MTM: c_sleeve_delta += qty×(cur-prev) [각 lot]

V21_cash constraint: Σ(lot.qty × cur_px) ≤ v21_alone_eq[t] × v21_cash_ratio[t-1]
초과 시 뒤 slot shrink.

최종 port_eq = v21_alone_eq + c_sleeve_cumulative
(c_sleeve_cumulative는 실제 '빌려온 cash 가치 변화 + tx'만 반영, V21은 자기 방식 유지)
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


def load_coin_daily(coins, start='2020-10-01', end='2026-03-30'):
    out = {}
    for c in coins:
        df = load_coin(c + 'USDT')
        if df is None: continue
        d = df['Close'].resample('D').last().ffill()
        out[c] = d
    return out


class Lot:
    __slots__ = ('coin','entry_date','exit_date','pnl_pct','cap_r',
                 'entry_px','qty','entered')
    def __init__(self, coin, ed, xd, pnl, cap_r, epx):
        self.coin = coin; self.entry_date = ed; self.exit_date = xd
        self.pnl_pct = pnl; self.cap_r = cap_r
        self.entry_px = epx; self.qty = 0.0; self.entered = False


def simulate_sleeve(v21, events, hist, coin_daily,
                    n_pick=1, cap_per_slot=0.30, universe_size=15,
                    tx_cost=0.003, swap_edge_threshold=0,
                    swap_cooldown_days=0, log_events=False):
    """
    C sleeve (value-based) + V21 alone sleeve.
    port_eq[t] = v21_alone_eq[t] + c_sleeve_cum[t]
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
    lots: list = []
    c_sleeve_cum = 0.0
    c_sleeve_series = []
    last_swap_date = None
    n_swaps = n_shrinks = n_entries = n_exits_natural = 0
    event_log = []
    prev_px_map = {}  # coin -> prev day close

    for date in idx:
        v21_eq_today = v21.loc[date, 'equity']  # V21 alone equity at this date
        v21_cash_today = v21.loc[date, 'prev_cash']  # 전일 기준 cash 사용 (look-ahead 방지)
        max_c_notional = v21_cash_today * v21_eq_today

        # ─── 1) MTM on open lots ───
        today_px_map = {}
        for lot in lots:
            if lot.coin in coin_daily and date in coin_daily[lot.coin].index:
                today_px_map[lot.coin] = float(coin_daily[lot.coin].loc[date])

        day_delta = 0.0  # c_sleeve 증감
        for lot in lots:
            if not lot.entered: continue
            cur_px = today_px_map.get(lot.coin)
            if cur_px is None: continue
            prev_px = prev_px_map.get(lot.coin, lot.entry_px)
            # MTM
            day_delta += lot.qty * (cur_px - prev_px)

        # ─── 2) Natural exits (TP/timeout) ───
        still_open = []
        for lot in lots:
            if lot.exit_date <= date and lot.entered:
                cur_px = today_px_map.get(lot.coin, lot.entry_px)
                # C sleeve는 이미 MTM 반영. exit 시 tx만 차감
                day_delta -= lot.qty * cur_px * tx_cost
                n_exits_natural += 1
                if log_events:
                    event_log.append({'date': date, 'type': 'exit_nat',
                                      'coin': lot.coin, 'pnl_pct': lot.pnl_pct})
                # lot은 제외
            else:
                still_open.append(lot)
        lots = still_open

        # ─── 3) Entry 후보 ───
        today = events_by_day.get(date, [])
        today_sorted = sorted(today, key=lambda x: x['cap_r'])
        open_coins = {lot.coin for lot in lots}
        swap_allowed = (last_swap_date is None or
                        (date - last_swap_date).days >= swap_cooldown_days)

        for ev in today_sorted:
            if ev['coin'] in open_coins: continue

            if len(lots) < n_pick:
                # 신규 진입 (나중 allocation에서 qty 결정)
                new = Lot(ev['coin'], date, ev['exit_date'], ev['pnl_pct'],
                          ev['cap_r'], float(ev['entry_px']))
                lots.append(new)
                n_entries += 1
                open_coins.add(ev['coin'])
                if log_events:
                    event_log.append({'date': date, 'type': 'entry_pending',
                                      'coin': ev['coin'], 'cap_r': ev['cap_r']})
            elif swap_allowed:
                worst = max(lots, key=lambda l: l.cap_r)
                if worst.entered and worst.cap_r - ev['cap_r'] > swap_edge_threshold:
                    # 기존 청산 (MTM 이미 반영, tx만)
                    cur_px = today_px_map.get(worst.coin, worst.entry_px)
                    day_delta -= worst.qty * cur_px * tx_cost
                    lots.remove(worst)
                    new = Lot(ev['coin'], date, ev['exit_date'], ev['pnl_pct'],
                              ev['cap_r'], float(ev['entry_px']))
                    lots.append(new)
                    n_swaps += 1
                    last_swap_date = date
                    open_coins.add(ev['coin'])
                    if log_events:
                        event_log.append({'date': date, 'type': 'swap',
                                          'out': worst.coin, 'in': ev['coin']})

        # ─── 4) Allocation ───
        lots.sort(key=lambda l: l.cap_r)
        cap_abs = cap_per_slot * v21_eq_today
        remaining = max_c_notional
        for lot in lots:
            target = min(cap_abs, remaining)
            if not lot.entered:
                # 신규 allocation
                if target <= 0:
                    lot.qty = 0.0  # 진입 보류
                    continue
                lot.qty = target / lot.entry_px
                lot.entered = True
                day_delta -= target * tx_cost
            else:
                # 기존 shrink 가능성
                cur_px = today_px_map.get(lot.coin, lot.entry_px)
                cur_value = lot.qty * cur_px
                if target < cur_value - 1e-9:
                    # Shrink: cur_value - target 만큼 매도
                    shrunk = cur_value - target
                    day_delta -= shrunk * tx_cost
                    new_qty = target / cur_px if cur_px > 0 else 0
                    lot.qty = new_qty
                    n_shrinks += 1
                    if log_events:
                        event_log.append({'date': date, 'type': 'shrink',
                                          'coin': lot.coin, 'from': cur_value, 'to': target})
            remaining -= target if target > 0 else 0

        # Remove zero-qty entered lots (shrunk to 0)
        lots = [lot for lot in lots if (lot.entered and lot.qty > 0) or not lot.entered]

        # ─── 5) c_sleeve 누적 ───
        c_sleeve_cum += day_delta
        c_sleeve_series.append(c_sleeve_cum)

        # ─── 6) prev_px 업데이트 ───
        for lot in lots:
            if lot.entered and lot.coin in today_px_map:
                prev_px_map[lot.coin] = today_px_map[lot.coin]

    c_sleeve_ser = pd.Series(c_sleeve_series, index=idx)
    port_eq = v21['equity'] + c_sleeve_ser
    stats = {'n_entries': n_entries, 'n_exits_natural': n_exits_natural,
             'n_swaps': n_swaps, 'n_shrinks': n_shrinks,
             'c_sleeve_final': round(float(c_sleeve_cum), 4)}
    if log_events:
        return port_eq, stats, event_log, c_sleeve_ser
    return port_eq, stats, c_sleeve_ser


if __name__ == '__main__':
    v21 = load_v21()
    hist = load_universe_hist()
    avail = list_available_futures()
    coin_daily = load_coin_daily(avail)
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    P = {'dip_bars':24, 'dip_thr':-0.18, 'tp':0.12, 'tstop':48}
    all_e = []
    for c in avail:
        df = load_coin(c + 'USDT')
        if df is None: continue
        _, evs = run_c_v5(df, **P)
        for e in evs:
            e['coin'] = c
            all_e.append(e)
    events = pd.DataFrame(all_e)
    print(f'Events: {len(events)}')

    print('\n=== Sleeve engine test ===')
    for n_pick in [1, 2, 3]:
        for cap in [0.10, 0.20, 0.30, 0.50]:
            port_eq, stats, c_ser = simulate_sleeve(v21, events, hist, coin_daily,
                                                     n_pick=n_pick, cap_per_slot=cap)
            m = metrics(port_eq, bpy=252)
            print(f'n={n_pick} cap={cap}: {m} stats={stats}')
