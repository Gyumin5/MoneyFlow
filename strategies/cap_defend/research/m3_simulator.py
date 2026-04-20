#!/usr/bin/env python3
"""Step 2: M3 dynamic cash simulator.

입력:
- V21 현물 daily: (equity, cash_ratio)
- C events: (coin, entry_ts, exit_ts, entry_px, exit_px, pnl_pct, bars_held, reason)

로직:
- 초기 total_eq = 1.0
- daily loop:
  1. V21 invested (1 - cash_ratio) 부분: V21 daily return 반영
  2. V21 cash (cash_ratio) 부분: 기본 0% 수익 (현금)
  3. C 활성 포지션 있으면:
     - C 슬롯 capital = min(hard_cap, v21_cash_ratio[t]) × total_eq
     - C 포지션 pnl_daily 반영 (이벤트 기간 동안)
  4. V21 cash_ratio 감소 (target 변경) → C 활성 포지션 강제 청산 (P1)
- Hard Cap: C 최대 전체 포트의 %
- Multiple coins 동시 발동 가능: 슬롯 분할 (1/n_pick 씩)

출력: daily total equity 시계열
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')


def load_inputs(events_csv='events_top10.csv', v21_csv='v21_daily.csv'):
    ev = pd.read_csv(os.path.join(STRAT_DIR, events_csv))
    ev['entry_ts'] = pd.to_datetime(ev['entry_ts'])
    ev['exit_ts'] = pd.to_datetime(ev['exit_ts'])
    v21 = pd.read_csv(os.path.join(STRAT_DIR, v21_csv), index_col=0, parse_dates=True)
    # 정규화
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    return ev, v21


def simulate_m3(v21, events, hard_cap=0.10, coins_include=None, n_pick=3,
                lev=1.0, tx_cost=0.003, force_close_on_cash_decrease=True):
    """
    v21: DataFrame(index=date, columns=['equity','cash_ratio','v21_ret'])
    events: DataFrame of C events (entry_ts, exit_ts, pnl_pct, coin, ...)
    hard_cap: 전체 포트의 몇 %를 C에 최대 할당
    coins_include: None(all) or list[coin]
    n_pick: 최대 동시 C 포지션 수
    force_close_on_cash_decrease: V21 cash_ratio 감소 시 C 포지션 강제 청산

    반환: DataFrame(index=date, columns=['total_eq','v21_eq','c_eq','c_used_ratio','c_open_n'])
    """
    if coins_include is not None:
        events = events[events['coin'].isin(coins_include)].copy()
    # events를 일 단위로 매핑 (하루에 entry_ts가 있는 이벤트 set)
    events = events.sort_values('entry_ts').reset_index(drop=True)

    total_eq = 1.0
    c_positions = []  # 활성 C: {'coin','entry_date','exit_date','pnl_pct','slot_cap_at_entry','bars_held'}
    rows = []
    prev_cash_ratio = None
    v21_invested_eq = 1.0  # V21 invested만의 누적
    c_eq_only = 1.0        # C만 단독 시

    for date, r in v21.iterrows():
        cash_ratio = r['cash_ratio']
        v21_ret = r['v21_ret']

        # 1) V21 cash 감소 시 강제 청산
        if force_close_on_cash_decrease and prev_cash_ratio is not None:
            if cash_ratio < prev_cash_ratio - 0.05:  # 5%p 이상 감소
                # 강제 청산 (손실/이익 그대로, tx_cost만 추가 차감)
                for pos in c_positions:
                    # realized pnl_so_far (부분 구현: pos['pnl_pct']에 도달했는지 여부 대신 간단히 bar_held로 비례)
                    # 간단화: 단순히 pnl=0 가정 후 tx 차감
                    total_eq -= pos['slot_cap_at_entry'] * tx_cost
                c_positions = []

        # 2) 이번 날 exit 되는 포지션 처리
        still_open = []
        for pos in c_positions:
            if pos['exit_date'] <= date:
                # 청산: pnl 반영 - tx
                realized = pos['slot_cap_at_entry'] * (pos['pnl_pct'] / 100.0) * lev
                total_eq += realized - pos['slot_cap_at_entry'] * tx_cost
                # C only eq 계산
                c_eq_only *= (1 + (pos['pnl_pct'] / 100.0) * lev - tx_cost)
            else:
                pos['bars_held'] += 1
                still_open.append(pos)
        c_positions = still_open

        # 3) 이번 날 entry 되는 이벤트 처리
        today_events = events[events['entry_ts'].dt.normalize() == pd.Timestamp(date)]
        open_slots = n_pick - len(c_positions)
        if open_slots > 0 and len(today_events) > 0:
            # 선정: pnl_pct 양수일 때 높은 쪽 우선? 실전에선 pnl 모름. 실전이면 deepest dip 기준. events는 pnl 이미 계산됐지만 선정기준은 dip 깊이로 사전에 했다고 가정 (events 이미 시간순)
            # 여기선 단순히 entry_ts 순서대로 slot 채움
            picked = today_events.head(open_slots)
            for _, ev in picked.iterrows():
                # c_slot_capital: 가용 C capital / 남은 slot
                available_cap = min(hard_cap, cash_ratio) * total_eq
                # 이미 사용 중 slot 차감
                already_used = sum(p['slot_cap_at_entry'] for p in c_positions)
                remaining_cap = max(0, available_cap - already_used)
                # slot = remaining_cap / open_slots (균등 분배)
                slot_cap = remaining_cap / (open_slots - (picked.index.get_loc(ev.name) - today_events.index[0]))
                if slot_cap <= 0:
                    continue
                # entry: tx 차감
                total_eq -= slot_cap * tx_cost
                c_positions.append({
                    'coin': ev['coin'],
                    'entry_date': date,
                    'exit_date': ev['exit_ts'].normalize(),
                    'pnl_pct': ev['pnl_pct'],
                    'slot_cap_at_entry': slot_cap,
                    'bars_held': 0,
                })

        # 4) V21 invested 부분 수익 반영 (C가 가져간 cash는 빠진 상태로)
        c_used = sum(p['slot_cap_at_entry'] for p in c_positions)
        v21_invested_portion = (1 - cash_ratio) * total_eq
        v21_cash_portion = cash_ratio * total_eq - c_used
        v21_cash_portion = max(0, v21_cash_portion)
        # V21 invested만 v21_ret 반영
        new_invested = v21_invested_portion * (1 + v21_ret)
        # C 부분은 bar 내 약간 변화하지만 단순화: daily pnl은 exit 시 일괄 반영 (여기선 0)
        total_eq = new_invested + v21_cash_portion + c_used
        v21_invested_eq *= (1 + v21_ret) if cash_ratio < 1.0 else 1.0

        rows.append({
            'date': date,
            'total_eq': total_eq,
            'v21_invested_eq': v21_invested_eq,
            'c_eq_only': c_eq_only,
            'cash_ratio': cash_ratio,
            'c_used_ratio': c_used / total_eq if total_eq > 0 else 0,
            'c_open_n': len(c_positions),
        })
        prev_cash_ratio = cash_ratio

    return pd.DataFrame(rows).set_index('date')


def metrics(eq_series):
    rets = eq_series.pct_change().dropna()
    if len(rets) == 0 or eq_series.iloc[-1] <= 0:
        return {'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0}
    bpy = 252
    sh = (rets.mean()*bpy) / (rets.std()*np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq_series.index[-1] - eq_series.index[0]).days
    years = days/365.25 if days > 0 else 0.001
    cagr = (eq_series.iloc[-1]/eq_series.iloc[0])**(1/years) - 1
    mdd = (eq_series / eq_series.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe':round(float(sh),3),'CAGR':round(float(cagr),4),
            'MDD':round(float(mdd),4),'Cal':round(float(cal),3),
            'Final':round(float(eq_series.iloc[-1]/eq_series.iloc[0]),3)}


def main():
    ev, v21 = load_inputs()
    print(f'Events: {len(ev)}, V21 days: {len(v21)}')
    print(f'V21 단독: {metrics(v21["equity"])}')

    # Sweep
    print('\n=== M3 시뮬 Hard Cap sweep (C universe=BTC+ADA만) ===')
    print('※ BTC+ADA: 보수 체결에서 유일하게 양수 Sharpe 코인')
    rows = []
    for uni_name, coins in [('BTC_only', ['BTCUSDT']),
                             ('BTC+ADA', ['BTCUSDT','ADAUSDT']),
                             ('Top5_all', ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT']),
                             ('Top10_all', None)]:
        for cap in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for n_pick in [1, 2, 3]:
                sim = simulate_m3(v21, ev, hard_cap=cap, coins_include=coins, n_pick=n_pick)
                m = metrics(sim['total_eq'])
                row = {'universe': uni_name, 'hard_cap': cap, 'n_pick': n_pick, **m}
                rows.append(row)
                if cap in [0.10, 0.20] and n_pick == 3:
                    print(f'  {uni_name:>10} cap={cap:.0%} n_pick={n_pick}: Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} MDD={m["MDD"]:.2%} Cal={m["Cal"]}')

    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(STRAT_DIR, 'm3_sweep.csv'), index=False)

    print('\n=== Top 10 by Cal ===')
    print(rdf.sort_values('Cal', ascending=False).head(10).to_string(index=False))

    # 2021 제거 ablation (best config)
    best = rdf.sort_values('Cal', ascending=False).iloc[0]
    print(f'\n=== 2021 제거 ablation (best: {best["universe"]} cap={best["hard_cap"]} n_pick={int(best["n_pick"])}) ===')
    coins_best = {'BTC_only':['BTCUSDT'],'BTC+ADA':['BTCUSDT','ADAUSDT'],
                  'Top5_all':['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT'],
                  'Top10_all':None}[best['universe']]
    ev_no2021 = ev[ev['entry_ts'] >= pd.Timestamp('2022-01-01')]
    v21_no2021 = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no2021['equity'] = v21_no2021['equity'] / v21_no2021['equity'].iloc[0]
    v21_no2021['v21_ret'] = v21_no2021['equity'].pct_change().fillna(0)
    sim2 = simulate_m3(v21_no2021, ev_no2021,
                       hard_cap=best['hard_cap'], coins_include=coins_best, n_pick=int(best['n_pick']))
    m2 = metrics(sim2['total_eq'])
    m_v21_no2021 = metrics(v21_no2021['equity'])
    print(f'  V21 단독 (2022+): {m_v21_no2021}')
    print(f'  V21+C M3 (2022+): {m2}')

    print(f'\n저장: {STRAT_DIR}/m3_sweep.csv')


if __name__ == '__main__':
    main()
