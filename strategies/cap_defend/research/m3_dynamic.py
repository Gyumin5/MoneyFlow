#!/usr/bin/env python3
"""M3 Dynamic universe simulator (정확한 버전).

설계:
- Dynamic universe: Top 10 또는 Top 20 고정 (BTC+ADA 같은 cherry-pick 금지)
- 매 1h 봉별 active 포지션 관리:
  * C 포지션 없는 bar: c_weight = 0 (V21이 전체 eq 운영)
  * C 포지션 있는 bar: c_weight = min(hard_cap, v21_cash_ratio) × slot_ratio
- n_pick × select_method 로 동시 포지션 수 & 선정 기준
- V21 cash_ratio 감소 시 C 포지션 강제 청산 (선택)

입력: 각 코인 v3 events (이미 보수 체결로 추출됨) + V21 daily
출력: daily total equity
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')


UNIVERSE_TOP10 = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                  'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT']
CAP_RANK = {c: i for i, c in enumerate(UNIVERSE_TOP10)}


def load_events(coins):
    ev = pd.read_csv(os.path.join(STRAT_DIR, 'events_top10.csv'))
    ev['entry_ts'] = pd.to_datetime(ev['entry_ts'])
    ev['exit_ts'] = pd.to_datetime(ev['exit_ts'])
    ev = ev[ev['coin'].isin(coins)].sort_values('entry_ts').reset_index(drop=True)
    return ev


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    return v21


def simulate(v21, events, hard_cap=0.20, n_pick=3, select_method='deepest',
             tx_cost=0.003, force_close_cash_decrease=True):
    """
    매 day:
    - 활성 C 포지션 exit 체크
    - 새 events 있으면 n_pick 범위 내 진입 (선정 기준)
    - c_weight_today = sum(active slot sizes)
    - port_ret = (1 - c_w) × v21_ret + c_w × c_ret(all active 평균)
    """
    # 인덱스: V21 일자
    idx = v21.index
    # 각 일자별 entry 이벤트 매핑
    # entry_date = entry_ts.normalize()
    events = events.copy()
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    # 선정 기준 키
    if select_method == 'deepest':
        # pnl_pct 는 사후 결과라 실전 선정 기준 아님. dip 깊이 대신 events엔 없음. 대체: bars_held 적은(빠른 bounce) 것이 좋지만 그것도 사후.
        # 실전에서 사용가능한 기준으로 entry_ts(먼저 들어온 것) 사용
        sort_key = 'entry_ts'
    elif select_method == 'cap':
        events['cap_r'] = events['coin'].map(CAP_RANK)
        sort_key = 'cap_r'
    else:
        sort_key = 'entry_ts'

    total_eq = 1.0
    positions = []  # [{'coin','entry_date','exit_date','pnl_pct','slot_cap'}]
    port_rets = []
    c_weight_series = []
    prev_cash = None

    for date in idx:
        cash_ratio = v21.loc[date, 'cash_ratio']
        v21_ret = v21.loc[date, 'v21_ret']

        # force close
        if force_close_cash_decrease and prev_cash is not None:
            if cash_ratio < prev_cash - 0.05:
                # 청산 (tx 비용만 반영, 실제 unrealized pnl은 손실 가능)
                for p in positions:
                    total_eq *= (1 - p['slot_cap_ratio'] * tx_cost)
                positions = []

        # exit check
        still_open = []
        day_exit_pnl_weighted = 0.0
        for p in positions:
            if p['exit_date'] <= date:
                realized = p['pnl_pct'] / 100.0
                day_exit_pnl_weighted += p['slot_cap_ratio'] * (realized - tx_cost)
            else:
                still_open.append(p)
        positions = still_open

        # entry check
        today_events = events[events['entry_date'] == date]
        open_slots = n_pick - len(positions)
        if open_slots > 0 and len(today_events) > 0:
            # 선정
            sorted_events = today_events.sort_values(sort_key)
            # 이미 열린 코인은 제외
            open_coins = {p['coin'] for p in positions}
            picks = sorted_events[~sorted_events['coin'].isin(open_coins)].head(open_slots)
            # 각 pick에 slot 할당: 모든 slot 동등 (C 전체의 hard_cap을 n_pick으로 나눔)
            target_slot_ratio = hard_cap / n_pick  # 각 slot이 차지하는 total_eq 비율 (C 내부 EW)
            actual_slot_ratio = min(target_slot_ratio, cash_ratio / max(1, open_slots))
            for _, ev in picks.iterrows():
                positions.append({
                    'coin': ev['coin'],
                    'entry_date': date,
                    'exit_date': ev['exit_date'],
                    'pnl_pct': ev['pnl_pct'],
                    'slot_cap_ratio': actual_slot_ratio,
                })

        # c_weight today = sum slot ratios
        c_w = sum(p['slot_cap_ratio'] for p in positions)
        c_w = min(c_w, cash_ratio)  # hard 제약
        port_ret = (1 - c_w) * v21_ret + day_exit_pnl_weighted  # exit pnl은 해당 day에 realize
        port_rets.append(port_ret)
        c_weight_series.append(c_w)
        # exit 비용 반영 + port_ret
        total_eq *= (1 + port_ret)
        prev_cash = cash_ratio

    port_eq = pd.Series(port_rets, index=idx)
    port_eq = (1 + port_eq).cumprod()
    return port_eq, pd.Series(c_weight_series, index=idx)


def metrics(eq):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0,'Final':0}
    bpy = 252
    sh = (rets.mean()*bpy) / (rets.std()*np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days/365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe':round(float(sh),3),'CAGR':round(float(cagr),4),
            'MDD':round(float(mdd),4),'Cal':round(float(cal),3),
            'Final':round(float(eq.iloc[-1]/eq.iloc[0]),3)}


def main():
    v21 = load_v21()
    print(f'V21 단독: {metrics(v21["equity"])}')

    # Universe 설정
    UNIVERSES = {
        'Top5_dyn': UNIVERSE_TOP10[:5],     # 시총 Top5
        'Top10_dyn': UNIVERSE_TOP10,        # 시총 Top10
    }

    print('\n=== Dynamic Universe M3 Sweep ===')
    rows = []
    for uname, coins in UNIVERSES.items():
        events = load_events(coins)
        for n_pick in [1, 2, 3, 5]:
            for method in ['entry_ts', 'cap']:  # deepest는 사전정보 없음
                for cap in [0.10, 0.20, 0.30, 0.50]:
                    port_eq, c_w = simulate(v21, events, hard_cap=cap, n_pick=n_pick,
                                            select_method=method)
                    m = metrics(port_eq)
                    row = {'universe': uname, 'n_pick': n_pick, 'method': method,
                          'hard_cap': cap, **m, 'c_w_mean': round(c_w.mean(), 4)}
                    rows.append(row)

    rdf = pd.DataFrame(rows)
    rdf = rdf.sort_values('Cal', ascending=False)
    rdf.to_csv(os.path.join(STRAT_DIR, 'm3_dynamic_sweep.csv'), index=False)

    print(f'\nTotal configs: {len(rdf)}')
    print(f'\nTop 15 by Cal:')
    cols = ['universe','n_pick','method','hard_cap','Sharpe','CAGR','MDD','Cal','c_w_mean','Final']
    print(rdf.head(15)[cols].to_string(index=False))

    # 2021 제거
    best = rdf.iloc[0]
    print(f'\n=== 2021 제거 ablation (best: {best["universe"]} n_pick={int(best["n_pick"])} method={best["method"]} cap={best["hard_cap"]}) ===')
    ev_2 = load_events({'Top5_dyn':UNIVERSE_TOP10[:5],'Top10_dyn':UNIVERSE_TOP10}[best['universe']])
    ev_2 = ev_2[ev_2['entry_ts'] >= pd.Timestamp('2022-01-01')]
    v21_2 = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_2['equity'] = v21_2['equity'] / v21_2['equity'].iloc[0]
    v21_2['v21_ret'] = v21_2['equity'].pct_change().fillna(0)
    port_2, _ = simulate(v21_2, ev_2, hard_cap=best['hard_cap'],
                         n_pick=int(best['n_pick']), select_method=best['method'])
    m_v21 = metrics(v21_2['equity'])
    m_port = metrics(port_2)
    print(f'  V21 단독 (2022+): {m_v21}')
    print(f'  V21+C M3 (2022+): {m_port}')


if __name__ == '__main__':
    main()
