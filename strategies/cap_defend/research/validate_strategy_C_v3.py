#!/usr/bin/env python3
"""Strategy C 보수 체결 기반 재검증 v3.

핵심 변경:
- 매수 체결: 다음 봉 High (사용자 지적)
- 매도 TP/Timeout: 다음 봉 Open
- TX 0.3% 기본

Phase 1: Param sweep + Slippage + 2021 ablation + Crash stress + Walk-forward
Phase 2: Universe (Top 5/10/20)
Phase 3: V21 + C 앙상블 (M3+P1, Hard Cap sweep, 동일일 공분산)
Phase 4: 최종 스트레스
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'

BEST_PARAMS = {
    'dip_bars': 24,
    'dip_threshold': -0.15,
    'take_profit': 0.08,
    'time_stop_bars': 24,
    'lev': 1.0,
}
TX_DEFAULT = 0.003    # 0.3% 편도


def load_coin(sym, interval='1h'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_strategy_v3(df, dip_bars, dip_threshold, take_profit, time_stop_bars,
                    lev=1.0, tx_cost=TX_DEFAULT,
                    buy_at='high', sell_at='open',
                    track_events=False):
    """v3: 보수 체결.
    buy_at: 'open' / 'high' — 매수 가격 참조
    sell_at: 'open' / 'low' — 매도 가격 참조
    """
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    df['dip_sig'] = df['dip_pct'].shift(1) <= dip_threshold

    eq = 10000.0
    equity = []
    position = 0.0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        # 청산 체크
        if position > 0:
            sell_px = row[sell_at.capitalize()] if sell_at in ['open', 'low'] else row['Open']
            pnl = sell_px / entry_price - 1.0
            if pnl >= take_profit or bars_held >= time_stop_bars:
                eq *= (1 + lev * pnl - tx_cost)
                if track_events:
                    events.append({'entry_ts': entry_ts, 'exit_ts': ts,
                                   'entry_px': entry_price, 'exit_px': sell_px,
                                   'pnl_pct': round(pnl*100, 2),
                                   'bars_held': bars_held,
                                   'reason': 'TP' if pnl >= take_profit else 'timeout'})
                position = 0.0; entry_price = 0.0; bars_held = 0; entry_ts = None

        # 진입 체크
        if position == 0 and row['dip_sig']:
            buy_px = row[buy_at.capitalize()] if buy_at in ['open', 'high'] else row['Open']
            entry_price = buy_px; entry_ts = ts; position = 1.0; bars_held = 0
            eq *= (1 - tx_cost)
            # bar 수익 (entry_price → Close)
            bar_ret = row['Close'] / entry_price - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1
        elif position > 0:
            prev_close = df.iloc[i-1]['Close'] if i > 0 else row['Open']
            bar_ret = row['Close'] / prev_close - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1

        if eq < 0: eq = 0
        equity.append(eq)

    return pd.Series(equity, index=df.index), events


def metrics(eq):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    bpy = 24 * 365
    sh = (rets.mean() * bpy) / (rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(sh, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3), 'Final': round(eq.iloc[-1]/eq.iloc[0], 3)}


# ─── Phase 1 ───

def phase1_param_sweep(df, out_csv):
    print('\n=== Phase 1-1: Param sweep (보수 체결) ===')
    rows = []
    for dip_bars in [12, 18, 24, 30, 36, 48]:
        for dip_thr in [-0.10, -0.12, -0.15, -0.18, -0.20]:
            for tp in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
                for tstop in [12, 24, 36, 48, 72]:
                    eq, _ = run_strategy_v3(df, dip_bars, dip_thr, tp, tstop,
                                            buy_at='high', sell_at='open')
                    m = metrics(eq)
                    m.update({'dip_bars':dip_bars,'dip_thr':dip_thr,'tp':tp,'tstop':tstop})
                    rows.append(m)
    rdf = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    rdf.to_csv(out_csv, index=False)
    print(f'Top 10 by Sharpe (보수 체결):')
    print(rdf.head(10).to_string(index=False))
    return rdf.iloc[0].to_dict()


def phase1_slippage(df, params):
    print('\n=== Phase 1-2: Slippage stress (보수 체결) ===')
    for tx in [0.0004, 0.003, 0.005, 0.010, 0.020]:
        eq, _ = run_strategy_v3(df, **{k:params[k] for k in BEST_PARAMS if k in params},
                                 tx_cost=tx, buy_at='high', sell_at='open')
        m = metrics(eq)
        print(f'  TX={tx*100:.2f}% buy=High: Sharpe={m["Sharpe"]:>6} CAGR={m["CAGR"]:>7.2%} MDD={m["MDD"]:>7.2%} Cal={m["Cal"]:>6}')
    print('---')
    # 비교: Open 체결
    for tx in [0.0004, 0.003]:
        eq, _ = run_strategy_v3(df, **{k:params[k] for k in BEST_PARAMS if k in params},
                                 tx_cost=tx, buy_at='open', sell_at='open')
        m = metrics(eq)
        print(f'  TX={tx*100:.2f}% buy=Open: Sharpe={m["Sharpe"]:>6} CAGR={m["CAGR"]:>7.2%} MDD={m["MDD"]:>7.2%} Cal={m["Cal"]:>6}')


def phase1_ablation_2021(df):
    print('\n=== Phase 1-3: 2021 제거 Ablation ===')
    df_no2021 = df[df.index >= '2022-01-01'].copy()
    eq, events = run_strategy_v3(df_no2021, **BEST_PARAMS,
                                  buy_at='high', sell_at='open', track_events=True)
    m = metrics(eq)
    print(f'  기간: {df_no2021.index[0].date()} ~ {df_no2021.index[-1].date()}')
    print(f'  발동 횟수: {len(events)}')
    print(f'  {m}')


def phase1_crash_weeks(df):
    print('\n=== Phase 1-4: Crash-week 스트레스 ===')
    crashes = [
        ('2020.3 코로나', '2020-03-01', '2020-03-31'),
        ('2021.5 5월 폭락', '2021-05-15', '2021-05-31'),
        ('2022.5 루나', '2022-05-08', '2022-05-25'),
        ('2022.11 FTX', '2022-11-06', '2022-11-20'),
        ('2024.8 엔캐리', '2024-08-01', '2024-08-15'),
    ]
    for name, s, e in crashes:
        sub = df.loc[s:e]
        if len(sub) < 10:
            print(f'  {name}: 데이터 없음'); continue
        eq, events = run_strategy_v3(sub, **BEST_PARAMS,
                                      buy_at='high', sell_at='open', track_events=True)
        pnl = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) else 0
        print(f'  {name}: 발동 {len(events)}회, 구간 수익 {pnl:.2%}')


def phase1_walkforward(df):
    print('\n=== Phase 1-5: Walk-forward (보수 체결) ===')
    for yr in [2021, 2022, 2023, 2024, 2025]:
        sub = df.loc[f'{yr}-01-01':f'{yr}-12-31']
        if len(sub) < 100:
            print(f'  {yr}: 데이터 부족'); continue
        eq, events = run_strategy_v3(sub, **BEST_PARAMS,
                                      buy_at='high', sell_at='open', track_events=True)
        m = metrics(eq)
        print(f'  {yr}: 발동 {len(events)}회, Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} Cal={m["Cal"]}')


# ─── Phase 2 ───

def phase2_universe(params):
    print('\n=== Phase 2: Universe 확장 (Top 5/10/20, 보수 체결) ===')
    # 유동성 기준 Top 20 (CoinGecko 시총 대략)
    coins = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT',
             'ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT','MATICUSDT',
             'LTCUSDT','BCHUSDT','TRXUSDT','NEARUSDT','ATOMUSDT',
             'APTUSDT','ARBUSDT','OPUSDT','SUIUSDT']
    rows = []
    for c in coins[:20]:
        df = load_coin(c, '1h')
        if df is None or len(df) < 1000: continue
        eq, events = run_strategy_v3(df, **{k:params[k] for k in BEST_PARAMS if k in params},
                                      buy_at='high', sell_at='open', track_events=True)
        m = metrics(eq)
        m['coin'] = c; m['events'] = len(events)
        rows.append(m)
    rdf = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    print(rdf[['coin','events','Sharpe','CAGR','MDD','Cal','Final']].to_string(index=False))
    # depth 요약
    for n in [5, 10, 20]:
        top_n = rdf.head(n)
        print(f'\n  Top{n} 평균: Sharpe={top_n["Sharpe"].mean():.3f} CAGR={top_n["CAGR"].mean():.3f} Cal={top_n["Cal"].mean():.3f}')
    return rdf


# ─── Phase 3 ───

def phase3_ensemble_with_V21(c_eq_daily):
    print('\n=== Phase 3: V21 + C 앙상블 (M3+P1, Hard Cap sweep) ===')
    import sys
    sys.path.insert(0, HERE)
    from phase4_3asset import build_ensemble_full_equity
    spot_top = pd.read_csv(os.path.join(HERE, 'phase3_10x', 'spot_top.csv'))
    ens = spot_top[spot_top['ensemble_tag'] == 'ENS_spot_k3_4b270476'].iloc[0]
    v21_eq = build_ensemble_full_equity(ens)
    if isinstance(v21_eq.index, pd.DatetimeIndex) and v21_eq.index.tz is not None:
        v21_eq = v21_eq.copy(); v21_eq.index = v21_eq.index.tz_localize(None)

    common = c_eq_daily.index.intersection(v21_eq.index)
    c = (c_eq_daily.loc[common] / c_eq_daily.loc[common].iloc[0])
    v = (v21_eq.loc[common] / v21_eq.loc[common].iloc[0])
    c_r = c.pct_change().dropna(); v_r = v.pct_change().dropna()
    common2 = c_r.index.intersection(v_r.index)
    c_r = c_r.loc[common2]; v_r = v_r.loc[common2]

    # 동일일 PnL 공분산 (Codex 지적)
    cov = c_r.cov(v_r)
    corr = c_r.corr(v_r)
    # 같은 날 둘 다 양수/음수 비율
    both_pos = ((c_r > 0) & (v_r > 0)).sum()
    both_neg = ((c_r < 0) & (v_r < 0)).sum()
    opposite = ((c_r * v_r) < 0).sum()
    print(f'  V21 vs C 일별: corr={corr:.4f} cov={cov:.6f}')
    print(f'  동시 양수: {both_pos}, 동시 음수: {both_neg}, 반대 방향: {opposite}')

    # Hard Cap sweep
    print(f'\n  Hard Cap (C 비중):')
    for cap in [0.05, 0.10, 0.15, 0.20, 0.30]:
        port_r = (1 - cap) * v_r + cap * c_r
        port_eq = (1 + port_r).cumprod()
        m = metrics_daily(port_eq)
        print(f'    C {cap:.0%}: Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} MDD={m["MDD"]:.2%} Cal={m["Cal"]}')

    # 2021 제거 후 앙상블
    print(f'\n  2021 제거 후 앙상블:')
    mask = c_r.index >= pd.Timestamp('2022-01-01')
    for cap in [0.10, 0.20]:
        port_r = (1 - cap) * v_r.loc[mask] + cap * c_r.loc[mask]
        port_eq = (1 + port_r).cumprod()
        m = metrics_daily(port_eq)
        print(f'    C {cap:.0%} (2022+): Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} MDD={m["MDD"]:.2%} Cal={m["Cal"]}')


def metrics_daily(eq):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0}
    bpy = 252
    sh = (rets.mean() * bpy) / (rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(sh, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3)}


def main():
    out_dir = os.path.join(HERE, 'strat_C_v3')
    os.makedirs(out_dir, exist_ok=True)

    # BTC 1h 데이터
    btc = load_coin('BTCUSDT', '1h')
    print(f'BTC 1h bars: {len(btc)}, {btc.index[0]} ~ {btc.index[-1]}')

    # Phase 1
    print('\n##### Phase 1: Strategy C 재검증 (보수 체결) #####')
    best = phase1_param_sweep(btc, os.path.join(out_dir, 'p1_sweep.csv'))
    print(f'\n최적 param (보수 체결): {best}')
    phase1_slippage(btc, best)
    phase1_ablation_2021(btc)
    phase1_crash_weeks(btc)
    phase1_walkforward(btc)

    # Phase 2
    print('\n##### Phase 2: Universe 확장 #####')
    rdf = phase2_universe(best)
    rdf.to_csv(os.path.join(out_dir, 'p2_coins.csv'), index=False)

    # Phase 3
    print('\n##### Phase 3: V21 + C 앙상블 #####')
    # C daily equity (BTC 기준)
    eq, _ = run_strategy_v3(btc, **{k:best[k] for k in BEST_PARAMS if k in best},
                             buy_at='high', sell_at='open')
    eq_daily = eq.resample('D').last().ffill()
    phase3_ensemble_with_V21(eq_daily)

    print(f'\n저장 완료: {out_dir}/')


if __name__ == '__main__':
    main()
