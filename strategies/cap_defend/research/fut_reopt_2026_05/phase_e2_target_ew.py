#!/usr/bin/env python3
"""DEPRECATED — DO NOT RUN. simulate() 가 backtest_futures_full.py 의 forced exit 로직 누락.

Phase E2 — Target-weight EW 앙상블 BT.
V23 와 C1 의 target weight history 를 trace 로 얻고,
EW merge 된 target 으로 단일 capital pool 에서 자체 simulate() 실집행 시뮬레이션.

실패 이유: simulate() 가 청산 (격리마진), crash breaker, DD exit, blacklist forced exit,
stop, maint_rate 로직 미구현. 결과 Cal 7.22 는 가짜 (실제 정식 BT 는 Cal 3.73).
ai-debate 코드 리뷰에서 신뢰 불가 판정.

정식 경로: phase_e3_target_ew_proper.py
- backtest_futures_full.py 의 external_target_schedule 모드 사용
- 청산/crash/DD/BL/stop/펀딩/tx/slip/drift 모두 기존 run() 내부 로직으로 처리
- V23/C1 replay parity 0% 검증 통과
"""
import os, sys
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import pandas as pd
import numpy as np
from backtest_futures_full import load_data, run, SLIPPAGE_MAP, get_close

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

V23_CFG = dict(sma_days=42, mom_short_days=30, mom_long_days=90,
               n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03)
C1_CFG  = dict(sma_days=38, mom_short_days=20, mom_long_days=122,
               n_snapshots=3, snap_interval_bars=111, drift_threshold=0.020)
COMMON = dict(
    interval='D', leverage=3.0,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015,
    health_mode='mom2vol',
    start_date='2020-10-01', end_date='2026-05-13',
)

def get_trace(cfg):
    trace = []
    run(LOAD['bars'], LOAD['funding'], _trace=trace, **{**COMMON, **cfg})
    return trace

LOAD = {}

def merge_targets(t_v23, t_c1, weight_v23=0.5):
    """각 bar 에서 target weights EW merge."""
    by_date_v = {row['date']: row['target'] for row in t_v23}
    by_date_c = {row['date']: row['target'] for row in t_c1}
    rebal_v   = {row['date']: row.get('rebal', False) for row in t_v23}
    rebal_c   = {row['date']: row.get('rebal', False) for row in t_c1}
    dates = sorted(set(by_date_v) & set(by_date_c))
    out = []
    for d in dates:
        v = by_date_v[d]
        c = by_date_c[d]
        merged = {}
        all_keys = set(v) | set(c)
        for k in all_keys:
            merged[k] = weight_v23 * v.get(k, 0.0) + (1 - weight_v23) * c.get(k, 0.0)
        # 정규화
        s = sum(merged.values())
        if s > 0:
            merged = {k: w/s for k, w in merged.items()}
        # rebal 트리거: 둘 중 하나라도 rebal 일 때
        rebal = rebal_v.get(d, False) or rebal_c.get(d, False)
        out.append(dict(date=d, target=merged, rebal_v23=rebal_v.get(d, False),
                        rebal_c1=rebal_c.get(d, False), rebal=rebal))
    return out

def simulate(merged, bars, funding, leverage=3.0, tx_cost=0.0006,
             drift_threshold=0.025, initial_capital=10000.0):
    """단일 capital pool 에서 merged target 실행 시뮬."""
    capital = initial_capital
    holdings = {}     # coin → qty
    entry_prices = {} # coin → avg entry
    margins = {}      # coin → margin
    pv_list = []
    rebal_count = 0
    trade_count = 0

    def _port_val(date):
        pv = capital
        for coin in holdings:
            ci = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
            cur = get_close(bars, coin, ci)
            pnl = holdings[coin] * (cur - entry_prices[coin])
            pv += margins[coin] + pnl
        return pv

    def _current_weights(date):
        pv = _port_val(date)
        if pv <= 0:
            return {'CASH': 1.0}
        w = {}
        for coin in holdings:
            ci = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
            cur = get_close(bars, coin, ci)
            val = margins[coin] + holdings[coin] * (cur - entry_prices[coin])
            if val > 0:
                w[coin] = val / pv
        cash_w = capital / pv
        if cash_w > 0.001:
            w['CASH'] = cash_w
        return w

    def _half_turnover(cur_w, tgt_w):
        all_k = set(cur_w.keys()) | set(tgt_w.keys())
        return sum(abs(tgt_w.get(k, 0) - cur_w.get(k, 0)) for k in all_k) / 2

    def _execute(target, date):
        nonlocal capital, rebal_count, trade_count
        pv = _port_val(date)
        if pv <= 0:
            return
        target_qty = {}
        target_margin = {}
        for coin, w in target.items():
            if coin == 'CASH' or w <= 0:
                continue
            if coin not in bars:
                continue
            ci = bars[coin].index.get_indexer([date], method='ffill')[0]
            cur = float(bars[coin]['Open'].iloc[ci]) if ci >= 0 else 0
            if cur <= 0:
                continue
            tmgn = pv * w * 0.95
            tnot = tmgn * leverage
            tqty = tnot / cur
            target_qty[coin] = tqty
            target_margin[coin] = tmgn

        # 매도
        for coin in list(holdings.keys()):
            ci = bars[coin].index.get_indexer([date], method='ffill')[0]
            cur = float(bars[coin]['Open'].iloc[ci])
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in target_qty:
                exit_p = cur * (1 - slip)
                pnl = holdings[coin] * (exit_p - entry_prices[coin])
                tx = holdings[coin] * cur * tx_cost
                capital += margins[coin] + pnl - tx
                del holdings[coin]; del entry_prices[coin]; del margins[coin]
                trade_count += 1
            else:
                delta = target_qty[coin] - holdings[coin]
                if delta < -holdings[coin] * 0.05:
                    sell_qty = -delta
                    sell_frac = sell_qty / holdings[coin]
                    sell_margin = margins[coin] * sell_frac
                    exit_p = cur * (1 - slip)
                    pnl = sell_qty * (exit_p - entry_prices[coin])
                    tx = sell_qty * cur * tx_cost
                    capital += sell_margin + pnl - tx
                    holdings[coin] -= sell_qty
                    margins[coin] -= sell_margin
                    trade_count += 1

        # 매수
        for coin, tqty in target_qty.items():
            ci = bars[coin].index.get_indexer([date], method='ffill')[0]
            cur = float(bars[coin]['Open'].iloc[ci])
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in holdings:
                entry_p = cur * (1 + slip)
                margin = target_margin[coin]
                notional = margin * leverage
                qty = notional / entry_p
                tx = notional * tx_cost
                if capital >= margin + tx:
                    capital -= margin + tx
                    holdings[coin] = qty
                    entry_prices[coin] = entry_p
                    margins[coin] = margin
                    trade_count += 1
            else:
                delta = tqty - holdings[coin]
                if delta > holdings[coin] * 0.05:
                    entry_p = cur * (1 + slip)
                    add_not = delta * entry_p
                    add_mgn = add_not / leverage
                    tx = add_not * tx_cost
                    if capital >= add_mgn + tx:
                        capital -= add_mgn + tx
                        total = holdings[coin] + delta
                        entry_prices[coin] = (entry_prices[coin] * holdings[coin] + entry_p * delta) / total
                        holdings[coin] = total
                        margins[coin] += add_mgn
                        trade_count += 1
        rebal_count += 1

    prev_date = None
    for row in merged:
        date = row['date']
        target = row['target']
        # 펀딩
        if prev_date is not None:
            for coin in list(holdings.keys()):
                fr = funding.get(coin)
                if fr is None: continue
                window = fr.loc[(fr.index > prev_date) & (fr.index <= date)]
                if len(window) == 0: continue
                fr_sum = float(window.sum())
                if fr_sum != 0 and not np.isnan(fr_sum):
                    ci = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
                    cur = get_close(bars, coin, ci)
                    if cur > 0:
                        capital -= holdings[coin] * cur * fr_sum
            capital = max(capital, 0)

        # rebal 트리거: V23/C1 중 하나라도 rebal 또는 half_turnover >= drift
        cur_w = _current_weights(date)
        ht = _half_turnover(cur_w, target)
        need = row['rebal'] or (ht >= drift_threshold)
        if need:
            _execute(target, date)
        pv_list.append({'Date': date, 'Value': _port_val(date)})
        prev_date = date

    if not pv_list:
        return {}
    pvdf = pd.DataFrame(pv_list).set_index('Date')
    eq = pvdf['Value']
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if eq.iloc[-1] <= 0 or yrs <= 0:
        return dict(Sharpe=0, CAGR=-1, MDD=-1, Cal=0, Rebal=rebal_count, Trades=trade_count)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(Sharpe=sh, CAGR=cagr, MDD=mdd, Cal=cal, Rebal=rebal_count, Trades=trade_count, _eq=eq)

def main():
    LOAD['bars'], LOAD['funding'] = load_data('D')
    print("trace 수집 V23...")
    t_v23 = get_trace(V23_CFG)
    print(f"  V23 trace: {len(t_v23)} bars")
    print("trace 수집 C1...")
    t_c1  = get_trace(C1_CFG)
    print(f"  C1 trace: {len(t_c1)} bars")

    merged = merge_targets(t_v23, t_c1, weight_v23=0.5)
    print(f"merged bars: {len(merged)}")

    print("\n=== EW Target 50/50, drift=0.025 ===")
    r = simulate(merged, LOAD['bars'], LOAD['funding'], drift_threshold=0.025)
    print(f"Sharpe={r['Sharpe']:.2f} CAGR={r['CAGR']:+.1%} MDD={r['MDD']:+.1%} Cal={r['Cal']:.2f} Rebal={r['Rebal']}")

    # 다른 weight sweep
    for w_v23 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        merged_w = merge_targets(t_v23, t_c1, weight_v23=w_v23)
        r_w = simulate(merged_w, LOAD['bars'], LOAD['funding'], drift_threshold=0.025)
        print(f"  V23 weight={w_v23:.1f}: Sharpe={r_w['Sharpe']:.2f} CAGR={r_w['CAGR']:+.1%} MDD={r_w['MDD']:+.1%} Cal={r_w['Cal']:.2f} Rebal={r_w['Rebal']}")

    # drift sweep at 50/50
    print("\n=== weight 50/50, drift sweep ===")
    for d in [0.020, 0.025, 0.030, 0.040, 0.050]:
        r_d = simulate(merged, LOAD['bars'], LOAD['funding'], drift_threshold=d)
        print(f"  drift={d:.3f}: Sharpe={r_d['Sharpe']:.2f} CAGR={r_d['CAGR']:+.1%} MDD={r_d['MDD']:+.1%} Cal={r_d['Cal']:.2f} Rebal={r_d['Rebal']}")

    # yearly
    if '_eq' in r:
        eq = r['_eq']
        print("\n=== Yearly (weight 50/50 drift=0.025) ===")
        for year, g in eq.groupby(eq.index.year):
            if len(g) < 20: continue
            ret = g.iloc[-1] / g.iloc[0] - 1
            dr = g.pct_change().dropna()
            sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
            mdd = (g / g.cummax() - 1).min()
            cal = ret / abs(mdd) if mdd != 0 else 0
            print(f"  {year}: ret={ret:+.1%} MDD={mdd:+.1%} Cal={cal:.2f} Sharpe={sh:.2f}")

if __name__ == '__main__':
    main()
