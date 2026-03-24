#!/usr/bin/env python3
"""V17 주식 전략 변형 비교 백테스트.

V17  : 기준선 (VT Crash -3%/3d, 월 1회 리밸런싱)
V17a : 개별 종목 Crash (-7% → 해당만 매도 → BIL)
V17b : 절대 모멘텀 필터 (Top 3 중 < SMA200 → BIL)
V17c : V17a + V17b 조합
V17d : 동적 Crash 복귀 (3일 고정 → VT > SMA10 회복)
V17e : 연속 3일 종목 변경 시 즉시 교체
V17f : Trailing Stop (60d peak -10% → 해당 매도 → BIL)

Usage:
  python3 backtest_stock_variants.py
"""

import sys, os, time
import numpy as np, pandas as pd
from dataclasses import dataclass, replace
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_engine import (
    SP, load_prices, precompute, _init, _run_one, run_bt,
    get_val, get_price, metrics, ALL_TICKERS,
    resolve_canary, select_offensive, select_defensive, filter_healthy,
)
import stock_engine as se

OFF_R7 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

# ─── V17 Baseline ────────────────────────────────────────
V17_BASE = SP(offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='zscore3', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=0.002,
              crash='vt', crash_thresh=0.03, crash_cool=3, sharpe_lookback=252)


# ─── Custom run_bt with variant logic ────────────────────
def run_bt_variant(prices_dict, ind, params, variant='V17'):
    """V17 변형 백테스트. 기본 엔진을 확장.
    variant flags: a=개별Crash, b=절대모멘텀, d=동적복귀, e=연속3일, f=TrailingStop
    """
    # 조합 파싱: 'V17bdf' → has_a=F, has_b=T, has_d=T, has_e=F, has_f=T
    flags = variant.replace('V17', '')
    has_a = 'a' in flags  # 개별 종목 Crash
    has_b = 'b' in flags  # 절대 모멘텀
    has_d = 'd' in flags  # 동적 복귀
    has_e = 'e' in flags  # 연속 3일 교체
    has_f = 'f' in flags  # Trailing Stop

    spy = ind.get('SPY')
    if spy is None:
        return None

    dates = spy.index[(spy.index >= params.start) & (spy.index <= params.end)]
    if len(dates) < 2:
        return None

    anchor = params._anchor
    holdings = {}  # {ticker: shares}
    buy_prices = {}  # {ticker: buy_price} for trailing stop
    cash = params.capital
    prev_month = None
    prev_risk_on = None
    history = []
    rebal_count = 0
    crash_cooldown = 0
    rebalanced_this_month = False
    prev_trading_date = None
    prev_top3 = []  # for V17e
    consecutive_change_days = 0  # for V17e
    peak_prices = {}  # for V17f: {ticker: 60d peak}

    for date in dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        if is_month_change:
            rebalanced_this_month = False

        # Portfolio value
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        sig_date = prev_trading_date if prev_trading_date is not None else date

        # ── Crash Breaker (VT -3%) ──
        crash_just_ended = False
        if crash_cooldown > 0:
            crash_cooldown -= 1

            # V17d: 동적 복귀 — VT > SMA10 회복 확인
            if has_d and crash_cooldown == 0:
                vt_p = get_val(ind, 'VT', sig_date, 'price')
                vt_sma10 = get_val(ind, 'VT', sig_date, 'sma10')
                if not (np.isnan(vt_p) or np.isnan(vt_sma10)) and vt_p < vt_sma10:
                    crash_cooldown = 1  # 아직 SMA10 아래 → 대기 연장
                else:
                    crash_just_ended = True
            elif crash_cooldown == 0:
                crash_just_ended = True

        elif not is_first and params.crash == 'vt':
            vt_ret = get_val(ind, 'VT', sig_date, 'ret')
            if not np.isnan(vt_ret) and vt_ret <= -params.crash_thresh:
                # 전량 매도
                for t in list(holdings.keys()):
                    p = get_price(ind, t, date)
                    shares = holdings.pop(t, 0)
                    if shares > 0 and not np.isnan(p):
                        cash += shares * p * (1 - params.tx_cost)
                crash_cooldown = params.crash_cool
                buy_prices.clear()
                peak_prices.clear()
                pv = cash

        # ── V17a/c: 개별 종목 Crash (-7%) ──
        if has_a and crash_cooldown <= 0 and not is_first:
            for t in list(holdings.keys()):
                if t in DEF:
                    continue
                ret = get_val(ind, t, sig_date, 'ret')
                if not np.isnan(ret) and ret <= -0.07:
                    p = get_price(ind, t, date)
                    shares = holdings.pop(t, 0)
                    if shares > 0 and not np.isnan(p):
                        cash += shares * p * (1 - params.tx_cost)
                        # BIL로 대체
                        bil_p = get_price(ind, 'BIL', date)
                        if not np.isnan(bil_p) and bil_p > 0:
                            bil_shares = int((shares * p * (1 - params.tx_cost)) / bil_p)
                            if bil_shares > 0:
                                holdings['BIL'] = holdings.get('BIL', 0) + bil_shares
                                cash -= bil_shares * bil_p

        # ── V17f: Trailing Stop (60d peak -10%) ──
        if has_f and crash_cooldown <= 0 and not is_first:
            for t in list(holdings.keys()):
                if t in DEF:
                    continue
                cur_p = get_val(ind, t, sig_date, 'price')
                if np.isnan(cur_p):
                    continue
                # 60d peak 갱신
                if t not in peak_prices:
                    peak_prices[t] = cur_p
                else:
                    peak_prices[t] = max(peak_prices[t], cur_p)
                # -10% 체크
                if cur_p < peak_prices[t] * 0.90:
                    p = get_price(ind, t, date)
                    shares = holdings.pop(t, 0)
                    peak_prices.pop(t, None)
                    if shares > 0 and not np.isnan(p):
                        cash += shares * p * (1 - params.tx_cost)
                        bil_p = get_price(ind, 'BIL', date)
                        if not np.isnan(bil_p) and bil_p > 0:
                            bil_shares = int((shares * p * (1 - params.tx_cost)) / bil_p)
                            if bil_shares > 0:
                                holdings['BIL'] = holdings.get('BIL', 0) + bil_shares
                                cash -= bil_shares * bil_p

        # ── V17e: 연속 3일 종목 변경 체크 ──
        force_rebal_e = False
        if has_e and not is_first and crash_cooldown <= 0:
            candidates = filter_healthy(params, ind, sig_date, params.offensive)
            off_w = select_offensive(params, ind, sig_date, candidates)
            today_top3 = sorted(off_w.keys()) if off_w else []
            if prev_top3 and set(today_top3) != set(prev_top3):
                consecutive_change_days += 1
            else:
                consecutive_change_days = 0
            if consecutive_change_days >= 3:
                force_rebal_e = True
                consecutive_change_days = 0
            prev_top3 = today_top3

        # ── Rebalance trigger ──
        is_rebal = False
        if is_first:
            is_rebal = True
        elif not rebalanced_this_month and date.day >= anchor:
            is_rebal = True
        if crash_just_ended and not holdings:
            is_rebal = True
        if force_rebal_e:
            is_rebal = True
        if crash_cooldown > 0:
            is_rebal = False

        if is_rebal:
            rebalanced_this_month = True
            # 현재 자산 합계
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

            risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)
            if prev_risk_on is not None and prev_risk_on != risk_on:
                rebal_count += 1
            prev_risk_on = risk_on

            if risk_on:
                candidates = filter_healthy(params, ind, sig_date, params.offensive)
                weights = select_offensive(params, ind, sig_date, candidates)

                # V17b/c: 절대 모멘텀 필터 (< SMA200 → BIL)
                if has_b and weights:
                    filtered = {}
                    bil_frac = 0
                    for t, w in weights.items():
                        t_p = get_val(ind, t, sig_date, 'price')
                        t_sma = get_val(ind, t, sig_date, 'sma200')
                        if not (np.isnan(t_p) or np.isnan(t_sma)) and t_p < t_sma:
                            bil_frac += w  # SMA200 아래 → BIL로 대체
                        else:
                            filtered[t] = w
                    if bil_frac > 0:
                        filtered['BIL'] = filtered.get('BIL', 0) + bil_frac
                    weights = filtered
            else:
                weights = select_defensive(params, ind, sig_date)

            if not weights:
                weights = {'BIL': 1.0}

            # 기존 매도
            for t in list(holdings.keys()):
                if t not in weights:
                    p = get_price(ind, t, date)
                    shares = holdings.pop(t, 0)
                    if shares > 0 and not np.isnan(p):
                        cash += shares * p * (1 - params.tx_cost)
                        peak_prices.pop(t, None)

            # 매수
            invest = pv  # 전액 투자
            for t, w in weights.items():
                target_val = invest * w
                p = get_price(ind, t, date)
                if np.isnan(p) or p <= 0:
                    continue
                current_val = holdings.get(t, 0) * p
                diff = target_val - current_val
                if abs(diff) < 50:  # $50 미만 무시
                    continue
                if diff > 0:  # 매수
                    shares_to_buy = int(diff / p)
                    cost = shares_to_buy * p * (1 + params.tx_cost)
                    if cost <= cash and shares_to_buy > 0:
                        holdings[t] = holdings.get(t, 0) + shares_to_buy
                        cash -= cost
                        buy_prices[t] = p
                        peak_prices[t] = p
                elif diff < 0:  # 매도
                    shares_to_sell = min(int(abs(diff) / p), holdings.get(t, 0))
                    if shares_to_sell > 0:
                        holdings[t] = holdings.get(t, 0) - shares_to_sell
                        cash += shares_to_sell * p * (1 - params.tx_cost)
                        if holdings.get(t, 0) <= 0:
                            holdings.pop(t, None)
                            peak_prices.pop(t, None)

            rebal_count += 1

        # Update peak prices for trailing stop
        if has_f:
            for t in holdings:
                cur_p = get_val(ind, t, date, 'price')
                if not np.isnan(cur_p):
                    peak_prices[t] = max(peak_prices.get(t, 0), cur_p)

        # Record
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p
        history.append({'Date': date, 'Value': pv})

        if not is_first and prev_risk_on is None:
            prev_risk_on = resolve_canary(params, ind, sig_date, None)
        prev_month = cur_month
        prev_trading_date = date

    if not history:
        return None

    df = pd.DataFrame(history).set_index('Date')
    m = metrics(df)
    m['Rebal'] = rebal_count
    return m


# ─── Main ────────────────────────────────────────────────
def run_one_variant(args):
    """(variant, anchor, prices, ind, start, end) → metrics"""
    variant, anchor, start, end = args
    prices = _prices_global
    ind = _ind_global
    sp = replace(V17_BASE, start=start, end=end, _anchor=anchor)
    if variant == 'V17':
        # 기본 엔진 사용
        return run_bt(prices, ind, sp)
    else:
        return run_bt_variant(prices, ind, sp, variant=variant)


_prices_global = None
_ind_global = None

def init_worker(prices, ind):
    global _prices_global, _ind_global
    _prices_global = prices
    _ind_global = ind


def main():
    t0 = time.time()
    print("데이터 로딩...")
    all_tickers = set(OFF_R7 + DEF + ('VT', 'EEM', 'BIL'))
    prices = load_prices(list(all_tickers))
    ind = precompute(prices)

    # SMA10 추가 (V17d용)
    for t in ind:
        if 'price' in ind[t].columns:
            ind[t]['sma10'] = ind[t]['price'].rolling(10).mean()

    _init(prices, ind)

    global _prices_global, _ind_global
    _prices_global = prices
    _ind_global = ind

    print(f"  로딩 완료 ({time.time()-t0:.1f}s)")

    VARIANTS = [
        'V17',    # 기준선 (동일 엔진)
        'V17a',   # 개별 Crash -7%
        'V17b',   # 절대 모멘텀 (SMA200)
        'V17d',   # 동적 복귀 (VT>SMA10)
        'V17e',   # 연속 3일 교체
        'V17f',   # Trailing Stop 60d -10%
        'V17bd',  # 절대모멘텀 + 동적복귀
        'V17ad',  # 개별Crash + 동적복귀
        'V17adf', # 개별Crash + 동적복귀 + TS
    ]
    PERIODS = [
        ('2017-01-01', '2025-12-31'),
        ('2019-01-01', '2025-12-31'),
        ('2021-01-01', '2025-12-31'),
    ]
    ANCHORS = list(range(1, 12))  # 11-anchor 평균

    print(f"\n{'='*110}")
    print("V17 주식 전략 변형 비교")
    print(f"{'='*110}")

    for start, end in PERIODS:
        print(f"\n  [{start} ~ {end}]")
        print(f"  {'전략':<8s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'σ(Sh)':>6s} {'Rebal':>6s}  설명")
        print(f"  {'-'*100}")

        for variant in VARIANTS:
            results = []
            for anchor in ANCHORS:
                sp = replace(V17_BASE, start=start, end=end, _anchor=anchor)
                # 모든 변형을 동일 엔진(run_bt_variant)으로 실행
                r = run_bt_variant(prices, ind, sp, variant=variant)
                if r is not None:
                    results.append(r)

            if results:
                sh = np.mean([r['Sharpe'] for r in results])
                sh_std = np.std([r['Sharpe'] for r in results])
                cagr = np.mean([r['CAGR'] for r in results])
                mdd = np.mean([r['MDD'] for r in results])
                cal = np.mean([r.get('Calmar', 0) for r in results])
                rebal = np.mean([r.get('Rebal', 0) for r in results])

                desc_map = {
                    'V17': 'VT Crash -3%/3d, 월1회 (tx 0.2%)',
                    'V17a': '+개별 -7%→BIL',
                    'V17b': '+절대모멘텀(SMA200)',
                    'V17c': 'a+b',
                    'V17d': 'Crash복귀:VT>SMA10',
                    'V17e': '연속3일→즉시교체',
                    'V17f': 'TrailingStop 60d-10%',
                    'V17bf': 'b+f 절대모멘텀+TS',
                    'V17bd': 'b+d 절대모멘텀+동적복귀',
                    'V17adf': 'a+d+f 개별+동적+TS',
                    'V17bdf': 'b+d+f 절대모멘텀+동적+TS',
                    'V17abdf': 'a+b+d+f 전부',
                    'V17be': 'b+e 절대모멘텀+3일교체',
                }
                desc = desc_map.get(variant, variant)

                print(f"  {variant:<8s} {sh:>7.3f} {cagr:>+8.1%} {mdd:>+8.1%} {cal:>7.2f} {sh_std:>6.3f} {rebal:>6.1f}  {desc}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
