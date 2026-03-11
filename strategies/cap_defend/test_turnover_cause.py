#!/usr/bin/env python3
"""Analyze what causes turnover: health changes vs universe rank changes."""

import os, sys, warnings
from collections import defaultdict
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_ret, get_volatility, STABLECOINS
)

EXCLUDE_SYMBOLS = STABLECOINS | {'PAXG', 'XAUT', 'WBTC', 'USD1', 'USDE'}
TOP_N = 50
SMA_PERIOD = 150
VOL_CAP = 0.05
N_PICKS = 5


def filter_universe_topn(universe_map, top_n=TOP_N):
    filtered = {}
    for month_key, tickers in universe_map.items():
        clean = [t for t in tickers if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
        filtered[month_key] = clean[:top_n]
    return filtered


def get_universe_for_date(universe_map, date):
    month_key = date.strftime('%Y-%m') + '-01'
    for mk in sorted(universe_map.keys(), reverse=True):
        if mk <= month_key:
            return universe_map[mk]
    if universe_map:
        return list(universe_map.values())[0]
    return []


def check_canary(prices, global_idx):
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < SMA_PERIOD: return False
    return close.iloc[-1] > close.rolling(SMA_PERIOD).mean().iloc[-1]


def check_health(ticker, prices, global_idx):
    if ticker not in prices: return False, {}
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90: return False, {}
    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)
    details = {
        'price_above_sma30': cur > sma30,
        'mom21_positive': mom21 > 0 if mom21 is not None else False,
        'vol90_ok': vol90 is not None and vol90 <= VOL_CAP,
        'sma30': sma30, 'mom21': mom21, 'vol90': vol90
    }
    passed = cur > sma30 and mom21 is not None and mom21 > 0 and vol90 is not None and vol90 <= VOL_CAP
    return passed, details


def get_healthy_picks(prices, universe_map, date, global_idx):
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        passed, _ = check_health(ticker, prices, global_idx)
        if passed:
            healthy.append(ticker)
    return healthy[:N_PICKS]


def main():
    print("Loading data...")
    universe_map = load_universe()
    filtered_map = filter_universe_topn(universe_map, TOP_N)

    all_tickers = set()
    for mt in filtered_map.values():
        for t in mt:
            all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    btc = prices.get('BTC-USD')
    all_dates = btc.index[(btc.index >= '2018-01-01') & (btc.index <= '2025-12-31')]

    prev_picks = []
    prev_month = None
    change_events = []
    total_change_days = 0
    cause_counts = defaultdict(int)  # health_fail, health_gain, rank_change

    print("=" * 130)
    print("  TURNOVER CAUSE ANALYSIS: 매일 이상적 포트폴리오(shadow) 변경 원인")
    print("=" * 130)
    print(f"\n  {'Date':<12} {'Canary':<7} {'Prev→New picks':<60} {'원인':<40}")
    print(f"  {'-'*125}")

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        if not risk_on:
            if prev_picks:
                # Canary off
                change_events.append((date, 'CANARY_OFF', prev_picks, []))
            prev_picks = []
            prev_month = current_month
            continue

        today_picks = get_healthy_picks(prices, filtered_map, date, global_idx)

        if set(today_picks) != set(prev_picks) and prev_picks:
            total_change_days += 1
            dropped = set(prev_picks) - set(today_picks)
            added = set(today_picks) - set(prev_picks)

            # Determine cause for each dropped coin
            causes = []
            for t in dropped:
                passed, details = check_health(t, prices, global_idx)
                if not passed:
                    # Which health condition failed?
                    fails = []
                    if not details.get('price_above_sma30'): fails.append('SMA30')
                    if not details.get('mom21_positive'): fails.append('Mom21')
                    if not details.get('vol90_ok'): fails.append(f"Vol{details.get('vol90',0):.1%}")
                    cause = f"{t.replace('-USD','')} 헬스탈락({','.join(fails)})"
                    cause_counts['health_fail'] += 1
                else:
                    # Coin is still healthy but pushed out by higher-ranked coin
                    cause = f"{t.replace('-USD','')} 순위밀림"
                    cause_counts['rank_pushed'] += 1
                causes.append(cause)

            for t in added:
                passed, details = check_health(t, prices, global_idx)
                if passed:
                    # Was it previously unhealthy or newly ranked higher?
                    prev_passed, _ = check_health(t, prices, global_idx - 1) if global_idx > 0 else (False, {})
                    if not prev_passed:
                        cause_counts['health_gain'] += 1
                    else:
                        cause_counts['rank_rise'] += 1

            n_changed = len(dropped)
            turnover_pct = n_changed / N_PICKS * 100

            prev_str = ','.join(t.replace('-USD','') for t in prev_picks)
            new_str = ','.join(t.replace('-USD','') for t in today_picks)
            cause_str = '; '.join(causes[:2])

            # Only print significant changes or first 80
            if total_change_days <= 80 or turnover_pct >= 40:
                uni_change = "📅월초" if is_month_change else ""
                print(f"  {date.strftime('%Y-%m-%d'):<12} {'ON':<7} {prev_str} → {new_str:<28} {cause_str} {uni_change}")

        prev_picks = today_picks if today_picks else prev_picks
        prev_month = current_month

    print(f"\n  ... (총 {total_change_days}일 picks 변경)")

    # ═══ Summary ═══
    print("\n" + "=" * 80)
    print("  CAUSE SUMMARY")
    print("=" * 80)
    total = sum(cause_counts.values())
    print(f"\n  총 변경 이벤트: {total}")
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        label = {
            'health_fail': '헬스체크 탈락 (보유코인이 건강 조건 실패)',
            'health_gain': '헬스체크 통과 (새 코인이 건강 조건 달성)',
            'rank_pushed': '순위 밀림 (건강하지만 상위코인에 밀려남)',
            'rank_rise': '순위 상승 (건강했지만 이제 Top5에 진입)',
        }.get(cause, cause)
        print(f"  {label:<50} {count:>5} ({pct:>5.1f}%)")

    # ═══ Which health condition fails most? ═══
    print("\n" + "=" * 80)
    print("  HEALTH CONDITION FAILURE ANALYSIS")
    print("=" * 80)

    # Check all coins on all risk-on days for which condition is most volatile
    sma30_flips = 0
    mom21_flips = 0
    vol90_flips = 0
    total_checks = 0

    prev_health = {}
    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        if not risk_on:
            prev_health.clear()
            continue

        uni_tickers = get_universe_for_date(filtered_map, date)
        uni_clean = [t.replace('-USD', '') for t in uni_tickers
                    if t.replace('-USD', '') not in EXCLUDE_SYMBOLS][:20]  # Top 20 for efficiency

        for sym in uni_clean:
            ticker = f"{sym}-USD"
            _, details = check_health(ticker, prices, global_idx)
            if not details:
                continue

            key = ticker
            if key in prev_health:
                pd_prev = prev_health[key]
                total_checks += 1
                if details.get('price_above_sma30') != pd_prev.get('price_above_sma30'):
                    sma30_flips += 1
                if details.get('mom21_positive') != pd_prev.get('mom21_positive'):
                    mom21_flips += 1
                if details.get('vol90_ok') != pd_prev.get('vol90_ok'):
                    vol90_flips += 1

            prev_health[key] = details

    print(f"\n  총 체크 횟수: {total_checks:,}")
    print(f"  SMA30 교차 (Price ⋛ SMA30):  {sma30_flips:>6} flips ({sma30_flips/max(1,total_checks)*100:.2f}%)")
    print(f"  Mom21 부호변경:               {mom21_flips:>6} flips ({mom21_flips/max(1,total_checks)*100:.2f}%)")
    print(f"  Vol90 임계값 교차 (≤5%):       {vol90_flips:>6} flips ({vol90_flips/max(1,total_checks)*100:.2f}%)")


if __name__ == '__main__':
    main()
