#!/usr/bin/env python3
"""V17 + RWA 방어 유니버스 백테스트.

코인 Risk-Off 시 주식 방어자산과 동일한 구조:
방어 유니버스 중 6M return Top N (양수만) → EW, 전부 음수면 현금.

방어 유니버스: PAXG(금), XAUT(금), KAG(은), ONDO(국채)
기간: 2024-01 ~ 2025-06

Usage:
  python3 backtest_rwa_defense.py
"""

import sys, os, time
import numpy as np
import pandas as pd
import yfinance as yf
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, load_all_prices, filter_universe
from coin_helpers import B, ANCHOR_DAYS
from backtest_official import run_coin_backtest


# ═══ 데이터 ═══
def load_rwa_assets():
    """방어 유니버스 자산 다운로드."""
    tickers = {
        'PAXG-USD': '금(Paxos)',
        'XAUT-USD': '금(Tether)',
        'KAG-USD': '은(Kinesis)',
        'ONDO-USD': '국채(Ondo)',
    }
    data = {}
    for t, desc in tickers.items():
        df = yf.download(t, start="2023-01-01", end="2025-12-31", progress=False)
        if len(df) > 0:
            s = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 0].squeeze()
            s = s.dropna()
            if len(s) > 0:
                data[t] = s
                print(f"  {t:<12} {desc:<15} {s.index[0].date()} ~ {s.index[-1].date()} ({len(s)}일)")
    return data


def select_defense(rwa_data, date, top_n=2, lookback=126):
    """방어 유니버스에서 6M return Top N (양수만) 선정."""
    rets = {}
    for ticker, series in rwa_data.items():
        idx = series.index.get_indexer([date], method='ffill')[0]
        if idx >= lookback and idx < len(series):
            ret = series.iloc[idx] / series.iloc[idx - lookback] - 1
            rets[ticker] = ret

    # 양수만, 높은 순
    positive = {k: v for k, v in rets.items() if v > 0}
    if not positive:
        return {}  # 전부 음수 → 현금

    sorted_picks = sorted(positive.items(), key=lambda x: x[1], reverse=True)[:top_n]
    n = len(sorted_picks)
    return {k: 1.0 / n for k, _ in sorted_picks}


def apply_defense_sleeve(eq, canary_hist, rwa_data, top_n=2, lookback=126,
                          use_sma=False, sma_period=30,
                          riskoff_weight=1.0):
    """V17 equity_curve의 Risk-Off 구간을 RWA 방어자산으로 교체."""
    canary_df = pd.DataFrame(canary_hist).set_index('Date')

    # 각 RWA 자산의 일간 수익률
    rwa_rets = {}
    for ticker, series in rwa_data.items():
        rwa_rets[ticker] = series.pct_change().fillna(0)

    dates = eq.index
    adj = np.empty(len(dates))
    adj[0] = eq.iloc[0]

    for i in range(1, len(dates)):
        date = dates[i]
        prev = adj[i - 1]
        v17_ret = eq.iloc[i] / eq.iloc[i - 1] - 1 if eq.iloc[i - 1] != 0 else 0

        cidx = canary_df.index.get_indexer([date], method='ffill')[0]
        is_off = not canary_df.iloc[cidx]['canary_on'] if 0 <= cidx < len(canary_df) else False

        if is_off:
            # 방어자산 선정
            picks = select_defense(rwa_data, date, top_n=top_n, lookback=lookback)

            if not picks:
                adj[i] = prev  # 현금
                continue

            # SMA 필터 (선정된 종목 중 SMA 위인 것만)
            if use_sma:
                filtered = {}
                for ticker, w in picks.items():
                    series = rwa_data.get(ticker)
                    if series is None:
                        continue
                    pidx = series.index.get_indexer([date], method='ffill')[0]
                    if pidx >= sma_period:
                        sma = series.iloc[pidx - sma_period + 1:pidx + 1].mean()
                        if series.iloc[pidx] > sma:
                            filtered[ticker] = w
                if filtered:
                    # 재정규화
                    total_w = sum(filtered.values())
                    picks = {k: v / total_w for k, v in filtered.items()}
                else:
                    adj[i] = prev  # SMA 아래 → 현금
                    continue

            # 수익률 계산
            day_ret = 0
            for ticker, w in picks.items():
                r = rwa_rets.get(ticker)
                if r is not None:
                    ridx = r.index.get_indexer([date], method='ffill')[0]
                    if 0 <= ridx < len(r):
                        day_ret += r.iloc[ridx] * w

            adj[i] = prev * (1 + day_ret * riskoff_weight)
        else:
            adj[i] = prev * (1 + v17_ret)

    return pd.Series(adj, index=dates)


def metrics(s):
    if s is None or len(s) < 2:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0}
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    dr = s.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (s / s.cummax() - 1).min()
    return {'Sharpe': sharpe, 'CAGR': cagr, 'MDD': mdd}


_GLOBAL = {}

def _run_one(args):
    base_d, start, end = args
    snap_days = tuple((base_d - 1 + j * 9) % 28 + 1 for j in range(3))
    p = B(**_GLOBAL['cfg'])
    p.start_date = start; p.end_date = end
    return base_d, run_coin_backtest(
        _GLOBAL['prices'], _GLOBAL['um'], snap_days,
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5, params_base=p)


SCENARIOS = {
    'A_baseline': dict(skip=True),

    # 방어 유니버스 Top N
    'B_top1': dict(top_n=1, lookback=126),
    'B_top2': dict(top_n=2, lookback=126),
    'B_top3': dict(top_n=3, lookback=126),

    # 모멘텀 lookback
    'C_3m': dict(top_n=2, lookback=63),
    'C_6m': dict(top_n=2, lookback=126),
    'C_12m': dict(top_n=2, lookback=252),

    # SMA 필터 추가
    'D_sma30': dict(top_n=2, lookback=126, use_sma=True, sma_period=30),
    'D_sma60': dict(top_n=2, lookback=126, use_sma=True, sma_period=60),

    # 비중
    'E_50pct': dict(top_n=2, lookback=126, riskoff_weight=0.5),
    'E_100pct': dict(top_n=2, lookback=126, riskoff_weight=1.0),

    # PAXG 단독 (비교용)
    'F_paxg_only': dict(top_n=1, lookback=126),  # PAXG가 6M return 1위면 단독
    'F_paxg_sma30': dict(top_n=1, lookback=126, use_sma=True, sma_period=30),
}


def main():
    global _GLOBAL
    t0 = time.time()
    print("=" * 90)
    print("V17 + RWA 방어 유니버스 (주식 방어자산 구조)")
    print("=" * 90)

    print("\n데이터 로딩...")
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))

    rwa_data = load_rwa_assets()
    print(f"  RWA 방어 유니버스: {list(rwa_data.keys())}")
    print(f"  완료 ({time.time()-t0:.1f}s)")

    _GLOBAL.update(prices=prices, um=um40, cfg=dict(
        sma_period=60, canary_band=1.0, health_sma=0, health_mom_short=30,
        selection='baseline', n_picks=5, weighting='WC', top_n=40, risk='G5'))

    PERIODS = [('2024-01-01', '2025-12-31')]

    for start, end in PERIODS:
        print(f"\n{'='*90}")
        print(f"  기간: {start} ~ {end}")
        print(f"{'='*90}")

        t1 = time.time()
        args = [(d, start, end) for d in ANCHOR_DAYS]
        with Pool(min(10, os.cpu_count() or 4)) as pool:
            v17 = dict(pool.map(_run_one, args))
        print(f"  V17 엔진 완료 ({time.time()-t1:.1f}s)")

        bl_sharpes = [v17[d]['metrics']['Sharpe'] for d in ANCHOR_DAYS]
        bl_cagrs = [v17[d]['metrics']['CAGR'] for d in ANCHOR_DAYS]
        bl_mdds = [v17[d]['metrics']['MDD'] for d in ANCHOR_DAYS]
        print(f"  Baseline: Sharpe {np.mean(bl_sharpes):.3f}, CAGR {np.mean(bl_cagrs):+.1%}, MDD {np.mean(bl_mdds):+.1%}")

        print(f"\n  {'시나리오':<20s} {'Sharpe':>7s} {'σ(Sh)':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
        print(f"  {'-'*60}")

        for name, kwargs in SCENARIOS.items():
            sharpes, cagrs, mdds = [], [], []

            for base_d in ANCHOR_DAYS:
                r = v17[base_d]

                if kwargs.get('skip'):
                    m = r['metrics']
                else:
                    kw = {k: v for k, v in kwargs.items() if k != 'skip'}
                    adj = apply_defense_sleeve(r['equity_curve'], r['canary_history'], rwa_data, **kw)
                    m = metrics(adj)

                sharpes.append(m['Sharpe'])
                cagrs.append(m['CAGR'])
                mdds.append(m['MDD'])

            s_mean = np.mean(sharpes)
            s_std = np.std(sharpes)
            c_mean = np.mean(cagrs)
            mdd_mean = np.mean(mdds)
            cal = c_mean / abs(mdd_mean) if mdd_mean != 0 else 0

            marker = " <-- 현재" if name == 'A_baseline' else ""
            print(f"  {name:<20s} {s_mean:>7.3f} {s_std:>7.3f} {c_mean:>+8.1%} {mdd_mean:>+8.1%} {cal:>7.2f}{marker}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
