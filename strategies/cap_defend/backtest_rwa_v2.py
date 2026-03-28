#!/usr/bin/env python3
"""V17 + RWA 방어자산 — 실전 검증 백테스트.

방어 유니버스: PAXG(금) + KAG(은)
재조정: 앵커일(Day 1/11/21)만 — 코인 전략과 동일
SMA 히스테리시스: 0~3%
EW + 소프트캡: 20~50%
거래비용: 0.4%
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


def load_rwa():
    data = {}
    for t in ['PAXG-USD', 'KAG-USD']:
        df = yf.download(t, start="2023-01-01", end="2025-12-31", progress=False)
        s = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 0].squeeze()
        s = s.dropna()
        if len(s) > 0:
            data[t] = s
            print(f"  {t}: {s.index[0].date()} ~ {s.index[-1].date()} ({len(s)}일)")
    return data


def select_defense_at_anchor(rwa_data, date, lookback=126, sma_period=30, hyst_pct=0.0, cap=0.5):
    """앵커일에 방어자산 선정 — 6M return 양수 + SMA 필터 + EW + 소프트캡."""
    picks = {}
    for ticker, series in rwa_data.items():
        idx = series.index.get_indexer([date], method='ffill')[0]
        if idx < max(lookback, sma_period):
            continue

        # 6M return > 0
        ret_6m = series.iloc[idx] / series.iloc[idx - lookback] - 1
        if ret_6m <= 0:
            continue

        # SMA 필터 + 히스테리시스
        sma = series.iloc[idx - sma_period + 1:idx + 1].mean()
        cur = series.iloc[idx]
        threshold = sma * (1 + hyst_pct)  # 진입: SMA × (1+hyst) 위
        if cur <= threshold:
            continue

        picks[ticker] = ret_6m

    if not picks:
        return {}

    # EW + 소프트캡
    n = len(picks)
    weights = {t: 1.0 / n for t in picks}

    # 소프트캡 적용
    if cap < 1.0:
        capped = {}
        excess = 0
        for t, w in weights.items():
            if w > cap:
                excess += w - cap
                capped[t] = cap
            else:
                capped[t] = w
        # excess를 cap 안 넘는 종목에 재분배
        uncapped = [t for t, w in capped.items() if w < cap]
        if uncapped and excess > 0:
            per = excess / len(uncapped)
            for t in uncapped:
                capped[t] = min(capped[t] + per, cap)
        weights = capped

    return weights


def apply_defense(eq, canary_hist, rwa_data, snap_days,
                  lookback=126, sma_period=30, hyst_pct=0.0, cap=0.5,
                  riskoff_weight=1.0, tx_cost=0.004):
    """V17 equity에 방어자산 적용. 앵커일에만 재조정."""
    canary_df = pd.DataFrame(canary_hist).set_index('Date')
    rwa_rets = {t: s.pct_change().fillna(0) for t, s in rwa_data.items()}

    dates = eq.index
    adj = np.empty(len(dates))
    adj[0] = eq.iloc[0]

    # 현재 방어 포지션 (앵커에서만 갱신)
    defense_weights = {}
    prev_defense = {}

    for i in range(1, len(dates)):
        date = dates[i]
        prev = adj[i - 1]
        v17_ret = eq.iloc[i] / eq.iloc[i - 1] - 1 if eq.iloc[i - 1] != 0 else 0

        cidx = canary_df.index.get_indexer([date], method='ffill')[0]
        is_off = not canary_df.iloc[cidx]['canary_on'] if 0 <= cidx < len(canary_df) else False

        if is_off:
            # 앵커일 체크 — 코인과 동일 (Day 1/11/21)
            day = date.day
            month_key = date.strftime('%Y-%m')

            is_anchor = False
            for sd in snap_days:
                if day >= sd:
                    is_anchor = True  # 해당 앵커 이후

            # 앵커일이면 방어자산 재선정
            # 단순화: 매월 앵커일(1,11,21)에만 재선정
            if day in snap_days or (i > 0 and dates[i-1].month != date.month):
                new_weights = select_defense_at_anchor(
                    rwa_data, date, lookback, sma_period, hyst_pct, cap)

                # 거래비용: 포지션 변경 시
                if new_weights != defense_weights:
                    # turnover 계산
                    all_tickers = set(list(defense_weights.keys()) + list(new_weights.keys()))
                    turnover = sum(abs(new_weights.get(t, 0) - defense_weights.get(t, 0)) for t in all_tickers) / 2
                    tx = turnover * tx_cost * riskoff_weight
                    prev = prev * (1 - tx)
                    defense_weights = new_weights

            # 방어 수익률 계산
            if defense_weights:
                day_ret = 0
                for ticker, w in defense_weights.items():
                    r = rwa_rets.get(ticker)
                    if r is not None:
                        ridx = r.index.get_indexer([date], method='ffill')[0]
                        if 0 <= ridx < len(r):
                            day_ret += r.iloc[ridx] * w
                adj[i] = prev * (1 + day_ret * riskoff_weight)
            else:
                adj[i] = prev  # 현금

            # SMA 이탈 체크 (히스테리시스 — 매일 확인하되 앵커에서만 매매)
            # 여기서는 앵커일에만 재선정하므로 자동으로 처리됨

        else:
            # Risk-On
            defense_weights = {}  # Risk-On 전환 시 방어 포지션 청산
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
    return base_d, snap_days, run_coin_backtest(
        _GLOBAL['prices'], _GLOBAL['um'], snap_days,
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5, params_base=p)


def run_scenarios(v17_results, rwa_data, scenarios, label):
    print(f"\n  --- {label} ---")
    print(f"  {'이름':<30s} {'Sharpe':>7s} {'σ(Sh)':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
    print(f"  {'-'*67}")

    results = {}
    for name, kwargs in scenarios.items():
        sharpes, cagrs, mdds = [], [], []
        for base_d, (snap_days, r) in v17_results.items():
            if kwargs.get('skip'):
                m = r['metrics']
            else:
                kw = {k: v for k, v in kwargs.items() if k != 'skip'}
                adj = apply_defense(r['equity_curve'], r['canary_history'], rwa_data,
                                    snap_days, **kw)
                m = metrics(adj)
            sharpes.append(m['Sharpe']); cagrs.append(m['CAGR']); mdds.append(m['MDD'])

        s, ss, c, md = np.mean(sharpes), np.std(sharpes), np.mean(cagrs), np.mean(mdds)
        cal = c / abs(md) if md != 0 else 0
        mk = " <--" if name == 'baseline' else ""
        print(f"  {name:<30s} {s:>7.3f} {ss:>7.3f} {c:>+8.1%} {md:>+8.1%} {cal:>7.2f}{mk}")
        results[name] = {'Sharpe': s, 'CAGR': c, 'MDD': md, 'Calmar': cal}
    return results


def main():
    global _GLOBAL
    t0 = time.time()
    print("=" * 90)
    print("V17 + RWA 방어 (PAXG+KAG, 앵커일 재조정, tx 0.4%)")
    print("=" * 90)

    print("\n데이터 로딩...")
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))
    rwa_data = load_rwa()
    print(f"  완료 ({time.time()-t0:.1f}s)")

    _GLOBAL.update(prices=prices, um=um40, cfg=dict(
        sma_period=60, canary_band=1.0, health_sma=0, health_mom_short=30,
        selection='baseline', n_picks=5, weighting='WC', top_n=40, risk='G5'))

    for start, end in [('2024-01-01', '2025-12-31'), ('2020-01-01', '2025-12-31')]:
        print(f"\n{'='*90}")
        print(f"  기간: {start} ~ {end}")
        print(f"{'='*90}")

        t1 = time.time()
        args = [(d, start, end) for d in ANCHOR_DAYS]
        with Pool(min(10, os.cpu_count() or 4)) as pool:
            raw = pool.map(_run_one, args)
        v17 = {d: (snaps, r) for d, snaps, r in raw}
        print(f"  V17 엔진 완료 ({time.time()-t1:.1f}s)")

        bl = [r['metrics'] for _, (_, r) in v17.items()]
        print(f"  Baseline: Sharpe {np.mean([m['Sharpe'] for m in bl]):.3f}, CAGR {np.mean([m['CAGR'] for m in bl]):+.1%}, MDD {np.mean([m['MDD'] for m in bl]):+.1%}")

        # A. SMA 히스테리시스
        hyst_scenarios = {'baseline': dict(skip=True)}
        for h in [0.0, 0.005, 0.01, 0.02, 0.03]:
            hyst_scenarios[f'hyst_{h:.1%}'] = dict(sma_period=30, hyst_pct=h, riskoff_weight=1.0, cap=0.5)
        run_scenarios(v17, rwa_data, hyst_scenarios, "A. SMA(30) 히스테리시스")

        # B. 소프트캡
        cap_scenarios = {'baseline': dict(skip=True)}
        for c in [0.20, 0.30, 0.40, 0.50, 1.0]:
            cap_scenarios[f'cap_{c:.0%}'] = dict(sma_period=30, hyst_pct=0.01, riskoff_weight=1.0, cap=c)
        run_scenarios(v17, rwa_data, cap_scenarios, "B. 소프트캡 (EW)")

        # C. Risk-Off 비중
        weight_scenarios = {'baseline': dict(skip=True)}
        for w in [0.3, 0.5, 0.7, 1.0]:
            weight_scenarios[f'weight_{w:.0%}'] = dict(sma_period=30, hyst_pct=0.01, riskoff_weight=w, cap=0.5)
        run_scenarios(v17, rwa_data, weight_scenarios, "C. Risk-Off 비중")

        # D. SMA 기간
        sma_scenarios = {'baseline': dict(skip=True)}
        for p in [20, 30, 40, 50, 60]:
            sma_scenarios[f'sma_{p}'] = dict(sma_period=p, hyst_pct=0.01, riskoff_weight=1.0, cap=0.5)
        run_scenarios(v17, rwa_data, sma_scenarios, "D. SMA 기간")

        # E. 최종 복합
        final_scenarios = {
            'baseline': dict(skip=True),
            'best_conservative': dict(sma_period=30, hyst_pct=0.01, riskoff_weight=0.5, cap=0.50),
            'best_balanced': dict(sma_period=30, hyst_pct=0.01, riskoff_weight=0.7, cap=0.50),
            'best_aggressive': dict(sma_period=30, hyst_pct=0.01, riskoff_weight=1.0, cap=0.50),
            'best_tight': dict(sma_period=30, hyst_pct=0.02, riskoff_weight=0.7, cap=0.40),
        }
        run_scenarios(v17, rwa_data, final_scenarios, "E. 최종 복합")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
