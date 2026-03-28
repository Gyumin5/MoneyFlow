#!/usr/bin/env python3
"""V17 + RWA Gold Sleeve 백테스트 (병렬화).

V17 엔진을 앵커별 1회씩만 실행 → equity_curve + canary_history 반환
→ 16개 RWA 시나리오는 후처리로 적용 (빠름)

Usage:
  python3 backtest_rwa_gold.py
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


# ═══ 데이터 로드 ═══
def load_paxg():
    df = yf.download("PAXG-USD", start="2019-01-01", end="2025-12-31", progress=False)
    return (df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 0].squeeze())

def load_eem():
    df = yf.download("EEM", start="2019-01-01", end="2025-12-31", progress=False)
    s = df['Close'].squeeze() if 'Close' in df.columns else df.iloc[:, 0].squeeze()
    return s, s.rolling(200).mean()


# ═══ RWA Sleeve 후처리 ═══
def apply_rwa(eq, canary_hist, paxg, eem, eem_sma,
              paxg_weight=1.0, health_params=None,
              use_equity_gate=False, equity_gate_inverse=False,
              use_paxg_sma=False, paxg_sma_period=60,
              use_dd_exit=False, dd_lookback=30, dd_threshold=-0.10,
              risk_on_paxg_pct=0.0):

    canary_df = pd.DataFrame(canary_hist).set_index('Date')
    paxg_rets = paxg.pct_change().fillna(0)
    dates = eq.index
    adj = np.empty(len(dates))
    adj[0] = eq.iloc[0]

    for i in range(1, len(dates)):
        date = dates[i]
        prev = adj[i - 1]
        v17_ret = eq.iloc[i] / eq.iloc[i - 1] - 1 if eq.iloc[i - 1] != 0 else 0

        # Risk-Off?
        cidx = canary_df.index.get_indexer([date], method='ffill')[0]
        is_off = not canary_df.iloc[cidx]['canary_on'] if 0 <= cidx < len(canary_df) else False

        if is_off:
            pidx = paxg_rets.index.get_indexer([date], method='ffill')[0]
            pr = paxg_rets.iloc[pidx] if 0 <= pidx < len(paxg_rets) else 0
            invest = True

            # 헬스
            if health_params and invest:
                ms, ml = health_params
                pi = paxg.index.get_indexer([date], method='ffill')[0]
                if pi >= ml and paxg.iloc[pi - ms] > 0 and paxg.iloc[pi - ml] > 0:
                    invest = (paxg.iloc[pi] / paxg.iloc[pi - ms] - 1 > 0) and (paxg.iloc[pi] / paxg.iloc[pi - ml] - 1 > 0)
                else:
                    invest = False

            # 주식 게이트
            if use_equity_gate and invest:
                ei = eem.index.get_indexer([date], method='ffill')[0]
                if 0 <= ei < len(eem_sma) and not pd.isna(eem_sma.iloc[ei]):
                    eem_on = eem.iloc[ei] > eem_sma.iloc[ei]
                    invest = (not eem_on) if equity_gate_inverse else eem_on

            # PAXG SMA
            if use_paxg_sma and invest:
                pi = paxg.index.get_indexer([date], method='ffill')[0]
                if pi >= paxg_sma_period:
                    invest = paxg.iloc[pi] > paxg.iloc[pi - paxg_sma_period + 1:pi + 1].mean()
                else:
                    invest = False

            # DD Exit
            if use_dd_exit and invest:
                pi = paxg.index.get_indexer([date], method='ffill')[0]
                if pi >= dd_lookback:
                    peak = paxg.iloc[pi - dd_lookback:pi + 1].max()
                    if peak > 0 and paxg.iloc[pi] / peak - 1 <= dd_threshold:
                        invest = False

            adj[i] = prev * (1 + pr * paxg_weight) if invest else prev
        else:
            if risk_on_paxg_pct > 0:
                pidx = paxg_rets.index.get_indexer([date], method='ffill')[0]
                pr = paxg_rets.iloc[pidx] if 0 <= pidx < len(paxg_rets) else 0
                adj[i] = prev * (1 + v17_ret * (1 - risk_on_paxg_pct) + pr * risk_on_paxg_pct)
            else:
                adj[i] = prev * (1 + v17_ret)

    return pd.Series(adj, index=dates)


def metrics(s):
    if s is None or len(s) < 2: return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0}
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    dr = s.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (s / s.cummax() - 1).min()
    return {'Sharpe': sharpe, 'CAGR': cagr, 'MDD': mdd}


# ═══ 병렬 V17 실행 ═══
_GLOBAL = {}

def _run_one_anchor(args):
    base_d, start, end = args
    snap_days = tuple((base_d - 1 + j * 9) % 28 + 1 for j in range(3))
    p = B(**_GLOBAL['cfg'])
    p.start_date = start
    p.end_date = end
    r = run_coin_backtest(
        _GLOBAL['prices'], _GLOBAL['um'], snap_days,
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5, params_base=p,
    )
    return base_d, r


SCENARIOS = {
    'A_baseline':       dict(),
    'B_full_gold':      dict(paxg_weight=1.0),
    'C_half_gold':      dict(paxg_weight=0.5),
    'C2_30pct':         dict(paxg_weight=0.3),
    'D_no_health':      dict(paxg_weight=1.0),
    'E_health_30_90':   dict(paxg_weight=1.0, health_params=(30, 90)),
    'F_health_63_126':  dict(paxg_weight=1.0, health_params=(63, 126)),
    'G_eq_gate_on':     dict(paxg_weight=1.0, health_params=(63, 126), use_equity_gate=True),
    'G2_eq_gate_off':   dict(paxg_weight=1.0, health_params=(63, 126), use_equity_gate=True, equity_gate_inverse=True),
    'H_paxg_sma60':     dict(paxg_weight=1.0, use_paxg_sma=True, paxg_sma_period=60),
    'H2_paxg_sma120':   dict(paxg_weight=1.0, use_paxg_sma=True, paxg_sma_period=120),
    'I_riskon_20':      dict(paxg_weight=1.0, health_params=(63, 126), risk_on_paxg_pct=0.20),
    'J_riskon_10':      dict(paxg_weight=1.0, health_params=(63, 126), risk_on_paxg_pct=0.10),
    'K_dd_10':          dict(paxg_weight=1.0, health_params=(63, 126), use_dd_exit=True, dd_lookback=30, dd_threshold=-0.10),
    'K2_dd_15':         dict(paxg_weight=1.0, health_params=(63, 126), use_dd_exit=True, dd_lookback=30, dd_threshold=-0.15),
    'M_always_gold':    dict(paxg_weight=1.0),
}


def main():
    global _GLOBAL
    t0 = time.time()
    print("=" * 90)
    print("V17 + RWA Gold Sleeve (병렬 10-anchor)")
    print("=" * 90)

    print("\n데이터 로딩...")
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))

    paxg = load_paxg()
    eem, eem_sma = load_eem()
    print(f"  PAXG: {paxg.index[0].date()} ~ {paxg.index[-1].date()}")
    print(f"  완료 ({time.time()-t0:.1f}s)")

    _GLOBAL['prices'] = prices
    _GLOBAL['um'] = um40
    _GLOBAL['cfg'] = dict(
        sma_period=60, canary_band=1.0, health_sma=0, health_mom_short=30,
        selection='baseline', n_picks=5, weighting='WC', top_n=40, risk='G5',
    )

    PERIODS = [('2020-01-01', '2025-06-30'), ('2021-01-01', '2025-06-30')]

    for start, end in PERIODS:
        print(f"\n{'='*90}")
        print(f"  기간: {start} ~ {end}")
        print(f"{'='*90}")

        # V17 엔진을 10-anchor 병렬 실행 (1회만)
        t1 = time.time()
        args = [(d, start, end) for d in ANCHOR_DAYS]
        with Pool(min(10, os.cpu_count() or 4)) as pool:
            v17_results = dict(pool.map(_run_one_anchor, args))
        print(f"  V17 엔진 {len(ANCHOR_DAYS)}앵커 완료 ({time.time()-t1:.1f}s)")

        # 시나리오별 후처리
        print(f"  {'시나리오':<20s} {'Sharpe':>7s} {'σ(Sh)':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
        print(f"  {'-'*60}")

        for name, kwargs in SCENARIOS.items():
            sharpes, cagrs, mdds = [], [], []

            for base_d in ANCHOR_DAYS:
                r = v17_results[base_d]
                eq = r.get('equity_curve')
                ch = r.get('canary_history')

                if name == 'A_baseline':
                    m = r['metrics']
                else:
                    adj = apply_rwa(eq, ch, paxg, eem, eem_sma, **kwargs)
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
