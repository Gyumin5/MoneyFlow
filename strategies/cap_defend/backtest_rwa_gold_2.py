#!/usr/bin/env python3
"""V17 + RWA Gold Sleeve — 2차 수치 그리드.

1차에서 유망한 축만 확장:
A. 헬스 Mom 조합 (9개)
B. PAXG SMA 기간 (8개)
C. 복합: 헬스+SMA (3개)
D. Risk-On 비중 (6개)
E. 최종 복합 (4개)

V17 엔진 결과 재사용 → 후처리만.
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
from backtest_rwa_gold import apply_rwa, metrics, load_paxg, load_eem

_GLOBAL = {}

def _run_one_anchor(args):
    base_d, start, end = args
    snap_days = tuple((base_d - 1 + j * 9) % 28 + 1 for j in range(3))
    p = B(**_GLOBAL['cfg'])
    p.start_date = start; p.end_date = end
    return base_d, run_coin_backtest(
        _GLOBAL['prices'], _GLOBAL['um'], snap_days,
        dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
        drift_threshold=0.10, post_flip_delay=5, params_base=p)


def run_grid(v17_results, paxg, eem, eem_sma, scenarios, label):
    print(f"\n  --- {label} ---")
    print(f"  {'이름':<28s} {'Sharpe':>7s} {'σ(Sh)':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
    print(f"  {'-'*65}")

    best = None
    for name, kwargs in scenarios.items():
        sharpes, cagrs, mdds = [], [], []
        for base_d in ANCHOR_DAYS:
            r = v17_results[base_d]
            adj = apply_rwa(r['equity_curve'], r['canary_history'], paxg, eem, eem_sma, **kwargs)
            m = metrics(adj)
            sharpes.append(m['Sharpe']); cagrs.append(m['CAGR']); mdds.append(m['MDD'])
        s, ss, c, md = np.mean(sharpes), np.std(sharpes), np.mean(cagrs), np.mean(mdds)
        cal = c / abs(md) if md != 0 else 0
        print(f"  {name:<28s} {s:>7.3f} {ss:>7.3f} {c:>+8.1%} {md:>+8.1%} {cal:>7.2f}")
        if best is None or cal > best[1]:
            best = (name, cal, s, c, md)
    print(f"  >>> Best: {best[0]} (Calmar {best[1]:.2f})")
    return best


def main():
    global _GLOBAL
    t0 = time.time()
    print("=" * 90)
    print("V17 + RWA Gold — 2차 수치 그리드")
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
    print(f"  완료 ({time.time()-t0:.1f}s)")

    _GLOBAL.update(prices=prices, um=um40, cfg=dict(
        sma_period=60, canary_band=1.0, health_sma=0, health_mom_short=30,
        selection='baseline', n_picks=5, weighting='WC', top_n=40, risk='G5'))

    for start, end in [('2020-01-01', '2025-06-30'), ('2021-01-01', '2025-06-30')]:
        print(f"\n{'='*90}")
        print(f"  기간: {start} ~ {end}")
        print(f"{'='*90}")

        t1 = time.time()
        args = [(d, start, end) for d in ANCHOR_DAYS]
        with Pool(min(10, os.cpu_count() or 4)) as pool:
            v17 = dict(pool.map(_run_one_anchor, args))
        print(f"  V17 엔진 완료 ({time.time()-t1:.1f}s)")

        # Baseline
        bl_sharpes = [v17[d]['metrics']['Sharpe'] for d in ANCHOR_DAYS]
        bl_cagrs = [v17[d]['metrics']['CAGR'] for d in ANCHOR_DAYS]
        bl_mdds = [v17[d]['metrics']['MDD'] for d in ANCHOR_DAYS]
        print(f"\n  Baseline: Sharpe {np.mean(bl_sharpes):.3f}, CAGR {np.mean(bl_cagrs):+.1%}, MDD {np.mean(bl_mdds):+.1%}")

        # A. 헬스 Mom 조합
        health_grid = {}
        for ms, ml in [(21,63),(21,90),(30,63),(30,90),(30,126),(42,90),(42,126),(63,90),(63,126)]:
            health_grid[f'H({ms},{ml})'] = dict(paxg_weight=1.0, health_params=(ms, ml))
        best_health = run_grid(v17, paxg, eem, eem_sma, health_grid, "A. 헬스 Mom 조합")

        # B. PAXG SMA 기간
        sma_grid = {}
        for p in [30, 40, 50, 60, 70, 80, 90, 120]:
            sma_grid[f'SMA({p})'] = dict(paxg_weight=1.0, use_paxg_sma=True, paxg_sma_period=p)
        best_sma = run_grid(v17, paxg, eem, eem_sma, sma_grid, "B. PAXG SMA 기간")

        # C. 복합: 헬스+SMA
        # best health params 추출
        bh_name = best_health[0]  # e.g. 'H(30,90)'
        bh_params = eval(bh_name.replace('H',''))  # (30,90)
        bs_name = best_sma[0]
        bs_period = int(bs_name.replace('SMA(','').replace(')',''))

        combo_grid = {
            f'{bh_name}+{bs_name}': dict(paxg_weight=1.0, health_params=bh_params, use_paxg_sma=True, paxg_sma_period=bs_period),
            f'H(30,90)+SMA(60)': dict(paxg_weight=1.0, health_params=(30,90), use_paxg_sma=True, paxg_sma_period=60),
            f'H(63,126)+SMA(60)': dict(paxg_weight=1.0, health_params=(63,126), use_paxg_sma=True, paxg_sma_period=60),
            f'H(30,90)+SMA(90)': dict(paxg_weight=1.0, health_params=(30,90), use_paxg_sma=True, paxg_sma_period=90),
        }
        best_combo = run_grid(v17, paxg, eem, eem_sma, combo_grid, "C. 복합: 헬스+SMA")

        # D. Risk-On 혼합 비중 (best 헬스 기반)
        riskon_grid = {}
        for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            riskon_grid[f'RiskOn {pct:.0%}'] = dict(paxg_weight=1.0, health_params=bh_params, risk_on_paxg_pct=pct)
        best_riskon = run_grid(v17, paxg, eem, eem_sma, riskon_grid, "D. Risk-On 혼합 비중")

        # E. Risk-Off 비중 (best 헬스 기반)
        riskoff_grid = {}
        for w in [0.3, 0.5, 0.7, 1.0]:
            riskoff_grid[f'RiskOff {w:.0%}'] = dict(paxg_weight=w, health_params=bh_params)
        run_grid(v17, paxg, eem, eem_sma, riskoff_grid, "E. Risk-Off PAXG 비중")

        # F. 최종 복합 (best들 조합)
        bc_name = best_combo[0]
        bc_kwargs = combo_grid.get(bc_name, combo_grid[list(combo_grid.keys())[0]])
        br_pct = float(best_riskon[0].split()[-1].replace('%','')) / 100

        final_grid = {
            f'FINAL: {bc_name}+RiskOn{br_pct:.0%}': {**bc_kwargs, 'risk_on_paxg_pct': br_pct},
            f'FINAL: {bc_name}+RiskOn10%': {**bc_kwargs, 'risk_on_paxg_pct': 0.10},
            f'FINAL: {bc_name}+RiskOn20%': {**bc_kwargs, 'risk_on_paxg_pct': 0.20},
            f'FINAL: {bh_name}+RiskOn{br_pct:.0%}': dict(paxg_weight=1.0, health_params=bh_params, risk_on_paxg_pct=br_pct),
        }
        run_grid(v17, paxg, eem, eem_sma, final_grid, "F. 최종 복합")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
