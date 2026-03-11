#!/usr/bin/env python3
"""K Top 5 × H Top 5 = 25 combination grid test.

K Top 5 (robustness-verified):
  K1: SMA(10)>SMA(65) crossover — best overall, broad plateau
  K2: SMA(60) single — best single SMA
  K3: SMA(10)>SMA(60) crossover
  K4: SMA(20)>SMA(60) crossover
  K5: 2/3 SMA(60,110)+Mom(120) hybrid vote

H Top 5:
  H1: H1 current (SMA30+Mom21+Vol5%+Mom90)
  H2: HM s30+m90+v5.0 (SMA30+Mom90+Vol5%, no mom21)
  H3: H1 vol4.5% (tighter vol cap)
  H4: HQ s30+v5.0 (SMA30+Vol5% only, simplest)
  H5: H5 vol accel (baseline+vol30≤vol90*1.5)
"""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ── K definitions ───────────────────────────────────────────────
K_CONFIGS = {
    'K1:10>65':   dict(canary='K9', cross_short=10, cross_long=65),
    'K2:SMA60':   dict(canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1),
    'K3:10>60':   dict(canary='K9', cross_short=10, cross_long=60),
    'K4:20>60':   dict(canary='K9', cross_short=20, cross_long=60),
    'K5:vote':    dict(canary='K8', vote_smas=(60,110), vote_moms=(120,), vote_threshold=2),
}

# ── H definitions ───────────────────────────────────────────────
H_CONFIGS = {
    'H1':          dict(health='H1', vol_cap=0.05),
    'HM:s30m90v5': dict(health='HM', health_sma=30, health_mom_long=90, vol_cap=0.05),
    'H1:v4.5%':   dict(health='H1', vol_cap=0.045),
    'HQ:s30v5':    dict(health='HQ', health_sma=30, vol_cap=0.05),
    'H5':          dict(health='H5', vol_cap=0.05),
}

K_NAMES = list(K_CONFIGS.keys())
H_NAMES = list(H_CONFIGS.keys())


def build_params(k_name, h_name):
    kc = K_CONFIGS[k_name]
    hc = H_CONFIGS[h_name]
    return Params(**{**kc, **hc})


def run_set(params_list, prices, universe, tx=0.004):
    full_params = []
    for p in params_list:
        full_params.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
            cross_short=p.cross_short, cross_long=p.cross_long,
            persist_period=p.persist_period, persist_months=p.persist_months,
            vol_cap=p.vol_cap,
            health_sma=p.health_sma, health_mom_short=p.health_mom_short,
            health_mom_long=p.health_mom_long, health_vol_window=p.health_vol_window,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, full_params)
    return results


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    # Build all 25 combinations
    combos = []
    combo_labels = []
    for k_name in K_NAMES:
        for h_name in H_NAMES:
            combos.append(build_params(k_name, h_name))
            combo_labels.append(f"{k_name} + {h_name}")

    t0 = time.time()
    results = run_set(combos, prices, universe)
    print(f"\n  25 combinations completed in {time.time()-t0:.1f}s")

    years = range(2018, 2026)

    # ══════════════════════════════════════════════════════════════
    # Grid: Sharpe
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  K × H Grid — Sharpe Ratio (tx=0.4%)")
    print(f"{'=' * 100}")

    print(f"\n  {'':>15}", end="")
    for h_name in H_NAMES:
        print(f" {h_name:>14}", end="")
    print(f" {'avg':>8}")
    print(f"  {'─' * 90}")

    idx = 0
    k_avgs = {}
    for k_name in K_NAMES:
        sharpes = []
        print(f"  {k_name:<15}", end="")
        for h_name in H_NAMES:
            m = results[idx]['metrics']
            s = m['Sharpe']
            sharpes.append(s)
            print(f" {s:>14.3f}", end="")
            idx += 1
        avg = sum(sharpes) / len(sharpes)
        k_avgs[k_name] = avg
        print(f" {avg:>8.3f}")

    # H averages
    print(f"  {'─' * 90}")
    print(f"  {'avg':<15}", end="")
    for j, h_name in enumerate(H_NAMES):
        vals = [results[i * len(H_NAMES) + j]['metrics']['Sharpe'] for i in range(len(K_NAMES))]
        avg = sum(vals) / len(vals)
        print(f" {avg:>14.3f}", end="")
    print()

    # ══════════════════════════════════════════════════════════════
    # Grid: CAGR
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  K × H Grid — CAGR (tx=0.4%)")
    print(f"{'=' * 100}")

    print(f"\n  {'':>15}", end="")
    for h_name in H_NAMES:
        print(f" {h_name:>14}", end="")
    print()
    print(f"  {'─' * 90}")

    idx = 0
    for k_name in K_NAMES:
        print(f"  {k_name:<15}", end="")
        for h_name in H_NAMES:
            m = results[idx]['metrics']
            print(f" {m['CAGR']:>+13.1%}", end="")
            idx += 1
        print()

    # ══════════════════════════════════════════════════════════════
    # Grid: MDD
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  K × H Grid — MDD (tx=0.4%)")
    print(f"{'=' * 100}")

    print(f"\n  {'':>15}", end="")
    for h_name in H_NAMES:
        print(f" {h_name:>14}", end="")
    print()
    print(f"  {'─' * 90}")

    idx = 0
    for k_name in K_NAMES:
        print(f"  {k_name:<15}", end="")
        for h_name in H_NAMES:
            m = results[idx]['metrics']
            print(f" {m['MDD']:>13.1%}", end="")
            idx += 1
        print()

    # ══════════════════════════════════════════════════════════════
    # Grid: Calmar (CAGR/MDD)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  K × H Grid — Calmar Ratio (CAGR/|MDD|)")
    print(f"{'=' * 100}")

    print(f"\n  {'':>15}", end="")
    for h_name in H_NAMES:
        print(f" {h_name:>14}", end="")
    print()
    print(f"  {'─' * 90}")

    idx = 0
    for k_name in K_NAMES:
        print(f"  {k_name:<15}", end="")
        for h_name in H_NAMES:
            m = results[idx]['metrics']
            calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
            print(f" {calmar:>14.2f}", end="")
            idx += 1
        print()

    # ══════════════════════════════════════════════════════════════
    # Top 10 overall
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print(f"  전체 Top 10 (Sharpe 기준)")
    print(f"{'=' * 130}")

    all_scored = []
    for i, (label, r) in enumerate(zip(combo_labels, results)):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        all_scored.append((label, m, r, calmar))

    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'K + H 조합':<35} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'Final':>12}")
    print(f"  {'─' * 85}")
    for i, (label, m, r, calmar) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {label:<35} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f} {m['Final']:>12,.0f}")

    # ══════════════════════════════════════════════════════════════
    # Top 10 by Calmar
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print(f"  전체 Top 10 (Calmar 기준 — 리스크 대비 수익)")
    print(f"{'=' * 130}")

    all_scored_calmar = sorted(all_scored, key=lambda x: x[3], reverse=True)
    print(f"\n  {'순위':>3} {'K + H 조합':<35} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'Final':>12}")
    print(f"  {'─' * 85}")
    for i, (label, m, r, calmar) in enumerate(all_scored_calmar[:10], 1):
        print(f"  {i:>3}. {label:<35} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f} {m['Final']:>12,.0f}")

    # ══════════════════════════════════════════════════════════════
    # Year-by-year for Top 5
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print(f"  Top 5 연도별 CAGR")
    print(f"{'=' * 130}")
    print(f"  {'K + H 조합':<35}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 105}")
    for label, m, r, calmar in all_scored[:5]:
        ym = r['yearly']
        row = f"  {label:<35}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)

    # ══════════════════════════════════════════════════════════════
    # Year-by-year MDD for Top 5
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Top 5 연도별 MDD")
    print(f"  {'K + H 조합':<35}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>7}")
    print(f"  {'─' * 105}")
    for label, m, r, calmar in all_scored[:5]:
        ym = r['yearly']
        row = f"  {label:<35}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['MDD']:>6.1%}"
        print(row)

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print(f"  종합 분석")
    print(f"{'=' * 130}")

    # Best K (averaged across all H)
    print(f"\n  K 전략 평균 Sharpe (모든 H에 걸친 평균):")
    for k_name in sorted(k_avgs.keys(), key=lambda x: k_avgs[x], reverse=True):
        bar = '█' * int(k_avgs[k_name] * 15)
        print(f"    {k_name:<15} {k_avgs[k_name]:.3f} {bar}")

    # Best H (averaged across all K)
    print(f"\n  H 전략 평균 Sharpe (모든 K에 걸친 평균):")
    h_avgs = {}
    for j, h_name in enumerate(H_NAMES):
        vals = [results[i * len(H_NAMES) + j]['metrics']['Sharpe'] for i in range(len(K_NAMES))]
        avg = sum(vals) / len(vals)
        h_avgs[h_name] = avg
    for h_name in sorted(h_avgs.keys(), key=lambda x: h_avgs[x], reverse=True):
        bar = '█' * int(h_avgs[h_name] * 15)
        print(f"    {h_name:<15} {h_avgs[h_name]:.3f} {bar}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
