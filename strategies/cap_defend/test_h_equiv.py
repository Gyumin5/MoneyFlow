#!/usr/bin/env python3
"""Test H equivalence: SMA vs Mom at same effective lookback.
   If SMA(30)≈Mom(15), then Mom(15)+Mom(90) should ≈ SMA(30)+Mom(90).
   Also test various short+long timeframe combos."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())


def K(**kw):
    return Params(canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1, **kw)


# ══════════════════════════════════════════════════════════════════
# Reference
# ══════════════════════════════════════════════════════════════════
REF = [
    ('H1 (현재)',            K(health='H1', vol_cap=0.05)),
    ('HM s30+m90+v5%',     K(health='HM', health_sma=30, health_mom_long=90, vol_cap=0.05)),
    ('Mom90만+v5%',         K(health='HL', health_mom_long=90, vol_cap=0.05)),
    ('SMA30만+v5%',         K(health='HQ', health_sma=30, vol_cap=0.05)),
    ('none',                K(health='none')),
]

# ══════════════════════════════════════════════════════════════════
# 1. SMA(short) + Mom(long) — 현재 구조
# ══════════════════════════════════════════════════════════════════
SMA_MOM = []
for sma_s in [10, 15, 20, 25, 30, 40]:
    for mom_l in [60, 90, 120]:
        SMA_MOM.append(
            (f'SMA{sma_s}+M{mom_l}+v5%',
             K(health='HM', health_sma=sma_s, health_mom_long=mom_l, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# 2. Mom(short) + Mom(long) — SMA 대신 Mom으로 대체
# ══════════════════════════════════════════════════════════════════
# HL only has mom_long, so I need a new health type for dual mom.
# Let me use HK which has health_mom_short + health_mom_long
# HK = cur > h_sma AND h_mom_s > 0 AND h_vol <= vol_cap [AND mom_long > 0]
# For mom-only, I need health_sma=1 to always pass SMA check.
# But SMA(1)=cur, and cur>cur is False. So I'll use health_sma=2 instead.
# Actually let me use a large number that always passes... or I can just test
# by setting health_sma to something very small.
# Actually the simplest: use HK with health_sma=2, which is almost always True.
MOM_MOM = []
for mom_s in [7, 10, 14, 21, 30]:
    for mom_l in [60, 90, 120]:
        if mom_s >= mom_l:
            continue
        MOM_MOM.append(
            (f'M{mom_s}+M{mom_l}+v5%',
             K(health='HK', health_sma=2, health_mom_short=mom_s,
               health_mom_long=mom_l, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# 3. SMA(short) + SMA(long) — 둘 다 SMA로
# ══════════════════════════════════════════════════════════════════
# HN = short SMA > long SMA + vol (crossover in health)
SMA_SMA = []
for sma_s in [10, 15, 20, 30]:
    for sma_l in [60, 80, 100]:
        if sma_s >= sma_l:
            continue
        SMA_SMA.append(
            (f'h:SMA{sma_s}>SMA{sma_l}+v5%',
             K(health='HN', health_mom_short=sma_s, health_sma=sma_l, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# 4. Equivalent lookback comparison
#    SMA(30)≈Mom(15), SMA(20)≈Mom(10), SMA(40)≈Mom(20)
# ══════════════════════════════════════════════════════════════════
EQUIV = [
    # ~15일 유효 + 90일 장기
    ('SMA30+M90+v5%',   K(health='HM', health_sma=30, health_mom_long=90, vol_cap=0.05)),
    ('M15+M90+v5%',     K(health='HK', health_sma=2, health_mom_short=15, health_mom_long=90, vol_cap=0.05)),
    # ~10일 유효 + 90일 장기
    ('SMA20+M90+v5%',   K(health='HM', health_sma=20, health_mom_long=90, vol_cap=0.05)),
    ('M10+M90+v5%',     K(health='HK', health_sma=2, health_mom_short=10, health_mom_long=90, vol_cap=0.05)),
    # ~20일 유효 + 90일 장기
    ('SMA40+M90+v5%',   K(health='HM', health_sma=40, health_mom_long=90, vol_cap=0.05)),
    ('M20+M90+v5%',     K(health='HK', health_sma=2, health_mom_short=20, health_mom_long=90, vol_cap=0.05)),
    # ~15일 유효 + 60일 장기
    ('SMA30+M60+v5%',   K(health='HM', health_sma=30, health_mom_long=60, vol_cap=0.05)),
    ('M15+M60+v5%',     K(health='HK', health_sma=2, health_mom_short=15, health_mom_long=60, vol_cap=0.05)),
]

ALL = REF + SMA_MOM + MOM_MOM + SMA_SMA + EQUIV


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
            cross_short=p.cross_short, cross_long=p.cross_long,
            vol_cap=p.vol_cap,
            health_sma=p.health_sma, health_mom_short=p.health_mom_short,
            health_mom_long=p.health_mom_long, health_vol_window=p.health_vol_window,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_section(title, strategies, results):
    print(f"\n  {title}")
    print(f"  {'─' * 75}")
    print(f"  {'조건':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 75}")
    best_s = max(r['metrics']['Sharpe'] for r in results) if results else 0
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = " ★" if m['Sharpe'] == best_s else ""
        print(f"  {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}{marker}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded, {len(ALL)} configs")

    t0 = time.time()
    results = run_set(ALL, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    idx = 0
    def get_slice(s):
        nonlocal idx
        r = results[idx:idx+len(s)]
        idx += len(s)
        return r

    r_ref = get_slice(REF)
    r_sm = get_slice(SMA_MOM)
    r_mm = get_slice(MOM_MOM)
    r_ss = get_slice(SMA_SMA)
    r_eq = get_slice(EQUIV)

    print(f"\n{'=' * 100}")
    print(f"  H 지표 등가 비교 — K=SMA(60) 고정")
    print(f"{'=' * 100}")

    print_section("기준", REF, r_ref)
    print_section("SMA(short) + Mom(long) + Vol5%", SMA_MOM, r_sm)
    print_section("Mom(short) + Mom(long) + Vol5%", MOM_MOM, r_mm)
    print_section("SMA crossover in H + Vol5%", SMA_SMA, r_ss)

    # ── Equivalent lookback comparison ──────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  등가 룩백 직접 비교: SMA(N) vs Mom(N/2)")
    print(f"{'=' * 100}")
    print(f"\n  {'유효 룩백':>10} {'SMA 조합':>20} {'Sharpe':>7}  │ {'Mom 조합':>20} {'Sharpe':>7}  │ {'차이':>7}")
    print(f"  {'─' * 85}")
    eq_pairs = [
        ('~15d + 90d', 0, 1),
        ('~10d + 90d', 2, 3),
        ('~20d + 90d', 4, 5),
        ('~15d + 60d', 6, 7),
    ]
    for label, i_sma, i_mom in eq_pairs:
        sma_name = EQUIV[i_sma][0]
        mom_name = EQUIV[i_mom][0]
        s_sma = r_eq[i_sma]['metrics']['Sharpe']
        s_mom = r_eq[i_mom]['metrics']['Sharpe']
        diff = s_sma - s_mom
        print(f"  {label:>10} {sma_name:>20} {s_sma:>7.3f}  │ {mom_name:>20} {s_mom:>7.3f}  │ {diff:>+6.3f}")

    # ── Grid: SMA+Mom vs Mom+Mom ────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  SMA+Mom vs Mom+Mom 그리드 (같은 유효 룩백)")
    print(f"{'=' * 100}")

    # Build lookups
    sm_lookup = {name: r['metrics']['Sharpe'] for (name, _), r in zip(SMA_MOM, r_sm)}
    mm_lookup = {name: r['metrics']['Sharpe'] for (name, _), r in zip(MOM_MOM, r_mm)}

    print(f"\n  SMA(short)+Mom(long) 그리드:")
    print(f"  {'':>15} {'Mom60':>8} {'Mom90':>8} {'Mom120':>8}")
    print(f"  {'─' * 42}")
    for sma_s in [10, 15, 20, 25, 30, 40]:
        print(f"  {'SMA'+str(sma_s):<15}", end="")
        for ml in [60, 90, 120]:
            key = f'SMA{sma_s}+M{ml}+v5%'
            if key in sm_lookup:
                print(f" {sm_lookup[key]:>8.3f}", end="")
            else:
                print(f" {'─':>8}", end="")
        print()

    print(f"\n  Mom(short)+Mom(long) 그리드:")
    print(f"  {'':>15} {'Mom60':>8} {'Mom90':>8} {'Mom120':>8}")
    print(f"  {'─' * 42}")
    for ms in [7, 10, 14, 21, 30]:
        print(f"  {'Mom'+str(ms):<15}", end="")
        for ml in [60, 90, 120]:
            key = f'M{ms}+M{ml}+v5%'
            if key in mm_lookup:
                print(f" {mm_lookup[key]:>8.3f}", end="")
            else:
                print(f" {'─':>8}", end="")
        print()

    # ── Top 10 overall ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  전체 Top 10")
    print(f"{'=' * 100}")
    all_scored = []
    for (name, _), r in zip(ALL, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        all_scored.append((name, m, calmar))
    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'H 조건':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 62}")
    for i, (name, m, calmar) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
