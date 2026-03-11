#!/usr/bin/env python3
"""Test remaining layers on new baseline: K=SMA(60) + H=Mom(21)+Mom(90)+Vol5%.
   Layers: Selection, Weighting, Rebalancing, Risk, n_picks."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# New baseline
def B(**kw):
    return Params(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05, **kw)

# ══════════════════════════════════════════════════════════════════
# 0. Baseline
# ══════════════════════════════════════════════════════════════════
BASELINE = [('BASELINE (균등5개)', B())]

# ══════════════════════════════════════════════════════════════════
# 1. Selection — 코인 선정 방식
# ══════════════════════════════════════════════════════════════════
SEL = [
    ('S:baseline (시총순)', B()),
    ('S1:cap15 mom top5', B(selection='S1')),
    ('S2:blend (cap+mom)', B(selection='S2')),
    ('S3:core3+mom2', B(selection='S3')),
    ('S4:BTC+ETH고정', B(selection='S4')),
    ('S5:유지보너스', B(selection='S5')),
    ('S6:Sharpe score', B(selection='S6')),
    ('S7:cap15 Sharpe', B(selection='S7')),
    ('S8:Sharpe+유지', B(selection='S8')),
    ('S9:히스테리시스', B(selection='S9')),
    ('S10:최소2개월보유', B(selection='S10')),
]

# ══════════════════════════════════════════════════════════════════
# 2. Weighting — 가중 방식
# ══════════════════════════════════════════════════════════════════
WGT = [
    ('W:baseline (균등)', B()),
    ('W1:순위감소', B(weighting='W1')),
    ('W2:역변동성70%', B(weighting='W2')),
    ('W3:모멘텀틸트', B(weighting='W3')),
    ('W4:breadth-scale', B(weighting='W4')),
    ('W5:BTC fill', B(weighting='W5')),
    ('W6:순수역변동성', B(weighting='W6')),
]

# ══════════════════════════════════════════════════════════════════
# 3. Rebalancing — 리밸런싱 규칙
# ══════════════════════════════════════════════════════════════════
REBAL = [
    ('R:baseline (월간)', B()),
    ('R2:월중급락-15%', B(rebalancing='R2')),
    ('R3:고TO>50%만', B(rebalancing='R3')),
    ('R4:밴드±5pp', B(rebalancing='R4')),
    ('R5:15일앵커', B(rebalancing='R5')),
    ('R6:trailing-20%', B(rebalancing='R6')),
    ('R7:월중급락-10%', B(rebalancing='R7')),
    ('R8:월중급락-20%', B(rebalancing='R8')),
    ('R9:trailing30d-15%', B(rebalancing='R9')),
]

# ══════════════════════════════════════════════════════════════════
# 4. Risk overlay — 리스크 관리
# ══════════════════════════════════════════════════════════════════
RISK = [
    ('G:baseline (없음)', B()),
    ('G1:DD>20% 반감', B(risk='G1')),
    ('G2:vol target80%', B(risk='G2')),
    ('G3:breadth ladder', B(risk='G3')),
    ('G4:rank floor', B(risk='G4')),
    ('G5:crash breaker', B(risk='G5')),
]

# ══════════════════════════════════════════════════════════════════
# 5. n_picks — 코인 개수
# ══════════════════════════════════════════════════════════════════
NPICKS = [
    (f'n={n}', B(n_picks=n))
    for n in [3, 4, 5, 6, 7, 8, 10]
]

# ══════════════════════════════════════════════════════════════════
# 6. Vol cap 미세 조정
# ══════════════════════════════════════════════════════════════════
VOLCAP = []
for v in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
    VOLCAP.append(
        (f'vol≤{v}%',
         Params(canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
                health='HK', health_sma=2, health_mom_short=21,
                health_mom_long=90, vol_cap=v/100)))

ALL = BASELINE + SEL + WGT + REBAL + RISK + NPICKS + VOLCAP


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
            selection=p.selection, weighting=p.weighting,
            rebalancing=p.rebalancing, risk=p.risk,
            n_picks=p.n_picks, top_n=p.top_n,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_section(title, strategies, results):
    print(f"\n  {title}")
    print(f"  {'─' * 85}")
    print(f"  {'조건':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'Rebal':>6}")
    print(f"  {'─' * 85}")
    best_s = max(r['metrics']['Sharpe'] for r in results) if results else 0
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = " ★" if m['Sharpe'] == best_s else ""
        rc = r.get('rebal_count', 0)
        print(f"  {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f} {rc:>6}{marker}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    print(f"  {len(ALL)} layer configs to test")

    t0 = time.time()
    results = run_set(ALL, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    idx = 0
    def get_slice(s):
        nonlocal idx
        r = results[idx:idx+len(s)]
        idx += len(s)
        return r

    r_base = get_slice(BASELINE)
    r_sel = get_slice(SEL)
    r_wgt = get_slice(WGT)
    r_reb = get_slice(REBAL)
    r_risk = get_slice(RISK)
    r_npick = get_slice(NPICKS)
    r_vol = get_slice(VOLCAP)

    print(f"\n{'=' * 100}")
    print(f"  레이어별 최적화 — 베이스라인: K=SMA(60) + H=Mom(21)&Mom(90)&Vol5%")
    print(f"{'=' * 100}")

    print_section("0. 베이스라인", BASELINE, r_base)
    print_section("1. 선정 방식 (Selection)", SEL, r_sel)
    print_section("2. 가중 방식 (Weighting)", WGT, r_wgt)
    print_section("3. 리밸런싱 (Rebalancing)", REBAL, r_reb)
    print_section("4. 리스크 오버레이 (Risk)", RISK, r_risk)
    print_section("5. 코인 개수 (n_picks)", NPICKS, r_npick)
    print_section("6. Vol Cap 미세조정", VOLCAP, r_vol)

    # ── 각 레이어별 최고 ──────────────────────────────────────────
    base_sharpe = r_base[0]['metrics']['Sharpe']
    base_mdd = r_base[0]['metrics']['MDD']
    base_calmar = r_base[0]['metrics']['CAGR'] / abs(base_mdd) if base_mdd != 0 else 0

    print(f"\n{'=' * 100}")
    print(f"  레이어별 최고 vs 베이스라인")
    print(f"{'=' * 100}")
    print(f"  베이스라인: Sharpe {base_sharpe:.3f}  MDD {base_mdd:.1%}  Calmar {base_calmar:.2f}\n")

    for name, strats, ress in [
        ('Selection', SEL[1:], r_sel[1:]),  # exclude baseline
        ('Weighting', WGT[1:], r_wgt[1:]),
        ('Rebalancing', REBAL[1:], r_reb[1:]),
        ('Risk', RISK[1:], r_risk[1:]),
        ('N picks', NPICKS, r_npick),
        ('Vol cap', VOLCAP, r_vol),
    ]:
        best_idx = max(range(len(ress)), key=lambda i: ress[i]['metrics']['Sharpe'])
        m = ress[best_idx]['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        diff = m['Sharpe'] - base_sharpe
        label = strats[best_idx][0]
        sig = "✓" if diff > 0.05 else "~" if diff > 0 else "✗"
        print(f"  {name:<12}: {label:<25} Sharpe {m['Sharpe']:.3f} ({diff:+.3f}) "
              f"MDD {m['MDD']:.1%}  Cal {calmar:.2f}  {sig}")

        # Also show best by Calmar
        best_c_idx = max(range(len(ress)),
                        key=lambda i: (ress[i]['metrics']['CAGR'] / abs(ress[i]['metrics']['MDD'])
                                       if ress[i]['metrics']['MDD'] != 0 else 0))
        mc = ress[best_c_idx]['metrics']
        calmar_c = mc['CAGR'] / abs(mc['MDD']) if mc['MDD'] != 0 else 0
        if best_c_idx != best_idx:
            print(f"  {'':12}  (Calmar최고: {strats[best_c_idx][0]} → Cal {calmar_c:.2f} MDD {mc['MDD']:.1%})")

    # ── Overall Top 10 ──────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  전체 Top 10")
    print(f"{'=' * 100}")
    all_scored = []
    for (name, _), r in zip(ALL, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        all_scored.append((name, m, r, calmar))
    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'조건':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for i, (name, m, r, calmar) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Top 10 by Calmar ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  Calmar Top 10")
    print(f"{'=' * 100}")
    all_scored_c = sorted(all_scored, key=lambda x: x[3], reverse=True)
    print(f"\n  {'순위':>3} {'조건':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for i, (name, m, r, calmar) in enumerate(all_scored_c[:10], 1):
        print(f"  {i:>3}. {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Year-by-year for top 5 ─────────────────────────────────────
    years = range(2018, 2026)
    print(f"\n  Top 5 연도별 CAGR")
    print(f"  {'조건':<25}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 100}")
    for name, m, r, calmar in all_scored[:5]:
        ym = r['yearly']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
