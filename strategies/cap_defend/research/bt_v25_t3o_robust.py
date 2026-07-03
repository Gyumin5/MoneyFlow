"""V25 — T3O 추가안 robustness: window rank-sum + 송금비용 stress + 시작일 일관성.

bt_v25_t1_t3u_t3o.py 의 sleeve 곡선·실 카나리·simulate_alloc 재사용.
홀드아웃/yearly-rank 금지(프로젝트 규칙) → 롤링 window unified rank-sum 사용.

후보:
  C0 base       : T1=20 T3U=20            (현행)
  C1 +T3O20     : T1=20 T3U=20 T3O=20 none(무게이트)
  C2 T1T3U-best : T1=23 T3U=15            (T1×T3U grid 1위)
  C3 +T3O15     : T1=20 T3U=20 T3O=15 none
  C4 +T3O25     : T1=20 T3U=20 T3O=25 none
  C5 +T3O20off  : T1=20 T3U=20 T3O=20 off (대칭 게이트, 대조군)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_t1_t3u_t3o import run_spot, run_fut, run_stock, load_canaries, simulate_alloc, metrics, EXCLUDE

CONFIGS = [
    ('C0 base',        dict(T1=0.20, T3U=0.20, T3O=None)),
    ('C1 +T3O20',      dict(T1=0.20, T3U=0.20, T3O=0.20, t3o_canary='none')),
    ('C2 T1T3U-best',  dict(T1=0.23, T3U=0.15, T3O=None)),
    ('C3 +T3O15',      dict(T1=0.20, T3U=0.20, T3O=0.15, t3o_canary='none')),
    ('C4 +T3O25',      dict(T1=0.20, T3U=0.20, T3O=0.25, t3o_canary='none')),
    ('C5 +T3O20off',   dict(T1=0.20, T3U=0.20, T3O=0.20, t3o_canary='off')),
]


def slice_metric(eq):
    """window 슬라이스용 Cal/Sharpe."""
    eq = eq.dropna()
    if len(eq) < 20: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0: return None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    peak = eq.cummax(); mdd = (eq / peak - 1).min()
    rets = eq.pct_change().dropna()
    sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else cagr / 0.01
    return cal, sh


def window_ranksum(eqs, labels, Ws=(252, 378), strides=(15, 21), by='cal'):
    """롤링 window rank-sum. 각 window 에서 후보별 Cal(or Sharpe) 랭킹(1=best) 합.
    낮을수록 robust 상위. top1 횟수도 반환."""
    common = eqs[0].index
    for e in eqs[1:]:
        common = common.intersection(e.index)
    common = sorted(common)
    n = len(common)
    ranksum = {l: 0 for l in labels}
    top1 = {l: 0 for l in labels}
    nwin = 0
    for W in Ws:
        for S in strides:
            for st in range(0, n - W, S):
                idx = common[st:st + W]
                vals = []
                for e in eqs:
                    m = slice_metric(e.loc[idx])
                    vals.append(m[0] if by == 'cal' else (m[1] if m else -9))
                order = np.argsort(-np.array(vals))  # 내림차순: 0=best
                ranks = np.empty(len(vals), dtype=int)
                for rk, pos in enumerate(order):
                    ranks[pos] = rk + 1
                for li, l in enumerate(labels):
                    ranksum[l] += int(ranks[li])
                top1[labels[int(order[0])]] += 1
                nwin += 1
    return ranksum, top1, nwin


def main():
    t0 = time.time()
    print("[load sleeves + real canaries]")
    eq_st = run_stock()
    eq_sp = run_spot(20, 127, 217, 7, EXCLUDE)
    eq_fu = run_fut(18, 127, 95, 5, EXCLUDE)
    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    can = load_canaries(common)

    def build(cfg, cost):
        res = simulate_alloc(eq_st, eq_sp, eq_fu, can, transfer_cost=cost, **cfg)
        return res[0], res[1], res[2], res[3]

    labels = [l for l, _ in CONFIGS]

    # === A. 전기간 metrics @ 비용 0 ===
    print("\n=== A. 전기간 (2020-11~2026-05), 송금비용 0 ===")
    print(f"  {'cfg':<15} {'CAGR':>7} {'MDD':>7} {'Sh':>6} {'Cal':>5} {'nT1':>4} {'nT3U':>5} {'nT3O':>5}")
    eqs0 = []
    for l, cfg in CONFIGS:
        eq, n1, n3u, n3o = build(cfg, 0.0)
        eqs0.append(eq)
        m = metrics(eq)
        print(f"  {l:<15} {m[0]:>6.1f}% {m[1]:>+6.1f}% {m[2]:>6.2f} {m[3]:>5.2f} {n1:>4} {n3u:>5} {n3o:>5}")

    # === B. window rank-sum (Cal & Sharpe) ===
    print("\n=== B. 롤링 window rank-sum (W=252,378 × stride=15,21) — 낮을수록 robust ===")
    for by in ['cal', 'sh']:
        rs, t1c, nwin = window_ranksum(eqs0, labels, by=by)
        print(f"  [{by}] (n_win={nwin})")
        for l in sorted(labels, key=lambda x: rs[x]):
            print(f"    {l:<15} ranksum {rs[l]:>5}  avg {rs[l]/nwin:>4.2f}  top1 {t1c[l]:>4} ({t1c[l]/nwin*100:>4.1f}%)")

    # === C. 송금비용 stress (전기간 Cal) ===
    print("\n=== C. 송금비용 stress (이동금액 대비, 전기간 Cal) ===")
    costs = [0.0, 0.001, 0.003, 0.005]
    print(f"  {'cfg':<15} " + " ".join(f"{int(c*1e4):>4}bp" for c in costs))
    for l, cfg in CONFIGS:
        row = []
        for c in costs:
            eq, *_ = build(cfg, c)
            m = metrics(eq)
            row.append(f"{m[3]:>5.2f}")
        print(f"  {l:<15} " + "  ".join(f"{v:>4}" for v in row))

    # === D. 시작일 일관성 (start 변화, 비용 0.003) ===
    print("\n=== D. 시작일 일관성 (송금비용 0.003=30bp, Cal) ===")
    starts = ['2020-11-13', '2021-06-01', '2022-01-01', '2022-06-01']
    print(f"  {'cfg':<15} " + " ".join(f"{s[2:7]:>7}" for s in starts))
    for l, cfg in CONFIGS:
        row = []
        for s in starts:
            sd = pd.Timestamp(s)
            est = eq_st[eq_st.index >= sd]; esp = eq_sp[eq_sp.index >= sd]; efu = eq_fu[eq_fu.index >= sd]
            res = simulate_alloc(est, esp, efu, can, transfer_cost=0.003, **cfg)
            m = metrics(res[0])
            row.append(f"{m[3]:>5.2f}" if m else "  - ")
        print(f"  {l:<15} " + "  ".join(f"{v:>5}" for v in row))

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
