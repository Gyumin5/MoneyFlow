"""합성(60/25/15)에서만 변형이 이겨 보이는 이유 분해 (2026-08-21 후속).

물음: sleeve 단독은 현행이 이기는데 합성은 변형이 이긴다. Cal 4.89 vs 5.17 이 어디서 왔나.

보는 것:
1. Cal 차이를 CAGR 기여 / MDD 기여로 분해
2. 두 곡선의 MDD 구간(peak~trough 날짜)이 같은 사건인가
3. Calmar 가 아닌 지표(Sharpe, 연평균초과)로 윈도우 rank-sum 을 다시 매기면 우위가 남나
4. 상관·추적오차 — 두 합성 곡선이 사실상 같은 시계열인가

주의: 이 스크립트는 엔진 변형 패치가 적용돼 있어야 의미가 있다.
  git apply strategies/cap_defend/research/bt_fut_daily_lev.engine.patch
안 하면 두 팔이 모두 현행이 되어 차이 0 이 나온다.

실행: cd strategies/cap_defend/research && python3 bt_fut_daily_lev_why_alloc.py
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_fut_daily_lev as base  # noqa: E402


def mdd_span(eq):
    """MDD 의 peak/trough 날짜와 값."""
    peak = eq.cummax()
    dd = eq / peak - 1
    t = dd.idxmin()
    p = eq.loc[:t].idxmax()
    return p, t, float(dd.min())


def cagr_of(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1


def window_rank(alloc, metric):
    """metric: 'cal' | 'sharpe' | 'cagr'"""
    common = sorted(alloc['fixed'].index.intersection(alloc['daily'].index))
    sums = defaultdict(float)
    wins = defaultdict(int)
    n = 0
    for size in base.WIN_SIZES:
        for stride in base.STRIDES:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                vals = {}
                for k, s in alloc.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        vals[k] = np.nan
                        continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    if metric == 'cagr':
                        vals[k] = cagr
                    elif metric == 'sharpe':
                        r = seg.pct_change().dropna()
                        vals[k] = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
                    else:
                        mdd = float((seg / seg.cummax() - 1).min())
                        vals[k] = cagr / abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in vals.values()):
                    continue
                ranked = sorted(vals.items(), key=lambda x: -x[1])
                for r_, (mk, _) in enumerate(ranked, 1):
                    sums[mk] += r_
                wins[ranked[0][0]] += 1
                n += 1
    return sums, wins, n


def main():
    eq_fu = {t: base.run_fut(dl)['_equity'] for t, dl in [('fixed', False), ('daily', True)]}
    eq_sp = base.run_spot_live()
    eq_st = base.run_stock_live()
    alloc = {t: base.build_alloc(eq_st, eq_sp, e) for t, e in eq_fu.items()}

    print("[1] Cal 차이 분해 (alloc 60/25/15)")
    info = {}
    for k, s in alloc.items():
        c = cagr_of(s)
        p, t, m = mdd_span(s)
        info[k] = (c, m, p, t)
        print(f"  {k:<6s} CAGR {c*100:+.2f}%  MDD {m*100:+.2f}%  Cal {c/abs(m):.3f}"
              f"   MDD 구간 {p.date()} → {t.date()}")
    cf, mf = info['fixed'][0], info['fixed'][1]
    cd, md = info['daily'][0], info['daily'][1]
    cal_f, cal_d = cf / abs(mf), cd / abs(md)
    # CAGR 만 바뀌었을 때 / MDD 만 바뀌었을 때
    only_cagr = cd / abs(mf)
    only_mdd = cf / abs(md)
    print(f"  현행 Cal {cal_f:.3f} → 변형 Cal {cal_d:.3f} (차이 {cal_d-cal_f:+.3f})")
    print(f"   · CAGR 만 변형 값으로: {only_cagr:.3f} (기여 {only_cagr-cal_f:+.3f})")
    print(f"   · MDD  만 변형 값으로: {only_mdd:.3f} (기여 {only_mdd-cal_f:+.3f})")

    print("\n[2] 두 합성 곡선이 얼마나 같은가")
    common = alloc['fixed'].index.intersection(alloc['daily'].index)
    rf = alloc['fixed'].loc[common].pct_change().dropna()
    rd = alloc['daily'].loc[common].pct_change().dropna()
    print(f"  일수익률 상관 {rf.corr(rd):.6f}")
    print(f"  추적오차(연) {(rd-rf).std()*np.sqrt(252)*100:.3f}%")
    print(f"  연평균 수익률 차이 {(cd-cf)*100:+.3f}%p")

    print("\n[3] 지표를 바꿔 윈도우 rank-sum 재계산")
    for metric in ['cal', 'sharpe', 'cagr']:
        sums, wins, n = window_rank(alloc, metric)
        order = sorted(sums, key=lambda k: sums[k])
        line = ' | '.join(f"{k} avg_rank {sums[k]/n:.3f} 승 {wins[k]}" for k in order)
        print(f"  {metric:<7s} n={n}  {line}")


if __name__ == '__main__':
    main()
