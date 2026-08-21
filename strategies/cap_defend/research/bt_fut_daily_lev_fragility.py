"""매일 L 재조정 — 우위의 취약성 진단 (2026-08-21).

sleeve 단독은 윈도우 rank-sum 에서 현행이 이기는데(1.429 vs 1.571) 60/25/15 합성에서는
변형이 이긴다(1.259 vs 1.741). 재조정 체결이 5.5년에 63건뿐이라 이 우위가 구조인지
소수 사건 운인지 가려야 한다.

보는 것:
1. 두 안의 일별 수익률 차이가 며칠에 몰려 있나 (top-N 일 기여 비중)
2. 재조정이 실제로 발생한 날짜 분포 (연도별)
3. 그 소수 사건을 제거하면 우위가 남나 (차이 상위 3/5/10일 제외 후 Cal 재계산)
4. 윈도우별 승패가 특정 구간에 몰리나

실행: cd strategies/cap_defend/research && python3 bt_fut_daily_lev_fragility.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_fut_daily_lev as base  # noqa: E402


def cal_of(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    return cagr / abs(mdd) if mdd < 0 else 0, cagr * 100, mdd * 100


def rebuild(ret_f, ret_d, drop_dates=()):
    """일별 수익률에서 특정 날짜를 양쪽 동일하게 제거하고 곡선 재구성."""
    keep = [d for d in ret_f.index if d not in set(drop_dates)]
    return (1 + ret_f.loc[keep]).cumprod(), (1 + ret_d.loc[keep]).cumprod()


def main():
    eq_fu = {}
    for tag, dl in [('fixed', False), ('daily', True)]:
        m = base.run_fut(dl)
        eq_fu[tag] = m['_equity']
    eq_sp = base.run_spot_live()
    eq_st = base.run_stock_live()

    alloc = {t: base.build_alloc(eq_st, eq_sp, e) for t, e in eq_fu.items()}
    common = alloc['fixed'].index.intersection(alloc['daily'].index)
    rf = alloc['fixed'].loc[common].pct_change().dropna()
    rd = alloc['daily'].loc[common].pct_change().dropna()
    diff = (rd - rf)

    print(f"공통 {len(common)}일 {common[0].date()}~{common[-1].date()}")
    c_f = cal_of(alloc['fixed'])
    c_d = cal_of(alloc['daily'])
    print(f"alloc fixed  Cal {c_f[0]:.2f} CAGR {c_f[1]:+.1f}% MDD {c_f[2]:+.1f}%")
    print(f"alloc daily  Cal {c_d[0]:.2f} CAGR {c_d[1]:+.1f}% MDD {c_d[2]:+.1f}%")

    print("\n[1] 수익률 차이의 집중도")
    nz = diff[diff.abs() > 1e-9]
    print(f"  차이가 있는 날 {len(nz)}일 / {len(diff)}일 ({len(nz)/len(diff)*100:.1f}%)")
    tot = diff.sum()
    for n in [1, 3, 5, 10, 20]:
        top = diff.abs().nlargest(n)
        share = diff.loc[top.index].sum() / tot * 100 if tot != 0 else np.nan
        print(f"  |차이| 상위 {n:>2d}일 합계 기여 {share:>6.1f}% (총 차이 {tot*100:+.2f}%p)")

    print("\n[2] 차이 상위 10일")
    for d in diff.abs().nlargest(10).index.sort_values():
        print(f"  {d.date()}  diff {diff[d]*100:+.3f}%p")

    print("\n[3] 소수 사건 제외 후 우위가 남나 (양쪽 동일 날짜 제거)")
    for n in [0, 3, 5, 10]:
        drop = diff.abs().nlargest(n).index if n else []
        a_f, a_d = rebuild(rf, rd, drop)
        cf, cd = cal_of(a_f)[0], cal_of(a_d)[0]
        winner = 'daily' if cd > cf else 'fixed'
        print(f"  상위 {n:>2d}일 제외 → fixed Cal {cf:.2f} / daily Cal {cd:.2f}  우세={winner}")

    print("\n[4] 연도별 차이 (alloc)")
    for y in sorted({d.year for d in diff.index}):
        seg = diff[diff.index.year == y]
        nzy = int((seg.abs() > 1e-9).sum())
        print(f"  {y}  누적차이 {seg.sum()*100:+.2f}%p  차이발생일 {nzy:>3d}")

    print("\n[5] 윈도우 승패 분포 (alloc, 시작연도별)")
    common_l = sorted(alloc['fixed'].index.intersection(alloc['daily'].index))
    win_by_year = {}
    for size in base.WIN_SIZES:
        for stride in base.STRIDES:
            for i in range(0, len(common_l) - size, stride):
                d0, d1 = common_l[i], common_l[i + size - 1]
                cals = {}
                for k, s in alloc.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        cals[k] = np.nan
                        continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    mdd = float((seg / seg.cummax() - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()):
                    continue
                w = max(cals, key=lambda k: cals[k])
                rec = win_by_year.setdefault(d0.year, {'fixed': 0, 'daily': 0})
                rec[w] += 1
    for y in sorted(win_by_year):
        r = win_by_year[y]
        print(f"  시작 {y}: fixed {r['fixed']:>3d} / daily {r['daily']:>3d}")


if __name__ == '__main__':
    main()
