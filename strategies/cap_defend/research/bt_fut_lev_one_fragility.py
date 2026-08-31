"""A 추세이탈 1배(thr=1.00) 의 합성 우위가 며칠에 얹혀 있는지 본다 (2026-08-31).

bt_fut_lev_one.py 에서 A 1.00 만 합성 rank-sum 1위(1.833, 41/60)로 나왔다. 그런데
비용 5배에서는 현행이 1위로 뒤집혔고, 같은 A 안에서 파라미터 순서가 슬리브(0.98 최상)와
합성(1.00 최상)이 서로 다르다. 우위가 구조적인지 몇 날에 얹혀 있는지를 가른다.

방법: 2026-08-21 bt_fut_daily_lev_fragility.py 와 같은 잣대. 합성 로그수익 차이에서
기여 상위 k일을 빼고도 우위가 남는지 본다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_lev_one_fragility.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_fut_lev_one as m                                  # noqa: E402
import bt_fut_rebal_band as base                            # noqa: E402
from bt_fut_rebal_band import build_alloc, metrics          # noqa: E402


def main():
    from unified_backtest import load_data
    m.BARS, m.FUNDING = load_data('D')
    base.START, base.END = m.START, m.END

    eq_b = m.run_fut_lev(m.build_lev(m.BARS, None, None))['_equity']
    eq_a = m.run_fut_lev(m.build_lev(m.BARS, 'A', 1.00))['_equity']
    eq_sp = base.run_spot_live()
    eq_st = base.run_stock_live()

    ab = build_alloc(eq_st, eq_sp, eq_b)
    aa = build_alloc(eq_st, eq_sp, eq_a)
    mb, ma = metrics(ab), metrics(aa)
    print(f"합성 현행   Cal {mb['Cal']:.2f} CAGR {mb['CAGR']:+.1f}% MDD {mb['MDD']:+.1f}%")
    print(f"합성 A 1.00 Cal {ma['Cal']:.2f} CAGR {ma['CAGR']:+.1f}% MDD {ma['MDD']:+.1f}%")

    rb = np.log(ab).diff().fillna(0)
    ra = np.log(aa).diff().fillna(0)
    d = (ra - rb).dropna()
    tot = float(d.sum())
    print(f"\n합성 로그수익 차이 합계 {tot:+.4f} (양수면 A 우세)")
    order = d.abs().sort_values(ascending=False)
    for k in (1, 3, 5, 10):
        drop = order.index[:k]
        rest = float(d.drop(drop).sum())
        print(f"  기여 상위 {k:>2d}일 제외 → 남은 차이 {rest:+.4f} "
              f"({rest/tot*100 if tot else 0:>6.1f}% 잔존)")
    top = d.sort_values(ascending=False).head(5)
    print("\n  기여 상위 5일")
    for dt, v in top.items():
        print(f"    {dt.date()} {v:+.4f}")
    worst = d.sort_values().head(3)
    print("  기여 하위 3일")
    for dt, v in worst.items():
        print(f"    {dt.date()} {v:+.4f}")
    nz = int((d.abs() > 1e-6).sum())
    print(f"\n  차이가 난 날 {nz} / {len(d)} ({nz/len(d)*100:.1f}%)")


if __name__ == '__main__':
    main()
