"""현물 체결 밴드 확대 구간 — 0 / 5 / 10 / 20 / 30% (2026-08-25 오후).

계기: 오전에 현물 밴드를 0/1/2/5/10% 로 재고, 라이브에 상대 밴드가 없으므로 BT 를 0 으로
맞췄다(라이브 무변경). 사용자가 그 뒤 물었다 — "5% 뿐 아니라 10/20/30 도 봐야 하는 것
아니냐. 밴드는 슬리피지 방지·거래 안정성 마진 역할도 하잖아."

현물에서 이 질문의 의미는 선물과 다르다. 선물은 밴드가 이미 라이브에 있으니 값의 문제지만,
현물은 라이브에 없으므로 "넓히자" 는 곧 "업비트 실매매에 없던 무반응 구간을 새로 넣자" 다.
그래서 채택 문턱을 높게 둔다.

판정 (결과 보기 전 고정):
  기본비용과 비용 5배 양쪽에서 같은 방향으로 뚜렷하게(순위차 0.3 이상, Calmar 0.3 이상)
  넓은 쪽이 이길 때만 라이브 도입 후보로 올린다. 한쪽만 좋거나 단조 개선이면 도입하지 않는다.
  단조 개선은 "거래를 덜 하면 비용이 준다" 는 동어반복일 수 있어 플래토가 아니다.

재현: git apply strategies/cap_defend/research/bt_spot_rebal_band.engine.patch (또는 엔진의
_band 를 REBAL_BAND env 로 덮어쓰게 임시 수정) 후 실행.
실행: cd strategies/cap_defend/research && python3 bt_spot_band_wide.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_spot_rebal_band import run_spot, window_rs, START, END, BASE_TX, STRESS_TX  # noqa: E402

BANDS = [0.0, 0.05, 0.10, 0.20, 0.30]


def tag(b):
    return f"밴드 {b*100:.0f}%" + ("(현행=라이브)" if b == 0.0 else "")


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 체결 밴드 (drift 0.10 고정)")
    print(f"후보 {[f'{b*100:.0f}%' for b in BANDS]} | tx {BASE_TX} (stress {STRESS_TX})\n")

    res, eqs = {}, {}
    print(f"[sleeve 단독 — tx {BASE_TX*100:.2f}%]")
    print(f"  {'arm':<18s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'체결':>6s} {'Rebal':>6s} {'최종':>9s}")
    for b in BANDS:
        m = run_spot(b)
        eq = m['_equity']
        res[tag(b)] = m
        eqs[tag(b)] = eq
        print(f"  {tag(b):<18s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>6d} {m['Rebal']:>6d} "
              f"{eq.iloc[-1]/eq.iloc[0]:>8.2f}x", flush=True)

    yrs = (list(eqs.values())[0].index[-1] - list(eqs.values())[0].index[0]).days / 365.25
    base_tr = res[tag(0.0)]['Trades']
    print(f"\n[회전 부담] {yrs:.1f}년")
    for k in eqs:
        tr = res[k]['Trades']
        print(f"  {k:<18s} 체결 {tr:>5d} ({tr/yrs:>6.1f}회/년, 현행 대비 {tr/base_tr:>5.2f}x)")

    print("\n[윈도우 rank-sum — 기본비용]")
    sums, wins, n = window_rs(eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"  {k:<18s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n} "
              f"({wins[k]/n*100:.0f}%)")
    print(f"  windows n={n}")

    print(f"\n[비용 stress — tx {STRESS_TX*100:.2f}% (5x)]")
    seqs = {}
    print(f"  {'arm':<18s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'체결':>6s}")
    for b in BANDS:
        m = run_spot(b, tx_cost=STRESS_TX)
        seqs[tag(b)] = m['_equity']
        print(f"  {tag(b):<18s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Trades']:>6d}", flush=True)
    ssums, swins, sn = window_rs(seqs)
    print("  [rank-sum — stress]")
    for k in sorted(ssums, key=lambda k: ssums[k]):
        print(f"    {k:<18s} avg_rank {ssums[k]/sn:.3f}  wins {swins[k]:>4d}/{sn}")

    order = [sums[tag(b)] / n for b in BANDS]
    sorder = [ssums[tag(b)] / sn for b in BANDS]
    bbest, sbest = BANDS[int(np.argmin(order))], BANDS[int(np.argmin(sorder))]
    cal_b = {b: res[tag(b)]['Cal'] for b in BANDS}
    print(f"\n[판정 보조] 기본비용 최상위 {bbest*100:.0f}% / 스트레스 최상위 {sbest*100:.0f}%")
    print(f"  전기간 Calmar 최대-최소 폭 {max(cal_b.values())-min(cal_b.values()):.2f}")
    agree = bbest == sbest and bbest > 0
    print(f"  두 비용 조건이 같은 넓은 밴드를 가리키나? {'예' if agree else '아니오'} "
          f"→ {'도입 후보' if agree else '도입 안 함(라이브 무변경)'}")
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
