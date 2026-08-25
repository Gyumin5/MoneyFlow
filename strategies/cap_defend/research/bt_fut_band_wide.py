"""선물 체결 밴드 확대 구간 검증 — 5 / 10 / 15 / 20 / 30% (2026-08-25 오후).

계기: 같은 날 오전 bt_fut_rebal_band.py 로 1/2/3/5/10% 를 재고 라이브를 1% → 5% 로 올렸다.
그때 10% 가 기본비용·비용5배 rank-sum 둘 다 1위였는데 "채택안 자체를 바꾸는 축" 이라 보류로
남겼다(history 2026-08-25 선물 체결 밴드 ADR 의 마지막 줄). 사용자가 그 보류를 다시 물었다 —
"5% 뿐만 아니라 10 / 20 / 30 도 다 확인해야 하는 것 아니냐".

축: 체결 밴드 하나. 나머지(선정·헬스·카나리·스냅·드리프트 0.03·동적 L·비용) 전부 고정.
오전 스크립트와 같은 하니스를 그대로 재사용한다(run_fut / window_rs / 합성). 기간만
데이터 끝까지 맞춘다.

읽는 법 (결과 보기 전 고정):
  1. 넓힐수록 단조 개선이면 플래토가 아니다. "거래를 덜 하면 비용이 준다" 는 동어반복일 수
     있으므로 채택 근거로 쓰지 않는다. 꺾이는 지점이 있어야 그 안쪽이 플래토다.
  2. 비용 5배에서 순서가 뒤집히면 유지(5%) 쪽으로 판정한다.
  3. 백테스트는 무조건 체결을 가정하므로 넓은 밴드의 진짜 이득(슬리피지·부분체결·거절 감소)을
     못 잰다. 반대로 손해(추적 오차)는 잰다. 그래서 동률로 나오면 넓은 쪽이 실제로는 유리하다.
     이 비대칭을 판정문에 명시한다.
  4. 추적 오차를 직접 본다 — 밴드가 넓으면 보유 명목이 목표에서 얼마나 벌어진 채로 방치되나.

재현: git apply strategies/cap_defend/research/bt_fut_rebal_band.engine.patch 먼저.
실행: cd strategies/cap_defend/research && python3 bt_fut_band_wide.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_fut_rebal_band as base                                    # noqa: E402
from bt_fut_rebal_band import run_fut, window_rs, metrics, build_alloc  # noqa: E402

base.END = END = '2026-08-24'
BANDS = [0.05, 0.10, 0.15, 0.20, 0.30]
BASE_TX = base.BASE_TX
STRESS_TX = base.STRESS_TX


def tag(b):
    return f"밴드 {b*100:.0f}%" + ("(현행)" if b == 0.05 else "")


def main():
    t0 = time.time()
    print(f"기간 {base.START} ~ {END} | 축: 체결 밴드 (drift 0.03 고정, 동적 L)")
    print(f"후보 {[f'{b*100:.0f}%' for b in BANDS]} | tx {BASE_TX} (stress {STRESS_TX})\n")

    res = {}
    print(f"[sleeve 단독 — tx {BASE_TX*100:.2f}%]")
    print(f"  {'arm':<14s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'체결':>6s} {'Rebal':>6s} {'청산':>4s} {'최종':>9s}")
    for b in BANDS:
        m = run_fut(b)
        eq = m['_equity']
        res[tag(b)] = dict(m=m, eq=eq, band=b)
        print(f"  {tag(b):<14s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>6d} {m['Rebal']:>6d} {m['Liq']:>4d} "
              f"{eq.iloc[-1]/eq.iloc[0]:>8.1f}x", flush=True)

    eqs = {k: v['eq'] for k, v in res.items()}
    yrs = (list(eqs.values())[0].index[-1] - list(eqs.values())[0].index[0]).days / 365.25
    base_tr = res[tag(0.05)]['m']['Trades']
    print(f"\n[회전 부담] {yrs:.1f}년")
    for k, v in res.items():
        tr = v['m']['Trades']
        print(f"  {k:<14s} 체결 {tr:>5d} ({tr/yrs:>6.1f}회/년, 현행 대비 {tr/base_tr:>5.2f}x)")

    print("\n[윈도우 rank-sum — sleeve, 기본비용]")
    sums, wins, n = window_rs(eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"  {k:<14s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n} "
              f"({wins[k]/n*100:.0f}%)")
    print(f"  windows n={n}")

    print(f"\n[비용 stress — tx {STRESS_TX*100:.2f}% (5x)]")
    seqs = {}
    print(f"  {'arm':<14s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'체결':>6s}")
    for b in BANDS:
        m = run_fut(b, tx_cost=STRESS_TX)
        seqs[tag(b)] = m['_equity']
        print(f"  {tag(b):<14s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Trades']:>6d}", flush=True)
    ssums, swins, sn = window_rs(seqs)
    print("  [rank-sum — stress]")
    for k in sorted(ssums, key=lambda k: ssums[k]):
        print(f"    {k:<14s} avg_rank {ssums[k]/sn:.3f}  wins {swins[k]:>4d}/{sn}")

    print("\n[자산배분 60/25/15 합성 — 선물만 교체]")
    try:
        eq_sp = base.run_spot_live()
        eq_st = base.run_stock_live()
        rows = {}
        for k, eq_fu in eqs.items():
            a = build_alloc(eq_st, eq_sp, eq_fu)
            rows[k] = (a, metrics(a))
        for k, (_a, mm) in rows.items():
            print(f"  {k:<14s} Cal {mm['Cal']:.2f} | CAGR {mm['CAGR']:+.1f}% | "
                  f"MDD {mm['MDD']:+.1f}% | Sharpe {mm['Sharpe']:.2f}")
        asums, awins, an = window_rs({k: v[0] for k, v in rows.items()})
        for k in sorted(asums, key=lambda k: asums[k]):
            print(f"    {k:<14s} alloc avg_rank {asums[k]/an:.3f}  wins {awins[k]}/{an}")
    except Exception as ex:
        print(f"  합성 실패(무시하고 sleeve 결론 유지): {ex}")

    # 판정 보조: 단조성 검사 (넓힐수록 좋아지기만 하면 플래토 아님)
    order = [sums[tag(b)] / n for b in BANDS]
    sorder = [ssums[tag(b)] / sn for b in BANDS]
    mono = all(order[i] >= order[i + 1] for i in range(len(order) - 1))
    smono = all(sorder[i] >= sorder[i + 1] for i in range(len(sorder) - 1))
    bbest = BANDS[int(np.argmin(order))]
    sbest = BANDS[int(np.argmin(sorder))]
    print(f"\n[판정 보조] 기본비용 최상위 {bbest*100:.0f}% / 스트레스 최상위 {sbest*100:.0f}%")
    print(f"  넓힐수록 단조 개선? 기본 {'예' if mono else '아니오'} / 스트레스 "
          f"{'예' if smono else '아니오'}")
    print("  단조면 = 꺾이는 지점이 이 구간 밖 → 플래토 미확인, 채택 근거로 쓰지 않는다.")
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
