"""위험을 줄이는 매도를 체결 밴드에서 뺄지 (2026-08-27).

계기: 08-26 라이브에서 레버리지 하향 3건(BNB 3→2, SOL 4→3, ETH 3→2) 뒤 목표 명목이 줄었는데
남은 과다노출 -2.1 ~ -3.0% 가 밴드(5%) 안이라 주문이 안 나갔다. 위험을 줄여야 하는 자리에서
"밴드 안이니 달성" 으로 넘어간 것이다.

게다가 밴드는 방향이 비대칭이다. delta = (목표 - 현재)/현재 라 분모가 현재값이어서
같은 금액이 어긋나도 부족한 쪽은 +5.0% 로 발화하고 넘치는 쪽은 -4.76% 라 발화하지 않는다.
구조가 과다노출에 더 관대한데, 레버리지 하향 국면에서는 정확히 반대여야 한다.

축은 하나 — 위험축소 매도(목표 명목이 줄어드는 쪽)의 밴드만 바꾼다. 매수는 5% 그대로.

  S0 대칭 5%(현행)   매도·매수 모두 5%
  S1 매도 1%         과다노출은 1% 만 넘으면 줄인다
  S2 매도 0%         과다노출은 무조건 줄인다(최소주문 제약만 남음)

무엇을 보고 정하나 (결과 보기 전 고정):
  1. 판정은 합성 60/25/15 기준이다(2026-08-25 채택 원칙). 슬리브 단독은 참고.
  2. 합성에서 S1·S2 가 S0 대비 나아지고 기본비용·비용5배가 같은 방향이면 라이브 변경 후보.
     방향이 갈리거나 차이가 Calmar 0.3 미만이면 현행 유지 — 성과가 아니라 구조가 이유라면
     그건 별도 판단이고, 그때도 회전수 증가폭을 같이 본다.
  3. 이건 파라미터 최적화가 아니다. 0/1/5% 셋만 보고 그 사이를 훑지 않는다.

변형은 정본 엔진 env 토글 FUT_SELL_BAND 로 넣었고 결론 후 되돌린다. 재현하려면 먼저
  git apply strategies/cap_defend/research/bt_fut_sell_band.engine.patch

실행: cd strategies/cap_defend/research && python3 bt_fut_sell_band.py
"""
from __future__ import annotations
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from bt_fut_rebal_band import (  # noqa: E402
    run_spot_live, run_stock_live, build_alloc, metrics, window_rs,
    WIN_SIZES, STRIDES, BASE_TX, STRESS_TX, START,
)

END = "2026-08-24"
ARMS = [("S0 대칭 5%(현행)", 0.05), ("S1 매도 1%", 0.01), ("S2 매도 0%", 0.0)]


def run_fut(sell_band: float, tx_cost: float = BASE_TX):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['FUT_SELL_BAND'] = f'{sell_band}'
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                tx_cost=tx_cost, maint_rate=0.004,
                sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                n_snapshots=5, snap_interval_bars=95,
                start_date=START, end_date=END)
    os.environ['FUT_SELL_BAND'] = '0.05'
    return m


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 위험축소 매도 밴드만 (매수 5% 고정)")
    print(f"팔 {[a[0] for a in ARMS]} | tx {BASE_TX} (stress {STRESS_TX})\n")

    print(f"[sleeve 단독 — tx {BASE_TX*100:.2f}%]")
    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'Trades':>7s} {'Rebal':>6s} {'Liq':>4s} {'최종':>9s}")
    res, eqs = {}, {}
    for name, sb in ARMS:
        m = run_fut(sb)
        res[name] = m
        eqs[name] = m['_equity']
        print(f"  {name:<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>7d} {m['Rebal']:>6d} {m['Liq']:>4d} "
              f"{eqs[name].iloc[-1]/eqs[name].iloc[0]:>8.1f}x", flush=True)

    base = ARMS[0][0]
    print("\n[회전 부담]")
    for k in eqs:
        print(f"  {k:<16s} 체결 {res[k]['Trades']:>5d} "
              f"(현행 대비 {res[k]['Trades']/res[base]['Trades']:>5.2f}x)")

    print("\n[윈도우 rank-sum — sleeve, 기본비용]")
    s1, w1, n1 = window_rs(eqs)
    for k in sorted(s1, key=lambda k: s1[k]):
        print(f"  {k:<16s} avg_rank {s1[k]/n1:.3f}  wins {w1[k]:>4d}/{n1}")
    print(f"  windows n={n1} (sizes {WIN_SIZES} × strides {STRIDES})")

    print(f"\n[비용 stress — tx {STRESS_TX*100:.2f}%]")
    st = {}
    for name, sb in ARMS:
        m = run_fut(sb, tx_cost=STRESS_TX)
        st[name] = m['_equity']
        print(f"  {name:<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>7d}", flush=True)
    s2, w2, n2 = window_rs(st)
    print("  [윈도우 rank-sum — stress]")
    for k in sorted(s2, key=lambda k: s2[k]):
        print(f"    {k:<16s} avg_rank {s2[k]/n2:.3f}  wins {w2[k]:>4d}/{n2}")

    print("\n[합성 60/25/15 — 결정층]")
    eq_sp, eq_st = run_spot_live(), run_stock_live()
    comp = {k: build_alloc(eq_st, eq_sp, v) for k, v in eqs.items()}
    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s}")
    for k, e in comp.items():
        mm = metrics(e)
        print(f"  {k:<16s} {mm['Cal']:>6.2f} {mm['CAGR']:>+7.1f}% {mm['MDD']:>+7.1f}% "
              f"{mm['Sharpe']:>7.2f}")
    s3, w3, n3 = window_rs(comp)
    print("  [윈도우 rank-sum — 합성]")
    for k in sorted(s3, key=lambda k: s3[k]):
        print(f"    {k:<16s} avg_rank {s3[k]/n3:.3f}  wins {w3[k]:>4d}/{n3}")

    comp_rank = sorted(s3, key=lambda k: s3[k])
    cal = {k: metrics(v)['Cal'] for k, v in comp.items()}
    print(f"\n[판정] 합성 1위 {comp_rank[0]} (Cal {cal[comp_rank[0]]:.2f} vs 현행 {cal[base]:.2f})")
    if comp_rank[0] == base:
        print("  현행이 합성 1위 → 성과 근거로는 변경 없음. 구조 근거(위험축소 지연)는 별도 판단.")
    elif abs(cal[comp_rank[0]] - cal[base]) < 0.3:
        print(f"  차이 {abs(cal[comp_rank[0]]-cal[base]):.2f} < 0.3 → 성과로는 동률. "
              f"회전수 {res[comp_rank[0]]['Trades']} vs 현행 {res[base]['Trades']} 로 판단.")
    else:
        print("  합성에서 뚜렷하게 앞선다 → 라이브 변경 후보. ai-debate 로 넘긴다.")
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
