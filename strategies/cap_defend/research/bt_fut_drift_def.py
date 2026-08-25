"""선물 드리프트 트리거 정의 3경로 판정 (P1-a, 2026-08-25).

문제: 라이브·BT 모두 드리프트를 half_turnover(cur_w, tgt_w) 로 재는데 cur_w 가
(진입마진+PnL)/equity 다. 레버리지가 걸려 있으면 목표 명목을 정확히 맞춘 직후에도
평가손익 때문에 cur_w 가 tgt_w 에서 벌어진다 — 08-22 실측 ht=0.12, 08-23 ht=0.16.
문턱 0.03 을 코인 일일 노이즈가 매일 넘으므로 트리거가 사실상 상시 ON 이고,
실제 게이트 역할은 체결 밴드(현행 5%)가 한다. 즉 지금 드리프트는 "언제 리밸런싱을
할지" 를 정하지 못하고 "매일 한다" 로 퇴화해 있다.

축은 정의 하나. 문턱은 안 건드린다(스윕 금지 — 다중검정).

  A 현행    cur_w = (진입마진 + 평가손익) / equity,  문턱 0.03
  B 명목편차 cur_w = 명목 / (equity × 0.95 × L),      문턱 0.03
            실행이 실제로 맞추는 값이 명목이라, 목표 명목을 맞춘 상태면 편차가 0 이 된다.
            가격이 움직여도 목표 명목이 같이 움직이므로 L 전환·목표 변경 때만 벌어진다.
  C 제거    드리프트 트리거 없음(문턱 0). 앵커(95봉)와 카나리만 리밸런싱을 만든다.

무엇을 보고 정하나 (결과 보기 전 고정):
  1. 세 팔의 sleeve 윈도우 rank-sum 이 기본비용·비용 5배 양쪽에서 같은 방향이어야 한다.
     양쪽이 갈리면 "현행 유지" 다 — 정의를 바꾸는 건 라이브 변경이라 근거가 갈리면 안 바꾼다.
  2. 성과가 사실상 동률(Calmar 차 0.3 미만)이면 회전수가 적은 쪽이 낫다.
     드리프트의 존재 이유가 회전을 늘리는 게 아니기 때문이다.
  3. C 가 이기면 그건 "드리프트는 선물에서 값을 못 낸다" 는 뜻이고, 그때는 A→C 가 아니라
     B 를 먼저 본다 — 트리거를 없애는 건 카나리 OFF 밖의 위험축소 경로를 지우는 것이라
     성과 동률로는 정당화되지 않는다.

변형은 정본 엔진 backtest_futures_v25.py 의 env 토글 FUT_DRIFT_DEF 로 구현했다(기본 margin=현행).
결론 후 SSoT 오염 방지로 엔진을 되돌리므로, 재현하려면 먼저
  git apply strategies/cap_defend/research/bt_fut_drift_def.engine.patch
를 적용한다. 적용 없이 돌리면 A·B 가 같은 값으로 나온다(가짜 동률).

실행: cd strategies/cap_defend/research && python3 bt_fut_drift_def.py
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
    window_rs, metrics, build_alloc, run_spot_live, run_stock_live,
    WIN_SIZES, STRIDES, BASE_TX, STRESS_TX, START,
)

END = "2026-08-24"          # 데이터 최신화 후 마지막 확정봉
ARMS = [
    ("A 현행(마진+PnL)", 'margin', 0.03),
    ("B 명목편차", 'notional', 0.03),
    ("C 드리프트 제거", 'margin', 0.0),
]


def run_fut(defn: str, drift: float, tx_cost: float = BASE_TX):
    """라이브 V25 선물 sleeve 설정 그대로. 드리프트 정의/유무만 바꾼다."""
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['FUT_DRIFT_DEF'] = defn
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                tx_cost=tx_cost, maint_rate=0.004,
                sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                canary_hyst=0.015, drift_threshold=drift, post_flip_delay=5,
                health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                n_snapshots=5, snap_interval_bars=95,
                start_date=START, end_date=END)
    os.environ['FUT_DRIFT_DEF'] = 'margin'
    return m


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 드리프트 정의만 (문턱 스윕 없음)")
    print(f"팔 {[a[0] for a in ARMS]} | tx {BASE_TX} (stress {STRESS_TX})\n")

    res = {}
    print(f"[sleeve 단독 — tx {BASE_TX*100:.2f}%]")
    print(f"  {'arm':<18s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'Trades':>7s} {'Rebal':>6s} {'Liq':>4s} {'최종':>9s} {'초':>5s}")
    for name, defn, drift in ARMS:
        t1 = time.time()
        m = run_fut(defn, drift)
        eq = m['_equity']
        res[name] = dict(m=m, eq=eq)
        print(f"  {name:<18s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>7d} {m['Rebal']:>6d} {m['Liq']:>4d} "
              f"{eq.iloc[-1]/eq.iloc[0]:>8.1f}x {time.time()-t1:>5.0f}", flush=True)

    eqs = {k: v['eq'] for k, v in res.items()}
    yrs = (list(eqs.values())[0].index[-1] - list(eqs.values())[0].index[0]).days / 365.25
    base = ARMS[0][0]
    print(f"\n[회전 부담] 기간 {yrs:.1f}년")
    for k, v in res.items():
        tr, rb = v['m']['Trades'], v['m']['Rebal']
        print(f"  {k:<18s} 체결 {tr:>5d} ({tr/yrs:>6.1f}회/년, 현행 대비 {tr/res[base]['m']['Trades']:>5.2f}x)"
              f"  리밸 {rb:>5d} ({rb/yrs:>6.1f}회/년, {rb/res[base]['m']['Rebal']:>5.2f}x)")

    print("\n[연도별 수익률 / MDD]")
    keys = list(eqs)
    print("  year   " + "".join(f"{k:>22s}" for k in keys))
    for year in sorted({d.year for d in list(eqs.values())[0].index}):
        cells = []
        for k in keys:
            seg = eqs[k][eqs[k].index.year == year].dropna()
            if len(seg) < 30:
                cells.append(f"{'-':>22s}")
                continue
            r = seg.iloc[-1] / seg.iloc[0] - 1
            mdd = float((seg / seg.cummax() - 1).min())
            cells.append(f"{r:>+11.1%}/{mdd:>+9.1%}")
        print(f"  {year:<6d} " + "".join(cells))

    print("\n[윈도우 rank-sum — sleeve, 기본비용]")
    s_base, w_base, n_base = window_rs(eqs)
    for k in sorted(s_base, key=lambda k: s_base[k]):
        print(f"  {k:<18s} avg_rank {s_base[k]/n_base:.3f}  wins {w_base[k]:>4d}/{n_base} "
              f"({w_base[k]/n_base*100:.0f}%)")
    print(f"  windows n={n_base} (sizes {WIN_SIZES} × strides {STRIDES})")

    print(f"\n[비용 stress — tx {STRESS_TX*100:.2f}% (5x)]")
    stress = {}
    print(f"  {'arm':<18s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Trades':>7s}")
    for name, defn, drift in ARMS:
        m = run_fut(defn, drift, tx_cost=STRESS_TX)
        stress[name] = m['_equity']
        print(f"  {name:<18s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>7d}", flush=True)
    s_st, w_st, n_st = window_rs(stress)
    print("  [윈도우 rank-sum — stress]")
    for k in sorted(s_st, key=lambda k: s_st[k]):
        print(f"    {k:<18s} avg_rank {s_st[k]/n_st:.3f}  wins {w_st[k]:>4d}/{n_st}")

    print("\n[합성 60/25/15 — 현물·주식 고정, 선물만 교체]")
    try:
        eq_sp = run_spot_live()
        eq_st = run_stock_live()
        comp = {k: build_alloc(eq_st, eq_sp, v) for k, v in eqs.items()}
        print(f"  {'arm':<18s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s}")
        for k, e in comp.items():
            mm = metrics(e)
            print(f"  {k:<18s} {mm['Cal']:>6.2f} {mm['CAGR']:>+7.1f}% {mm['MDD']:>+7.1f}% "
                  f"{mm['Sharpe']:>7.2f}")
        s_c, w_c, n_c = window_rs(comp)
        print("  [윈도우 rank-sum — 합성]")
        for k in sorted(s_c, key=lambda k: s_c[k]):
            print(f"    {k:<18s} avg_rank {s_c[k]/n_c:.3f}  wins {w_c[k]:>4d}/{n_c}")
    except Exception as e:                                    # noqa: BLE001
        print(f"  합성 계산 실패({type(e).__name__}: {e}) — sleeve 판정만 사용한다.")

    b_rank = sorted(s_base, key=lambda k: s_base[k])
    st_rank = sorted(s_st, key=lambda k: s_st[k])
    cal = {k: v['m']['Cal'] for k, v in res.items()}
    print(f"\n[판정] 기본비용 1위 {b_rank[0]} / 비용5배 1위 {st_rank[0]}")
    if b_rank[0] != st_rank[0]:
        print("  두 비용 조건이 다른 팔을 가리킨다 → 현행 유지(정의 변경 안 함).")
    elif abs(cal[b_rank[0]] - cal[base]) < 0.3:
        print(f"  성과 동률(Calmar 차 {abs(cal[b_rank[0]]-cal[base]):.2f}) → 회전수로 판단. "
              f"현행 체결 {res[base]['m']['Trades']} vs 1위 {res[b_rank[0]]['m']['Trades']}")
    else:
        print(f"  {b_rank[0]} 이 양 조건 모두 우세하고 Calmar 차 "
              f"{abs(cal[b_rank[0]]-cal[base]):.2f} → 라이브 변경 후보. ai-debate 로 넘긴다.")
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
