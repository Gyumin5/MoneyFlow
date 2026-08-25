"""현물 리밸런싱 체결 밴드 비교 — BT 5% (현행) vs 라이브 밴드없음 (2026-08-25).

계기: 라이브 현물(trade/executor_coin.py)에는 체결 밴드가 없다. 목표와 현재 비중이 조금만
달라도 최소주문(5,000원)만 넘으면 그대로 낸다. 반면 채택 BT(unified_backtest._execute_rebalance)는
수량 ±5% 밴드를 벗어날 때만 체결한다. 선물은 같은 축을 이미 쟀지만(bt_fut_rebal_band.py)
현물은 안 쟀다. 라이브가 BT 가정보다 자주 돈다면 실제 비용이 BT 보다 크다.

가설: 밴드를 없애면 목표 추종은 정확해지지만 체결·슬리피지 비용이 늘어 순손해다.
반대로 손해가 아니면 BT 를 라이브에 맞추는 게 옳고, 손해면 라이브에 밴드를 넣는 게
논의 대상이 된다. (라이브 변경은 별도 승인 사항 — 이 스크립트는 측정만 한다.)

축: 체결 밴드 하나. 선정·헬스·카나리·스냅샷·드리프트·비용 전부 채택 V24 현물 설정 고정.
변형은 정본 엔진 unified_backtest.py 의 env 토글 REBAL_BAND 로 구현했다(기본 0.05=현행).
측정 후 SSoT 오염 방지를 위해 엔진 패치를 되돌리므로, 재현하려면 먼저
  git apply strategies/cap_defend/research/bt_spot_rebal_band.engine.patch
를 적용해야 한다. 적용 없이 돌리면 전 팔이 0.05 로 동작해 차이 0 이 나온다(가짜 동률).

판정 (결과 보기 전 고정):
  윈도우 rank-sum 평균순위에서 밴드없음(0%)이 현행(5%)보다 나쁘지 않고(차이 0.1 미만)
  거래비용 5배에서도 뒤집히지 않으면 → 밴드는 무의미. BT 를 라이브에 맞춘다(문서·엔진 기본값).
  밴드없음이 유의하게 나쁘면 → 라이브 무밴드가 실제 손실원. 밴드 도입 검토를 연다.
  밴드없음이 유의하게 좋으면 → 현행 BT 성과표가 과소평가. 채택 수치를 다시 낸다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_rebal_band.py
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-08-24"
WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]
BANDS = [0.0, 0.01, 0.02, 0.05, 0.10]
BASE_TX = 0.004
STRESS_TX = 0.02  # 5x


def _tag(band):
    suffix = "(라이브)" if band == 0.0 else "(현행BT)" if band == 0.05 else ""
    return f"band {band*100:.0f}%{suffix}"


def run_spot(band, tx_cost=BASE_TX):
    """채택 V24 현물 sleeve 설정 그대로. 체결 밴드만 바꾼다."""
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['REBAL_BAND'] = f'{band}'
    bars, funding = load_data('D')
    m = bt_run(bars, funding, interval='D', asset_type='spot', leverage=1.0,
               tx_cost=tx_cost, start_date=START, end_date=END,
               sma_bars=42, mom_short_bars=20, mom_long_bars=127,
               vol_threshold=0.05, vol_mode='daily',
               n_snapshots=7, snap_interval_bars=217,
               canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
               universe_size=3, cap=1 / 3, health_mode='mom2vol')
    os.environ['REBAL_BAND'] = '0.05'
    return m


def window_rs(eq_dict):
    """윈도우 rank-sum (홀드아웃 금지 규약). 낮은 avg_rank 가 우세."""
    common = None
    for s in eq_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    sums = defaultdict(float)
    wins = defaultdict(int)
    n = 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                cals = {}
                for k, s in eq_dict.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        cals[k] = np.nan
                        continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0:
                        cals[k] = np.nan
                        continue
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    mdd = float((seg / seg.cummax() - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()):
                    continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1):
                    sums[mk] += r
                wins[ranked[0][0]] += 1
                n += 1
    return sums, wins, n


def report(res, tx_label):
    eqs = {k: v['eq'] for k, v in res.items()}
    any_eq = list(eqs.values())[0]
    yrs = (any_eq.index[-1] - any_eq.index[0]).days / 365.25
    base_tr = res[_tag(0.05)]['m']['Trades']
    sums, wins, n = window_rs(eqs)
    print(f"\n[{tx_label}] 기간 {yrs:.1f}년 · 윈도우 {n}개")
    print(f"  {'arm':<18s} {'순위':>6s} {'승률':>6s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} "
          f"{'Sharpe':>7s} {'체결':>6s} {'회/년':>6s} {'대비':>6s}")
    for band in BANDS:
        k = _tag(band)
        m = res[k]['m']
        tr = m['Trades']
        print(f"  {k:<18s} {sums[k]/n:>6.3f} {wins[k]/n*100:>5.1f}% {m['Cal']:>6.2f} "
              f"{m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} {m['Sharpe']:>7.2f} "
              f"{tr:>6d} {tr/yrs:>6.1f} {tr/base_tr:>5.2f}x")
    return sums, n


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 체결 밴드만 (현물 V24, drift 0.10 고정)")
    print(f"밴드 후보 {[f'{b*100:.0f}%' for b in BANDS]} | tx {BASE_TX} (stress {STRESS_TX})")

    out = {}
    for tx, label in ((BASE_TX, f"거래비용 편도 {BASE_TX*100:.1f}%"),
                      (STRESS_TX, f"거래비용 5배 스트레스 {STRESS_TX*100:.1f}%")):
        res = {}
        for band in BANDS:
            t1 = time.time()
            m = run_spot(band, tx)
            res[_tag(band)] = dict(m=m, eq=m.get('_equity'), band=band)
            print(f"  {_tag(band):<18s} 완료 ({time.time()-t1:.0f}s)", flush=True)
        out[label] = report(res, label)

    print("\n[판정]")
    for label, (sums, n) in out.items():
        gap = (sums[_tag(0.0)] - sums[_tag(0.05)]) / n
        verdict = ("무차별" if abs(gap) < 0.1 else
                   "밴드없음(라이브) 열위" if gap > 0 else "밴드없음(라이브) 우위")
        print(f"  {label}: 라이브-현행 평균순위 차 {gap:+.3f} → {verdict}")
    print(f"\n총 소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
