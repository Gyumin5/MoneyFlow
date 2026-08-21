"""앵커 갱신 시점 — 시간조건(현행) vs 사건조건(드리프트 발화) 비교 (2026-08-21).

가설: 앵커 슬롯 갱신을 (bar_i % 95 == 슬롯오프셋) 이라는 달력 조건에서
"드리프트가 발화할 때마다 다음 슬롯을 갱신" 이라는 사건 조건으로 바꾸면,
포트폴리오가 실제로 어긋난 시점에만 종목을 새로 뽑으므로 반응성이 좋아질 수 있다.
반대 가설: 선물 드리프트는 2009봉 중 760일 발화한다 — 사건구동은 갱신 빈도가 폭증해
종목교체 비용이 늘고, 5슬롯이 사실상 같은 시점을 보게 되어 스태거 분산이 붕괴한다.

축: 그것 하나만. 선정·헬스·카나리·드리프트 문턱·레버리지·비용 전부 동일.
팔 4개 (정본 엔진 env 토글 ANCHOR_EVENT):
  off    = 현행 (95봉 시간조건만)
  pure   = 시간조건 제거, 드리프트 발화 때만 다음 슬롯 갱신
  spaced = pure + 최소간격 19봉(=snap_interval/n, 현행 스태거 유지)
  hybrid = 시간 OR 사건 (먼저 오는 쪽)

결론 후 SSoT 오염 방지를 위해 엔진 패치는 되돌린다 — 재현하려면 먼저
  git apply strategies/cap_defend/research/bt_anchor_event.engine.patch
를 적용해야 한다. 적용 없이 돌리면 네 팔이 모두 현행이 되어 차이 0 이 나온다(가짜 동률).

판정: 윈도우 rank-sum 을 Calmar·Sharpe·CAGR 세 지표로 각각 매기고(Cal 단독 판정 금지 —
2026-08-21 L 적용시점 건에서 Cal 이 단일 저점 하나에 뒤집혔다), 거래비용 5배 stress,
연도별 일관성까지 본다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_anchor_event.py
출력: state/<job>/ 로 리다이렉트 (원시 로그는 리포트가 아니다).
env: WITH_ALLOC=on 이면 자산배분 60/25/15 합성까지, ANCHOR_OUT=<csv> 면 equity 저장.
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

from bt_fut_daily_lev import (START, END, WIN_SIZES, STRIDES,  # noqa: E402
                             run_spot_live, run_stock_live, build_alloc, metrics)

ARMS = [('off(현행)', 'off'), ('pure', 'pure'), ('spaced19', 'spaced'), ('hybrid', 'hybrid')]


def run_fut(arm: str, tx=0.0006, drift=0.03, sn=95, n_snap=5):
    """라이브 V25 선물 sleeve 설정 그대로. ANCHOR_EVENT 만 바꾼다."""
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['ANCHOR_EVENT'] = arm
    try:
        bars, funding = load_data('D')
        k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                             btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                             l_min=2.0, l_mid=3.0, l_max=4.0)
        return fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                       tx_cost=tx, maint_rate=0.004,
                       sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                       canary_hyst=0.015, drift_threshold=drift, post_flip_delay=5,
                       health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                       n_snapshots=n_snap, snap_interval_bars=sn,
                       start_date=START, end_date=END)
    finally:
        os.environ['ANCHOR_EVENT'] = 'off'


def window_rs(eq_dict, metric='cal'):
    """윈도우 rank-sum (홀드아웃 금지 규약). metric: cal | sharpe | cagr. 낮은 avg_rank 우세."""
    common = None
    for s in eq_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    sums, wins, n = defaultdict(float), defaultdict(int), 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                vals = {}
                for k, s in eq_dict.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        vals[k] = np.nan
                        continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0:
                        vals[k] = np.nan
                        continue
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
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 앵커 갱신 시점만 (선정·헬스·드리프트·비용 동일)")

    res = {}
    for tag, arm in ARMS:
        t1 = time.time()
        m = run_fut(arm)
        eq = m.get('_equity')
        res[tag] = dict(m=m, eq=eq)
        print(f"\n[{tag}] {time.time()-t1:.0f}s  bars={len(eq)} "
              f"{eq.index[0].date()}~{eq.index[-1].date()}")
        print(f"  Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | MDD {m['MDD']:+.1%} | "
              f"Sharpe {m['Sharpe']:.2f}")
        print(f"  Trades {m['Trades']} | Rebal {m['Rebal']} | Liq {m['Liq']} | "
              f"사건갱신 {m.get('AnchorEvents', 0)} | 간격스킵 {m.get('AnchorSkipped', 0)} | "
              f"최종 {eq.iloc[-1]/eq.iloc[0]:.2f}x")

    eqs = {k: v['eq'] for k, v in res.items()}
    tags = [t for t, _ in ARMS]

    print("\n[연도별 수익률 / MDD]")
    hdr = ' '.join(f"{t:>16s}" for t in tags)
    print(f"  {'year':<6s} {hdr}")
    for year in sorted({d.year for d in eqs[tags[0]].index}):
        cells = []
        for t in tags:
            seg = eqs[t][eqs[t].index.year == year].dropna()
            if len(seg) < 30:
                cells.append(f"{'-':>16s}")
                continue
            r = seg.iloc[-1] / seg.iloc[0] - 1
            mdd = float((seg / seg.cummax() - 1).min())
            cells.append(f"{r:>+8.1%}/{mdd:>+7.1%}")
        print(f"  {year:<6d} " + ' '.join(cells))

    print("\n[윈도우 rank-sum — 지표 3종 (Cal 단독 판정 금지)]")
    for metric in ['cal', 'sharpe', 'cagr']:
        sums, wins, n = window_rs(eqs, metric)
        line = ' | '.join(f"{k} {sums[k]/n:.3f}(승{wins[k]})"
                          for k in sorted(sums, key=lambda k: sums[k]))
        print(f"  {metric:<7s} n={n}  {line}")
    print(f"  windows sizes {WIN_SIZES} × strides {STRIDES}, 낮은 avg_rank 우세")

    print("\n[비용 stress — tx 5x (0.30%)]")
    for tag, arm in ARMS:
        m = run_fut(arm, tx=0.003)
        print(f"  {tag:<12s} Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | "
              f"MDD {m['MDD']:+.1%} | Sharpe {m['Sharpe']:.2f} | Trades {m['Trades']}")

    print("\n[드리프트 문턱 민감도 — 사건 빈도가 축의 세기다]")
    for thr in [0.03, 0.10]:
        for tag, arm in ARMS:
            m = run_fut(arm, drift=thr)
            print(f"  drift={thr:.2f} {tag:<12s} Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | "
                  f"MDD {m['MDD']:+.1%} | Sharpe {m['Sharpe']:.2f} | "
                  f"사건 {m.get('AnchorEvents', 0):>4d} | Trades {m['Trades']:>5d}")

    out = os.environ.get('ANCHOR_OUT', '')
    if out:
        pd.DataFrame(eqs).to_csv(out)
        print(f"\nequity 저장: {out}")

    if os.environ.get('WITH_ALLOC', 'off') == 'on':
        print("\n[자산배분 60/25/15 합성 — 선물 sleeve 만 교체]")
        eq_sp = run_spot_live()
        eq_st = run_stock_live()
        rows = {t: build_alloc(eq_st, eq_sp, e) for t, e in eqs.items()}
        for t, a in rows.items():
            mm = metrics(a)
            print(f"  {t:<12s} Cal {mm['Cal']:.2f} | CAGR {mm['CAGR']:+.1f}% | "
                  f"MDD {mm['MDD']:+.1f}% | Sharpe {mm['Sharpe']:.2f}")
        for metric in ['cal', 'sharpe', 'cagr']:
            sums, wins, n = window_rs(rows, metric)
            line = ' | '.join(f"{k} {sums[k]/n:.3f}(승{wins[k]})"
                              for k in sorted(sums, key=lambda k: sums[k]))
            print(f"  alloc {metric:<7s} n={n}  {line}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
