"""앵커 갱신 시점 — 현물 sleeve (V24) 팔 4개 (2026-08-21, 사용자 요청 "선물과 현물 모두").

선물편과 같은 축·같은 팔. 엔진은 현물 정본 unified_backtest.py 의 env 토글 ANCHOR_EVENT.
현물 라이브: n_snapshots=7, snap_interval=217(스태거 31), drift=0.10, tx 0.4%.
선물(드리프트 0.03, 스태거 19)보다 문턱이 높고 스태거가 길어 사건 빈도·간격이 다르다 —
같은 결론이 두 슬리브에서 독립적으로 나오는지가 과적합 판별의 핵심이다.

재현: git apply strategies/cap_defend/research/bt_anchor_event.engine.patch (선물+현물 둘 다 포함)
적용 없이 돌리면 네 팔이 모두 현행이 되어 차이 0 (가짜 동률).

실행: cd .../research && python3 bt_anchor_event_spot.py
env: ANCHOR_SPOT_OUT=<csv> 면 equity 저장.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from bt_fut_daily_lev import START, END, WIN_SIZES, STRIDES  # noqa: E402,F401
from bt_anchor_event import ARMS, window_rs  # noqa: E402


def run_spot(arm: str, tx=0.004, drift=0.10, sn=217, n_snap=7):
    """현물 sleeve 라이브 V24 설정 그대로. ANCHOR_EVENT 만 바꾼다."""
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['ANCHOR_EVENT'] = arm
    try:
        bars, funding = load_data('D')
        return bt_run(bars, funding, interval='D', asset_type='spot', leverage=1.0,
                      tx_cost=tx, start_date=START, end_date=END,
                      sma_bars=42, mom_short_bars=20, mom_long_bars=127,
                      vol_threshold=0.05, vol_mode='daily',
                      n_snapshots=n_snap, snap_interval_bars=sn,
                      canary_hyst=0.015, drift_threshold=drift, post_flip_delay=5,
                      universe_size=3, cap=1 / 3, selection='greedy',
                      stop_kind='none', stop_pct=0.0,
                      dd_lookback=60, dd_threshold=-99.0,
                      bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
                      health_mode='mom2vol')
    finally:
        os.environ['ANCHOR_EVENT'] = 'off'


def main():
    t0 = time.time()
    print(f"[현물 sleeve V24] 기간 {START} ~ {END} | 축: 앵커 갱신 시점만")
    print("  라이브 설정: n_snap=7 snap=217(스태거31) drift=0.10 tx=0.4%")

    eqs = {}
    for tag, arm in ARMS:
        t1 = time.time()
        m = run_spot(arm)
        eq = m.get('_equity')
        eqs[tag] = eq
        print(f"\n[{tag}] {time.time()-t1:.0f}s  bars={len(eq)}")
        print(f"  Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | MDD {m['MDD']:+.1%} | "
              f"Sharpe {m['Sharpe']:.2f}")
        print(f"  Trades {m['Trades']} | Rebal {m['Rebal']} | "
              f"사건갱신 {m.get('AnchorEvents', 0)} | 간격스킵 {m.get('AnchorSkipped', 0)} | "
              f"최종 {eq.iloc[-1]/eq.iloc[0]:.2f}x")

    tags = [t for t, _ in ARMS]
    print("\n[연도별 수익률 / MDD]")
    print(f"  {'year':<6s} " + ' '.join(f"{t:>16s}" for t in tags))
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

    print("\n[윈도우 rank-sum — 지표 3종]")
    for metric in ['cal', 'sharpe', 'cagr']:
        sums, wins, n = window_rs(eqs, metric)
        line = ' | '.join(f"{k} {sums[k]/n:.3f}(승{wins[k]})"
                          for k in sorted(sums, key=lambda k: sums[k]))
        print(f"  {metric:<7s} n={n}  {line}")

    print("\n[비용 stress — tx 5x (2.0%)]")
    for tag, arm in ARMS:
        m = run_spot(arm, tx=0.02)
        print(f"  {tag:<12s} Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | "
              f"MDD {m['MDD']:+.1%} | Sharpe {m['Sharpe']:.2f} | Trades {m['Trades']}")

    print("\n[드리프트 문턱 민감도 — 사건 빈도가 축의 세기다]")
    for thr in [0.05, 0.10, 0.20]:
        for tag, arm in ARMS:
            m = run_spot(arm, drift=thr)
            print(f"  drift={thr:.2f} {tag:<12s} Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | "
                  f"MDD {m['MDD']:+.1%} | Sharpe {m['Sharpe']:.2f} | "
                  f"사건 {m.get('AnchorEvents', 0):>4d} | Trades {m['Trades']:>5d}")

    out = os.environ.get('ANCHOR_SPOT_OUT', '')
    if out:
        pd.DataFrame(eqs).to_csv(out)
        print(f"\nequity 저장: {out}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
