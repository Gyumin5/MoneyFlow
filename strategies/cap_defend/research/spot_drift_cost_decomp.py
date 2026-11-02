"""드리프트 0.30 이 왜 이기나 — 비용·거래 분해와 경계 확장.

토론(run-20261102T022434Z) action_step 5: "드리프트 0.30에는 '비용·거래 분해'와
'경계 확장'을 건다. 편도 비용 0·0.2%·0.4%·0.8%에서 총수익·순수익·거래량을
비교하고, 문턱 0.10·0.20·0.30·0.40·0.50 및 드리프트 발화만 끈 대조군을 측정한다.
끄는 인자의 의미는 실제 분기로 확인하며 0을 임의로 끔으로 해석하지 않는다."

끄는 인자를 코드에서 확인했다. `unified_backtest.run` 1022행이
    if not need_rebal and canary_on and drift_threshold > 0 and holdings and is_daily_bar:
라서 `drift_threshold=0` 이 정확히 "드리프트 발화만 끔" 이다. 스냅샷 재조정과 카나리
청산은 그 분기 밖이라 그대로 남는다 — 토론이 유지하라고 한 둘이다.

무엇을 가르나. 드리프트 문턱을 올리면 갈아타기가 줄어드는데, 그 이득이 둘 중 어느
쪽인지가 안 갈려 있다.
  ⑴ 거래비용 절감 — 같은 매매를 덜 해서 수수료를 아낀다. 비용 0 에서는 사라져야 한다.
  ⑵ 승자 비중 유지 — 오른 코인을 덜 팔아 더 들고 간다. 비용 0 에서도 남아야 한다.
둘은 성격이 다르다. ⑴ 뿐이면 실제 비용이 바뀔 때 결론이 같이 바뀌고, ⑵ 가 있으면
비용과 무관한 구조적 이득이다. 토론의 소수의견대로 비용 0 에서 우세가 사라지는 것만으로
기각하지는 않는다 — 실비용 절감도 유효한 개선이다. 가르는 것이 목적이다.

경계 확장은 0.40·0.50 까지 본다. 거기까지 계속 좋아지면 "0.30 이 최적" 이라는 주장은
철회하고 "드리프트를 줄이거나 아예 끈다" 는 가설로 바꾼다.

문턱 6 × 비용 4 × 앵커 7 = 168칸.

실행: python3 spot_drift_cost_decomp.py
      python3 spot_drift_cost_decomp.py --summary
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import unified_backtest as ub  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-drift-cost')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1 / 3,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=217,
)

# 0 = 드리프트 발화만 끔(1022행 가드). 스냅샷 재조정·카나리 청산은 남는다.
DRIFTS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
COSTS = [0.0, 0.002, 0.004, 0.008]


def main():
    os.makedirs(OUT, exist_ok=True)
    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    path = os.path.join(OUT, 'cells.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['drift'], r['tx'], r['phase']))
            except Exception:
                pass
    print(f"이미 끝난 칸 {len(done)}개 / 전체 {len(DRIFTS)*len(COSTS)*len(PHASES)}",
          flush=True)

    t_start = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for d in DRIFTS:
            for tx in COSTS:
                for ph in PHASES:
                    if (d, tx, ph) in done:
                        continue
                    kw = dict(BASE, drift_threshold=d, tx_cost=tx,
                              phase_offset_bars=ph)
                    m = ub.run(bars, fund, start_date=START, end_date=END, **kw)
                    row = dict(drift=d, tx=tx, phase=ph,
                               **{k: m.get(k) for k in
                                  ('CAGR', 'MDD', 'Sharpe', 'Trades', 'Rebal')})
                    dd = abs(float(row['MDD'] or 0.0))
                    row['Cal'] = (float(row['CAGR']) / dd) if dd > 0 else None
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                    f.flush()
            print(f"  문턱 {d} 끝 ({(time.time()-t_start)/60:.1f}분)", flush=True)
    summary(path)
    return 0


def summary(path=None):
    path = path or os.path.join(OUT, 'cells.jsonl')
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    agg = {}
    for r in rows:
        agg.setdefault((r['drift'], r['tx']), []).append(r)

    print("\n문턱 × 비용 — 연복리 (앵커 7곳 평균)")
    header = "  문턱  " + "".join(f"{int(c*10000):>9}bp" for c in COSTS)
    print(header)
    for d in DRIFTS:
        cells = []
        for tx in COSTS:
            g = agg.get((d, tx), [])
            cells.append(f"{np.mean([x['CAGR'] for x in g]):10.4f}"
                         if len(g) >= len(PHASES) else "         -")
        tag = ' (끔)' if d == 0 else ''
        print(f"  {d:.2f}{tag:<5}" + "".join(cells))

    print("\n문턱 × 비용 — 칼마")
    print(header)
    for d in DRIFTS:
        cells = []
        for tx in COSTS:
            g = agg.get((d, tx), [])
            cells.append(f"{np.mean([x['Cal'] for x in g]):10.3f}"
                         if len(g) >= len(PHASES) else "         -")
        tag = ' (끔)' if d == 0 else ''
        print(f"  {d:.2f}{tag:<5}" + "".join(cells))

    print("\n거래 수 (비용과 무관하게 같아야 정상 — 다르면 비용이 체결 가능성을 바꾼 것)")
    print(header)
    for d in DRIFTS:
        cells = []
        for tx in COSTS:
            g = agg.get((d, tx), [])
            cells.append(f"{np.mean([x['Trades'] for x in g]):10.1f}"
                         if len(g) >= len(PHASES) else "         -")
        tag = ' (끔)' if d == 0 else ''
        print(f"  {d:.2f}{tag:<5}" + "".join(cells))

    # 비용 0 에서 0.30 이 0.10 을 이기나 — 이기면 구조적 이득, 아니면 비용 절감만.
    g0 = agg.get((0.10, 0.0), [])
    g3 = agg.get((0.30, 0.0), [])
    if len(g0) >= len(PHASES) and len(g3) >= len(PHASES):
        a = np.mean([x['CAGR'] for x in g0])
        b = np.mean([x['CAGR'] for x in g3])
        print(f"\n비용 0 에서 문턱 0.30 - 문턱 0.10 연복리 차이 {b-a:+.4f}")
        print("  양수면 승자 비중 유지라는 구조적 이득이 있다는 뜻이고,")
        print("  0 근처거나 음수면 드리프트 0.30 의 이득은 거래비용 절감이 전부다.")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
