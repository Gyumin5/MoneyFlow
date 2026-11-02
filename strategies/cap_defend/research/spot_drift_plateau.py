"""문턱 0.40 이 고원인가 봉우리인가 — 이웃을 촘촘히 쓸어본다.

앞선 측정에서 문턱 0.40 이 네 축(전 구간 평균·창 순위합·자료 출처별 분할·앵커 분산)
모두에서 가장 깨끗하게 나왔다. 그런데 0.40 과 끔(또는 0.50)은 9.8년 리밸런싱 횟수가
235.9 대 234.3 으로 사실상 같다 — 우위가 앵커당 발화 한두 번에 걸려 있다는 뜻이다.

사건 수가 얇으면 봉우리 하나가 우연일 수 있다. 프로젝트 채택 기준이 "최고점 하나보다
고원 존재 여부를 본다" 이므로 0.32~0.48 을 0.02 간격으로 쓴다.

두 갈래로 갈린다.
  ⑴ 그 구간 전체가 고르게 높다(고원) — 0.40 이라는 숫자가 아니라 "드리프트를 가끔만
     쓴다" 는 성질이 이득의 원천이다. 채택 근거가 된다.
  ⑵ 0.40 만 솟아 있고 이웃이 낮다(봉우리) — 우연히 잘 맞은 발화 몇 개가 만든 값이다.
     그러면 0.40 을 고르지 않고, 사실상 같은 전략인 끔 쪽을 본다.

비용은 실비용 40bp 하나만 쓴다. 앞 측정에서 순위가 네 비용 칸 전부 같았으므로
비용 축을 다시 펼칠 이유가 없다.

문턱 9 × 앵커 7 = 63칸.

실행: python3 spot_drift_plateau.py
      python3 spot_drift_plateau.py --summary
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-drift-plateau')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]
TX = 0.004

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1 / 3, tx_cost=TX,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=217,
)

DRIFTS = [0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48]


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
                done.add((r['drift'], r['phase']))
            except Exception:
                pass
    print(f"이미 끝난 칸 {len(done)}개 / 전체 {len(DRIFTS)*len(PHASES)}", flush=True)

    t0 = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for d in DRIFTS:
            for ph in PHASES:
                if (d, ph) in done:
                    continue
                kw = dict(BASE, drift_threshold=d, phase_offset_bars=ph)
                m = ub.run(bars, fund, start_date=START, end_date=END, **kw)
                row = dict(drift=d, phase=ph,
                           **{k: m.get(k) for k in
                              ('CAGR', 'MDD', 'Sharpe', 'Trades', 'Rebal')})
                dd = abs(float(row['MDD'] or 0.0))
                row['Cal'] = (float(row['CAGR']) / dd) if dd > 0 else None
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                f.flush()
            print(f"  문턱 {d} 끝 ({(time.time()-t0)/60:.1f}분)", flush=True)
    summary(path)
    return 0


def summary(path=None):
    path = path or os.path.join(OUT, 'cells.jsonl')
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    agg = {}
    for r in rows:
        agg.setdefault(r['drift'], []).append(r)

    print("\n문턱 이웃 (비용 40bp, 앵커 7곳)")
    print(f"  {'문턱':>6}{'칼마평균':>10}{'칼마sd':>9}{'칼마최저':>10}"
          f"{'연복리':>9}{'샤프':>8}{'리밸':>8}{'거래':>8}")
    for d in DRIFTS:
        g = agg.get(d, [])
        if len(g) < len(PHASES):
            print(f"  {d:6.2f}{'  미완':>10}")
            continue
        cal = [x['Cal'] for x in g]
        print(f"  {d:6.2f}{np.mean(cal):10.3f}{np.std(cal):9.3f}{min(cal):10.3f}"
              f"{np.mean([x['CAGR'] for x in g]):9.4f}"
              f"{np.mean([x['Sharpe'] for x in g]):8.3f}"
              f"{np.mean([x['Rebal'] for x in g]):8.1f}"
              f"{np.mean([x['Trades'] for x in g]):8.1f}")

    full = [d for d in DRIFTS if len(agg.get(d, [])) >= len(PHASES)]
    if len(full) >= 3:
        means = [float(np.mean([x['Cal'] for x in agg[d]])) for d in full]
        best = full[int(np.argmax(means))]
        spread = max(means) - min(means)
        print(f"\n이웃 전체 칼마 폭 {spread:.3f} (최고 {max(means):.3f} @ {best:.2f},"
              f" 최저 {min(means):.3f})")
        print("  폭이 좁으면 고원이다 — 0.40 이라는 숫자가 아니라 '가끔만 쓴다' 는")
        print("  성질이 이득의 원천이라는 뜻이고 채택 근거가 된다.")
        print("  0.40 만 솟고 이웃이 낮으면 봉우리다 — 우연히 맞은 발화가 만든 값이라")
        print("  0.40 을 고르지 않고 사실상 같은 전략인 끔 쪽을 본다.")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
