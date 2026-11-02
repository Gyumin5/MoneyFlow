"""연구 호출부와 채택 호출부의 설정 차이가 수를 바꾸나 — 실측.

토론(run-20261102T022434Z) 첫째 지적: "연구 호출부가 방어 해제 인자와
DRIFT_HEALTH_MODE를 명시하지 않아 현행 V24와 다른 기본값으로 실행될 가능성".

대조해 보니 둘로 갈린다.
  가드 인자(dd_lookback·bl_drop·crash_threshold) — 채택 호출부
    `trade/v24_shadow_today.py` 도 안 넘긴다. 엔진 기본값이 양쪽에 똑같이 걸리므로
    연구와 채택 사이의 차이가 아니다. (CLAUDE.md 의 "전략 가드 없음" 과 엔진
    기본값이 어긋나는 것은 별개 문제다 — 여기서 다루지 않는다.)
  DRIFT_HEALTH_MODE — 채택 호출부는 'refill', 연구 세 파일은 설정을 안 해 'off'.
    이것만 진짜 불일치다.

2026-06-06 검증(bt_spot_refill_vs_anchor.py)은 드리프트 0.10 에서 refill ≡ anchor-only
라고 적었다. 그런데 이번 격자는 0.20·0.30 도 돌렸고, 창 판정을 통과한 후보 둘 중
하나가 바로 드리프트 0.30 이다. 문턱을 올리면 발화가 드물어지는 대신 발화할 때의
괴리가 커져서, 그 자리에서 refill 이 실제로 종목을 갈아탈 수 있다.

그래서 세 문턱 × 두 모드를 같은 앵커 일곱으로 돌려 수가 바뀌는지 본다.
바뀌면 긴 구간 연구 전체를 refill 켜고 다시 돌려야 하고, 안 바뀌면 기존 결과를
그대로 쓴다.

실행: python3 spot_refill_setting_check.py
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-refill-check')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1 / 3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=217,
)

DRIFTS = [0.10, 0.20, 0.30]
MODES = ['off', 'refill']


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
                done.add((r['mode'], r['drift'], r['phase']))
            except Exception:
                pass
    print(f"이미 끝난 칸 {len(done)}개", flush=True)

    with open(path, 'a', encoding='utf-8') as f:
        for mode in MODES:
            # 환경변수는 엔진이 run() 안에서 매번 읽는다(모듈 적재 시점이 아니다).
            os.environ['DRIFT_HEALTH_MODE'] = mode
            import unified_backtest as ub
            for d in DRIFTS:
                for ph in PHASES:
                    if (mode, d, ph) in done:
                        continue
                    kw = dict(BASE, drift_threshold=d, phase_offset_bars=ph)
                    t0 = time.time()
                    m = ub.run(bars, fund, start_date=START, end_date=END, **kw)
                    row = dict(mode=mode, drift=d, phase=ph,
                               sec=round(time.time() - t0, 1),
                               **{k: m.get(k) for k in
                                  ('CAGR', 'MDD', 'Sharpe', 'Trades', 'Rebal')})
                    dd = abs(float(row['MDD'] or 0.0))
                    row['Cal'] = (float(row['CAGR']) / dd) if dd > 0 else None
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                    f.flush()
                    print(f"  {mode} d={d} p={ph}: Cal {row['Cal']:.3f} "
                          f"CAGR {row['CAGR']:.4f} ({row['sec']}초)", flush=True)
    summary(path)
    return 0


def summary(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    agg = {}
    for r in rows:
        agg.setdefault((r['mode'], r['drift']), []).append(r)
    print("\n드리프트별 두 모드 비교 (앵커 7곳 평균)")
    print("  문턱   모드     칼마    연복리      낙폭    거래    리밸")
    for d in DRIFTS:
        vals = {}
        for mode in MODES:
            g = agg.get((mode, d), [])
            if len(g) < len(PHASES):
                continue
            a = {k: float(np.mean([x[k] for x in g]))
                 for k in ('Cal', 'CAGR', 'MDD', 'Sharpe', 'Trades', 'Rebal')}
            vals[mode] = a
            print(f"  {d:.2f}  {mode:<7} {a['Cal']:7.3f} {a['CAGR']:9.4f} "
                  f"{a['MDD']:9.4f} {a['Trades']:7.1f} {a['Rebal']:7.1f}")
        if len(vals) == 2:
            same = all(abs(vals['off'][k] - vals['refill'][k]) < 1e-9
                       for k in ('Cal', 'CAGR', 'MDD', 'Trades'))
            print(f"         → {'동일' if same else '다르다'}")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary(os.path.join(OUT, 'cells.jsonl'))
        raise SystemExit(0)
    raise SystemExit(main())
