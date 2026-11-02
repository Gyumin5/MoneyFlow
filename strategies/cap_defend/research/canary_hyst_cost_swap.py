#!/usr/bin/env python3
"""수수료를 맞바꿔 재면 두 자산군의 카나리 모양이 자리를 바꾸나.

왜 본다. 같은 자료·같은 구간·같은 유니버스인데 카나리 히스테리시스 축의 모양이
두 엔진에서 반대로 나왔다. 현물은 0.015 가 최고이고 0 으로 내리면 1.282 로
무너지는데, 선물은 0 이 3.979 로 가장 높다.

두 엔진이 다른 것은 여섯이다 — 거래비용(0.4% 대 0.06%), 스냅샷(7개 217봉 대
5개 95봉), 드리프트 문턱(0.10 대 0.015), 단기 모멘텀(20일 대 18일), 레버리지
(1배 대 2~4배 동적), 유지증거금·청산(선물만). 이 중 어느 것이 모양을 뒤집는지는
한 축만 바꿔 봐야 갈린다.

거래비용이 첫 용의자인 이유. 히스테리시스를 내리면 카나리가 자주 뒤집히고 전환이
는다. 전환의 대가는 수수료에 비례하므로 6.7배 비싼 쪽이 6.7배로 치른다.

그래서 수수료만 맞바꾼다. 다른 조건은 하나도 안 건드린다.
  현물에 선물 수수료(0.0006)를 물린다 → 모양이 선물처럼 되면 비용이 원인이다.
  선물에 현물 수수료(0.004)를 물린다  → 모양이 현물처럼 되면 비용이 원인이다.

양쪽이 다 넘어가야 비용으로 설명된다. 한쪽만 넘어가면 비용은 일부이고 나머지
다섯 중에 다른 원인이 있다.

쓰는 법:
  python3 canary_hyst_cost_swap.py
  python3 canary_hyst_cost_swap.py --summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

from canary_hyst_cost import HYSTS, run_one, trade_stats  # noqa: E402,F401

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(ROOT, 'state', '20261102-canary-hyst-swap', 'cells.jsonl')

# 자기 수수료가 아니라 상대 자산군의 수수료를 물린다.
SWAP_TX = {'spot': 0.0006, 'fut': 0.004}
# 같은 표에 나란히 놓고 읽으려고 자기 수수료도 적어 둔다(이미 잰 값과 대조용).
OWN_TX = {'spot': 0.004, 'fut': 0.0006}


def summarize(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    if not rows:
        print("잰 것이 없다")
        return 1
    own = os.path.join(ROOT, 'state', '20261102-canary-hyst-cost', 'cells.jsonl')
    base = {}
    if os.path.exists(own):
        for l in open(own, encoding='utf-8'):
            if not l.strip():
                continue
            d = json.loads(l)
            if d['mult'] == 1:
                base.setdefault((d['engine'], d['hyst']), []).append(d)

    for eng in ('spot', 'fut'):
        g = [r for r in rows if r['engine'] == eng]
        if not g:
            continue
        other = 'fut' if eng == 'spot' else 'spot'
        print(f"\n  {eng} 엔진에 {other} 수수료({SWAP_TX[eng]})를 물렸다"
              f" — 자기 수수료는 {OWN_TX[eng]}")
        print(f"    {'히스테리시스':>12}{'자기수수료':>11}{'맞바꾼뒤':>10}"
              f"{'차이':>9}{'전환/년':>9}")
        for h in sorted({r['hyst'] for r in g}):
            rs = [r for r in g if r['hyst'] == h]
            swapped = float(np.mean([r['metrics'].get('calmar', 0) for r in rs]))
            flips = float(np.mean([r['trade']['flips_per_year'] for r in rs]))
            b = base.get((eng, h))
            if b:
                ownv = float(np.mean([r['metrics'].get('calmar', 0) for r in b]))
                print(f"    {h:>12}{ownv:>11.3f}{swapped:>10.3f}"
                      f"{swapped - ownv:>+9.3f}{flips:>9.2f}")
            else:
                print(f"    {h:>12}{'—':>11}{swapped:>10.3f}{'—':>9}{flips:>9.2f}")

    print("\n  읽는 법. 맞바꾼 뒤의 열에서 어느 히스테리시스가 최고인지를 본다.")
    print("  현물이 선물 수수료를 쓰자 0 쪽이 최고가 되고, 선물이 현물 수수료를")
    print("  쓰자 0.015 쪽이 최고가 되면 모양 차이는 비용으로 설명된다. 한쪽만")
    print("  넘어가면 비용은 일부이고 나머지 다섯 축에 다른 원인이 있다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', type=int, default=7)
    ap.add_argument('--out', default='')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()

    out = a.out or OUT
    if a.summary:
        return summarize(out)

    os.makedirs(os.path.dirname(out), exist_ok=True)

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    cfg = hashlib.sha1(json.dumps([HYSTS, SWAP_TX], sort_keys=True)
                       .encode()).hexdigest()[:12]
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding='utf-8'):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('cfg') == cfg:
                done.add((d['engine'], d['hyst'], d['phase']))
        if done:
            print(f"  이미 잰 칸 {len(done)} — 건너뛴다", flush=True)

    phases = [round(i * 19 / a.anchors) for i in range(a.anchors)]
    total = 2 * len(HYSTS) * len(phases)
    print(f"엔진 2 · 히스테리시스 {len(HYSTS)} · 앵커 {len(phases)} · 칸 {total}",
          flush=True)

    with open(out, 'a', encoding='utf-8') as fh:
        for eng in ('spot', 'fut'):
            if eng == 'fut':
                import backtest_futures_v25 as f25
                from fut_param_sensitivity import LEV_BASE
                lev_map = f25.build_K2_signal(bars, **LEV_BASE)
            else:
                lev_map = None
            tx = SWAP_TX[eng]
            for h in HYSTS:
                m = s = None
                for ph in phases:
                    if (eng, h, ph) in done:
                        continue
                    t0 = time.time()
                    m, s = run_one(eng, bars, fund, lev_map, ph, h, tx)
                    fh.write(json.dumps(dict(engine=eng, hyst=h, tx=tx, phase=ph,
                                             metrics=m, trade=s, cfg=cfg,
                                             sec=round(time.time() - t0, 2)),
                                        ensure_ascii=False) + '\n')
                    fh.flush()
                if m is None:
                    continue
                print(f"  {eng} hyst={h} 수수료 {tx}: 칼마 {m.get('calmar', 0):.3f} "
                      f"연복리 {m.get('cagr', 0):.4f} "
                      f"전환/년 {s['flips_per_year']:.2f}", flush=True)

    print(f"끝. {out}")
    return summarize(out)


if __name__ == '__main__':
    raise SystemExit(main())
