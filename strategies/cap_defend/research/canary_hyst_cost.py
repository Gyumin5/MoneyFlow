#!/usr/bin/env python3
"""카나리 히스테리시스를 내리면 비용이 얼마나 느나 — 전환 횟수와 비용 스트레스로 본다.

왜 본다. 민감도(#124)에서 선물 canary_hyst 가 0.015 → 0.005 에서 칼마 3.294 → 3.929 로
올랐다. 앵커 흔들림 0.398 의 1.6배라 잡음이 아니다. 현물에서도 같은 값이 1.687 → 1.729 로
중립이라, 두 자산군이 공유하는 규칙을 깨지 않고 내릴 수 있는 유일한 값이다.

그런데 히스테리시스는 카나리가 42일선 둘레에서 떠는 것을 막는 장치다. 줄이면 전환이
잦아지고, 전환은 선물에서 가장 비싸다 — 카나리가 켜지면 전 스냅샷이 한꺼번에 들어가고
꺼지면 한꺼번에 빠진다. 백테스트는 그 체결을 수수료 한 값으로만 치지만 라이브에는 슬리피지와
최소 주문 단위가 붙는다. 그래서 "칼마가 올랐다" 만으로 올릴 수 없다.

무엇을 재나.
  전환      전액 현금 ↔ 투자 가 뒤집힌 횟수. 카나리 전환의 대리값이다.
  반회전율  봉마다 sum(|목표 - 직전목표|)/2 의 합. 실제로 사고판 양이다.
  비용 민감 수수료를 기본·두 배·다섯 배로 올려 순위가 뒤집히는지 본다. 이것이 판정이다 —
            비용에 약한 변경이면 백테스트 가정 밖에서 무너진다.

쓰는 법:
  python3 canary_hyst_cost.py --engine fut
  python3 canary_hyst_cost.py --engine spot
  python3 canary_hyst_cost.py --engine both --summary
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

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(ROOT, 'state', '20261102-canary-hyst-cost', 'cells.jsonl')

HYSTS = [0.0, 0.005, 0.015, 0.025]
# 기본·두 배·다섯 배. 선물 기본은 0.0006, 현물은 0.004 다.
COST_MULT = [1, 2, 5]


def cash_of(tgt):
    return float(tgt.get('CASH', tgt.get('Cash', 0.0)))


def trade_stats(trace):
    """전환 횟수와 반회전율. 목표 비중의 변화만 쓴다 — 엔진 내부를 안 건드린다."""
    prev = None
    flips = 0
    turn = 0.0
    in_market = None
    cash_bars = 0
    for t in trace:
        tgt = {k: float(v) for k, v in (t.get('target') or {}).items()}
        c = cash_of(tgt)
        now_in = c < 0.999
        if not now_in:
            cash_bars += 1
        if in_market is not None and now_in != in_market:
            flips += 1
        in_market = now_in
        if prev is not None:
            keys = set(prev) | set(tgt)
            turn += sum(abs(tgt.get(k, 0.0) - prev.get(k, 0.0)) for k in keys) / 2.0
        prev = tgt
    n = max(len(trace), 1)
    return dict(bars=n, flips=flips, turnover=turn,
                turnover_per_year=turn / (n / 365.0),
                flips_per_year=flips / (n / 365.0),
                cash_pct=100.0 * cash_bars / n)


# 두 엔진을 같은 구간에서 잰다. 현물 민감도 모듈의 기본 구간은 2020-10 시작이라
# 선물(2017-01)과 길이가 달라, 그대로 두면 "현물에서 중립" 이 5.8년 얘기가 되고
# 선물 판정은 9.7년 얘기가 된다. 같은 자료로 같은 구간을 재야 공통 규칙을 논한다.
START = '2017-01-01'
END = '2026-11-01'


def run_one(engine, bars, fund, lev_map, ph, hyst, tx):
    tr = []
    if engine == 'fut':
        import backtest_futures_v25 as f25
        from fut_param_sensitivity import BASE, metrics
        kw = dict(BASE)
        kw['canary_hyst'] = hyst
        kw['tx_cost'] = tx
        r = f25.run(bars, fund, leverage=lev_map, start_date=START, end_date=END,
                    phase_offset_bars=ph, _trace=tr, **kw)
    else:
        import unified_backtest as ub
        from spot_param_sensitivity import BASE, metrics  # type: ignore
        kw = dict(BASE)
        kw['canary_hyst'] = hyst
        kw['tx_cost'] = tx
        kw['phase_offset_bars'] = ph
        kw['start_date'] = START
        kw['end_date'] = END
        kw['_trace'] = tr
        r = ub.run(bars, fund, **kw)
    return metrics(r), trade_stats(tr)


def summarize(path, engine=None):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    if engine:
        rows = [r for r in rows if r['engine'] == engine]
    if not rows:
        print("잰 것이 없다")
        return 1
    for eng in sorted({r['engine'] for r in rows}):
        g = [r for r in rows if r['engine'] == eng]
        print(f"\n  {eng} — 수수료 배수별 카나리 히스테리시스")
        for mult in sorted({r['mult'] for r in g}):
            print(f"    수수료 x{mult}")
            print(f"      {'히스테리시스':>12}{'칼마':>8}{'연복리':>9}{'낙폭':>9}"
                  f"{'샤프':>8}{'전환/년':>9}{'회전/년':>9}{'현금%':>8}")
            for h in sorted({r['hyst'] for r in g}):
                rs = [r for r in g if r['mult'] == mult and r['hyst'] == h]
                if not rs:
                    continue
                f = lambda k: np.mean([r['metrics'].get(k, 0) for r in rs])  # noqa: E731
                s = lambda k: np.mean([r['trade'][k] for r in rs])  # noqa: E731
                print(f"      {h:>12}{f('calmar'):>8.3f}{f('cagr'):>9.4f}{f('mdd'):>9.4f}"
                      f"{f('sharpe'):>8.3f}{s('flips_per_year'):>9.2f}"
                      f"{s('turnover_per_year'):>9.2f}{s('cash_pct'):>8.1f}")
    print("\n  읽는 법. 수수료를 올려도 순위가 그대로여야 그 값을 믿는다. 전환과 회전이")
    print("  같이 늘면 백테스트 밖 비용(슬리피지·최소 주문)이 그만큼 더 붙으므로,")
    print("  이득이 회전 증가에 비례해 깎여도 남는지를 본다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default='fut', choices=['spot', 'fut', 'both'])
    ap.add_argument('--anchors', type=int, default=7)
    ap.add_argument('--out', default='')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()

    out = a.out or OUT
    if a.summary:
        return summarize(out, None if a.engine == 'both' else a.engine)

    engines = ['spot', 'fut'] if a.engine == 'both' else [a.engine]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    cfg = hashlib.sha1(json.dumps([HYSTS, COST_MULT, engines],
                                  sort_keys=True).encode()).hexdigest()[:12]
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
                done.add((d['engine'], d['hyst'], d['mult'], d['phase']))
        if done:
            print(f"  이미 잰 칸 {len(done)} — 건너뛴다", flush=True)

    phases = [round(i * 19 / a.anchors) for i in range(a.anchors)]
    total = len(engines) * len(HYSTS) * len(COST_MULT) * len(phases)
    print(f"엔진 {len(engines)} · 히스테리시스 {len(HYSTS)} · 비용 {len(COST_MULT)} "
          f"· 앵커 {len(phases)} · 칸 {total}", flush=True)

    with open(out, 'a', encoding='utf-8') as fh:
        for eng in engines:
            if eng == 'fut':
                import backtest_futures_v25 as f25
                from fut_param_sensitivity import LEV_BASE, BASE
                lev_map = f25.build_K2_signal(bars, **LEV_BASE)
            else:
                lev_map = None
                from spot_param_sensitivity import BASE  # type: ignore
            base_tx = float(BASE.get('tx_cost', 0.0006))
            for h in HYSTS:
                for mult in COST_MULT:
                    tx = base_tx * mult
                    m = s = None
                    for ph in phases:
                        if (eng, h, mult, ph) in done:
                            continue
                        t0 = time.time()
                        m, s = run_one(eng, bars, fund, lev_map, ph, h, tx)
                        fh.write(json.dumps(dict(engine=eng, hyst=h, mult=mult, tx=tx,
                                                 phase=ph, metrics=m, trade=s, cfg=cfg,
                                                 sec=round(time.time() - t0, 2)),
                                            ensure_ascii=False) + '\n')
                        fh.flush()
                    if m is None:
                        continue
                    print(f"  {eng} hyst={h} 비용x{mult}: 칼마 {m.get('calmar', 0):.3f} "
                          f"연복리 {m.get('cagr', 0):.4f} 전환/년 {s['flips_per_year']:.2f} "
                          f"회전/년 {s['turnover_per_year']:.2f}", flush=True)

    print(f"끝. {out}")
    return summarize(out)


if __name__ == '__main__':
    raise SystemExit(main())
