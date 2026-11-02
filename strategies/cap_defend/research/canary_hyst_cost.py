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


def fixed_lev_map(f25, bars, lev_base, lev):
    """레버리지를 한 값으로 고정한 L 맵.

    왜 둔다. 같은 카나리 축이 현물에서는 0.015 가 최고인데 선물에서는 0.005 가 최고다.
    수수료 가설은 맞바꿈으로 기각했다(48e81aa). 남은 후보 중 레버리지가 가장 그럴듯하다 —
    배수가 걸리면 카나리를 늦게 끄는 손실이 배수만큼 커지므로 선물은 빨리 꺼야 한다.
    이 함수로 배수를 1 로 눕히고 같은 축을 다시 재서, 모양이 현물 쪽으로 넘어가면
    배수가 원인이고 안 넘어가면 다른 축을 봐야 한다.

    구현은 세 문턱을 같은 값으로 주는 것이다 — min/mid/max 가 같으면 어느 가지로
    가든 그 값이 나온다. 맵의 모양(코인·날짜 키)은 그대로라 엔진 쪽은 안 건드린다.
    """
    kw = dict(lev_base)
    kw['l_min'] = kw['l_mid'] = kw['l_max'] = float(lev)
    return f25.build_K2_signal(bars, **kw)


def run_one(engine, bars, fund, lev_map, ph, hyst, tx, override=None):
    tr = []
    if engine == 'fut':
        import backtest_futures_v25 as f25
        from fut_param_sensitivity import BASE, metrics
        kw = dict(BASE)
        kw.update(override or {})
        kw['canary_hyst'] = hyst
        kw['tx_cost'] = tx
        r = f25.run(bars, fund, leverage=lev_map, start_date=START, end_date=END,
                    phase_offset_bars=ph, _trace=tr, **kw)
    else:
        import unified_backtest as ub
        from spot_param_sensitivity import BASE, metrics  # type: ignore
        kw = dict(BASE)
        kw.update(override or {})
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
    ap.add_argument('--lev-fixed', type=float, default=0.0,
                    help='선물 레버리지를 이 값으로 고정한다(0 이면 동적 L 그대로). '
                         '두 자산군이 반대로 서는 원인이 배수인지 가리는 용도다.')
    ap.add_argument('--base-override', default='',
                    help='BASE 를 덮어쓸 JSON. 축 하나씩 상대 자산군 값으로 바꿔 '
                         '어디서 봉우리 자리가 넘어가는지 가린다. '
                         '예: \'{"drift_threshold": 0.1}\'')
    ap.add_argument('--cost-mults', default='',
                    help='쉼표로 나눈 수수료 배수(기본 1,2,5). 봉우리 자리만 볼 때는 1 로 줄인다.')
    ap.add_argument('--no-funding', action='store_true',
                    help='선물 펀딩비를 끈다. 엔진은 코인별 요율 계열이 없으면 그냥 넘어가므로 '
                         '빈 표를 주면 된다. 현물 엔진은 원래 안 쓰니 선물 전용이다.')
    a = ap.parse_args()

    cost_mults = ([float(x) for x in a.cost_mults.split(',') if x.strip()]
                  if a.cost_mults else list(COST_MULT))
    override = json.loads(a.base_override) if a.base_override else {}

    if a.lev_fixed and a.engine == 'spot':
        print("현물에는 레버리지가 없다 — --lev-fixed 는 선물에만 쓴다", file=sys.stderr)
        return 2
    if a.no_funding and a.engine == 'spot':
        print("현물 엔진은 펀딩비를 안 쓴다 — --no-funding 은 선물에만 쓴다", file=sys.stderr)
        return 2
    if (a.lev_fixed or override or a.no_funding) and not a.out:
        print("--lev-fixed/--base-override/--no-funding 은 --out 을 따로 준다 — "
              "기준 칸과 섞으면 안 된다", file=sys.stderr)
        return 2

    out = a.out or OUT
    if a.summary:
        return summarize(out, None if a.engine == 'both' else a.engine)

    engines = ['spot', 'fut'] if a.engine == 'both' else [a.engine]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    cfg = hashlib.sha1(json.dumps([HYSTS, cost_mults, engines, a.lev_fixed, override,
                                   a.no_funding], sort_keys=True).encode()).hexdigest()[:12]
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
    total = len(engines) * len(HYSTS) * len(cost_mults) * len(phases)
    print(f"엔진 {len(engines)} · 히스테리시스 {len(HYSTS)} · 비용 {len(cost_mults)} "
          f"· 앵커 {len(phases)} · 칸 {total}", flush=True)
    if override:
        print(f"  BASE 덮어쓰기 {override}", flush=True)
    if a.no_funding:
        print("  선물 펀딩비 끔", flush=True)
    # 엔진은 코인별 요율 계열을 못 찾으면 그 코인을 그냥 넘어간다. 빈 표를 주면
    # 모든 코인이 그 길로 가므로 펀딩비가 0 이 된다.
    fund_use = {} if a.no_funding else fund

    with open(out, 'a', encoding='utf-8') as fh:
        for eng in engines:
            if eng == 'fut':
                import backtest_futures_v25 as f25
                from fut_param_sensitivity import LEV_BASE, BASE
                if a.lev_fixed:
                    lev_map = fixed_lev_map(f25, bars, LEV_BASE, a.lev_fixed)
                    print(f"  선물 레버리지 {a.lev_fixed} 배 고정", flush=True)
                else:
                    lev_map = f25.build_K2_signal(bars, **LEV_BASE)
            else:
                lev_map = None
                from spot_param_sensitivity import BASE  # type: ignore
            base_tx = float(BASE.get('tx_cost', 0.0006))
            for h in HYSTS:
                for mult in cost_mults:
                    tx = base_tx * mult
                    m = s = None
                    for ph in phases:
                        if (eng, h, mult, ph) in done:
                            continue
                        t0 = time.time()
                        m, s = run_one(eng, bars, fund_use, lev_map, ph, h, tx, override)
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
