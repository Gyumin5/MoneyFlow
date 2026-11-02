#!/usr/bin/env python3
"""선물 레버리지 축 — 노출까지 재서 "잘 골라서" 와 "크게 걸어서" 를 가른다.

fut_param_sensitivity.py 가 레버리지 축 일곱을 같이 재지만, 그 측정만으로는 판정이
안 선다. 노출을 안 담기 때문이다.

왜 안 담기나. 선물 엔진의 target 은 자본 대비 비중이고 레버리지는 그 비중에 곱해져
명목이 된다. 그래서 fut_cap_exposure_decomp.py 의 노출 지표(1 - 현금)는 배수를
올려도 그대로다 — l_max 를 4 에서 8 로 올리면 명목이 두 배가 되는데 그 지표는 한
자리도 안 움직인다. 그 상태로 노출당 수익(연복리/평균투자)을 내면 분모가 안 커져서
"크게 걸어서 번 것" 이 "잘 골라서 번 것" 으로 읽힌다. 레버리지 축에서는 그게 바로
판정 대상이라 치명적이다.

그래서 여기서는 명목으로 잰다.
  평균투자   = 봉마다 (1 - 현금). 자본 대비 비중. 기존 지표와 같다.
  평균명목   = 봉마다 sum(코인 비중 × 그 코인의 그날 배수). 실제로 걸린 크기다.
  평균배수   = 평균명목 / 평균투자. 들어간 자리에서 평균 몇 배를 걸었나.
  노출당수익 = 연복리 / 평균명목. 이것만 올라야 선정이 좋아진 것이다.

배수는 trace 에 안 담겨서 build_K2_signal 결과(dict[코인 -> 시계열])를 그 날짜로
조회해 곱한다. SSoT(backtest_futures_v25.py)는 안 건드린다.

축·기준선은 fut_param_sensitivity.py 에서 그대로 가져온다 — 두 벌로 두면 한쪽만
고쳐져 같은 이름의 축이 다른 값을 재게 된다.

산출물은 현물 민감도와 같은 모양이라 spot_axis_shape.py 가 그대로 읽는다. 모양·성격
판정은 거기서 하고, 노출당 수익은 이 파일의 요약이 낸다.

쓰는 법:
  python3 fut_lev_exposure_sens.py                 (레버리지 축 일곱 전부)
  python3 fut_lev_exposure_sens.py --axes l_max    (축 지목)
  python3 fut_lev_exposure_sens.py --summary       (이미 잰 것 요약만)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import backtest_futures_v25 as f25  # noqa: E402
from fut_param_sensitivity import BASE, LEV_AXES, LEV_BASE, START, END, metrics  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(ROOT, 'state', '20261102-fut-lev-exposure', 'cells.jsonl')


def exposure(trace, lev_map):
    """봉마다 자본 대비 비중과 명목을 모은다.

    명목은 코인마다 다르다 — V25 의 배수가 per-coin min(BTC_cap, K2) 라 같은 봉에서도
    코인에 따라 2배와 4배가 섞인다. 그래서 봉 하나의 명목은 비중 가중합이다.
    """
    inv, notional = [], []
    for t in trace:
        tgt = t.get('target') or {}
        date = t.get('date')
        cash = float(tgt.get('CASH', tgt.get('Cash', 0.0)))
        inv.append(1.0 - cash)
        n = 0.0
        for coin, w in tgt.items():
            if coin in ('CASH', 'Cash'):
                continue
            s = lev_map.get(coin)
            if s is None:
                continue
            try:
                lv = float(s.asof(date))
            except (KeyError, TypeError, ValueError):
                continue
            if lv == lv:  # NaN 이면 그 코인은 센 적이 없다
                n += float(w) * lv
        notional.append(n)
    v = np.asarray(inv, dtype=float)
    q = np.asarray(notional, dtype=float)
    on = v > 1e-9
    mi = float(np.mean(v)) if len(v) else 0.0
    mn = float(np.mean(q)) if len(q) else 0.0
    return dict(
        bars=int(len(v)),
        mean_inv=mi,
        mean_notional=mn,
        mean_lev=float(mn / mi) if mi > 1e-9 else 0.0,
        mean_lev_when_in=float(np.mean(q[on] / v[on])) if on.any() else 0.0,
        cash_only_pct=float(100.0 * (1.0 - on.mean())) if len(v) else 0.0,
    )


def run_cell(bars, fund, lev_map, ph, **over):
    tr = []
    r = f25.run(bars, fund, leverage=lev_map, start_date=START, end_date=END,
                phase_offset_bars=ph, _trace=tr, **dict(BASE, **over))
    return metrics(r), exposure(tr, lev_map)


def summarize(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    if not rows:
        print("잰 것이 없다")
        return 1
    base = [r for r in rows if r['axis'] == '_baseline']
    if base:
        c = [r['metrics']['calmar'] for r in base]
        print(f"기준선 앵커 폭(칼마) {max(c) - min(c):.3f} · 앵커 {len(base)}곳\n")

    axes = sorted({r['axis'] for r in rows if r['axis'] != '_baseline'})
    for ax in axes:
        g = {}
        for r in rows:
            if r['axis'] == ax:
                g.setdefault(r['value'], []).append(r)
        print(f"  {ax}")
        print(f"    {'값':>9}{'칼마':>8}{'연복리':>9}{'낙폭':>9}"
              f"{'평균명목':>10}{'평균배수':>9}{'노출당수익':>11}{'청산':>6}")
        for v in sorted(g):
            rs = g[v]
            cal = np.mean([r['metrics']['calmar'] for r in rs])
            cagr = np.mean([r['metrics']['cagr'] for r in rs])
            mdd = np.mean([r['metrics']['mdd'] for r in rs])
            mn = np.mean([r['exposure']['mean_notional'] for r in rs])
            ml = np.mean([r['exposure']['mean_lev'] for r in rs])
            liq = np.mean([r['metrics'].get('liq', 0) for r in rs])
            per = cagr / mn if mn > 1e-9 else float('nan')
            print(f"    {v:>9}{cal:>8.3f}{cagr:>9.4f}{mdd:>9.4f}"
                  f"{mn:>10.4f}{ml:>9.3f}{per:>11.3f}{liq:>6.1f}")
        print()

    print("  읽는 법. 노출당 수익이 올라야 선정이 좋아진 것이다. 연복리만 오르고 노출당")
    print("  수익이 제자리면 크게 걸어서 번 것이고, 그건 배수를 올리면 언제든 살 수 있다.")
    print("  청산 칸을 같이 본다 — 칼마가 못 담는 위험이 거기서만 보인다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axes', default='all', help="쉼표로 나눈 축 이름, 또는 all")
    ap.add_argument('--anchors', type=int, default=7)
    ap.add_argument('--out', default='')
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()

    out = a.out or OUT
    if a.summary:
        return summarize(out)

    want = a.axes.strip()
    if want == 'all':
        axes = dict(LEV_AXES)
    else:
        names = [s.strip() for s in want.split(',') if s.strip()]
        axes = {n: LEV_AXES[n] for n in names if n in LEV_AXES}
        missing = [n for n in names if n not in LEV_AXES]
        if missing:
            print(f"레버리지 축이 아니다: {', '.join(missing)}")
            return 1

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    cache = {}

    def lev(**over):
        kw = dict(LEV_BASE)
        kw.update(over)
        key = json.dumps(kw, sort_keys=True)
        if key not in cache:
            cache[key] = f25.build_K2_signal(bars, **kw)
        return cache[key]

    phases = [round(i * 19 / a.anchors) for i in range(a.anchors)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = sum(len(v) for v in axes.values()) * len(phases) + len(phases)
    print(f"레버리지 축 {len(axes)} · 앵커 {len(phases)} · 칸 {total}", flush=True)

    with open(out, 'w', encoding='utf-8') as fh:
        for ph in phases:
            t0 = time.time()
            m, e = run_cell(bars, fund, lev(), ph)
            fh.write(json.dumps(dict(axis='_baseline', value=None, phase=ph, metrics=m,
                                     exposure=e, sec=round(time.time() - t0, 2)),
                                ensure_ascii=False) + '\n')
            fh.flush()
            print(f"  _baseline 위상 {ph}: 칼마 {m.get('calmar', 0):.3f} "
                  f"명목 {e['mean_notional']:.4f} 배수 {e['mean_lev']:.3f} "
                  f"청산 {m.get('liq', 0)}", flush=True)

        for ax, values in axes.items():
            for v in values:
                m = e = None
                for ph in phases:
                    t0 = time.time()
                    m, e = run_cell(bars, fund, lev(**{ax: v}), ph)
                    fh.write(json.dumps(dict(axis=ax, value=v, phase=ph, metrics=m,
                                             exposure=e, sec=round(time.time() - t0, 2)),
                                        ensure_ascii=False) + '\n')
                    fh.flush()
                if m is None or e is None:
                    continue
                per = m.get('cagr', 0) / e['mean_notional'] if e['mean_notional'] > 1e-9 else 0
                print(f"  {ax}={v}: 칼마 {m.get('calmar', 0):.3f} "
                      f"연복리 {m.get('cagr', 0):.4f} 명목 {e['mean_notional']:.4f} "
                      f"배수 {e['mean_lev']:.3f} 노출당 {per:.3f} "
                      f"청산 {m.get('liq', 0)}", flush=True)

    print(f"끝. {out}")
    return summarize(out)


if __name__ == '__main__':
    raise SystemExit(main())
