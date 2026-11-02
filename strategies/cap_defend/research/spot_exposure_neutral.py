"""종목 수를 노출 중립으로 다시 잰다.

왜 다시 재나. 단일축 민감도(spot_param_sensitivity.py)에서 종목 수를 2로 줄이면
칼마가 1.687 에서 2.811 로 뛰었는데, 그건 종목을 잘 골라서가 아니다. 종목당 상한이
1/3 로 고정돼 있어서 둘만 고르면 2/3 만 투자하고 1/3 은 현금으로 남는다. 즉 그 축은
"몇 개를 고르나" 와 "얼마나 투자하나" 를 같이 바꾸고 있어서 칼마 비교가 성립하지 않는다.

그래서 여기서는 종목 수 N 마다 상한을 1/N 로 같이 움직여 항상 전액 투자로 맞춘다.
남는 차이는 분산 정도 하나다 — 하나에 몰 것인가 열에 나눌 것인가.

짧은 구간과 긴 구간(2017~)을 둘 다 돌린다. 긴 구간은 2017~2018 약세장이 들어와서
집중 투자가 거기서 어떻게 되는지가 같이 보인다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 spot_exposure_neutral.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, CAP)

import unified_backtest as ub  # noqa: E402
from spot_param_sensitivity import BASE, END, metrics  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-exposure-neutral')

SIZES = [1, 2, 3, 5, 7, 10]
ANCHORS = 7


def _phases():
    gap = max(BASE['snap_interval_bars'] // max(BASE['n_snapshots'], 1), 1)
    step = max(gap // ANCHORS, 1)
    return [i * step for i in range(ANCHORS)]


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'cells.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['window'], r['n'], r['phase']))
            except Exception:
                pass

    from long_history import load_extended
    windows = [('짧게', '2020-10-01', ub.load_data('D')),
               ('길게', '2017-01-01', load_extended(verbose=False))]

    with open(path, 'a', encoding='utf-8') as f:
        for wname, start, (bars, funding) in windows:
            for n in SIZES:
                for ph in _phases():
                    if (wname, n, ph) in done:
                        continue
                    kw = dict(BASE)
                    kw.update(universe_size=n, cap=1.0 / n,
                              phase_offset_bars=ph, start_date=start, end_date=END)
                    t0 = time.time()
                    m = metrics(ub.run(bars, funding, **kw))
                    row = dict(window=wname, n=n, phase=ph,
                               sec=round(time.time() - t0, 1), **m)
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                    f.flush()
                print(f"{wname} 종목수 {n} 끝", flush=True)
    _summary(path)
    return 0


def _summary(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    for wname in ('짧게', '길게'):
        rs = [r for r in rows if r['window'] == wname]
        if not rs:
            continue
        print(f"\n[{wname}] 종목 N개 · 각 1/N (전액 투자)")
        print("  N   칼마    연복리    낙폭     샤프   거래")
        for n in SIZES:
            g = [r for r in rs if r['n'] == n]
            if not g:
                continue
            a = {k: float(np.mean([x[k] for x in g])) for k in
                 ('calmar', 'cagr', 'mdd', 'sharpe', 'trades')}
            print(f"  {n:<3} {a['calmar']:6.3f} {a['cagr']:8.4f} "
                  f"{a['mdd']:8.4f} {a['sharpe']:6.3f} {a['trades']:6.0f}")


if __name__ == '__main__':
    raise SystemExit(main())
