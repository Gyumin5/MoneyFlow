"""숏 셋째 격자 — 숏 노출을 잘게 흔들고, 긴 구간(2017~)에서도 돌린다.

둘째 격자가 낸 것. 전환 비용을 0 으로 깎고 나니 숏이 수익은 늘리고 낙폭은 더 늘린다.
숏 총노출을 줄일수록 롱 전용에 가까워지는데, 수익 증가분은 생각보다 덜 준다.

  총노출 75%: 칼마 1.4762 · 연복리 +4.16%p · 낙폭 -0.4119
  총노출 40%: 칼마 1.7770 · 연복리 +3.06%p · 낙폭 -0.3361
  총노출 20%: 칼마 1.9493 · 연복리 +2.52%p · 낙폭 -0.3037 · 샤프 1.6587 (롱 전용 1.6330)
  총노출  0%: 칼마 2.0965 · 연복리 +0     · 낙폭 -0.2704 · 샤프 1.6330

20% 에서 샤프는 이미 롱 전용을 넘었다. 칼마만 아직 아래다. 그래서 두 가지를 더 본다.

⑴ 노출을 더 잘게 — 5·10·15·20·30%. 칼마가 롱 전용을 넘는 자리가 있나.
⑵ 긴 구간 — 지금 구간(2020-10~)에는 길게 이어진 약세장이 없다. 숏이 값을 하는 자리가
   원래 거기인데 표본에 없다. 2017년까지 늘리면 2018년 약세장이 통째로 들어온다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 short_grid3.py
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

import bt_coin_param_resweep as R  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-grid3')

_FLIP = dict(allow_direct_flip=True, flip_preserve_snapshots=True)
_SH = dict(enable_short=True, short_entry_discount=0.02, **_FLIP)

EXPOSURES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75]

ARMS = [('B', dict(enable_short=False))] + [
    (f"E{int(e*100):02d}", dict(_SH, short_gross_exposure_cap=e,
                                short_per_coin_cap=min(0.20, e)))
    for e in EXPOSURES
]

WINDOWS = [('짧게', '2020-10-01'), ('길게', '2017-01-01')]


def _kw(anchor, start):
    kw = dict(R.SPOT_BASE)
    return dict(
        interval='D', asset_type='spot', leverage=1.0,
        tx_cost=R.TX_SPOT, start_date=start, end_date=R.END,
        vol_mode='daily', phase_offset_bars=R.PH_SPOT[anchor],
        post_flip_delay=5, selection='greedy', stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
        crash_threshold=-99.0, health_mode='mom2vol', **kw)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from short_engine import run as bt
    from unified_backtest import load_data
    from long_history import load_extended

    data = {'짧게': load_data('D'), '길게': load_extended(verbose=False)}
    print("자료 준비 끝", flush=True)

    path = os.path.join(OUT, 'cells.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['window'], r['arm'], r['anchor']))
            except Exception:
                pass
    print(f"이미 끝난 칸 {len(done)}개", flush=True)

    keys = ['CAGR', 'Cal', 'MDD', 'Sharpe', 'Trades', 'Rebal', 'Liq', 'Stops']
    with open(path, 'a', encoding='utf-8') as f:
        for wname, start in WINDOWS:
            bars, fund = data[wname]
            for name, over in ARMS:
                for anchor in R.ANCHORS:
                    if (wname, name, anchor) in done:
                        continue
                    t0 = time.time()
                    m = bt(bars, fund, **_kw(anchor, start), **over)
                    row = dict(window=wname, arm=name, anchor=anchor,
                               sec=round(time.time() - t0, 1),
                               **{k: m.get(k) for k in keys})
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                    f.flush()
                print(f"{wname} {name} 끝 (앵커 {len(R.ANCHORS)}곳)", flush=True)
    _summary(path)
    return 0


def _summary(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    for wname, _ in WINDOWS:
        rs = [r for r in rows if r['window'] == wname]
        if not rs:
            continue
        print(f"\n[{wname}] 앵커 평균 (칼마 / 연복리 / 낙폭 / 샤프 / 거래)")
        base = None
        for name, _o in ARMS:
            g = [x for x in rs if x['arm'] == name]
            if not g:
                continue
            a = {k: float(np.mean([x[k] for x in g]))
                 for k in ('Cal', 'CAGR', 'MDD', 'Sharpe')}
            tr = float(np.mean([x['Trades'] for x in g]))
            if name == 'B':
                base = a
            d = ''
            if base and name != 'B':
                d = (f"   B 대비 칼마 {a['Cal']-base['Cal']:+.4f}"
                     f" 연복리 {a['CAGR']-base['CAGR']:+.4f}"
                     f" 샤프 {a['Sharpe']-base['Sharpe']:+.4f}")
            print(f"  {name:<5} {a['Cal']:7.4f} {a['CAGR']:8.4f} "
                  f"{a['MDD']:8.4f} {a['Sharpe']:7.4f} {tr:7.1f}{d}")


if __name__ == '__main__':
    raise SystemExit(main())
