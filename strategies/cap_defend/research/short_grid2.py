"""숏 둘째 격자 — 방향 전환을 아끼는 축으로 롱 전용을 넘을 수 있나.

첫 격자(short_grid.py)가 낸 것: 숏을 하나도 안 걸고 전환 일정만 따라간 대조군 D 가
이미 롱 전용 B 보다 칼마 0.69 낮았다. 숏 주문 자체는 수익을 0.5%p 더하고 낙폭을
12.4%p 키웠다. 즉 손해의 절반 이상이 "숏이 나빠서" 가 아니라 "방향을 바꾸느라" 났다.

그래서 이 격자가 흔드는 것은 전환 비용 쪽 셋이다.
  ⑴ 직접 전환 — 현금을 한 봉 거치지 않고 같은 봉에 뒤집는다(allow_direct_flip).
  ⑵ 스냅샷 보존 — 롱으로 돌아갈 때 스냅샷을 현금으로 지우지 않는다. 지우면 스태거
     일곱 자리가 전환마다 한꺼번에 무너지고, 그 손해는 숏과 무관하다.
  ⑶ 숏 노출 상한과 복귀 문턱 — 숏에 머무는 양과 시간을 줄인다.

대조군을 같이 돌린다. 이름 끝의 D 계열은 전환 일정만 쓰고 숏 주문을 뺀 것이라,
위 셋이 전환 비용을 실제로 깎았는지를 숏과 분리해 보여준다.

판정: 어느 칸도 B 를 못 넘으면 숏은 기각이다. 전환 비용을 0 으로 깎아도 못 넘으면
남는 것은 숏 자체의 손해뿐이고, 그건 첫 격자가 이미 쟀다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 short_grid2.py
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-grid2')

_FLIP = dict(allow_direct_flip=True, flip_preserve_snapshots=True)
_SH = dict(enable_short=True, short_entry_discount=0.02)

ARMS = [
    ('B',      dict(enable_short=False)),
    # 전환 비용만 보는 대조군 — 숏 주문 없음
    ('D',      dict(_SH, short_universe_size=0)),
    ('Dd',     dict(_SH, short_universe_size=0, allow_direct_flip=True)),
    ('Dp',     dict(_SH, short_universe_size=0, flip_preserve_snapshots=True)),
    ('Ddp',    dict(_SH, short_universe_size=0, **_FLIP)),
    # 숏을 실제로 거는 칸
    ('C',      dict(_SH)),
    ('Cd',     dict(_SH, allow_direct_flip=True)),
    ('Cp',     dict(_SH, flip_preserve_snapshots=True)),
    ('Cdp',    dict(_SH, **_FLIP)),
    ('Cdp40',  dict(_SH, short_gross_exposure_cap=0.40, **_FLIP)),
    ('Cdp20',  dict(_SH, short_gross_exposure_cap=0.20, **_FLIP)),
    ('Cdpx',   dict(_SH, short_exit_margin=0.005, **_FLIP)),
]

_BARS = None
_FUND = None


def _load():
    global _BARS, _FUND
    if _BARS is None:
        from unified_backtest import load_data
        t0 = time.time()
        _BARS, _FUND = load_data('D')
        print(f"일봉 {len(_BARS)}종 로드 ({time.time()-t0:.0f}초)", flush=True)
    return _BARS, _FUND


def _kw(anchor: int):
    kw = dict(R.SPOT_BASE)
    return dict(
        interval='D', asset_type='spot', leverage=1.0,
        tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
        vol_mode='daily', phase_offset_bars=R.PH_SPOT[anchor],
        post_flip_delay=5, selection='greedy', stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
        crash_threshold=-99.0, health_mode='mom2vol', **kw)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from short_engine import run as bt
    bars, fund = _load()
    path = os.path.join(OUT, 'cells.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['arm'], r['anchor']))
            except Exception:
                pass
    print(f"이미 끝난 칸 {len(done)}개", flush=True)

    keys = ['CAGR', 'Cal', 'MDD', 'Sharpe', 'Trades', 'Rebal', 'Liq', 'Stops']
    with open(path, 'a', encoding='utf-8') as f:
        for name, over in ARMS:
            for anchor in R.ANCHORS:
                if (name, anchor) in done:
                    continue
                t0 = time.time()
                m = bt(bars, fund, **_kw(anchor), **over)
                row = dict(arm=name, anchor=anchor, sec=round(time.time() - t0, 1),
                           **{k: m.get(k) for k in keys})
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                f.flush()
                print(f"{name} 앵커{anchor}: 칼마 {m.get('Cal'):.4f}  "
                      f"연복리 {m.get('CAGR'):.4f}  낙폭 {m.get('MDD'):.4f}  "
                      f"샤프 {m.get('Sharpe'):.4f}  거래 {m.get('Trades')}  "
                      f"({row['sec']}초)", flush=True)
    _summary(path)
    return 0


def _summary(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    by = {}
    for r in rows:
        by.setdefault(r['arm'], []).append(r)
    print("\n앵커 평균 (칼마 / 연복리 / 낙폭 / 샤프 / 거래)")
    base = None
    for name, _ in ARMS:
        rs = by.get(name, [])
        if not rs:
            continue
        avg = {k: float(np.mean([x[k] for x in rs])) for k in ('Cal', 'CAGR', 'MDD', 'Sharpe')}
        tr = float(np.mean([x['Trades'] for x in rs]))
        if name == 'B':
            base = avg
        d = ''
        if base and name != 'B':
            d = (f"   B 대비 칼마 {avg['Cal']-base['Cal']:+.4f}"
                 f" 연복리 {avg['CAGR']-base['CAGR']:+.4f}")
        print(f"  {name:<6} {avg['Cal']:7.4f} {avg['CAGR']:8.4f} "
              f"{avg['MDD']:8.4f} {avg['Sharpe']:7.4f} {tr:7.1f}  (앵커 {len(rs)}곳){d}")


if __name__ == '__main__':
    raise SystemExit(main())
