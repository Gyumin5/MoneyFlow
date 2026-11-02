"""숏 첫 격자 — 구성 B·C0·C1·C2 와 대조군 D 를 앵커 일곱 곳에서 돌린다.

명세 short_design_spec.md 9·10·13절.

  B  = 연구 엔진의 롱 전용 (enable_short=False). 기준선이다.
  C0 = 롱·숏 상태기계, 이격 3% · 기울기 문 켬
  C1 = 같은 것, 이격 2%
  C2 = 같은 것, 이격 4%
  D  = C0 의 방향 전환 일정은 그대로 두고 숏 주문만 뺀 대조군
       (숏 구간을 현금으로 보낸다. 숏이 번 몫과 전환 기회비용을 가른다)

읽는 법.
  C − B = 전략을 바꿔서 달라진 몫
  C − D = 숏을 더해서 번 몫
  D − B = 방향 전환 때문에 롱에서 잃은 몫

헬스·종목 수·노출·손절은 전부 고정한다. 이 격자에서 바꾸는 축은 숏 진입 이격 하나다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 short_grid.py
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-grid')

# 바꾸는 축은 이격 하나. 나머지는 명세 출발값으로 고정한다.
ARMS = [
    ('B',  dict(enable_short=False)),
    ('C0', dict(enable_short=True, short_entry_discount=0.03)),
    ('C1', dict(enable_short=True, short_entry_discount=0.02)),
    ('C2', dict(enable_short=True, short_entry_discount=0.04)),
    ('D',  dict(enable_short=True, short_entry_discount=0.03, short_universe_size=0)),
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
        print(f"  {name:<3} {avg['Cal']:7.4f} {avg['CAGR']:8.4f} "
              f"{avg['MDD']:8.4f} {avg['Sharpe']:7.4f} {tr:7.1f}  (앵커 {len(rs)}곳){d}")


if __name__ == '__main__':
    raise SystemExit(main())
