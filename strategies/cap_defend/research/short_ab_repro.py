"""구성 A 와 B 가 같은 결과를 내는지 본다 — 숏 연구의 첫 관문.

명세 short_design_spec.md 9절.
  A = 지금 쓰는 정본 롱 전용 (unified_backtest.run)
  B = 숏을 넣으려고 고친 연구 엔진의 롱 전용 (short_engine.run, enable_short=False)

B 가 A 를 재현 못 하면 거기서 멈춘다. 성과 탐색으로 안 넘어간다 — 나중에 숏이 좋아 보여도
그게 숏 때문인지 엔진을 고쳐서 생긴 차이인지 못 가른다.

재는 법은 자산곡선 전체 대조다. 성과 네 값(칼마·연복리·낙폭·샤프)만 같은 것으로는 모자라다 —
서로 다른 두 곡선이 같은 요약값을 낼 수 있다. 봉마다 상대차의 최대를 본다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 short_ab_repro.py
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-ab')

# 부동소수점 누적 오차만 허용한다. 이보다 크면 엔진이 실제로 달라진 것이다.
TOL = 1e-12


def _kw(anchor: int):
    """A·B 에 똑같이 넘기는 인자. 채택 호출부(v24_shadow_today)와 같은 설정이다."""
    kw = dict(R.SPOT_BASE)
    return dict(
        interval='D', asset_type='spot', leverage=1.0,
        tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
        vol_mode='daily', phase_offset_bars=R.PH_SPOT[anchor],
        post_flip_delay=5, selection='greedy', stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
        crash_threshold=-99.0, health_mode='mom2vol', **kw)


_BARS = None
_FUND = None


def _load():
    """자료는 한 번만 읽고 A·B 가 같은 객체를 쓴다 — 입력 차이를 원천적으로 뺀다."""
    global _BARS, _FUND
    if _BARS is None:
        from unified_backtest import load_data
        t0 = time.time()
        _BARS, _FUND = load_data('D')
        print(f"일봉 {len(_BARS)}종 로드 ({time.time()-t0:.0f}초)", flush=True)
    return _BARS, _FUND


def run_pair(anchor: int):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from unified_backtest import run as run_a
    from short_engine import run as run_b
    bars, fund = _load()
    kw = _kw(anchor)
    t0 = time.time()
    ma = run_a(bars, fund, **kw)
    t1 = time.time()
    mb = run_b(bars, fund, enable_short=False, **kw)
    t2 = time.time()
    return ma, mb, t1 - t0, t2 - t1


def curve_gap(ma, mb):
    """두 자산곡선의 봉별 상대차 최대. 길이가 다르면 그 자체가 불일치다."""
    ea, eb = ma.get('_equity'), mb.get('_equity')
    if ea is None or eb is None:
        return None, '자산곡선 없음'
    a = np.asarray(ea, dtype=float)
    b = np.asarray(eb, dtype=float)
    if a.shape != b.shape:
        return None, f'길이 다름 A={a.shape} B={b.shape}'
    denom = np.maximum(np.abs(a), 1e-12)
    return float(np.max(np.abs(a - b) / denom)), ''


def main():
    os.makedirs(OUT, exist_ok=True)
    keys = ['CAGR', 'Cal', 'MDD', 'Sharpe', 'Trades', 'Rebal', 'Liq', 'Stops']
    rows = []
    bad = 0
    for anchor in R.ANCHORS:
        ma, mb, ta, tb = run_pair(anchor)
        gap, note = curve_gap(ma, mb)
        diffs = {k: (ma.get(k), mb.get(k)) for k in keys
                 if not _same(ma.get(k), mb.get(k))}
        ok = (gap is not None and gap <= TOL and not diffs)
        if not ok:
            bad += 1
        rows.append(dict(anchor=anchor, ok=ok, curve_gap=gap, note=note,
                         diffs={k: list(v) for k, v in diffs.items()},
                         a={k: ma.get(k) for k in keys},
                         b={k: mb.get(k) for k in keys},
                         sec_a=round(ta, 1), sec_b=round(tb, 1)))
        mark = '같음' if ok else '다름'
        g = '-' if gap is None else f'{gap:.2e}'
        print(f"앵커 {anchor}: {mark}  곡선최대차 {g}  "
              f"A 칼마 {ma.get('Cal'):.4f} / B 칼마 {mb.get('Cal'):.4f}"
              f"{'  ' + note if note else ''}", flush=True)
        for k, (va, vb) in diffs.items():
            print(f"    {k}: A={va} B={vb}", flush=True)

    path = os.path.join(OUT, 'ab_repro.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n앵커 {len(rows)}곳 중 다른 곳 {bad}곳. 기록 {os.path.abspath(path)}")
    if bad:
        print("B 가 A 를 재현 못 했다. 성과 격자로 넘어가지 않는다 — 원인부터 찾는다.")
    return 1 if bad else 0


def _same(x, y):
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    try:
        return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(x)))
    except (TypeError, ValueError):
        return x == y


if __name__ == '__main__':
    raise SystemExit(main())
