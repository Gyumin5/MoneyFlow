"""숏 단독 성적 — 배수 축을 제대로 흔든다 (운영자 지시 9539·9540).

롱과 섞지 않는다. short_only=True 라 카나리가 켜져도 현금에 머문다. 그래서 여기
나오는 수치는 전부 "숏만 했을 때" 의 성적이고, 롱 전용 기준선과 직접 비교하는 수가
아니다 — 비교는 뒤에 합쳐서 잰다. 지금 보는 것은 배수가 숏의 수익·낙폭·샤프를 어떻게
바꾸나 하나다.

숏 다리는 선물 회계로 돈다(커밋 0b791a4) — 수수료 0.06% 편도, 선물 틱 슬리피지,
펀딩, 증거금, 청산가. 롱 다리는 어차피 안 선다.

축 둘.
⑴ 고정 배수 1·2·3·4.
⑵ 동적 배수. V25 롱 규칙의 거울이다 — 비트코인이 42일선 아래로 깊을수록, 코인이
   7일선 아래로 깊을수록 배수를 올린다. 범위 셋 × 비트코인 문턱 셋 × 코인 문턱 둘.

진입 이격은 2% 로 고정한다. 그 축은 따로 재는 자리가 있고, 한 격자에서 둘을 같이
흔들면 어느 쪽이 수를 움직였는지 못 가른다.

실행: python3 short_grid4.py   (앵커 7곳 × 구성 22개 = 154칸)
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-only-lev')

# 숏 단독. 방향 전환 유예는 구현 결함이었던 것이 밝혀져 직접 전환으로 둔다(커밋 e94a8ba).
_SH = dict(enable_short=True, short_only=True, short_entry_discount=0.02,
           allow_direct_flip=True, flip_preserve_snapshots=True)

# 동적 배수 범위 (min, mid, max)
RANGES = [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (1.0, 3.0, 5.0)]
# 비트코인 42일선 이격 문턱 (mid, max)
BTC_DEV = [(0.015, 0.05), (0.03, 0.08), (0.05, 0.12)]
# 코인 7일선 이격 문턱 (mid, max)
K2_DEV = [(0.025, 0.075), (0.05, 0.12)]

ARMS = [(f"고정{int(L)}배", dict(_SH, short_leverage=L)) for L in (1.0, 2.0, 3.0, 4.0)]
for lo, mid, hi in RANGES:
    for bm, bx in BTC_DEV:
        for km, kx in K2_DEV:
            ARMS.append((
                f"동적{int(lo)}{int(mid)}{int(hi)}-B{int(bm*1000):03d}{int(bx*1000):03d}"
                f"-K{int(km*1000):03d}{int(kx*1000):03d}",
                dict(_SH, short_lev_mode='dynamic',
                     short_lev_min=lo, short_lev_mid=mid, short_lev_max=hi,
                     short_btc_dev_mid=bm, short_btc_dev_max=bx,
                     short_k2_dev_mid=km, short_k2_dev_max=kx)))


def _kw(anchor):
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
    from unified_backtest import load_data

    bars, fund = load_data('D')
    print(f"자료 준비 끝. 구성 {len(ARMS)}개 × 앵커 {len(R.ANCHORS)}곳", flush=True)

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
                row = dict(arm=name, anchor=anchor,
                           sec=round(time.time() - t0, 1),
                           **{k: m.get(k) for k in keys})
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                f.flush()
            print(f"{name} 끝", flush=True)
    summary(path)
    return 0


def summary(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    print("\n앵커 평균 (칼마 / 연복리 / 낙폭 / 샤프 / 거래 / 청산 / 손절)")
    out = []
    for name, _o in ARMS:
        g = [x for x in rows if x['arm'] == name]
        if not g:
            continue
        a = {k: float(np.mean([x[k] for x in g]))
             for k in ('Cal', 'CAGR', 'MDD', 'Sharpe', 'Trades', 'Liq', 'Stops')}
        out.append((name, a))
    for name, a in out:
        print(f"  {name:<28} {a['Cal']:7.3f} {a['CAGR']:8.4f} {a['MDD']:8.4f} "
              f"{a['Sharpe']:7.3f} {a['Trades']:7.1f} {a['Liq']:5.1f} {a['Stops']:6.1f}")
    if out:
        print("\n연복리 상위 다섯")
        for name, a in sorted(out, key=lambda t: -t[1]['CAGR'])[:5]:
            print(f"  {name:<28} 연복리 {a['CAGR']:.4f} 칼마 {a['Cal']:.3f} "
                  f"낙폭 {a['MDD']:.4f} 샤프 {a['Sharpe']:.3f}")
        print("\n샤프 상위 다섯")
        for name, a in sorted(out, key=lambda t: -t[1]['Sharpe'])[:5]:
            print(f"  {name:<28} 샤프 {a['Sharpe']:.3f} 연복리 {a['CAGR']:.4f} "
                  f"칼마 {a['Cal']:.3f} 낙폭 {a['MDD']:.4f}")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary(os.path.join(OUT, 'cells.jsonl'))
        raise SystemExit(0)
    raise SystemExit(main())
