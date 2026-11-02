"""숏 단독 본 격자 — 축 여섯을 한 번에 건다 (운영자 지시 9529·9539·9540).

왜 한 번에 거나. 숏은 축이 서로 얽혀 있다. 이격을 깊게 잡으면 들어가는 자리가 줄어
종목 수와 노출의 뜻이 같이 바뀌고, 배수를 올리면 손절 폭의 뜻이 바뀐다. 한 축씩
흔들면 그 상호작용 자리를 통째로 못 본다(토론 run-20261102T005717Z 는 얽힌 쌍만
작게 교차하라고 했는데, 한 칸이 1초라 전부 교차해도 두 시간이다).

롱과 섞지 않는다(short_only=True). 숏 다리는 선물 회계다 — 수수료 0.06% 편도,
선물 틱 슬리피지, 펀딩, 증거금, 청산가(커밋 0b791a4).

축 여섯 = 배수 5 × 진입이격 3 × 총노출 4 × 종목수 3 × 선정 2 × 손절 3 = 1,080 구성.
앵커 7곳이라 7,560 칸.

실행: python3 short_grid5.py        (이어 돌리기 가능 — 끝난 칸은 건너뛴다)
      python3 short_grid5.py --summary
"""
from __future__ import annotations

import itertools
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

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-only-grid5')

BASE_SH = dict(enable_short=True, short_only=True,
               allow_direct_flip=True, flip_preserve_snapshots=True,
               short_slope_gate=True, short_exit_margin=0.015,
               short_rebalance_mode='anchor_fixed_qty')

LEV = [('L1', dict(short_leverage=1.0)),
       ('L2', dict(short_leverage=2.0)),
       ('L3', dict(short_leverage=3.0)),
       ('L4', dict(short_leverage=4.0)),
       ('Ldyn', dict(short_lev_mode='dynamic', short_lev_min=2.0,
                     short_lev_mid=3.0, short_lev_max=4.0))]
DISC = [('x00', 0.0), ('x02', 0.02), ('x04', 0.04)]
GROSS = [('g25', 0.25), ('g50', 0.50), ('g75', 0.75), ('g100', 1.00)]
NSEL = [('n1', 1), ('n3', 3), ('n5', 5)]
HEALTH = [('inv', 'inv_mom2vol'), ('fail', 'failed_rally')]
STOP = [('s_none', dict(short_stop_kind='none', short_stop_pct=0.0)),
        ('s_e12', dict(short_stop_kind='entry_price_pct', short_stop_pct=0.12)),
        ('s_l12', dict(short_stop_kind='lowest_close_since_entry_pct',
                       short_stop_pct=0.12))]


def arms():
    for (ln, lo), (dn, dv), (gn, gv), (nn, nv), (hn, hv), (sn, sv) in itertools.product(
            LEV, DISC, GROSS, NSEL, HEALTH, STOP):
        name = f"{ln}-{dn}-{gn}-{nn}-{hn}-{sn}"
        over = dict(BASE_SH, **lo)
        over.update(short_entry_discount=dv,
                    short_gross_exposure_cap=gv,
                    short_universe_size=nv,
                    short_cap=gv / nv,
                    short_per_coin_cap=gv / nv,
                    short_health_mode=hv, **sv)
        yield name, over


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
    all_arms = list(arms())
    print(f"자료 준비 끝. 구성 {len(all_arms)}개 × 앵커 {len(R.ANCHORS)}곳 "
          f"= {len(all_arms) * len(R.ANCHORS)}칸", flush=True)

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
    t_start = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for i, (name, over) in enumerate(all_arms, 1):
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
            if i % 25 == 0:
                el = time.time() - t_start
                print(f"  구성 {i}/{len(all_arms)} ({el/60:.1f}분)", flush=True)
    summary(path)
    return 0


def summary(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    if not rows:
        print("칸이 없다")
        return
    agg = {}
    for r in rows:
        agg.setdefault(r['arm'], []).append(r)
    out = []
    for name, g in agg.items():
        if len(g) < len(R.ANCHORS):
            continue
        a = {k: float(np.mean([x[k] for x in g]))
             for k in ('Cal', 'CAGR', 'MDD', 'Sharpe', 'Trades', 'Liq', 'Stops')}
        a['CAGR_min'] = float(np.min([x['CAGR'] for x in g]))
        out.append((name, a))
    print(f"\n완성된 구성 {len(out)}개 (앵커 {len(R.ANCHORS)}곳 평균)")

    def show(title, key, rev=True, n=12):
        print(f"\n{title}")
        print("  구성                              칼마   연복리     낙폭    샤프"
              "   앵커최저연복리")
        for name, a in sorted(out, key=lambda t: -t[1][key] if rev else t[1][key])[:n]:
            print(f"  {name:<32} {a['Cal']:6.3f} {a['CAGR']:8.4f} {a['MDD']:8.4f} "
                  f"{a['Sharpe']:7.3f} {a['CAGR_min']:9.4f}")

    show("연복리 상위", 'CAGR')
    show("샤프 상위", 'Sharpe')
    show("칼마 상위", 'Cal')

    # 축별 평균 — 어느 축이 수를 움직이나
    print("\n축별 평균 연복리 / 샤프")
    pos = {'배수': 0, '이격': 1, '노출': 2, '종목수': 3, '선정': 4, '손절': 5}
    for axis, i in pos.items():
        buckets = {}
        for name, a in out:
            buckets.setdefault(name.split('-')[i], []).append(a)
        print(f"  [{axis}]")
        for k, v in sorted(buckets.items()):
            print(f"    {k:<8} 연복리 {np.mean([x['CAGR'] for x in v]):8.4f}  "
                  f"샤프 {np.mean([x['Sharpe'] for x in v]):7.3f}  "
                  f"낙폭 {np.mean([x['MDD'] for x in v]):8.4f}")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary(os.path.join(OUT, 'cells.jsonl'))
        raise SystemExit(0)
    raise SystemExit(main())
