#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""위약 다중비교 보정 — 격자의 1등이 실력인지 자릿수인지 가른다.

후보가 94만이면 그 중 하나는 우연만으로도 현행을 이긴다. 위약 한 벌은 전이 사건
신호만 무의미하게 만든 채 같은 격자를 통째로 돌려서 "우연이 낼 수 있는 최대 개선"
한 줄을 남긴다. 여기서는 그 줄들을 모아 분포를 만들고 사전등록한 문턱을 적용한다.

사전등록한 것(바꾸지 않는다): 실제 격자의 최대 개선이 위약 최대 개선의 95분위를
넘어야 한다. 순위는 작을수록 좋으므로 반대쪽 끝 — 위약 최소순위의 5분위다.

여기서 더한 것은 그 문턱의 기계적 적용 둘이고 새 기준이 아니다.
  · 후보별 걸러내기 — 자기 개선폭이 위약 95분위를 못 넘는 후보는 우연과 구분이
    안 되므로 떨어뜨린다. 격자 최대에 대한 문턱을 후보마다 적용한 것이고,
    스트레스 단계가 받을 색인이 이것이다.
  · 가족 둘(shift·rand)의 문턱 중 엄한 쪽을 쓴다. 둘 다 같은 신호를 지우므로
    한쪽만 넘는 것은 그 가족의 귀무가정이 느슨했다는 뜻이다.

위약은 앵커 0 · 기본비용 한 칸에서만 돈다. 그래서 이 보정도 그 칸의 통계에만
건다 — 다른 열세 칸의 우연 크기는 재지 않았고, 그 사실을 판정문에 적는다.

실행: python3 alloc_placebo_agg.py [--idx <본채점 통과색인>] [--min-each N]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import alloc_grid_run as RUN      # noqa: E402

OUT = RUN.OUT
PL = os.path.join(OUT, "placebo")
KINDS = ['shift', 'rand']
ANCHOR, COST = 0, 'base'          # 위약을 돌린 칸. 보정도 이 칸에서만 건다.


def load_placebos():
    """가족별로 위약 줄을 모은다. 기준선이 벌마다 다르면 섞인 것이므로 막는다."""
    rows = {k: [] for k in KINDS}
    for p in sorted(glob.glob(os.path.join(PL, "pl_*.json"))):
        with open(p) as f:
            r = json.load(f)
        if r.get('anchor') != ANCHOR or r.get('stress') is not False:
            continue
        if r['kind'] in rows:
            rows[r['kind']].append(r)
    return rows


def null_thresholds(rows):
    """위약 분포에서 사전등록 문턱을 뽑는다 — 개선은 95분위, 순위는 5분위."""
    out = {}
    for k, rs in rows.items():
        if not rs:
            continue
        b = {f: np.array([r['best'][f] for r in rs], dtype=np.float64)
             for f in ('rank', 'dCal', 'dMDD', 'dht95')}
        out[k] = dict(n=len(rs),
                      rank=float(np.percentile(b['rank'], 5)),
                      dCal=float(np.percentile(b['dCal'], 95)),
                      dMDD=float(np.percentile(b['dMDD'], 95)),
                      dht95=float(np.percentile(b['dht95'], 95)),
                      npass=np.array([r['npass'] for r in rs], dtype=np.int64),
                      n140=np.array([r['n140'] for r in rs], dtype=np.int64),
                      raw=b)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', default=os.path.join(OUT, "pass_idx.npy"),
                    help='본 채점 통과 색인. 없으면 격자 전체에 건다.')
    ap.add_argument('--min-each', type=int, default=100,
                    help='가족마다 이 수를 못 채우면 판정이 아니라 중간값이라고 적는다')
    a = ap.parse_args()

    rows = load_placebos()
    th = null_thresholds(rows)
    if not th:
        print(f"위약이 한 벌도 없다: {PL}")
        return 1

    print(f"위약 {sum(v['n'] for v in th.values())}벌 "
          + " ".join(f"{k} {v['n']}" for k, v in th.items()))
    print("\n가족별 귀무분포 (격자 전체의 최대값 분포)")
    print(f"{'가족':<8}{'벌':>5}{'최소순위 5분위':>15}{'칼마개선 95분위':>16}"
          f"{'낙폭개선 95분위':>16}{'이탈개선 95분위':>16}{'전기준통과 중앙':>16}")
    for k, v in th.items():
        print(f"{k:<8}{v['n']:>5}{v['rank']:>15.4f}{v['dCal']:>16.3f}"
              f"{v['dMDD'] * 100:>15.2f}pp{v['dht95'] * 100:>15.2f}pp"
              f"{np.median(v['npass']):>16.1f}")

    short = [k for k in KINDS if th.get(k, {}).get('n', 0) < a.min_each]
    if short:
        print(f"\n[경고] {', '.join(short)} 가족이 {a.min_each}벌을 못 채웠다 — "
              "아래는 중간 집계이고 사전등록 판정이 아니다. 벌이 다 모이면 같은 명령으로 다시 본다.")

    cell = os.path.join(OUT, f"grid_a{ANCHOR}_{COST}.npz")
    meta = os.path.join(OUT, f"grid_a{ANCHOR}_{COST}.json")
    if not (os.path.exists(cell) and os.path.exists(meta)):
        print(f"\n본 채점 칸이 아직 없다: {cell}")
        return 1
    z = np.load(cell)
    base = json.load(open(meta))['base']
    n = z['rank'].shape[0]

    # 위약이 재는 것과 같은 축으로 실제 격자의 개선폭을 센다
    real = dict(rank=z['rank'],
                dCal=z['Cal'] - base['Cal'],
                dMDD=abs(base['MDD']) - np.abs(z['MDD']),
                dht95=base['ht95'] - z['ht95'])

    print(f"\n실제 격자(앵커 {ANCHOR} {COST}, 후보 {n:,}) 대 위약 문턱")
    print(f"{'축':<10}{'실제 최대':>12}{'엄한 문턱':>12}{'가족':>8}{'넘나':>8}")
    strict = {}
    for f in ('rank', 'dCal', 'dMDD', 'dht95'):
        if f == 'rank':
            t = min(v[f] for v in th.values())        # 순위는 작을수록 좋다
            who = min(th, key=lambda k: th[k][f])
            got = float(real[f].min())
            beat = got < t
        else:
            t = max(v[f] for v in th.values())
            who = max(th, key=lambda k: th[k][f])
            got = float(real[f].max())
            beat = got > t
        strict[f] = t
        sc = 100 if f in ('dMDD', 'dht95') else 1
        unit = 'pp' if sc == 100 else ''
        print(f"{f:<10}{got * sc:>12.3f}{unit}{t * sc:>11.3f}{unit}{who:>8}"
              f"{'예' if beat else '아니오':>8}")

    # 후보별 보정 — 네 축 전부에서 위약 문턱을 넘어야 남는다
    keep = ((real['rank'] < strict['rank']) & (real['dCal'] > strict['dCal'])
            & (real['dMDD'] > strict['dMDD']) & (real['dht95'] > strict['dht95']))
    print(f"\n네 축 전부에서 위약 문턱을 넘은 후보 {int(keep.sum()):,} / {n:,}")
    for f in ('rank', 'dCal', 'dMDD', 'dht95'):
        m = (real[f] < strict[f]) if f == 'rank' else (real[f] > strict[f])
        print(f"  {f} 단독 통과 {int(m.sum()):,}")

    if os.path.exists(a.idx):
        pre = np.load(a.idx).astype(np.int64)
        both = np.intersect1d(pre, np.flatnonzero(keep))
        print(f"\n본 채점 통과 {pre.size:,}개 중 보정까지 넘은 것 {both.size:,}개")
        np.save(os.path.join(OUT, "pass_idx_corrected.npy"), both)
        print(f"저장: {OUT}/pass_idx_corrected.npy")
        if both.size == 0:
            print("보정 뒤 남는 후보가 0개면 이 연구의 결론은 현행 유지다 — "
                  "스트레스 단계는 잴 것이 없다.")
    else:
        print(f"\n본 채점 통과 색인이 아직 없다({a.idx}) — 격자 전체에만 걸었다. "
              "채점이 끝나면 같은 명령으로 다시 본다.")
        np.save(os.path.join(OUT, "placebo_keep_mask.npy"), keep)
        print(f"저장: {OUT}/placebo_keep_mask.npy")

    print(f"\n확인하지 못한 것: 위약은 앵커 {ANCHOR} {COST} 한 칸에서만 돌았다. "
          "나머지 열세 칸의 우연 크기는 안 쟀고, 이 보정도 그 칸 통계에만 건다.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
