"""축 민감도가 조건을 바꾸면 어떻게 달라지나 — 두 측정을 나란히 놓는다.

쓰는 자리. 2026-11-02 에 연구 판정기 네 곳이 라이브에 없는 전략 가드를 켠 채 돌고
있었다(history #114). 그중 민감도 측정은 "어느 축이 실제로 움직이나" 를 보고 교차
격자에 넣을 축 넷을 고른 자리라, 결함이 수치가 아니라 설계에 닿는다. 가드를 끄고
다시 잰 뒤 축 선택을 다시 하려면 "어느 축의 움직임이 커졌고 어느 축이 죽었나" 를
봐야 한다. 평균표 둘을 눈으로 번갈아 보는 것으로는 안 된다.

재는 것. 축마다 그 축의 값들이 만든 칼마·연복리의 폭(최고 - 최저)을 낸다. 폭이
앵커 흔들림(기준선 한 구성이 앵커 일곱에서 벌어지는 폭)보다 작으면 그 축은 잡음과
못 가른다 — 두 조건에서 그 판정이 뒤집혔는지가 축 재선택의 근거다.

읽는 법. "폭" 이 그 축을 흔들었을 때 얻을 수 있는 최대 차이다. 앵커폭 대비가 1
아래면 그 축의 봉우리는 앵커를 바꾸면 사라지는 크기다. 두 조건의 대비가 1 을
사이에 두고 갈리는 축이 재선택 대상이다.

실행: python3 spot_axis_shift.py                     (가드 켠 것 대 끈 것)
      python3 spot_axis_shift.py <옛 디렉토리> <새 디렉토리>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

OLD = os.path.join(ROOT, 'state', '20261102-spot-long')
NEW = os.path.join(ROOT, 'state', '20261102-spot-long-noguard')


def load(d):
    p = os.path.join(d, 'cells.jsonl')
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p, encoding='utf-8'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def digest(rows):
    """축별 폭과 기준선 앵커 폭."""
    by = {}
    for r in rows:
        by.setdefault((r['axis'], json.dumps(r['value'])), []).append(r['metrics'])

    base = [v for (ax, _val), v in by.items() if ax == '_baseline']
    anchor_cal = anchor_cagr = float('nan')
    if base:
        ms = base[0]
        anchor_cal = float(np.max([m['calmar'] for m in ms]) -
                           np.min([m['calmar'] for m in ms]))
        anchor_cagr = float(np.max([m['cagr'] for m in ms]) -
                            np.min([m['cagr'] for m in ms]))

    axes = {}
    for (ax, val), ms in by.items():
        if ax == '_baseline':
            continue
        axes.setdefault(ax, []).append(
            (json.loads(val),
             float(np.mean([m['calmar'] for m in ms])),
             float(np.mean([m['cagr'] for m in ms]))))

    out = {}
    for ax, pts in axes.items():
        cals = [c for _v, c, _g in pts]
        cagrs = [g for _v, _c, g in pts]
        best = max(pts, key=lambda t: t[1])
        out[ax] = dict(
            n=len(pts),
            cal_span=max(cals) - min(cals),
            cagr_span=max(cagrs) - min(cagrs),
            best_val=best[0], best_cal=best[1], best_cagr=best[2],
        )
    return out, anchor_cal, anchor_cagr


def main():
    old_dir, new_dir = OLD, NEW
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if len(args) == 2:
        old_dir, new_dir = args

    ro, rn = load(old_dir), load(new_dir)
    if ro is None:
        print(f"옛 측정이 없다: {old_dir}")
        return 1
    if rn is None:
        print(f"새 측정이 아직 없다: {new_dir}")
        print("민감도 재측정이 도는 중이면 끝난 뒤에 다시 불러라.")
        return 1

    do, ac_o, ag_o = digest(ro)
    dn, ac_n, ag_n = digest(rn)
    print(f"옛 측정 {len(ro)}칸 · 새 측정 {len(rn)}칸")
    print(f"기준선 앵커 폭 — 칼마 {ac_o:.3f} → {ac_n:.3f},"
          f" 연복리 {ag_o:.4f} → {ag_n:.4f}\n")

    print("축별 칼마 폭과 앵커폭 대비 (1 아래면 잡음과 못 가른다)")
    print(f"  {'축':<20}{'옛폭':>8}{'옛대비':>8}{'새폭':>8}{'새대비':>8}  판정")
    rows = []
    for ax in sorted(set(do) | set(dn)):
        o, n = do.get(ax), dn.get(ax)
        so = o['cal_span'] if o else float('nan')
        sn = n['cal_span'] if n else float('nan')
        ro_ = so / ac_o if ac_o == ac_o and ac_o > 0 else float('nan')
        rn_ = sn / ac_n if ac_n == ac_n and ac_n > 0 else float('nan')
        if ro_ != ro_ or rn_ != rn_:
            verdict = "한쪽 없음"
        elif ro_ < 1 <= rn_:
            verdict = "← 새로 살아남 (교차에 넣어야 한다)"
        elif rn_ < 1 <= ro_:
            verdict = "← 죽음 (교차에서 빼도 된다)"
        elif rn_ >= 1:
            verdict = "둘 다 유의"
        else:
            verdict = "둘 다 잡음"
        rows.append((ax, so, ro_, sn, rn_, verdict))
    for ax, so, ro_, sn, rn_, v in sorted(rows, key=lambda t: -(t[4] if t[4] == t[4] else -1)):
        print(f"  {ax:<20}{so:8.3f}{ro_:8.2f}{sn:8.3f}{rn_:8.2f}  {v}")

    print("\n축별 최고 칼마 값이 옮겨졌나")
    print(f"  {'축':<20}{'옛최고값':>12}{'옛칼마':>9}{'새최고값':>12}{'새칼마':>9}")
    for ax in sorted(set(do) & set(dn)):
        o, n = do[ax], dn[ax]
        mark = "" if str(o['best_val']) == str(n['best_val']) else "  ← 옮겨졌다"
        print(f"  {ax:<20}{str(o['best_val']):>12}{o['best_cal']:9.3f}"
              f"{str(n['best_val']):>12}{n['best_cal']:9.3f}{mark}")

    print("\n연복리 폭도 같이 본다 — 칼마만 보면 노출을 줄여 낙폭을 깎은 축이 좋아 보인다.")
    print(f"  {'축':<20}{'옛연복리폭':>12}{'새연복리폭':>12}")
    for ax in sorted(set(do) & set(dn)):
        print(f"  {ax:<20}{do[ax]['cagr_span']:12.4f}{dn[ax]['cagr_span']:12.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
