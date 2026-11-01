#!/usr/bin/env python3
"""현물 민감도 격자를 축별 지도로 읽는다.

spot_param_sensitivity.py 가 낸 JSONL 을 받아, 축마다 값별로 앵커 평균과 흔들림을 적고
기준선과의 차이를 낸다. 판정은 안 한다 — 지도만 그린다.

읽는 법 둘.
  · 평탄면이 있나. 이웃한 값에서 성과가 비슷해야 그 축의 선택이 과적합이 아니다.
    한 값만 뾰족하게 높으면 그건 봉우리가 아니라 흔들림일 수 있다.
  · 앵커 흔들림보다 큰 차이인가. 기준선의 앵커 표준편차보다 작은 차이는 잡음이다.
    그래서 축의 효과를 그 표준편차로 나눈 값(배수)을 같이 적는다.

쓰는 법:
  python3 spot_sensitivity_report.py <cells.jsonl> [--metric calmar] [--md <출력.md>]
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict

LABEL = {'calmar': '칼마', 'cagr': '연복리', 'mdd': '낙폭', 'sharpe': '샤프'}
PCT = {'cagr', 'mdd'}


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def agg(cells, metric):
    """같은 (축, 값) 의 앵커들을 묶어 평균·표준편차·최소·최대를 낸다."""
    vals = [c['metrics'][metric] for c in cells
            if c.get('metrics') and c['metrics'].get(metric) is not None]
    if not vals:
        return None
    return {'mean': st.mean(vals), 'sd': st.pstdev(vals),
            'min': min(vals), 'max': max(vals), 'n': len(vals)}


def fmt(v, metric):
    return f"{v*100:+.1f}%" if metric in PCT else f"{v:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cells')
    ap.add_argument('--metric', default='calmar', choices=list(LABEL))
    ap.add_argument('--md', default='')
    a = ap.parse_args()

    rows = load(a.cells)
    if not rows:
        print("빈 파일이다")
        return 1

    base_cells = [r for r in rows if r['axis'] == '_baseline']
    base = agg(base_cells, a.metric)
    if base is None:
        print("기준선 칸이 없다 — 격자가 아직 시작 단계일 수 있다")
        return 1

    out = []
    def w(s=''):
        out.append(s)
        print(s)

    m = a.metric
    w(f"# 현물 롱 변수 민감도 — {LABEL[m]}")
    w()
    w(f"자료: {a.cells}  (칸 {len(rows)}개)")
    sd_txt = fmt(base['sd'], m) if m in PCT else f"{base['sd']:.3f}"
    w(f"기준선: {fmt(base['mean'], m)}  앵커 {base['n']}곳  표준편차 {sd_txt}  "
      f"폭 {fmt(base['min'], m)} ~ {fmt(base['max'], m)}")
    w()
    w("차이는 기준선 평균 대비다. 배수는 그 차이를 기준선 앵커 표준편차로 나눈 값이고,")
    w("1 미만이면 앵커를 흔들었을 때 생기는 폭 안이라 축의 효과로 읽을 수 없다.")
    w()

    by_axis = defaultdict(list)
    for r in rows:
        if r['axis'] != '_baseline':
            by_axis[r['axis']].append(r)

    sd = base['sd'] or 1e-12
    for ax in by_axis:
        groups = defaultdict(list)
        for r in by_axis[ax]:
            groups[json.dumps(r['value'])].append(r)
        w(f"## {ax}")
        w()
        w(f"| 값 | {LABEL[m]} 평균 | 차이 | 배수 | 앵커 폭 |")
        w("|---|---:|---:|---:|---|")
        best = None
        for key in groups:
            g = agg(groups[key], m)
            if g is None:
                continue
            val = json.loads(key)
            d = g['mean'] - base['mean']
            mult = abs(d) / sd
            mark = ''
            if best is None or g['mean'] > best[1]:
                best = (val, g['mean'])
            w(f"| {val} | {fmt(g['mean'], m)} | {fmt(d, m)} | {mult:.1f}x | "
              f"{fmt(g['min'], m)} ~ {fmt(g['max'], m)} |{mark}")
        if best:
            w()
            w(f"이 축의 최고값: {best[0]} ({fmt(best[1], m)})")
        w()

    if a.md:
        with open(a.md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        print(f"\n적었다: {a.md}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
