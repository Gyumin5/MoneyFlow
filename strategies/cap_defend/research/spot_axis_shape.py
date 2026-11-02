"""축마다 모양을 판정한다 — 어디가 진짜 자리이고 어디가 과적합인가.

42일선 하나만 보던 판정(spot_sma_peak_check.py, history #117)을 민감도 측정의 모든
축으로 넓힌다. 운영자 지시(2026-11-02): 과적합인지 생존편향인지 다 판단하라.

판정선은 기준선 앵커 폭이다 — 같은 구성 하나가 앵커 일곱 곳에서 벌어지는 칼마 폭.
그보다 작은 차이는 앵커를 바꾸면 사라지므로 어느 쪽이 높다고 말하지 않는다.

모양 넷으로 가른다.
  평평   축 전체 폭이 앵커 폭 이하. 그 축은 성과를 안 가른다 — 아무 값이나 써도 같다.
  톱니   이웃 간 차이가 앵커 폭을 넘는 자리가 절반을 넘는다. 값을 고르는 행위 자체가
         과적합이다(42일선이 이 꼴이다).
  단조   최고값이 잰 범위의 끝에 있다. 더 밀면 더 좋아진다는 뜻이라 봉우리를 못 봤다 —
         범위 밖을 재기 전엔 그 끝값을 고르지 않는다.
  단봉   최고값이 안쪽이고 톱니가 아니다. 양쪽으로 내려가므로 진짜 자리다.

값이 셋 미만인 축은 모양을 말할 수 없어 건너뛴다.

읽는 법. 단봉인 축만 값을 고를 자격이 있다. 톱니·단조는 지금 값을 유지하거나 범위를
넓혀 다시 재는 쪽이고, 평평은 아예 격자에서 뺀다(칸만 늘리고 아무것도 안 가른다).

함정 하나. 이 판정은 한 축만 움직인 측정을 읽으므로, 다른 축과 얽힌 축은 모양을 잘못
낸다. 실제로 걸렸다 — universe_size 가 여기선 톱니(4/5)인데 상한을 50% 로 고정한
2x2 측정에서는 1<2>3>4>5 단봉이었다(history #118). 민감도는 상한을 현행 1/3 에 두고
종목 수만 바꾸는데, 그러면 투자 비중이 같이 움직인다(2개면 67%, 3개면 100%). 선정
효과와 노출 효과가 한 축에 겹쳐서 톱니로 보인 것이다.

그래서 축마다 성격도 같이 낸다. 노출을 올리면 연복리와 낙폭이 같이 커지고, 선정이
좋아지면 연복리만 오르고 낙폭은 안 커진다. 축을 따라가며 연복리와 낙폭 절댓값의
상관을 재서 +0.7 이상이면 노출축으로 부른다. 노출축의 모양은 액면 그대로 읽지 않고,
상대 축을 고정한 2x2 로 다시 잰다. 모양만 보고 과적합이라고 적지 않는다.

값이 숫자가 아닌 축(health_mode 처럼 이름 몇 개 중 고르는 것)은 모양을 안 본다.
이웃이라는 게 없어서 톱니·단봉이 뜻을 갖지 않는다 — 글자순으로 늘어놓고 이웃 차이를
세면 아무 말이나 나온다.

실행: python3 spot_axis_shape.py [측정디렉토리]
기본 측정: state/20261102-spot-long-noguard (가드를 끈 라이브 정합 민감도)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

DEFAULT = os.path.join(ROOT, 'state', '20261102-spot-long-noguard')


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, 'cells.jsonl')
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def anchor_span(rows):
    """기준선 한 구성이 앵커 사이에서 벌어지는 칼마 폭. 판정선이다."""
    c = [r['metrics']['calmar'] for r in rows if r['axis'] == '_baseline']
    return float(max(c) - min(c)) if c else float('nan')


def by_value(rows, axis):
    g = {}
    for r in rows:
        if r['axis'] == axis:
            g.setdefault(json.dumps(r['value']), []).append(r['metrics'])
    out = {}
    for k, ms in g.items():
        v = json.loads(k)
        # 수치 축만 모양을 본다. 이름 중에 고르는 축은 이웃이 없어 톱니가 뜻을 안 갖는다.
        out[v if isinstance(v, (int, float)) and not isinstance(v, bool) else k] = dict(
            cal=float(np.mean([m['calmar'] for m in ms])),
            cagr=float(np.mean([m['cagr'] for m in ms])),
            mdd=float(np.mean([abs(m['mdd']) for m in ms])),
            n=len(ms))
    return out


def nature(vals):
    """노출축인가 선정축인가. 연복리와 낙폭이 같이 커지면 노출을 산 것이다."""
    try:
        ks = sorted(vals)
    except TypeError:
        return '—', float('nan')
    a = np.asarray([vals[k]['cagr'] for k in ks], dtype=float)
    b = np.asarray([vals[k]['mdd'] for k in ks], dtype=float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return '—', float('nan')
    r = float(np.corrcoef(a, b)[0, 1])
    return ('노출' if r >= 0.7 else '선정'), r


def shape(vals, span):
    """값→칼마 사전을 받아 모양 이름과 근거 수치를 돌려준다."""
    ks = sorted(vals)
    if not all(isinstance(k, (int, float)) for k in ks):
        return '범주형', {}
    cals = [vals[k]['cal'] for k in ks]
    width = max(cals) - min(cals)
    if width <= span:
        return '평평', dict(width=width, best=ks[int(np.argmax(cals))])
    steps = [abs(cals[i + 1] - cals[i]) for i in range(len(cals) - 1)]
    big = sum(1 for s in steps if s > span)
    bi = int(np.argmax(cals))
    info = dict(width=width, best=ks[bi], jag=big, steps=len(steps))
    if big > len(steps) / 2:
        return '톱니', info
    if bi == 0 or bi == len(ks) - 1:
        return '단조', info
    return '단봉', info


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = load(src)
    if rows is None:
        print(f"측정이 없다: {src}")
        return 1
    span = anchor_span(rows)
    axes = sorted({r['axis'] for r in rows if r['axis'] != '_baseline'})
    print(f"측정 {os.path.relpath(src, ROOT)} · 칸 {len(rows)} · 축 {len(axes)}")
    print(f"기준선 앵커 폭(칼마) {span:.3f} — 이보다 작은 차이는 잡음으로 본다\n")

    print(f"  {'축':<22}{'값수':>5}{'폭':>8}{'최고값':>10}{'톱니/계단':>11}  {'모양':<6}{'성격':>6}{'상관':>8}")
    verdict = {}
    tangled = []
    for ax in axes:
        vals = by_value(rows, ax)
        if len(vals) < 3:
            print(f"  {ax:<22}{len(vals):>5}       —         —          —  값이 셋 미만")
            continue
        name, info = shape(vals, span)
        nat, r = nature(vals)
        verdict.setdefault(name, []).append(ax)
        if nat == '노출':
            tangled.append(ax)
        jag = f"{info.get('jag', 0)}/{info.get('steps', 0)}" if 'jag' in info else '—'
        w = f"{info['width']:.3f}" if 'width' in info else '—'
        best = str(info.get('best', '—'))
        rs = '—' if r != r else f"{r:+.2f}"
        print(f"  {ax:<22}{len(vals):>5}{w:>8}{best:>10}{jag:>11}  {name:<6}{nat:>6}{rs:>8}")

    print("\n  모양별")
    for name in ('단봉', '톱니', '단조', '평평', '범주형'):
        if name in verdict:
            print(f"    {name}: {', '.join(verdict[name])}")
    if tangled:
        print(f"\n  노출축(모양을 액면으로 안 읽는다): {', '.join(tangled)}")

    print("\n  쓰는 법. 단봉이면서 선정축인 것만 값을 고를 자격이 있다. 톱니는 값 고르기")
    print("  자체가 과적합이라 지금 값을 유지한다. 단조는 봉우리를 못 본 것이라 범위를")
    print("  넓혀 다시 재기 전엔 끝값을 안 고른다. 평평은 격자에서 뺀다 — 칸만 늘린다.")
    print("  노출축은 상대 축을 고정한 2x2 로 다시 잰다 — 연복리가 오른 게 잘 골라서인지")
    print("  현금을 덜 들어서인지 한 축만 움직인 측정으로는 안 갈린다.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
