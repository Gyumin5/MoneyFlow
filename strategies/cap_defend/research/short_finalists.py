"""숏 후보를 평균표가 아니라 사건으로 판정한다.

왜 따로 두나. 본 격자(short_grid5.py)는 칸마다 지표만 적고 자산곡선을 안 남긴다.
그런데 숏에서 먼저 물어야 하는 것은 "평균이 얼마냐" 가 아니라 셋이다 —
들어간 적이 몇 번이나 있나, 그 성과가 며칠에 몰려 있나, 구간을 옮겨도 양수인가.
셋 다 곡선이 있어야 잰다. 그래서 지목한 후보만 다시 돌려 곡선을 받는다.

현물에서 배운 순서를 그대로 건다(history #104·#106·#108·#109).
  ⑴ 평균표 1위는 후보 자격이지 채택이 아니다.
  ⑵ 창을 옮겨 가며 재서 한 구간 쏠림을 가른다.
  ⑶ 우위가 며칠에 걸려 있으면 그건 전략이 아니라 사건이다.

후보를 전 구간 칼마 상위로 고르지 않는다. 그 선별 자체가 여기서 잡으려는 함정이다
(spot_long_windows.py 가 같은 이유로 그 길을 폐지했다). 고르는 길은 둘이다 —
이름을 손으로 지목하거나, 성과가 아닌 구조 조건으로 거른다(청산 0회, 진입 사건
최소 수). 구조 조건은 "잴 수 있는 칸인가" 를 묻는 것이지 "좋은 칸인가" 가 아니다.

기준선을 안 둔다. 현물은 라이브 전략이 기준선이지만 숏은 지금 아무것도 안 하고
있으므로 비교 대상이 현금이다. 그래서 쌍대 순위합 대신 절대 판정을 쓴다 —
창마다 양수인가, 상위 10일을 빼도 양수인가.

판정선은 앵커 일곱 중 다섯이다(현물과 같다).

값을 미리 적어 둔다. 구조 조건은 넓다 — 부분 자료 831개 중 617개가 통과했다.
그걸 그대로 돌리면 4천 칸이 넘어 본 격자를 한 번 더 도는 셈이다. 실제로 쓰는 길은
격자 요약에서 축별로 살아남은 값을 보고, 그 축 값을 고루 덮는 후보를 손으로 지목해
`--arms` 로 거는 것이다. `--valid` 는 후보가 적을 때와 구조 조건 자체를 볼 때만 쓴다.

쓰는 법:
  python3 short_finalists.py --arms L1-x00-g25-n1-inv-s_none,L2-x02-g50-n3-inv-s_e12
  python3 short_finalists.py --valid          (청산 0 + 진입 사건 충분한 칸 전부)
  python3 short_finalists.py --summary
"""
from __future__ import annotations

import argparse
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
import short_grid5 as G  # noqa: E402

GRID = os.path.join(CAP, '..', '..', 'state',
                    '20261102-short-only-grid5', 'cells.jsonl')
OUT = os.path.join(CAP, '..', '..', 'state', '20261102-short-finalists')

# (창 길이, 이동폭) — 봉 단위. 격자 구간이 2020-10 ~ 2026-09 라 약 2,150봉이다.
# 1008봉 창은 자리가 둘밖에 안 나와 뺀다.
WINDOWS = [(504, 63), (756, 126)]
TOP_DAYS = 10        # 사건 집중도 — 상위 며칠을 빼고 다시 보나
MIN_TRADES = 20      # 구조 조건: 이보다 적게 들어간 칸은 수를 믿을 수 없다


def arm_overrides(name):
    """격자 이름(L2-x02-g50-n3-inv-s_e12)을 run() 인자로 되돌린다.

    이름을 손으로 뜯지 않고 격자 생성기가 만든 짝에서 찾는다 — 축이 늘거나
    이름 규칙이 바뀌어도 여기만 따로 낡지 않는다.
    """
    for n, over in G.arms():
        if n == name:
            return over
    raise KeyError(f"격자에 없는 이름: {name}")


def load_cells():
    agg = {}
    with open(GRID, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            agg.setdefault(r['arm'], {})[r['anchor']] = r
    return agg


def pick_valid(agg):
    """성과가 아니라 구조로 거른다 — 청산 0회, 진입 사건 충분, 앵커 일곱 완료."""
    out = []
    for name, by in sorted(agg.items()):
        if len(by) < len(R.ANCHORS):
            continue
        rows = list(by.values())
        if any((r.get('Liq') or 0) > 0 for r in rows):
            continue
        if np.mean([r.get('Trades') or 0 for r in rows]) < MIN_TRADES:
            continue
        out.append(name)
    return out


def curve(bars, fund, name, anchor):
    kw = G._kw(anchor)
    over = arm_overrides(name)
    from short_engine import run as bt
    m = bt(bars, fund, **kw, **over)
    eq = m.get('_equity')
    if eq is None:
        return None, m
    v = np.asarray(eq.values if hasattr(eq, 'values') else eq, dtype=float)
    return v, m


def concentration(v):
    """상위 며칠을 빼면 무엇이 남나.

    일별 로그수익을 더한 것이 전체다. 거기서 가장 큰 열흘을 빼고 다시 더한다.
    남은 것이 음수면 그 성과는 전략이 아니라 사건 열 개다.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 2:
        return None
    lr = np.diff(np.log(v))
    tot = float(np.sum(lr))
    k = min(TOP_DAYS, len(lr))
    resid = float(tot - np.sum(np.sort(lr)[-k:]))
    yrs = len(v) / 365.0
    return dict(total=tot, resid=resid, bars=len(v),
                cagr=float(np.exp(tot / yrs) - 1.0) if yrs > 0 else 0.0,
                cagr_resid=float(np.exp(resid / yrs) - 1.0) if yrs > 0 else 0.0,
                share=(float(np.sum(np.sort(lr)[-k:]) / tot) if abs(tot) > 1e-12
                       else float('nan')))


def window_positive(v):
    """창을 밀어가며 양수 비율을 센다. 기준선이 현금이라 0 이 문턱이다."""
    v = np.asarray(v, dtype=float)
    pos = tot = 0
    bylen = {}
    for wlen, step in WINDOWS:
        if len(v) < wlen:
            continue
        p = n = 0
        for s in range(0, len(v) - wlen + 1, step):
            seg = v[s:s + wlen]
            if seg[0] <= 0 or not np.isfinite(seg).all():
                continue
            n += 1
            if seg[-1] > seg[0]:
                p += 1
        if n:
            bylen[wlen] = (p, n, 100.0 * p / n)
            pos += p
            tot += n
    return bylen, (100.0 * pos / tot if tot else 0.0), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='')
    ap.add_argument('--valid', action='store_true',
                    help='청산 0 + 진입 사건 충분 + 앵커 일곱 완료인 칸 전부')
    a = ap.parse_args()

    agg = load_cells()
    if a.arms:
        arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    elif a.valid:
        arms = pick_valid(agg)
    else:
        print("후보를 지목해라: --arms <이름들> 또는 --valid")
        print("(전 구간 칼마 상위만 골라 보내는 길은 안 만들었다 — 그 선별 자체가"
              " 여기서 잡으려는 함정이다.)")
        return 2
    if not arms:
        print("후보가 없다. 격자가 덜 끝났거나 구조 조건에 다 걸렸다.")
        return 2
    print(f"후보 {len(arms)}개 × 앵커 {len(R.ANCHORS)}곳 "
          f"= {len(arms)*len(R.ANCHORS)}칸", flush=True)

    os.makedirs(OUT, exist_ok=True)
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from unified_backtest import load_data
    bars, fund = load_data('D')

    path = os.path.join(OUT, 'curves.npz')
    curves = {}
    if os.path.exists(path):
        z = np.load(path)
        for k in z.files:
            n, ph = k.rsplit('|', 1)
            curves[(n, int(ph))] = z[k]
        print(f"이미 받아 둔 곡선 {len(curves)}개", flush=True)

    t0 = time.time()
    for name in arms:
        for anchor in R.ANCHORS:
            if (name, anchor) in curves:
                continue
            v, _m = curve(bars, fund, name, anchor)
            if v is None:
                print(f"  {name} 앵커 {anchor}: 자산곡선 없음", flush=True)
                continue
            curves[(name, anchor)] = v
        np.savez_compressed(path,
                            **{f"{n}|{p}": x for (n, p), x in curves.items()})
        print(f"  {name} 끝 ({(time.time()-t0)/60:.1f}분)", flush=True)

    judge(curves, arms, agg)
    return 0


def judge(curves, arms, agg):
    print(f"\n숏 후보 판정 (앵커 {len(R.ANCHORS)}곳, 상위 {TOP_DAYS}일 제외)")
    print("  순서는 사건 수 → 창 양수율 → 상위일 제외 잔여다. 평균표는 맨 뒤에 둔다.")
    print(f"\n  {'구성':<26}{'진입':>6}{'창양수율':>9}{'연복리':>9}"
          f"{'잔여':>9}{'상위10일몫':>11}{'통과앵커':>9}")
    rows = []
    for name in arms:
        cs = [(ph, curves[(name, ph)]) for ph in R.ANCHORS
              if (name, ph) in curves]
        if not cs:
            continue
        trades = np.mean([r.get('Trades') or 0
                          for r in agg.get(name, {}).values()]) if agg.get(name) else float('nan')
        wp, cg, cr, sh, ok = [], [], [], [], 0
        for ph, v in cs:
            bylen, pct, n = window_positive(v)
            c = concentration(v)
            if c is None or not n:
                continue
            wp.append(pct)
            cg.append(c['cagr'])
            cr.append(c['cagr_resid'])
            sh.append(c['share'])
            if pct > 50.0 and c['resid'] > 0:
                ok += 1
        if not wp:
            continue
        s = dict(arm=name, trades=float(trades), winpct=float(np.mean(wp)),
                 cagr=float(np.mean(cg)), cagr_resid=float(np.mean(cr)),
                 share=float(np.nanmean(sh)), anchors=ok)
        rows.append(s)

    rows.sort(key=lambda r: (-r['anchors'], -r['cagr_resid']))
    for s in rows:
        mark = '통과' if s['anchors'] >= 5 else '미달'
        print(f"  {s['arm']:<26}{s['trades']:6.0f}{s['winpct']:8.1f}%"
              f"{s['cagr']:+9.2%}{s['cagr_resid']:+9.2%}{s['share']:10.1%}"
              f"   {s['anchors']}/7 {mark}")

    print("\n읽는 법. 잔여는 가장 좋았던 열흘을 빼고 다시 연복리로 환산한 값이다.")
    print("  잔여가 음수면 그 성과는 전략이 아니라 사건 열 개다 — 다음에 같은 열흘이")
    print("  온다는 보장이 없으므로 채택 근거가 못 된다. 상위10일몫은 전체 로그수익")
    print("  중 그 열흘이 차지한 비율이고, 1.0 을 넘으면 나머지 날이 합쳐서 손실이다.")
    print("  통과앵커는 창 양수율이 절반을 넘고 잔여가 양수인 앵커 수이고 판정선은")
    print("  다섯이다. 진입은 앵커 평균 거래 수다 — 수가 적으면 평균표를 믿지 않는다.")
    return rows


def summary():
    path = os.path.join(OUT, 'curves.npz')
    if not os.path.exists(path):
        print("아직 곡선이 없다. 먼저 --arms 또는 --valid 로 돌려라.")
        return 2
    z = np.load(path)
    curves = {}
    for k in z.files:
        n, ph = k.rsplit('|', 1)
        curves[(n, int(ph))] = z[k]
    arms = sorted({n for n, _ in curves})
    judge(curves, arms, load_cells())
    return 0


if __name__ == '__main__':
    if '--summary' in sys.argv:
        raise SystemExit(summary())
    raise SystemExit(main())
