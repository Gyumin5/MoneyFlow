"""긴 구간 현물 후보를 창 순위합으로 판정한다.

운영자 지적(2026-11-02 텔레그램 9550): "과적합 의심은? 특정 구간의 데이터 쏠림으로
극단적 분석이 된거라면 / 평가는 윈도우 기반으로하고있나?"

안 하고 있었다. 민감도와 교차 격자는 전 구간 하나로 잰 지도다. 한 사건이 전체를
설명해도 그 수에는 안 보인다. 여기서 창 순위합을 건다 — 자산배분 때 쓴 그 방식이다.

어떻게 하나. 후보 구성과 기준선을 같은 앵커·같은 자료로 돌려 일별 자산곡선을 받고,
창마다 네 지표(칼마·샤프·연복리·낙폭)의 순위를 매겨 합산한다. 창은 셋이다 —
504봉을 63봉씩, 756봉을 126봉씩, 1008봉을 252봉씩 민다. 짧은 창은 자리를 촘촘히
보고 긴 창은 한 사건에 덜 흔들린다.

판정은 셋을 같이 본다.
  ⑴ 순위합이 기준선보다 낮은가(낮을수록 좋다).
  ⑵ 창마다 기준선을 이긴 비율이 얼마인가 — 같은 창에서 짝으로 비교한다.
  ⑶ 앵커 일곱 중 몇 곳에서 이기나. 토론 판정선은 최소 다섯이다.
한 사건이 순개선의 절반 이상을 설명하면 채택하지 않는다 — 창별 승률이 그걸 드러낸다.

쓰는 법:
  python3 spot_long_windows.py --arms n2-c33-d10-m127,n2-c50-d20-m90
  python3 spot_long_windows.py --top 8     (교차 격자 칼마 상위에서 자동으로 고른다)
  python3 spot_long_windows.py --summary
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

import unified_backtest as ub  # noqa: E402

CROSS = os.path.join(CAP, '..', '..', 'state', '20261102-spot-long-cross', 'cells.jsonl')
OUT = os.path.join(CAP, '..', '..', 'state', '20261102-spot-long-windows')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1 / 3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
)

# (창 길이, 이동폭) — 봉 단위. 일봉이라 504봉 ≈ 2년, 1008봉 ≈ 4년.
WINDOWS = [(504, 63), (756, 126), (1008, 252)]
CAPS = {'c25': 0.25, 'c33': 1 / 3, 'c50': 0.50}


def parse_arm(name):
    """교차 격자의 이름(n2-c33-d10-m127)을 run() 인자로 되돌린다."""
    if name == 'BASE':
        return {}
    n, c, d, m = name.split('-')
    return dict(universe_size=int(n[1:]), cap=CAPS[c],
                drift_threshold=int(d[1:]) / 100.0, mom_long_days=int(m[1:]))


def win_metrics(eq):
    """자산곡선 조각에서 네 지표를 낸다. 봉 수가 모자라면 None."""
    v = np.asarray(eq, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 60 or v[0] <= 0:
        return None
    yrs = len(v) / 365.0
    cagr = (v[-1] / v[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    peak = np.maximum.accumulate(v)
    mdd = float(np.min(v / peak - 1.0))
    r = np.diff(v) / v[:-1]
    sd = float(np.std(r))
    sharpe = (float(np.mean(r)) / sd * np.sqrt(365.0)) if sd > 0 else 0.0
    cal = cagr / abs(mdd) if mdd < 0 else 0.0
    return dict(cagr=float(cagr), mdd=mdd, sharpe=sharpe, cal=float(cal))


def run_arm(bars, fund, name, phase):
    kw = dict(BASE, **parse_arm(name))
    kw['phase_offset_bars'] = phase
    res = ub.run(bars, fund, start_date=START, end_date=END, **kw)
    eq = res.get('_equity')
    if eq is None:
        return None
    return np.asarray(eq.values if hasattr(eq, 'values') else eq, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='')
    ap.add_argument('--top', type=int, default=8)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if a.arms:
        arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    else:
        agg = {}
        for line in open(CROSS, encoding='utf-8'):
            r = json.loads(line)
            if r.get('Cal') is not None:
                agg.setdefault(r['arm'], []).append(r['Cal'])
        full = {k: float(np.mean(v)) for k, v in agg.items() if len(v) >= len(PHASES)}
        arms = [k for k, _ in sorted(full.items(), key=lambda t: -t[1])[:a.top]]
    if 'BASE' not in arms:
        arms = ['BASE'] + arms
    print(f"후보 {len(arms)}개: {', '.join(arms)}", flush=True)

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    curves = {}
    for name in arms:
        for ph in PHASES:
            t0 = time.time()
            eq = run_arm(bars, fund, name, ph)
            if eq is None:
                print(f"  {name} 위상 {ph}: 자산곡선 없음", flush=True)
                continue
            curves[(name, ph)] = eq
            print(f"  {name} 위상 {ph}: {len(eq)}봉 {time.time()-t0:.1f}초", flush=True)

    np.savez_compressed(os.path.join(OUT, 'curves.npz'),
                        **{f"{n}|{p}": v for (n, p), v in curves.items()})
    judge(curves, arms)
    return 0


def judge(curves, arms):
    """창마다 순위를 매겨 합산하고, 기준선과의 짝 비교 승률도 낸다."""
    rows = []          # (창id, 구성, 지표dict)
    for ph in PHASES:
        have = [n for n in arms if (n, ph) in curves]
        if len(have) < 2:
            continue
        L = min(len(curves[(n, ph)]) for n in have)
        for wlen, step in WINDOWS:
            if L < wlen:
                continue
            for s in range(0, L - wlen + 1, step):
                wid = f"p{ph}-w{wlen}-s{s}"
                for n in have:
                    m = win_metrics(curves[(n, ph)][s:s + wlen])
                    if m:
                        rows.append((wid, n, m))
    if not rows:
        print("창이 없다")
        return

    by_win = {}
    for wid, n, m in rows:
        by_win.setdefault(wid, {})[n] = m

    ranks = {n: [] for n in arms}
    wins = {n: [0, 0] for n in arms}   # [기준선을 이긴 창, 전체 창]
    for wid, d in by_win.items():
        if 'BASE' not in d or len(d) < 2:
            continue
        for key, rev in (('cal', True), ('sharpe', True), ('cagr', True), ('mdd', True)):
            order = sorted(d.items(), key=lambda t: -t[1][key] if rev else t[1][key])
            for i, (n, _m) in enumerate(order, 1):
                ranks[n].append(i)
        b = d['BASE']
        for n, m in d.items():
            if n == 'BASE':
                continue
            wins[n][1] += 1
            better = sum(1 for k in ('cal', 'sharpe', 'cagr', 'mdd') if m[k] >= b[k])
            if better >= 3:
                wins[n][0] += 1

    print(f"\n창 {len(by_win)}개 (앵커 {len(PHASES)}곳 × 창길이 셋)")
    print("  구성                 순위합평균  기준선대비 우세창   네지표 전구간평균")
    out = []
    for n in arms:
        if not ranks[n]:
            continue
        rs = float(np.mean(ranks[n]))
        w = wins[n]
        pct = (100.0 * w[0] / w[1]) if w[1] else 0.0
        out.append((n, rs, pct))
    for n, rs, pct in sorted(out, key=lambda t: t[1]):
        tag = ' (기준선)' if n == 'BASE' else ''
        print(f"  {n:<20} {rs:9.3f} {pct:14.1f}%{tag}")

    # 앵커별로 기준선을 이겼나 — 토론 판정선은 일곱 중 다섯이다.
    print("\n앵커별 우세 (창 순위합 기준, 기준선보다 낮으면 우세)")
    for n in arms:
        if n == 'BASE':
            continue
        cnt = 0
        for ph in PHASES:
            rb, rn = [], []
            for wid, d in by_win.items():
                if not wid.startswith(f"p{ph}-") or 'BASE' not in d or n not in d:
                    continue
                for key in ('cal', 'sharpe', 'cagr', 'mdd'):
                    order = sorted(d.items(), key=lambda t: -t[1][key])
                    pos = {k: i for i, (k, _) in enumerate(order, 1)}
                    rb.append(pos['BASE'])
                    rn.append(pos[n])
            if rb and float(np.mean(rn)) < float(np.mean(rb)):
                cnt += 1
        mark = '통과' if cnt >= 5 else '미달'
        print(f"  {n:<20} {cnt}/7  {mark}")


def summary():
    p = os.path.join(OUT, 'curves.npz')
    if not os.path.exists(p):
        print("자산곡선 파일이 없다 — 먼저 돌려라")
        return
    z = np.load(p)
    curves, arms = {}, []
    for k in z.files:
        n, ph = k.split('|')
        curves[(n, int(ph))] = z[k]
        if n not in arms:
            arms.append(n)
    judge(curves, arms)


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
