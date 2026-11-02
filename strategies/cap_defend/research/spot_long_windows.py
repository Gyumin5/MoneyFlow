"""긴 구간 현물 후보를 창 순위합으로 판정한다.

운영자 지적(2026-11-02 텔레그램 9550): "과적합 의심은? 특정 구간의 데이터 쏠림으로
극단적 분석이 된거라면 / 평가는 윈도우 기반으로하고있나?"

안 하고 있었다. 민감도와 교차 격자는 전 구간 하나로 잰 지도다. 한 사건이 전체를
설명해도 그 수에는 안 보인다. 여기서 창 순위합을 건다 — 자산배분 때 쓴 그 방식이다.

어떻게 하나. 후보와 기준선을 같은 앵커·같은 자료로 돌려 일별 자산곡선을 받고,
창마다 네 지표(칼마·샤프·연복리·낙폭)를 재서 둘을 비교한다. 창은 셋이다 —
504봉을 63봉씩, 756봉을 126봉씩, 1008봉을 252봉씩 민다. 짧은 창은 자리를 촘촘히
보고 긴 창은 한 사건에 덜 흔들린다.

판정은 쌍대다 — 후보 하나와 기준선 둘만 놓고 잰다. 여럿을 한 번에 줄 세우면 끼어든
제3 구성이 후보의 점수와 앵커 판정을 바꾼다(토론 run-20261102T022434Z 지적). 쌍대로
재면 순위합이 1.0(모든 창 모든 지표에서 이김) ~ 2.0(다 짐) 사이의 절대 척도가 되어
회차를 넘어 비교할 수 있다. 1.5 가 동률이다.

동률은 양쪽에 같은 평균순위(1.5)를 준다. 전에는 후보에게 유리한 쪽으로 셌다 —
`m[k] >= b[k]` 라 두 곡선이 같은 창(둘 다 현금인 구간)이 전부 후보 우세로 집계됐다.
같은 이유로 창 판정을 승·무·패 셋으로 나눈다.

창 길이별 결과를 따로 내고, 길이별 동일 가중으로도 집계한다. 504봉 창이 1008봉보다
개수가 훨씬 많아 그냥 합치면 짧은 창이 판정을 지배한다.

판정선은 앵커 일곱 중 다섯이다.

쓰는 법:
  python3 spot_long_windows.py --arms n2-c33-d10-m127,n2-c50-d20-m90
  python3 spot_long_windows.py --all        (교차 격자의 완성된 구성 전부)
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

# 채택 호출부(trade/v24_shadow_today.py)와 같은 설정으로 돌린다. 이 줄이 없으면
# 엔진이 'off' 로 떨어져 드리프트 발화 때 종목을 안 갈아끼운다 — 라이브와 다른
# 전략을 재게 된다(토론 run-20261102T022434Z 가 짚은 자리, 2026-11-02 에 고쳤다).
os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import unified_backtest as ub  # noqa: E402

CROSS = os.path.join(CAP, '..', '..', 'state',
                     '20261102-spot-long-cross-refill', 'cells.jsonl')
# 옛 'off' 곡선은 state/20261102-spot-long-windows 에 그대로 두고 여기서는 안 읽는다.
OUT = os.path.join(CAP, '..', '..', 'state', '20261102-spot-long-windows-refill')
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
CAPS = {'c25': 0.25, 'c33': 1 / 3, 'c50': 0.50, 'c100': 1.0}
KEYS = ('cal', 'sharpe', 'cagr', 'mdd')
EPS = 1e-12


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
    ap.add_argument('--all', action='store_true',
                    help='교차 격자에서 앵커 일곱이 다 끝난 구성 전부')
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if a.arms:
        arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    elif a.all:
        agg = {}
        for line in open(CROSS, encoding='utf-8'):
            r = json.loads(line)
            agg.setdefault(r['arm'], []).append(r)
        arms = sorted(k for k, v in agg.items() if len(v) >= len(PHASES))
    else:
        print("후보를 지목해라: --arms <이름들> 또는 --all")
        print("(전 구간 칼마 상위만 골라 보내는 길은 폐지했다 — 그 선별 자체가"
              " 창 판정이 잡으려는 함정이다.)")
        return 2
    if 'BASE' not in arms:
        arms = ['BASE'] + arms
    print(f"후보 {len(arms)}개: {', '.join(arms)}", flush=True)

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)

    # 이미 받아 둔 곡선은 다시 안 돌린다.
    path = os.path.join(OUT, 'curves.npz')
    curves = {}
    if os.path.exists(path):
        z = np.load(path)
        for k in z.files:
            n, ph = k.split('|')
            curves[(n, int(ph))] = z[k]

    for name in arms:
        for ph in PHASES:
            if (name, ph) in curves:
                continue
            t0 = time.time()
            eq = run_arm(bars, fund, name, ph)
            if eq is None:
                print(f"  {name} 위상 {ph}: 자산곡선 없음", flush=True)
                continue
            curves[(name, ph)] = eq
            print(f"  {name} 위상 {ph}: {len(eq)}봉 {time.time()-t0:.1f}초", flush=True)

    np.savez_compressed(path,
                        **{f"{n}|{p}": v for (n, p), v in curves.items()})
    judge(curves, arms)
    return 0


def pair_windows(curves, name, ph):
    """한 앵커에서 후보와 기준선의 창별 지표 짝을 낸다 — (창길이, 후보m, 기준선m)."""
    if (name, ph) not in curves or ('BASE', ph) not in curves:
        return []
    a, b = curves[(name, ph)], curves[('BASE', ph)]
    L = min(len(a), len(b))
    out = []
    for wlen, step in WINDOWS:
        if L < wlen:
            continue
        for s in range(0, L - wlen + 1, step):
            ma = win_metrics(a[s:s + wlen])
            mb = win_metrics(b[s:s + wlen])
            if ma and mb:
                out.append((wlen, ma, mb))
    return out


def score_pair(pairs):
    """쌍대 순위합과 승·무·패. 순위는 1.0/1.5/2.0 (동률은 양쪽 1.5)."""
    rs, w, d, l = [], 0, 0, 0
    for _wlen, ma, mb in pairs:
        better = worse = 0
        for k in KEYS:
            if ma[k] > mb[k] + EPS:
                rs.append(1.0)
                better += 1
            elif ma[k] < mb[k] - EPS:
                rs.append(2.0)
                worse += 1
            else:
                rs.append(1.5)
        if better > worse:
            w += 1
        elif better < worse:
            l += 1
        else:
            d += 1
    n = w + d + l
    return dict(rank=float(np.mean(rs)) if rs else float('nan'),
                win=w, draw=d, loss=l, n=n,
                winpct=(100.0 * w / n) if n else 0.0)


def judge(curves, arms):
    """후보마다 기준선과 둘씩 비교한다. 제3 구성이 점수에 끼어들지 않는다."""
    print(f"\n쌍대 창 판정 (앵커 {len(PHASES)}곳 × 창길이 셋)")
    print("  순위합은 1.0 이 모든 창·모든 지표 우세, 1.5 가 동률, 2.0 이 전패다.")
    print("\n  구성                 순위합  승     무     패    승률   앵커")
    results = []
    for name in arms:
        if name == 'BASE':
            continue
        all_pairs, per_len, anchors = [], {}, 0
        for ph in PHASES:
            p = pair_windows(curves, name, ph)
            if not p:
                continue
            all_pairs += p
            for wlen, ma, mb in p:
                per_len.setdefault(wlen, []).append((wlen, ma, mb))
            if score_pair(p)['rank'] < 1.5 - EPS:
                anchors += 1
        if not all_pairs:
            continue
        s = score_pair(all_pairs)
        # 창 길이별 동일 가중 — 504봉 창이 개수로 판정을 지배하지 않게 한다.
        lens = sorted(per_len)
        bylen = {wl: score_pair(per_len[wl]) for wl in lens}
        s['rank_eq'] = float(np.mean([bylen[wl]['rank'] for wl in lens])) if lens else s['rank']
        s['anchors'] = anchors
        s['bylen'] = bylen
        results.append((name, s))
        mark = '통과' if anchors >= 5 else '미달'
        print(f"  {name:<20} {s['rank']:6.3f} {s['win']:5d} {s['draw']:6d} "
              f"{s['loss']:6d} {s['winpct']:6.1f}% {anchors}/7 {mark}")

    print("\n창 길이별 (순위합 / 승률) · 맨 끝은 길이 동일가중 순위합")
    print("  구성                  504봉          756봉         1008봉     동일가중")
    for name, s in results:
        cells = []
        for wl in (504, 756, 1008):
            b = s['bylen'].get(wl)
            cells.append(f"{b['rank']:5.3f}/{b['winpct']:4.1f}%" if b else "    -     ")
        print(f"  {name:<20} {cells[0]}  {cells[1]}  {cells[2]}  {s['rank_eq']:8.3f}")

    print("\n읽는 법. 승·무·패는 창 하나에서 네 지표 중 엄격 우세 수와 엄격 열세 수를")
    print("비교해 가른다. 무는 양쪽이 같은 창이다 — 둘 다 현금인 구간이 여기 모인다.")
    print("앵커 칸은 그 앵커만의 쌍대 순위합이 1.5 보다 낮은 곳 수이고 판정선은 다섯이다.")
    return results


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
