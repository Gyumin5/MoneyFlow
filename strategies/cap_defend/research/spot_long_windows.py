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
# 옛 곡선은 두 벌 다 그대로 두고 여기서는 안 읽는다 — state/20261102-spot-long-windows
# ('off' 리필), state/20261102-spot-long-windows-refill (리필은 맞지만 가드가 켜진 채).
# 설정을 입증할 수 없는 캐시는 재사용하지 않는다(토론 run-20261102T062159Z).
OUT = os.path.join(CAP, '..', '..', 'state', '20261102-spot-long-windows-noguard')
# 2차 격자는 다른 디렉토리에 쌓는다. 환경변수로 덮는 이유는 spot_cross_metrics.py 가
# 이 상수를 import 해서 쓰기 때문이다 — 인자로 받으면 그쪽이 못 따라온다.
OUT = os.environ.get('SPOT_WIN_OUT') or OUT
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
    # 라이브는 전략 가드가 없다(CLAUDE.md 현물 V24 절). 2026-11-02 오후부터는
    # 엔진 기본값도 꺼짐이라(커밋 7de5788, history #115) 이 셋은 중복이다 —
    # 그래도 남겨 둔다. 이 파일이 어느 조건으로 쟀는지를 파일 안에 적어 두는
    # 줄이고, 기본값이 다시 바뀌어도 이 판정기는 안 흔들린다.
    # 그 전 이력: 오전에 채택 호출부에서 같은 결함을 잡아 놓고(#107) 이 판정기에
    # 안 넣어서, 그날 오후 세 관문이 통째로 가드 켠 곡선 위에서 돌았다(#114).
    dd_lookback=0, bl_drop=0, crash_threshold=-10,
)

# (창 길이, 이동폭) — 봉 단위. 일봉이라 504봉 ≈ 2년, 1008봉 ≈ 4년.
WINDOWS = [(504, 63), (756, 126), (1008, 252)]
CAPS = {'c25': 0.25, 'c33': 1 / 3, 'c50': 0.50, 'c100': 1.0}
KEYS = ('cal', 'sharpe', 'cagr', 'mdd')
EPS = 1e-12


def parse_arm(name):
    """격자의 이름을 run() 인자로 되돌린다.

    1차 격자는 축 넷이다 — n2-c33-d10-m127.
    2차 격자는 뒤에 둘을 더 붙인다 — n2-c50-d10-m127-s217-v90 (앵커 간격 · 변동성 기간).
    꼬리가 없으면 그 축은 BASE 값이라 1차 이름이 그대로 산다. 두 격자의 같은 구성이
    같은 곡선을 내야 하므로 이름만 보고 갈라지지 않게 한다.
    """
    if name == 'BASE':
        return {}
    parts = name.split('-')
    n, c, d, m = parts[:4]
    out = dict(universe_size=int(n[1:]), cap=CAPS[c],
               drift_threshold=int(d[1:]) / 100.0, mom_long_days=int(m[1:]))
    for tail in parts[4:]:
        if tail.startswith('s'):
            out['snap_interval_bars'] = int(tail[1:])
        elif tail.startswith('v'):
            out['vol_days'] = int(tail[1:])
        else:
            raise ValueError(f"모르는 꼬리: {tail} (이름 {name})")
    return out


# ─── 2차 격자 ───
# 1차 격자(축 넷)가 고른 축은 가드를 켠 채 돌린 민감도에서 나왔다. 가드를 끄고 다시
# 재니 앵커 간격(앵커폭 대비 0.35 → 1.63)과 변동성 기간(0.80 → 1.54)이 잡음선 위로
# 올라왔다(커밋 a157c2a). 그 둘을 더해 다시 짠다.
#
# 값에서 뺀 것 셋은 1차에서 명확히 나빴고 축별 평균이 단조로웠다 — 종목 5개(칼마
# 1.454 대 2개 2.032), 상한 25%(1.568 대 100% 2.017), 장기 모멘텀 240일(1.099 대
# 127일 2.180). 되살릴 이유가 없고, 안 빼면 칸이 두 배를 넘는다.
# 현행 구성(n3-c33-d10-m127-s217-v90)과 1차 통과 구성(n2-c50-d10-m127-s217-v90)은
# 둘 다 이 격자 안에 그대로 있다.
G2_NSEL = [2, 3]
G2_CAPS = ['c33', 'c50', 'c100']
G2_DRIFT = [10, 20, 30]
G2_MOML = [90, 127]
G2_SNAP = [170, 217, 280]
G2_VOL = [60, 90, 120]


def grid2_arms():
    import itertools
    for n, c, d, m, s, v in itertools.product(
            G2_NSEL, G2_CAPS, G2_DRIFT, G2_MOML, G2_SNAP, G2_VOL):
        yield f"n{n}-{c}-d{d:02d}-m{m}-s{s}-v{v}"


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
    ap.add_argument('--grid2', action='store_true',
                    help='2차 격자(축 여섯) 전부. 교차 격자 파일을 안 읽고 이름을 직접 만든다')
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if a.arms:
        arms = [x.strip() for x in a.arms.split(',') if x.strip()]
    elif a.grid2:
        arms = sorted(grid2_arms())
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

    def save():
        """같은 디렉토리에 새로 쓰고 rename 한다 — 저장 도중 죽어도 옛 파일이 남는다.

        임시 이름이 .npz 로 끝나야 한다. np.savez_compressed 는 확장자가 없으면
        제 손으로 붙이므로 'curves.npz.tmp' 를 주면 'curves.npz.tmp.npz' 를 만들고
        rename 이 없는 파일을 찾는다(2026-11-02 에 실제로 그렇게 터뜨렸다).
        """
        tmp = path[:-4] + '.tmp.npz'
        np.savez_compressed(tmp,
                            **{f"{n}|{p}": v for (n, p), v in curves.items()})
        os.replace(tmp, path)

    # 중간 저장. 캐시를 읽는 코드가 위에 있는데 쓰기만 끝에 한 번이라 비대칭이었다 —
    # 2,275칸짜리 격자가 네 시간째에 죽으면 전부 날아간다. 묶음마다 저장하면 재실행이
    # 끝난 칸을 건너뛴다. 묶음을 열로 잡은 이유는 저장 자체가 공짜가 아니어서다
    # (곡선 하나 3,549봉이라 전부 모이면 수십 메가바이트, 압축에 몇 초 걸린다).
    done_arms = 0
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
        done_arms += 1
        if done_arms % 10 == 0:
            save()
            print(f"  — 중간 저장 {len(curves)}곡선", flush=True)

    save()
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
