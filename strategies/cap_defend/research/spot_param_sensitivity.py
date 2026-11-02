#!/usr/bin/env python3
"""현물 롱(V24) 파라미터 민감도 — 한 번에 한 축만 흔들어 얼마나 달라지는지 잰다.

운영자 요청(2026-11-01 텔레그램 9511): "현물로 롱전략 변수 확인도하자 최대한 긴기간부터해서
변수를 변화를 줘서 얼마나 달라지는지 확인해봐".

무엇을 하나. 채택 현물 V24 를 기준선으로 고정하고, 축 하나씩만 값을 바꿔 가며
칼마·샤프·연복리·최대낙폭 넷을 전부 적는다. 축끼리 섞지 않는다(프로젝트 규칙:
한 번에 한 가설만). 앵커는 phase_offset_bars 로 흔들어 평균과 분산을 같이 본다 —
앵커 하나에서만 좋은 값은 기각 대상이라 단일 앵커 수치는 결론에 쓰지 않는다.

구간은 지금 가진 자료의 최대다. 바이낸스 선물 1시간봉을 일봉으로 리샘플해 쓰므로
BTC 가 2019-09 에 시작해도 유니버스 전체가 서는 것은 2020-10 부터다. 그 앞으로 더
늘리려면 자료 출처 자체를 바꿔야 하고(야후 일봉은 2018-01 부터) 그건 체결·유니버스
정합이 달라지는 별도 작업이다.

쓰는 법:
  python3 spot_param_sensitivity.py --axes all --anchors 7 --out <경로>
  python3 spot_param_sensitivity.py --axes drift,vol_threshold --anchors 3   (좁혀 보기)
  python3 spot_param_sensitivity.py --timing                                  (한 칸 시간만 잰다)

산출물은 JSONL 한 줄에 한 칸이다. 분석은 따로 한다 — 이 파일은 재기만 한다.
SSoT(unified_backtest.py)는 안 건드린다. 호출만 한다.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# 채택 호출부(trade/v24_shadow_today.py)와 같은 설정으로 돌린다. 이 줄이 없으면
# 엔진이 'off' 로 떨어져 드리프트 발화 때 종목을 안 갈아끼운다 — 라이브와 다른
# 전략을 재게 된다(2026-11-02, 토론 run-20261102T022434Z 가 짚었다).
# 2026-06-06 검증이 "문턱 0.10 에서 리필 ≡ 앵커전용" 이라고 적은 것은 2020-10 부터의
# 5.4년 구간 얘기다. 9.8년에서는 세 문턱 모두 수가 달라진다 — 0.10 에서도 칼마
# 1.647 → 1.561 (state/20261102-refill-check).
os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import unified_backtest as ub  # noqa: E402

# ─── 채택 현물 V24 기준선 (trade/v24_shadow_today.py 와 같은 값) ───
BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
    # 라이브는 전략 가드가 없다(CLAUDE.md 현물 V24 절). 2026-11-02 오후부터는
    # 엔진 기본값도 꺼짐이라(커밋 7de5788, history #115) 이 셋은 중복이다 —
    # 그래도 남겨 둔다. 이 측정이 어느 조건으로 쟀는지를 파일 안에 적어 두는 줄이다.
    # 그 전 이력: 2026-11-02 에 같은 결함을 네 자리에서 찾았다 —
    # 채택 호출부(history #107) · 창 판정기와 집중도(#114) · 교차 격자, 그리고 여기.
    # 이 자리가 특히 아픈 것은 여기서 "어느 축이 움직이나" 를 보고 교차할 축 넷을
    # 골랐기 때문이다. 안 고른 축(42일선 · 단기 모멘텀 · 변동성 상한 · 앵커 간격 ·
    # 카나리 이격 · 스냅샷 개수)이 가드 없이는 달리 움직일 수 있다.
    # state/20261102-spot-long 의 칸은 전부 가드 켠 값이라 인용하지 않는다.
    dd_lookback=0, bl_drop=0, crash_threshold=-10,
)

START = '2020-10-01'
END = '2026-11-01'

# ─── 흔들 축과 값 ───
# 기준선 값은 목록 안에 반드시 들어간다 — 같은 조건에서 비교해야 차이가 축 때문인지 안다.
AXES = {
    # 42 양옆이 30·60 으로 멀어 "42만 솟은 바늘인지 36~48 이 다 괜찮은 평평한 자리인지"
    # 를 못 가린다는 지적(운영자 9550)을 받아 사이값을 채웠다. 넓힌 목록은 별도 출력으로
    # 돌린다 — 이미 끝난 칸과 같은 파일에 섞으면 평균이 두 목록의 혼합이 된다.
    'sma_days':          [21, 30, 34, 36, 38, 40, 42, 45, 48, 52, 60, 90, 120],
    'mom_short_days':    [10, 15, 20, 30, 45, 60],
    'mom_long_days':     [90, 110, 127, 150, 180, 240],
    'vol_days':          [30, 60, 90, 120, 180],
    'vol_threshold':     [0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    'canary_hyst':       [0.0, 0.005, 0.015, 0.025, 0.04],
    'drift_threshold':   [0.05, 0.075, 0.10, 0.15, 0.20, 0.30],
    'n_snapshots':       [1, 3, 5, 7, 9, 12],
    'snap_interval_bars':[63, 126, 170, 217, 280, 365],
    'universe_size':     [1, 2, 3, 5, 7, 10],
    'cap':               [0.20, 0.25, 1/3, 0.50, 1.0],
    'health_mode':       ['none', 'mom1', 'mom2', 'mom1vol', 'mom2vol'],
    'post_flip_delay':   [0, 2, 5, 10, 20],
    'tx_cost':           [0.001, 0.002, 0.004, 0.008, 0.020],
}


def metrics(res):
    """run() 이 돌려준 것에서 네 지표를 꺼낸다. 키 이름이 판본마다 달라 넓게 받는다."""
    if not isinstance(res, dict):
        return None
    out = {}
    for want, names in (
        ('cagr',   ('CAGR', 'cagr', 'annual_return')),
        ('mdd',    ('MDD', 'mdd', 'max_drawdown')),
        ('calmar', ('Cal', 'calmar', 'Calmar')),
        ('sharpe', ('Sharpe', 'sharpe')),
    ):
        for n in names:
            if n in res:
                out[want] = res[n]
                break
    for extra in ('Trades', 'Rebal', 'Liq', 'Stops'):
        if extra in res:
            out[extra.lower()] = res[extra]
    # run() 은 칼마를 안 돌려준다 — 연복리를 낙폭 절댓값으로 나눠 직접 만든다.
    if 'calmar' not in out and out.get('cagr') is not None and out.get('mdd'):
        try:
            d = abs(float(out['mdd']))
            out['calmar'] = float(out['cagr']) / d if d > 0 else None
        except (TypeError, ValueError):
            pass
    return out or dict(res)


def one(bars, funding, overrides, phase):
    kw = dict(BASE)
    kw.update(overrides)
    kw['phase_offset_bars'] = phase
    kw['start_date'] = START
    kw['end_date'] = END
    t0 = time.time()
    res = ub.run(bars, funding, **kw)
    return metrics(res), round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axes', default='all', help="쉼표로 나눈 축 이름, 또는 all")
    ap.add_argument('--anchors', type=int, default=7, help="앵커(위상) 개수")
    ap.add_argument('--out', default='')
    ap.add_argument('--timing', action='store_true', help="기준선 한 칸만 돌려 시간을 잰다")
    ap.add_argument('--long', action='store_true',
                    help="2017년까지 이어 붙인 일봉으로 돈다(long_history.load_extended)")
    ap.add_argument('--start', default='', help="시작일을 손으로 지정한다")
    a = ap.parse_args()

    global START
    if a.long and not a.start:
        START = '2017-01-01'
    if a.start:
        START = a.start

    print(f"자료 로드 중 ({START} ~ {END})", flush=True)
    t0 = time.time()
    if a.long:
        from long_history import load_extended
        bars, funding = load_extended(verbose=False)
    else:
        bars, funding = ub.load_data('D')
    print(f"  심볼 {len(bars)}개, {round(time.time()-t0,1)}초", flush=True)

    if a.timing:
        m, sec = one(bars, funding, {}, 0)
        print(f"기준선 한 칸: {sec}초")
        print(f"  {json.dumps(m, ensure_ascii=False, default=str)}")
        return 0

    axes = list(AXES) if a.axes == 'all' else [x.strip() for x in a.axes.split(',')]
    bad = [x for x in axes if x not in AXES]
    if bad:
        print(f"모르는 축: {bad}")
        return 2

    # 앵커 위상은 "스냅 사이의 틈" 을 나눈 자리로 잡는다. 스냅 간격 전체를 나누면 안 된다 —
    # 스냅 7칸의 발화 자리가 int(si × 217/7) = 31의 배수라, 위상을 31의 배수만큼 밀면
    # 그 집합이 자기 자신으로 돌아가 모든 앵커가 글자 그대로 같은 결과를 낸다(2026-11-02 실측:
    # 7곳 전부 칼마 1.6871, 표준편차 0). 틈(간격/스냅수) 안에서 나눠야 발화일이 실제로 달라진다.
    gap = max(BASE['snap_interval_bars'] // max(BASE['n_snapshots'], 1), 1)
    step = max(gap // max(a.anchors, 1), 1)
    phases = [i * step for i in range(a.anchors)]

    out = a.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
        'state', f'{datetime.now():%Y%m%d-%H%M%S}-spot-sensitivity', 'cells.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"산출물: {out}", flush=True)

    total = sum(len(AXES[x]) for x in axes) * len(phases)
    print(f"총 {total}칸 (축 {len(axes)}개 × 값 × 앵커 {len(phases)}곳)\n", flush=True)

    done = 0
    t_start = time.time()
    with open(out, 'w', encoding='utf-8') as f:
        # 기준선을 앵커마다 먼저 적어 둔다 — 모든 비교의 바닥이다.
        for ph in phases:
            m, sec = one(bars, funding, {}, ph)
            f.write(json.dumps({'axis': '_baseline', 'value': None, 'phase': ph,
                                'metrics': m, 'sec': sec}, ensure_ascii=False, default=str) + '\n')
            f.flush()
            done += 1
        print(f"기준선 {len(phases)}칸 끝 ({round(time.time()-t_start,1)}초)", flush=True)

        for ax in axes:
            for val in AXES[ax]:
                for ph in phases:
                    m, sec = one(bars, funding, {ax: val}, ph)
                    f.write(json.dumps({'axis': ax, 'value': val, 'phase': ph,
                                        'metrics': m, 'sec': sec},
                                       ensure_ascii=False, default=str) + '\n')
                    f.flush()
                    done += 1
                print(f"  [{done}/{total + len(phases)}] {ax}={val} "
                      f"({round(time.time()-t_start,1)}초 누적)", flush=True)

    print(f"\n끝. {done}칸, {round((time.time()-t_start)/60,1)}분")
    print(f"산출물: {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
