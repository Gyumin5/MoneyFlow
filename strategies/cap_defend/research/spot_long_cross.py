"""긴 구간 현물 교차 격자 — 민감도가 움직인 축만 골라 같이 흔든다.

운영자 지시(2026-11-02 텔레그램 9546): "바꿀수있는 모든 변수를 다 바꿔서확인중이야?
기간 오래보기 테스트는? 그것부터 하고나서 선물 숏확인하는게좋지않나".

민감도(spot_param_sensitivity.py --long, state/20261102-spot-long)는 한 번에 한 축씩
재서 다른 축을 전부 지금 값에 묶어 뒀다. 그래서 거기 나온 봉우리는 "다른 축이 현행일
때의 봉우리" 지 진짜 최고가 아니다. 여기서는 수가 실제로 움직인 축만 골라 전부 교차한다.

고른 축과 이유.
  universe_size — 9.8년 칼마 1.65 → 2.83 (앵커 흔들림의 17배). 제일 큰 자리다.
  cap           — 종목 수와 한 짝이다. 종목 2개에 상한 3분의 1이면 최대 67%만 투자하고
                  나머지는 현금이라, 종목 수를 바꾸면 노출까지 같이 바뀐다. 가르려면 같이 흔든다.
  drift_threshold — 0.30 에서 칼마 2.11·연복리 54.7%. 단조라 봉우리는 아니지만 크다.
  mom_long_days   — 90 에서 연복리 +6.6%p, 240 에서 칼마 +0.11. 둘이 반대 방향이라 교차에서 본다.

안 고른 축. 42일선·단기 모멘텀 20일·변동성 상한 5%·앵커 간격 217 은 지금 값이 최고이고
이웃이 전부 크게 나쁘다. 카나리 이격은 0.005 가 칼마 +0.124(1.8배)인데 앵커 폭이
1.50~2.06 으로 넓어 잡음과 못 가른다 — 교차 대상에서 빼고 현행 0.015 로 묶는다.
스냅샷 개수도 뺀다(1개가 칼마 +0.115, 1.7배인데 앵커 폭 1.56~2.09 로 넓다).

3 × 3 × 3 × 3 = 81 구성 × 앵커 7곳 = 567칸.

실행: python3 spot_long_cross.py
      python3 spot_long_cross.py --summary
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

# 채택 호출부(trade/v24_shadow_today.py)와 같은 설정으로 돌린다. 이 줄이 없으면
# 엔진이 'off' 로 떨어져 드리프트 발화 때 종목을 안 갈아끼운다 — 라이브와 다른
# 전략을 재게 된다(2026-11-02, 토론 run-20261102T022434Z 가 짚었다).
# 2026-06-06 검증이 "문턱 0.10 에서 리필 ≡ 앵커전용" 이라고 적은 것은 2020-10 부터의
# 5.4년 구간 얘기다. 9.8년에서는 세 문턱 모두 수가 달라진다 — 0.10 에서도 칼마
# 1.647 → 1.561 (state/20261102-refill-check). 그래서 이 줄이 긴 구간에서는 필수다.
os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import unified_backtest as ub  # noqa: E402

# 옛 'off' 결과는 state/20261102-spot-long-cross 에 그대로 두고 여기서는 안 읽는다.
OUT = os.path.join(CAP, '..', '..', 'state', '20261102-spot-long-cross-refill')
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

NSEL = [2, 3, 5]
# c100 은 1차 설계에서 빠져 있었다. 민감도 마지막 축이 끝나고 나서야 상한 1.0 이
# 9.8년 연복리를 49.9% → 99.7% 로 올리는 제일 큰 자리라는 게 나왔다(칼마 2.50).
# 같은 출력 파일에 이어 돌리면 끝난 칸은 건너뛰므로 새 27구성만 더 돈다.
CAPS = [('c25', 0.25), ('c33', 1 / 3), ('c50', 0.50), ('c100', 1.0)]
DRIFT = [0.10, 0.20, 0.30]
MOML = [90, 127, 240]


def arms():
    for n, (cn, cv), d, m in itertools.product(NSEL, CAPS, DRIFT, MOML):
        name = f"n{n}-{cn}-d{int(d*100):02d}-m{m}"
        yield name, dict(universe_size=n, cap=cv,
                         drift_threshold=d, mom_long_days=m)


def main():
    os.makedirs(OUT, exist_ok=True)
    from long_history import load_extended
    bars, fund = load_extended(verbose=False)
    all_arms = list(arms())
    print(f"자료 준비 끝. 구성 {len(all_arms)}개 × 앵커 {len(PHASES)}곳 "
          f"= {len(all_arms) * len(PHASES)}칸", flush=True)

    path = os.path.join(OUT, 'cells.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['arm'], r['phase']))
            except Exception:
                pass
    print(f"이미 끝난 칸 {len(done)}개", flush=True)

    t_start = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for i, (name, over) in enumerate(all_arms, 1):
            for ph in PHASES:
                if (name, ph) in done:
                    continue
                kw = dict(BASE, **over)
                kw['phase_offset_bars'] = ph
                t0 = time.time()
                m = ub.run(bars, fund, start_date=START, end_date=END, **kw)
                row = dict(arm=name, phase=ph, sec=round(time.time() - t0, 1),
                           **{k: m.get(k) for k in
                              ('CAGR', 'MDD', 'Sharpe', 'Trades', 'Rebal')})
                # run() 은 칼마를 안 돌려준다 — 연복리를 낙폭 절댓값으로 나눠 만든다
                # (spot_param_sensitivity.metrics 와 같은 식).
                d = abs(float(row['MDD'] or 0.0))
                row['Cal'] = (float(row['CAGR']) / d) if d > 0 else None
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                f.flush()
            if i % 10 == 0:
                print(f"  구성 {i}/{len(all_arms)} ({(time.time()-t_start)/60:.1f}분)",
                      flush=True)
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
        if len(g) < len(PHASES):
            continue
        a = {k: float(np.mean([x[k] for x in g]))
             for k in ('Cal', 'CAGR', 'MDD', 'Sharpe', 'Trades')}
        a['Cal_min'] = float(np.min([x['Cal'] for x in g]))
        a['CAGR_min'] = float(np.min([x['CAGR'] for x in g]))
        out.append((name, a))
    print(f"\n완성된 구성 {len(out)}개 (앵커 {len(PHASES)}곳 평균)")

    def show(title, key, n=15):
        print(f"\n{title}")
        print("  구성                 칼마   연복리     낙폭    샤프  앵커최저칼마 앵커최저연복리")
        for name, a in sorted(out, key=lambda t: -t[1][key])[:n]:
            print(f"  {name:<20} {a['Cal']:6.3f} {a['CAGR']:8.4f} {a['MDD']:8.4f} "
                  f"{a['Sharpe']:7.3f} {a['Cal_min']:9.3f} {a['CAGR_min']:11.4f}")

    show("칼마 상위", 'Cal')
    show("연복리 상위", 'CAGR')

    print("\n축별 평균 칼마 / 연복리 / 낙폭")
    pos = {'종목수': 0, '상한': 1, '드리프트': 2, '장기모멘텀': 3}
    for axis, i in pos.items():
        buckets = {}
        for name, a in out:
            buckets.setdefault(name.split('-')[i], []).append(a)
        print(f"  [{axis}]")
        for k, v in sorted(buckets.items()):
            print(f"    {k:<8} 칼마 {np.mean([x['Cal'] for x in v]):6.3f}  "
                  f"연복리 {np.mean([x['CAGR'] for x in v]):8.4f}  "
                  f"낙폭 {np.mean([x['MDD'] for x in v]):8.4f}")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary(os.path.join(OUT, 'cells.jsonl'))
        raise SystemExit(0)
    raise SystemExit(main())
