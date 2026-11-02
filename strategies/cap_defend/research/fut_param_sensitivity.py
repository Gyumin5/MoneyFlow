#!/usr/bin/env python3
"""선물(V25) 파라미터 민감도 — 한 번에 한 축만 흔들어 얼마나 달라지는지 잰다.

운영자 물음(2026-11-02 텔레그램 9635): "선물도 동일한 테스트해봐 선물은 할때 또
다양한 그리드로확인해봐야할까". 답은 그렇다 — 선물에는 현물에 없는 축이 여섯 더
있다. V25 의 레버리지가 고정값이 아니라 코인마다 min(BTC_cap, K2) 로 2~4배를
오가는 규칙이고, 그 규칙 자체가 파라미터 여섯을 갖는다.

그래서 축을 두 묶음으로 나눠 잰다.
  전략 축 열다섯 — 현물과 같은 것들(추세선·모멘텀·변동성·카나리·앵커·종목 수·상한·
    드리프트·헬스·전환 후 재평가·거래비용)에 선물만 갖는 유지증거금률을 더한 것.
  레버리지 축 여섯 — build_K2_signal 의 인자다. K2 쪽 기간·이격, BTC 쪽 기간과
    문턱 둘, 그리고 배수 상한.

레버리지 축을 따로 두는 이유는 그것이 노출 축이기 때문이다. 배수를 올리면 연복리와
낙폭이 같이 커지는데, 그건 잘 골라서 번 것이 아니라 더 크게 걸어서 번 것이다. 한
격자에 전략 축과 섞으면 어느 쪽이 효과를 냈는지 안 갈린다(현물에서 상한과 종목 수가
겹쳐 모양을 잘못 읽은 것과 같은 기제 — history #118·#120).

구간은 2017년까지 이어 붙인 자료의 최대다(long_history.load_extended, 3,549봉).

기준선은 CLAUDE.md "선물 전략 규칙 (V25)" 과 같은 값이다. 축마다 기준선 값이 목록
안에 반드시 들어간다 — 같은 조건에서 비교해야 차이가 축 때문인지 안다.

레버리지는 build_K2_signal 로 만들어 dict 로 넘긴다. 숫자만 넘기면 고정 배수가 되어
V25 가 아니다 — 2026-11-02 에 한 칸을 그렇게 잘못 재서 낙폭 -69.45% 를 얻었다.

쓰는 법:
  python3 fut_param_sensitivity.py --axes all --anchors 7 --out <경로>
  python3 fut_param_sensitivity.py --axes lev          (레버리지 축만)
  python3 fut_param_sensitivity.py --axes strat        (전략 축만)
  python3 fut_param_sensitivity.py --timing            (한 칸 시간만 잰다)

산출물은 JSONL 한 줄에 한 칸이고, 현물 민감도와 같은 모양이라 spot_axis_shape.py 가
그대로 읽는다. SSoT(backtest_futures_v25.py)는 안 건드린다. 호출만 한다.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import backtest_futures_v25 as f25  # noqa: E402

# ─── 채택 선물 V25 기준선 ───
BASE = dict(
    interval='D',
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=5, universe_size=3, cap=1 / 3,
    snap_interval_bars=95, drift_threshold=0.015,
    post_flip_delay=5, health_mode='mom2vol', vol_mode='daily',
    tx_cost=0.0006, maint_rate=0.004,
)

# build_K2_signal 기준선. 이 여섯이 V25 의 동적 배수 규칙 전부다.
LEV_BASE = dict(
    btc_cap_sma_period=42, btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
    k2_sma_period=7, k2_hyst=0.025, l_min=2.0, l_mid=3.0, l_max=4.0,
)

START = '2017-01-01'
END = '2026-11-01'

# ─── 전략 축 ───
STRAT_AXES = {
    'sma_days':          [21, 30, 36, 42, 48, 60, 90],
    'mom_short_days':    [10, 14, 18, 25, 35, 50],
    'mom_long_days':     [90, 110, 127, 150, 180, 240],
    'vol_days':          [30, 60, 90, 120, 180],
    'vol_threshold':     [0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    'canary_hyst':       [0.0, 0.005, 0.015, 0.025, 0.04],
    'drift_threshold':   [0.005, 0.010, 0.015, 0.03, 0.06, 0.10],
    'n_snapshots':       [1, 3, 5, 7, 9, 12],
    'snap_interval_bars': [40, 63, 95, 126, 170, 217],
    'universe_size':     [1, 2, 3, 5, 7, 10],
    'cap':               [0.20, 0.25, 1 / 3, 0.50, 1.0],
    'health_mode':       ['none', 'mom1', 'mom2', 'mom1vol', 'mom2vol'],
    'post_flip_delay':   [0, 2, 5, 10, 20],
    'tx_cost':           [0.0002, 0.0004, 0.0006, 0.0012, 0.0030],
    'maint_rate':        [0.002, 0.003, 0.004, 0.006, 0.010],
}

# ─── 레버리지 축 (build_K2_signal 인자) ───
LEV_AXES = {
    'k2_sma_period':     [3, 5, 7, 10, 14, 21],
    'k2_hyst':           [0.0, 0.0125, 0.025, 0.05, 0.075],
    'btc_cap_sma_period': [21, 30, 42, 60, 90],
    'btc_cap_thr_mid':   [1.0, 1.0075, 1.015, 1.03, 1.05],
    'btc_cap_thr_max':   [1.025, 1.05, 1.075, 1.10, 1.15],
    # 배수 셋은 l_min <= l_mid <= l_max 를 지켜야 한다. 상한을 중간보다 낮게 두면
    # 중간 단계가 상한을 넘어서 축이 뜻을 잃는다 — 그래서 l_max 는 l_mid(3) 부터,
    # l_mid 는 l_min(2) 과 l_max(4) 사이에서만 흔든다.
    'l_max':             [3.0, 4.0, 5.0, 6.0, 8.0],
    'l_mid':             [2.0, 2.5, 3.0, 3.5, 4.0],
}


def metrics(res):
    """run() 이 돌려준 것에서 지표를 꺼낸다. 현물 민감도와 같은 키 이름으로 맞춘다."""
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
    if 'calmar' not in out and out.get('cagr') is not None and out.get('mdd'):
        try:
            d = abs(float(out['mdd']))
            out['calmar'] = float(out['cagr']) / d if d > 0 else None
        except (TypeError, ValueError):
            pass
    return out or dict(res)


class LevCache:
    """같은 인자면 배수 시계열을 다시 안 만든다. 축 하나 안에서 앵커 일곱 번 쓰인다."""

    def __init__(self, bars):
        self.bars = bars
        self.store = {}

    def get(self, **over):
        kw = dict(LEV_BASE)
        kw.update(over)
        key = json.dumps(kw, sort_keys=True)
        if key not in self.store:
            self.store[key] = f25.build_K2_signal(self.bars, **kw)
        return self.store[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axes', default='all',
                    help="쉼표로 나눈 축 이름, 또는 all / strat / lev")
    ap.add_argument('--anchors', type=int, default=7, help="앵커(위상) 개수")
    ap.add_argument('--out', default='')
    ap.add_argument('--timing', action='store_true', help="기준선 한 칸만 돌려 시간을 잰다")
    a = ap.parse_args()

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)
    cache = LevCache(bars)

    # 선물 스태거 19 안쪽을 고르게. 현물(31)과 다른 값이다.
    phases = [round(i * 19 / a.anchors) for i in range(a.anchors)]

    if a.timing:
        t0 = time.time()
        r = f25.run(bars, fund, leverage=cache.get(), start_date=START, end_date=END,
                    phase_offset_bars=0, **BASE)
        print(f"한 칸 {time.time() - t0:.1f}초 · {metrics(r)}")
        return 0

    want = a.axes.strip()
    if want == 'all':
        axes = dict(STRAT_AXES, **LEV_AXES)
    elif want == 'strat':
        axes = dict(STRAT_AXES)
    elif want == 'lev':
        axes = dict(LEV_AXES)
    else:
        names = [s.strip() for s in want.split(',') if s.strip()]
        both = dict(STRAT_AXES, **LEV_AXES)
        axes = {n: both[n] for n in names if n in both}
        missing = [n for n in names if n not in both]
        if missing:
            print(f"모르는 축: {', '.join(missing)}")
            return 1

    # HERE 는 research/ 라 루트는 세 단계 위다. 두 단계로 쓰면 strategies/state/ 에
    # 떨어진다 — 2026-11-02 에 실제로 그렇게 떨어뜨렸다(gitignore 의 state/ 가 깊이
    # 무관이라 커밋 사고는 안 났지만 산출물이 규약 밖에 생긴다).
    out = a.out or os.path.join(HERE, '..', '..', '..', 'state',
                                '20261102-fut-long-sens', 'cells.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = sum(len(v) for v in axes.values()) * len(phases) + len(phases)
    print(f"축 {len(axes)} · 앵커 {len(phases)} · 칸 {total}", flush=True)

    with open(out, 'w', encoding='utf-8') as fh:
        # 기준선을 먼저 적는다 — 앵커 폭이 판정선이라 이 칸 없이는 아무것도 못 읽는다.
        for ph in phases:
            t0 = time.time()
            r = f25.run(bars, fund, leverage=cache.get(), start_date=START,
                        end_date=END, phase_offset_bars=ph, **BASE)
            m = metrics(r)
            fh.write(json.dumps(dict(axis='_baseline', value=None, phase=ph,
                                     metrics=m, sec=round(time.time() - t0, 2)),
                                ensure_ascii=False) + '\n')
            fh.flush()
            print(f"  _baseline 위상 {ph}: 칼마 {m.get('calmar', 0):.3f} "
                  f"연복리 {m.get('cagr', 0):.4f} 낙폭 {m.get('mdd', 0):.4f} "
                  f"청산 {m.get('liq', 0)}", flush=True)

        for ax, values in axes.items():
            is_lev = ax in LEV_AXES
            for v in values:
                for ph in phases:
                    t0 = time.time()
                    if is_lev:
                        lev = cache.get(**{ax: v})
                        kw = dict(BASE)
                    else:
                        lev = cache.get()
                        kw = dict(BASE)
                        kw[ax] = v
                    r = f25.run(bars, fund, leverage=lev, start_date=START,
                                end_date=END, phase_offset_bars=ph, **kw)
                    m = metrics(r)
                    fh.write(json.dumps(dict(axis=ax, value=v, phase=ph, metrics=m,
                                             sec=round(time.time() - t0, 2)),
                                        ensure_ascii=False) + '\n')
                    fh.flush()
                print(f"  {ax}={v}: 칼마 {m.get('calmar', 0):.3f} "
                      f"연복리 {m.get('cagr', 0):.4f} 낙폭 {m.get('mdd', 0):.4f} "
                      f"청산 {m.get('liq', 0)}", flush=True)

    print(f"끝. {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
