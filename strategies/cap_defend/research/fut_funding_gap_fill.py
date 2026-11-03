#!/usr/bin/env python3
"""늘린 구간의 펀딩 공백을 보수적인 값으로 채워 선물 성과를 다시 잰다.

운영자 지시(2026-11-03 텔레그램 9773·9774): 상장 여부로 그 구간을 버리지 말고,
"펀딩을 보수적으로 잡아서 안 좋게 했을 때 어떤가를 분석" 하라.

왜 채우나. 늘린 자료는 2017-02-13 부터인데 바이낸스 펀딩 기록은 비트코인
2019-09-10 이 가장 이르다. 그 앞 약 2년 7개월에는 펀딩 시계열이 없고,
backtest_futures_v25.py 885~900행은 시계열이 없거나 윈도우가 비면 그냥 건너뛴다.
레버리지 2~4배 롱은 상승장에서 펀딩을 내는 쪽이라, 비워 두면 그 구간이 비용 하나를
뺀 채로 계산된다. 그래서 공백을 0 이 아니라 불리한 상수로 채운다.

보수값은 실측 분포에서 골랐다(2020-01-01 이후 102코인 602,730건).
  중앙  +0.000100 / 8시간 = 연환산 +8.27%   (바이낸스 기본 펀딩률, 가장 흔한 값)
  75분위 +0.000100                = 연환산 +10.95%
  90분위 +0.000128                = 연환산 +14.06%
  평균  +0.000044                = 연환산  +4.87%
  양수 비율 76.7% — 대부분의 시간에 롱이 낸다.
칸을 셋 둔다. 0(현행·기준선) · 중앙 · 90분위. 민감도를 보려는 것이라 한 값만
쓰지 않는다. 중앙이 평균보다 높은 것은 분포가 0.0001 에 몰려 있고 음수 꼬리가
평균을 끌어내리기 때문이다 — 보수 쪽으로는 중앙이 맞다.

채우는 규칙. 코인마다 그 코인 봉이 시작하는 날부터 그 코인 펀딩이 시작하는 날
직전까지 8시간 간격으로 상수를 넣는다. 펀딩 기록이 아예 없는 코인은 봉 전 구간을
채운다. 실측 구간(2019-09 이후)은 한 줄도 안 건드린다.

SSoT(backtest_futures_v25.py)는 안 건드린다. 호출만 한다.

쓰는 법:
  python3 fut_funding_gap_fill.py --timing          한 칸 시간만 잰다
  python3 fut_funding_gap_fill.py --anchors 7       기본
산출물: state/20261103-funding-gap/cells.jsonl + 표준출력 요약
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import backtest_futures_v25 as f25  # noqa: E402

BASE = dict(
    interval='D',
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=5, universe_size=3, cap=1 / 3,
    snap_interval_bars=95, drift_threshold=0.015,
    post_flip_delay=5, health_mode='mom2vol', vol_mode='daily',
    tx_cost=0.0006, maint_rate=0.004,
)

LEV_BASE = dict(
    btc_cap_sma_period=42, btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
    k2_sma_period=7, k2_hyst=0.025, l_min=2.0, l_mid=3.0, l_max=4.0,
)

START = '2017-01-01'
END = '2026-11-01'

# (이름, 8시간당 비율). 0 은 현행이다 — 비교 기준으로 반드시 둔다.
FILLS = [
    ('공백=0 (현행)', 0.0),
    ('공백=중앙 0.0001 (연 8.27%)', 0.000100),
    ('공백=90분위 0.000128 (연 14.06%)', 0.000128),
]


def fill_funding(bars, funding, rate):
    """펀딩 공백을 상수로 채운 새 dict 를 돌려준다. 원본은 안 건드린다."""
    if rate == 0:
        return funding, {'filled_coins': 0, 'filled_rows': 0}
    out = dict(funding)
    n_coin = n_row = 0
    for coin, bar in bars.items():
        if bar is None or len(bar) == 0:
            continue
        b0 = pd.Timestamp(bar.index[0])
        if getattr(b0, 'tz', None) is not None:
            b0 = b0.tz_localize(None)
        cur = funding.get(coin)
        if cur is not None and len(cur) > 0:
            f0 = pd.Timestamp(cur.index[0])
            if getattr(f0, 'tz', None) is not None:
                f0 = f0.tz_localize(None)
            if f0 <= b0:
                continue  # 공백 없음
            end = f0 - pd.Timedelta(hours=8)
        else:
            end = pd.Timestamp(bar.index[-1])
            if getattr(end, 'tz', None) is not None:
                end = end.tz_localize(None)
        idx = pd.date_range(b0.normalize(), end, freq='8h')
        if len(idx) == 0:
            continue
        head = pd.Series(rate, index=idx, dtype=float)
        if cur is not None and len(cur) > 0:
            c = cur.copy()
            if getattr(c.index, 'tz', None) is not None:
                c.index = c.index.tz_localize(None)
            merged = pd.concat([head, c])
            merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        else:
            merged = head
        out[coin] = merged
        n_coin += 1
        n_row += len(head)
    return out, {'filled_coins': n_coin, 'filled_rows': n_row}


def metrics(res):
    if not isinstance(res, dict):
        return None
    out = {}
    for want, names in (
        ('cagr', ('CAGR', 'cagr', 'annual_return')),
        ('mdd', ('MDD', 'mdd', 'max_drawdown')),
        ('calmar', ('Cal', 'calmar', 'Calmar')),
        ('sharpe', ('Sharpe', 'sharpe')),
    ):
        for n in names:
            if n in res:
                out[want] = res[n]
                break
    if 'calmar' not in out and out.get('cagr') is not None and out.get('mdd'):
        try:
            d = abs(float(out['mdd']))
            out['calmar'] = float(out['cagr']) / d if d > 0 else None
        except (TypeError, ValueError):
            pass
    return out or dict(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', type=int, default=7)
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    from long_history import load_extended
    bars, fund = load_extended(verbose=False)
    lev = f25.build_K2_signal(bars, **LEV_BASE)

    if a.timing:
        t0 = time.time()
        r = f25.run(bars, fund, leverage=lev, start_date=START, end_date=END,
                    phase_offset_bars=0, **BASE)
        print(f"한 칸 {time.time() - t0:.1f}초 · {metrics(r)}")
        return 0

    phases = [round(i * 19 / a.anchors) for i in range(a.anchors)]
    out = a.out or os.path.join(HERE, '..', '..', '..', 'state',
                                '20261103-funding-gap', 'cells.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"칸 {len(FILLS)} × 앵커 {len(phases)} = {len(FILLS) * len(phases)}", flush=True)

    rows = []
    with open(out, 'a', encoding='utf-8') as fh:
        for name, rate in FILLS:
            f2, info = fill_funding(bars, fund, rate)
            print(f"\n[{name}] 채운 코인 {info['filled_coins']} · 줄 {info['filled_rows']:,}",
                  flush=True)
            got = []
            for ph in phases:
                t0 = time.time()
                r = f25.run(bars, f2, leverage=lev, start_date=START, end_date=END,
                            phase_offset_bars=ph, **BASE)
                m = metrics(r) or {}
                m.update(fill=name, rate=rate, phase=ph, secs=round(time.time() - t0, 1))
                fh.write(json.dumps(m, ensure_ascii=False, default=str) + '\n')
                fh.flush()
                got.append(m)
                print(f"  위상 {ph:>2} · 연복리 {m.get('cagr', 0):+.1%} "
                      f"낙폭 {m.get('mdd', 0):+.1%} 칼마 {m.get('calmar', 0):.2f} "
                      f"샤프 {m.get('sharpe', 0):.2f} ({m['secs']}초)", flush=True)
            rows.append((name, got))

    print("\n=== 앵커 평균 ===", flush=True)
    base = None
    for name, got in rows:
        def avg(k):
            v = [g[k] for g in got if g.get(k) is not None]
            return sum(v) / len(v) if v else 0.0
        c, d, cal, sh = avg('cagr'), avg('mdd'), avg('calmar'), avg('sharpe')
        if base is None:
            base = (c, cal)
            delta = ''
        else:
            delta = (f"  (연복리 {(c - base[0]) * 100:+.1f}%p · "
                     f"칼마 {cal - base[1]:+.2f})")
        print(f"{name:<34} 연복리 {c:+.1%} 낙폭 {d:+.1%} "
              f"칼마 {cal:.2f} 샤프 {sh:.2f}{delta}", flush=True)
    print(f"\n산출물 {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
