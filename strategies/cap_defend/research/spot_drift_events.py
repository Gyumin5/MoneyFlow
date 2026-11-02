"""드리프트가 실제로 몇 번, 어떤 날 켜졌나 — 사건을 날짜로 꺼낸다.

운영자 지적(텔레그램 9588): "큰 드리프트는 발동 안 할 거라 의미없다. 실제로 몇 번
켜졌고 어떤 상황이었는지 말해 봐."

리밸런싱 횟수로 역산하면 문턱 0.40 은 9.8년에 앵커당 한두 번이다. 그 정도면 성과 차이가
사건 몇 개에 걸려 있다는 뜻이라 평균표로는 판단할 수 없다. 날짜를 꺼내 하나씩 본다.

## 어떻게 가려내나

엔진은 드리프트 발화를 따로 기록하지 않는다. 대신 같은 앵커·같은 설정으로 두 번 돌린다.
  ⑴ 문턱 T 로 한 번 — 앵커 갱신과 드리프트 발화가 둘 다 든다.
  ⑵ 문턱 0 으로 한 번 — 발화 분기가 통째로 꺼지고(unified_backtest.py:1022 의
     `drift_threshold > 0` 가드) 앵커 갱신만 남는다.
두 실행에서 리밸 플래그가 갈리는 날이 드리프트 발화일이다. 엔진을 안 고치고 가려낸다.

## 사건마다 적는 것

- 날짜, 그날 카나리 상태
- 직전 30봉 비트코인 수익률 — 어떤 장이었나
- 그날 목표의 보유 종목 수와 단일 종목 최대 비중
- 발화 직전과 직후 목표가 어떻게 달라졌나(들어온 종목·나간 종목)
- 발화 뒤 60봉 동안 두 곡선의 로그수익 차이

마지막 항목은 사건 하나만의 몫이 아니다. 한 번 갈라진 포트폴리오는 계속 다르게 가므로
앞 사건의 영향이 섞인다. 그래서 "이 사건이 얼마를 벌었다" 가 아니라 "이 사건 뒤 구간에서
어느 쪽이 앞섰나" 로만 읽는다. 사건 하나를 떼어내 재려면 엔진에 손을 대야 한다.

문턱 넷(0.10 · 0.30 · 0.40 · 0.50) × 앵커 일곱. 비용은 실비용 40bp 하나.
가드는 라이브 정합으로 끈다(dd_lookback=0, bl_drop=0, crash_threshold=-10) — 채택
호출부가 켠 채 돌고 있던 것이 별도로 확인됐고, 라이브 엔진에는 그 가드가 없다.

실행: python3 spot_drift_events.py
      python3 spot_drift_events.py --summary
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

os.environ.setdefault('DRIFT_HEALTH_MODE', 'refill')

import unified_backtest as ub  # noqa: E402

OUT = os.path.join(CAP, '..', '..', 'state', '20261102-drift-events')
START = '2017-01-01'
END = '2026-11-01'
PHASES = [0, 4, 8, 12, 16, 20, 24]
TX = 0.004
HORIZON = 60          # 발화 뒤 몇 봉을 묶어 보나
LOOKBACK = 30         # 직전 장세를 재는 봉 수

BASE = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015,
    n_snapshots=7, universe_size=3, cap=1 / 3, tx_cost=TX,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=217,
    dd_lookback=0, bl_drop=0, crash_threshold=-10,   # 라이브 정합(가드 없음)
)

DRIFTS = [0.10, 0.30, 0.40, 0.50]


def names(tgt):
    return {k for k, v in tgt.items()
            if k not in ('CASH', 'Cash') and v > 1e-9}


def main():
    os.makedirs(OUT, exist_ok=True)
    from long_history import load_extended
    bars, fund = load_extended(verbose=False)
    btc = bars['BTC']['Close'] if 'Close' in bars['BTC'] else bars['BTC'].iloc[:, 0]

    path = os.path.join(OUT, 'events.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['drift'], r['phase']))
            except Exception:
                pass
    print(f"이미 끝난 조합 {len(done)}개 / 전체 {len(DRIFTS)*len(PHASES)}", flush=True)

    t0 = time.time()
    off_cache = {}
    with open(path, 'a', encoding='utf-8') as f:
        for ph in PHASES:
            if ph not in off_cache:
                tr0 = []
                m0 = ub.run(bars, fund, start_date=START, end_date=END,
                            _trace=tr0, **dict(BASE, drift_threshold=0.0,
                                               phase_offset_bars=ph))
                off_cache[ph] = (tr0, np.asarray(m0['_equity'].values, float),
                                 m0['_equity'].index)
            tr0, eq0, idx0 = off_cache[ph]

            for d in DRIFTS:
                if (d, ph) in done:
                    continue
                tr = []
                m = ub.run(bars, fund, start_date=START, end_date=END,
                           _trace=tr, **dict(BASE, drift_threshold=d,
                                             phase_offset_bars=ph))
                eq = np.asarray(m['_equity'].values, float)

                by0 = {t['date']: t for t in tr0}
                evs = []
                for i, t in enumerate(tr):
                    o = by0.get(t['date'])
                    if o is None or not t['rebal'] or o['rebal']:
                        continue
                    prev = tr[i - 1]['target'] if i else {}
                    cur = t['target']
                    w = [v for k, v in cur.items() if k not in ('CASH', 'Cash')]
                    dt = pd.Timestamp(t['date'])
                    j = int(np.searchsorted(idx0, dt))
                    k = min(j + HORIZON, len(eq) - 1, len(eq0) - 1)
                    fwd = (float(np.log(eq[k] / eq[j]) - np.log(eq0[k] / eq0[j]))
                           if 0 <= j < k else None)
                    bi = int(btc.index.searchsorted(dt))
                    pre = (float(btc.iloc[bi] / btc.iloc[bi - LOOKBACK] - 1)
                           if bi >= LOOKBACK and bi < len(btc) else None)
                    evs.append(dict(
                        date=str(t['date'])[:10],
                        held=len(w), wmax=float(max(w)) if w else 0.0,
                        cash=float(cur.get('CASH', cur.get('Cash', 0.0))),
                        added=sorted(names(cur) - names(prev)),
                        dropped=sorted(names(prev) - names(cur)),
                        btc30=pre, fwd60=fwd))
                f.write(json.dumps(dict(drift=d, phase=ph, n=len(evs),
                                        events=evs), ensure_ascii=False) + '\n')
                f.flush()
            print(f"  앵커 {ph} 끝 ({(time.time()-t0)/60:.1f}분)", flush=True)
    summary(path)
    return 0


def summary(path=None):
    path = path or os.path.join(OUT, 'events.jsonl')
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    agg = {}
    for r in rows:
        agg.setdefault(r['drift'], []).append(r)

    print("\n발화 횟수 (9.8년, 앵커 7곳)")
    print(f"  {'문턱':>6}{'평균':>8}{'최저':>7}{'최고':>7}{'연평균':>9}"
          f"{'직후60봉 우세율':>16}{'직후60봉 평균':>15}")
    for d in DRIFTS:
        g = agg.get(d, [])
        if len(g) < len(PHASES):
            print(f"  {d:6.2f}{'  미완':>8}")
            continue
        ns = [r['n'] for r in g]
        fw = [e['fwd60'] for r in g for e in r['events']
              if e['fwd60'] is not None]
        win = (100.0 * sum(1 for x in fw if x > 0) / len(fw)) if fw else 0.0
        print(f"  {d:6.2f}{np.mean(ns):8.1f}{min(ns):7d}{max(ns):7d}"
              f"{np.mean(ns)/9.8:9.2f}{win:15.1f}%{np.mean(fw) if fw else 0:15.4f}")

    print("\n발화일 장세 (직전 30봉 비트코인 수익률 분포)")
    for d in DRIFTS:
        g = agg.get(d, [])
        if len(g) < len(PHASES):
            continue
        pre = [e['btc30'] for r in g for e in r['events']
               if e['btc30'] is not None]
        if not pre:
            continue
        neg = 100.0 * sum(1 for x in pre if x < 0) / len(pre)
        print(f"  {d:.2f}: 중앙값 {np.median(pre):+.3f}, "
              f"하락장 비율 {neg:.1f}%, 최저 {min(pre):+.3f} 최고 {max(pre):+.3f}")

    print("\n앵커 0 의 발화 날짜 (문턱별)")
    for d in DRIFTS:
        g = [r for r in agg.get(d, []) if r['phase'] == 0]
        if not g:
            continue
        evs = g[0]['events']
        print(f"\n  문턱 {d:.2f} — {len(evs)}회")
        for e in evs[:40]:
            ad = ','.join(e['added']) or '-'
            dr = ','.join(e['dropped']) or '-'
            print(f"    {e['date']}  보유{e['held']} 최대{e['wmax']:.3f} "
                  f"현금{e['cash']:.2f}  직전30봉 "
                  f"{e['btc30'] if e['btc30'] is not None else float('nan'):+.3f}  "
                  f"직후60봉 {e['fwd60'] if e['fwd60'] is not None else float('nan'):+.4f}  "
                  f"들어옴[{ad}] 나감[{dr}]")
        if len(evs) > 40:
            print(f"    … 외 {len(evs)-40}회")

    print("\n읽는 법. 직후 60봉 차이는 그 사건 하나의 몫이 아니다 — 한 번 갈라진")
    print("  포트폴리오는 계속 다르게 가므로 앞 사건의 영향이 섞인다. '이 사건이")
    print("  얼마를 벌었다' 가 아니라 '이 사건 뒤 구간에서 어느 쪽이 앞섰나' 로만 읽는다.")


if __name__ == '__main__':
    if '--summary' in sys.argv:
        summary()
        raise SystemExit(0)
    raise SystemExit(main())
