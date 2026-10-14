"""현물 시가 주문의 당일 종가 불변성 검사 (주문 인과성).

ai-debate run-20261014T232843Z 가 설계한 사전등록 측정이다. 결과를 보기 전에 아래
판정 기준을 못박는다.

## 무엇을 의심하나

unified_backtest.py 의 _execute_rebalance 는 주문 수량을 정할 때 _port_val(date) 로
계좌를 평가한다. _port_val 은 그날 종가를 쓰고, 체결가 _get_fill_price 는 그날 시가를
쓴다. 그래서 시가에 내는 주문의 크기가 그날 종가에 의존한다 — 체결 시점에 알 수 없는
값이다. 선물 엔진(backtest_futures_v25.py:505)에는 같은 자리에 pv_date 인자가 있고
호출부가 pv_date=prev_date 를 넘긴다. 현물 쪽에만 안 들어갔다.

## 어떻게 재나

라이브 V24 현물 설정 그대로 돌린다. 리밸런싱이 일어나고 그 전에 보유 종목이 있는 날
하나를 고른다. 그날 보유 코인의 종가만 각각 저가＋당일 가격폭×0.25 와 저가＋가격폭×0.75
로 바꿔 두 번 재생한다. 시가·고가·저가와 다른 날은 건드리지 않는다.

비교 대상은 그날 체결 직후의 보유 수량과 주문 목표 수량이다.

## 판정 (결과 보기 전 고정)

- 실패(시간 누수 확인): 주문 대상이나 방향이 바뀌거나, 수량 차이가
  max(1e-10, 1e-10×기준 수량 절댓값) 를 넘는다.
- 통과: 위 허용오차 안이다.
- 측정불가: 조건을 만족하는 날을 못 찾거나 두 재생이 같은 종가를 받는다.

종가 기반 평가 자체(pv_list, _current_weights)는 비교에서 뺀다. 그건 성과 집계용이지
주문 경로가 아니다.

## 두 번째 축 — 고치면 사라지나

같은 검사를 SPOT_PV_DATE=prev (전일 종가로 평가) 로 한 번 더 돌린다. 이쪽이 통과해야
원인이 평가 시점이라는 것이 선다. 이쪽도 실패하면 다른 누수가 더 있다는 뜻이다.
"""
import os
import sys
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unified_backtest as ub  # noqa: E402

START = '2020-10-01'
END = '2026-08-24'
TOL_REL = 1e-10


def spot_kwargs():
    return dict(
        interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1 / 3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
    )


def run_probe(bars, funding, probe_date, pv_mode):
    ub.ORDER_PROBE.clear()
    os.environ['ORDER_PROBE_DATE'] = probe_date
    os.environ['SPOT_PV_DATE'] = pv_mode
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    try:
        ub.run(bars, funding, **spot_kwargs())
    finally:
        os.environ.pop('ORDER_PROBE_DATE', None)
        os.environ['SPOT_PV_DATE'] = 'same'
    return list(ub.ORDER_PROBE)


def find_candidate_days(bars, funding, limit=400):
    """보유 종목이 있는 리밸런싱 날 목록 (trace 로 후보만 좁힌다)."""
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    trace = []
    ub.run(bars, funding, _trace=trace, **spot_kwargs())
    days = [t['date'] for t in trace if t.get('rebal')]
    return days[:limit]


def perturb(bars, date, coins, frac):
    """date 의 Close 만 저가＋폭×frac 으로. 다른 봉·다른 열은 불변."""
    out = {}
    for c, df in bars.items():
        if c in coins and date in df.index:
            d2 = df.copy()
            lo = float(d2.loc[date, 'Low'])
            hi = float(d2.loc[date, 'High'])
            d2.loc[date, 'Close'] = lo + (hi - lo) * frac
            out[c] = d2
        else:
            out[c] = df
    return out


def compare(a, b):
    """두 probe 결과의 주문 목표·보유 수량 비교. 차이 목록 반환."""
    diffs = []
    if not a or not b:
        return [('probe_missing', len(a), len(b))]
    pa, pb = a[0], b[0]
    for key in ('target_qty', 'holdings_after'):
        ka, kb = pa[key], pb[key]
        for coin in sorted(set(ka) | set(kb)):
            va, vb = ka.get(coin, 0.0), kb.get(coin, 0.0)
            if (coin in ka) != (coin in kb):
                diffs.append((f'{key}:{coin}:membership', va, vb))
                continue
            tol = max(1e-10, TOL_REL * abs(va))
            if abs(va - vb) > tol:
                diffs.append((f'{key}:{coin}', va, vb))
    return diffs


def main():
    bars, funding = ub.load_data('D')
    print('후보 리밸런싱 날 탐색 중...')
    days = find_candidate_days(bars, funding)
    print(f'리밸런싱 날 {len(days)} 개')

    chosen = None
    for d in days:
        probe = run_probe(bars, funding, str(d.date()), 'same')
        if not probe:
            continue
        p = probe[0]
        if p['holdings_before'] and p['target_qty']:
            chosen = (d, p)
            break
    if chosen is None:
        print('측정불가: 보유 종목이 있는 리밸런싱 날을 못 찾음')
        return 2

    d, p = chosen
    coins = sorted(set(p['holdings_before']) | set(p['target_qty']))
    print(f'\n선택한 날: {d.date()}')
    print(f'  직전 보유: {sorted(p["holdings_before"])}')
    print(f'  주문 목표: {sorted(p["target_qty"])}')
    print(f'  사용한 평가액(당일 종가): {p["pv_used"]:.6f}')

    for pv_mode, label in (('same', '현행(당일 종가로 평가)'), ('prev', '수정안(전일 종가로 평가)')):
        print(f'\n=== {label} ===')
        res = {}
        for frac in (0.25, 0.75):
            b2 = perturb(copy.copy(bars), d, coins, frac)
            res[frac] = run_probe(b2, funding, str(d.date()), pv_mode)
            if res[frac]:
                print(f'  종가 {frac:.2f} 자리 → 평가액 {res[frac][0]["pv_used"]:.6f}')
        diffs = compare(res[0.25], res[0.75])
        if not diffs:
            print('  통과 — 종가를 바꿔도 주문 수량이 안 바뀐다')
        else:
            print(f'  실패 — 차이 {len(diffs)} 건 (시간 누수)')
            for name, va, vb in diffs[:8]:
                print(f'    {name}: {va!r} vs {vb!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
