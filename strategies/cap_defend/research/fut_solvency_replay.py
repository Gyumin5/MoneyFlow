"""선물 지급능력(청산 여유) 1시간봉 재생 — 읽기전용 측정 (2026-08-25).

계기: 채택 V25 선물 sleeve 의 최대낙폭 -38.3% 는 일봉 종가 기준 자산곡선의 낙폭이고,
청산은 그 사이 어느 순간의 계좌 상태로 결정된다. 엔진은 일봉 저가로 청산을 판정하지만
(Liq=0), 일봉 저가는 하루에 한 점이고 코인별 저가가 같은 시각이라는 가정이 들어간다.
실제로 계좌가 유지증거금에 얼마나 가까이 갔는지를 더 촘촘한 해상도로 잰다.

방법: 채택 설정으로 BT 를 돌리며 일별 포지션 상태(지갑현금·코인별 수량/진입가/증거금)를
덤프하고(엔진 env 토글 FUT_SOLVENCY_DUMP, 측정 후 원복), 그 상태를 다음 날까지의 1시간봉에
대해 재평가한다. CROSS 이므로 계좌 하나로 합산한다.

  equity(t) = 지갑현금 + 코인별 증거금 합 + 코인별 평가손익 합
  maint(t)  = 코인별 (수량 × 가격 × 유지증거금률) 합
  여유비율   = equity / maint          (1.0 이하 = 청산)

보수적 가정: 각 1시간봉에서 보유 코인이 모두 그 봉의 저가를 동시에 찍는다고 본다.
실제보다 나쁘게 잡는 방향이다.

판정 기준 (결과를 보기 전에 고정한다 — 2026-08-25):
  FAIL  어느 봉에서든 여유비율 <= 1.0        → 역사 경로에서 청산. 선물 성과 주장 철회.
  WARN  최소 여유비율 < 2.0                  → 유지증거금의 2배 미만까지 근접. 원인 구간 정밀 조사.
  PASS  전 구간 최소 여유비율 >= 2.0
  스트레스 S1: 보유 전 코인 추가 -20% 즉시 하락에서도 여유비율 > 1.0 이어야 한다.
  스트레스 S2: 실행 3일 누락(그날 포지션을 3일 더 그대로 보유)에서도 여유비율 > 1.0 이어야 한다.
  S1/S2 위반은 즉시 실패가 아니라 조건부 트리거 — 레버리지 상한·마진 안전규칙 검토를 연다.

실행: python3 strategies/cap_defend/research/fut_solvency_replay.py
입력: state/solvency/daily_state.json  (dump_run.py 가 만든다)
출력: state/solvency/replay_result.json + 표준출력 요약
"""
from __future__ import annotations
import json
import os
import sys

import pandas as pd

ROOT = '/home/gmoh/mon/251229'
CAP = os.path.join(ROOT, 'strategies', 'cap_defend')
sys.path.insert(0, CAP)

STATE = os.path.join(ROOT, 'state', 'solvency', 'daily_state.json')
OUT = os.path.join(ROOT, 'state', 'solvency', 'replay_result.json')

FAIL_RATIO = 1.0
WARN_RATIO = 2.0
STRESS_DROP = 0.20      # S1
STRESS_HOLD_DAYS = 3    # S2


def load_hourly():
    from backtest_futures_v25 import TICKER_MAP, DATA_DIR
    out = {}
    for coin, sym in TICKER_MAP.items():
        p = os.path.join(DATA_DIR, f'{sym}_1h.csv')
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['Date'], index_col='Date')
            out[coin] = df[['Low', 'Close']]
    return out


def cushion(pos, wallet, maint_rate, price_of, drop=0.0):
    """여유비율 = equity / maint. price_of(coin) 이 None 이면 그 코인은 건너뛴다."""
    equity = wallet
    maint = 0.0
    used = 0
    for coin, d in pos.items():
        p = price_of(coin)
        if p is None or p <= 0:
            continue
        p = p * (1.0 - drop)
        equity += d['margin'] + d['qty'] * (p - d['entry'])
        maint += d['qty'] * p * maint_rate
        used += 1
    if maint <= 0 or used == 0:
        return None
    return equity / maint



def critical_drop(pos, wallet, maint_rate, price_of):
    """지금 가격에서 보유 전 코인이 동시에 몇 % 더 떨어지면 청산인가.

    equity(d) = B + (1-d)N,  maint(d) = r(1-d)N
      B = 지갑현금 + 증거금합 - 진입원가합,  N = 현재 명목합
    두 값이 같아지는 지점: (1-d) = B / (N(r-1)) → d* = 1 + B/((1-r)N)
    B >= 0 이면 어떤 하락에도 청산되지 않는다(무한대로 본다).
    """
    B = wallet
    N = 0.0
    for coin, dd in pos.items():
        p = price_of(coin)
        if p is None or p <= 0:
            continue
        B += dd['margin'] - dd['qty'] * dd['entry']
        N += dd['qty'] * p
    if N <= 0:
        return None
    if B >= 0:
        return 1.0
    d = 1.0 + B / ((1.0 - maint_rate) * N)
    return max(0.0, min(1.0, d))


def main():
    rows = json.load(open(STATE))
    hourly = load_hourly()
    print(f"[입력] 일별 상태 {len(rows)}일, 1시간봉 코인 {len(hourly)}종", flush=True)

    worst = {'ratio': float('inf'), 'date': None, 'ts': None, 'coins': None}
    worst_s1 = {'ratio': float('inf'), 'date': None}
    worst_s2 = {'ratio': float('inf'), 'date': None}
    worst_drop = {'drop': 1.0, 'date': None, 'ts': None, 'coins': None}
    liq_bars = []
    warn_days = []
    days_with_pos = 0
    series = []

    idx = {c: hourly[c].index for c in hourly}

    for i, r in enumerate(rows):
        pos = r['pos']
        if not pos:
            continue
        days_with_pos += 1
        d0 = pd.Timestamp(r['date'])
        d1 = d0 + pd.Timedelta(days=1)
        d_s2 = d0 + pd.Timedelta(days=STRESS_HOLD_DAYS)

        # 이 날의 1시간봉 구간 (d0, d1]
        day_min = float('inf'); day_ts = None
        s2_min = float('inf')
        for coin in pos:
            if coin not in hourly:
                # 1시간봉이 없는 코인이 하나라도 있으면 그 날은 판정 불가로 남긴다
                pass
        # 공통 시간축 = 보유 코인들의 1시간봉 교집합
        common = None
        for coin in pos:
            if coin not in idx:
                common = None
                break
            sel = idx[coin][(idx[coin] > d0) & (idx[coin] <= d_s2)]
            common = sel if common is None else common.intersection(sel)
        if common is None or len(common) == 0:
            continue

        lows = {c: hourly[c]['Low'].reindex(common) for c in pos}
        for ts in common:
            def price_of(coin, _ts=ts):
                v = lows[coin].get(_ts)
                return None if v is None or pd.isna(v) else float(v)
            cu = cushion(pos, r['wallet'], r['maint_rate'], price_of)
            if cu is None:
                continue
            if ts <= d1:
                if cu < day_min:
                    day_min, day_ts = cu, ts
                if cu <= FAIL_RATIO:
                    liq_bars.append({'date': r['date'], 'ts': str(ts), 'ratio': cu})
            if cu < s2_min:
                s2_min = cu

            # S1: 같은 봉에서 추가 급락
            if ts <= d1:
                cd = critical_drop(pos, r['wallet'], r['maint_rate'], price_of)
                if cd is not None and cd < worst_drop['drop']:
                    worst_drop = {'drop': cd, 'date': r['date'], 'ts': str(ts),
                                  'coins': sorted(pos)}
                cs = cushion(pos, r['wallet'], r['maint_rate'], price_of, drop=STRESS_DROP)
                if cs is not None and cs < worst_s1['ratio']:
                    worst_s1 = {'ratio': cs, 'date': r['date'], 'ts': str(ts)}

        if day_min < float('inf'):
            series.append({'date': r['date'], 'min_ratio': day_min})
            if day_min < worst['ratio']:
                worst = {'ratio': day_min, 'date': r['date'], 'ts': str(day_ts),
                         'coins': sorted(pos)}
            if day_min < WARN_RATIO:
                warn_days.append({'date': r['date'], 'ratio': day_min, 'coins': sorted(pos)})
        if s2_min < worst_s2['ratio']:
            worst_s2 = {'ratio': s2_min, 'date': r['date']}

        if (i + 1) % 300 == 0:
            print(f"  … {i+1}/{len(rows)}일 처리", flush=True)

    verdict = ('FAIL' if worst['ratio'] <= FAIL_RATIO else
               'WARN' if worst['ratio'] < WARN_RATIO else 'PASS')
    res = {
        'verdict': verdict,
        'days_with_position': days_with_pos,
        'evaluated_days': len(series),
        'min_cushion_ratio': worst,
        'liq_bars': liq_bars[:20],
        'liq_bar_count': len(liq_bars),
        'warn_day_count': len(warn_days),
        'warn_days_worst10': sorted(warn_days, key=lambda x: x['ratio'])[:10],
        'min_critical_drop': worst_drop,
        'stress_S1_drop20': worst_s1,
        'stress_S2_hold3d': worst_s2,
        'criteria': {'FAIL<=': FAIL_RATIO, 'WARN<': WARN_RATIO,
                     'S1_drop': STRESS_DROP, 'S2_hold_days': STRESS_HOLD_DAYS},
    }
    json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)

    print(f"\n판정: {verdict}")
    print(f"  보유일 {days_with_pos} / 평가일 {len(series)}")
    print(f"  최소 여유비율 {worst['ratio']:.2f} ({worst['date']} {worst['ts']}, {worst['coins']})")
    print(f"  청산(<=1.0) 봉 수: {len(liq_bars)}")
    print(f"  경고(<2.0) 일수: {len(warn_days)}")
    print(f"  최소 임계하락(추가로 이만큼 동시 하락하면 청산): {worst_drop['drop']*100:.1f}% "
          f"({worst_drop['date']} {worst_drop['ts']}, {worst_drop['coins']})")
    print(f"  S1 전코인 -20% 추가하락 최소 여유: {worst_s1['ratio']:.2f} ({worst_s1.get('date')})")
    print(f"  S2 3일 방치 최소 여유: {worst_s2['ratio']:.2f} ({worst_s2.get('date')})")
    print(f"  결과: {OUT}")


if __name__ == '__main__':
    main()
