#!/usr/bin/env python3
"""신호 수준 일별 대조 — 라이브 선정함수와 채택 백테스트 엔진이 같은 답을 내는지 매일 본다.

폐기한 월간 replay 자리를 메우는 감시다. 라이브 실행 로그를 읽는 게 아니라,
라이브가 쓰는 선정함수 자체를 읽기 전용으로 다시 돌려 백테스트 엔진과 맞춰 본다.
매매도 상태 저장도 하지 않는다.

지금 덮는 것 = 선물(V25). 현물은 v24_shadow_today.py 가 이미 있고, 주식은 다음 층.

판정 셋:
  0 = 비교한 모든 신호일에서 종목·비중이 같다
  1 = 하나라도 다르다 (사람이 봐야 한다)
  2 = 비교 자체를 못 했다 (데이터가 밀렸거나 불러오기 실패)

2 를 0 처럼 넘기지 않는 게 요점이다. 데이터가 밀린 채 옛 날짜만 비교하면
감시는 매일 통과를 찍으면서 아무것도 안 보게 된다.

쓰기:
  python3 trade/ops/signal_parity_daily.py            # 최근 3 신호일
  python3 trade/ops/signal_parity_daily.py --days 7
  python3 trade/ops/signal_parity_daily.py --allow-stale-days 2
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'strategies', 'cap_defend'))
sys.path.insert(0, os.path.join(_ROOT, 'trade'))

BT_START = '2020-10-01'

# 백테스트 쪽에 넘길 선물 전략값은 여기서 따로 적지 않고 파라미터 대조표에서 가져온다.
# 같은 숫자를 세 번째로 베껴 적으면 그 사본만 낡아서, 라이브가 바뀐 게 아니라 이 감시가
# 낡은 탓에 불일치가 나는 일이 생긴다. 이름이 다른 것만 여기서 옮긴다(bars → days).
from check_params_drift import CANONICAL as _CANON      # noqa: E402

_FUT = _CANON['fut']['params']
BT_FUT_PARAMS = dict(
    sma_days=_FUT['sma_bars'],
    mom_short_days=_FUT['mom_short_bars'],
    mom_long_days=_FUT['mom_long_bars'],
    vol_threshold=_FUT['vol_threshold'],
    canary_hyst=_FUT['canary_hyst'],
    health_mode=_FUT['health_mode'],
    vol_mode=_FUT['vol_mode'],
)

# 고정 유니버스 — 라이브는 CoinGecko 실시간 상위 40 종, 백테스트는 시점별 시가총액을
# 쓴다. 그 차이까지 대조하면 선정 로직이 아니라 입력 차이를 매일 불일치로 신고하게 된다
# (2026-06-06 통제 패리티에서 5.69% 갈림을 확인했고 의도된 갭으로 판정했다).
# 그래서 여기서는 양쪽에 같은 목록을 주입하고 선정 로직만 본다.
REF_UNIVERSE = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC',
                'TRX', 'LINK', 'LTC', 'UNI', 'ATOM', 'BCH', 'XLM', 'VET', 'FIL', 'THETA',
                'EOS', 'AAVE', 'ALGO', 'GRT', 'NEAR', 'FTM', 'SAND', 'MANA', 'AXS', 'GALA',
                'CHZ', 'APE', 'FLOW', 'ICP', 'APT', 'ARB', 'OP', 'SUI']


class ParityDataStale(Exception):
    """비교에 쓸 데이터가 기대한 신호일까지 안 와 있다."""


def _norm(weights: dict) -> dict:
    """현금 키 표기를 합치고 0 비중을 버린다. 비중은 소수점 6자리에서 끊는다."""
    out = {}
    for key, val in weights.items():
        norm_key = 'CASH' if key in ('CASH', 'Cash') else key
        if val and abs(val) > 1e-9:
            out[norm_key] = round(float(val), 6)
    return out


def _check_freshness(bars, allow_stale_days: int):
    """BTC 일봉 마지막 날짜가 기대 신호일에 닿는지 본다. 못 닿으면 비교를 안 한다.

    기대 신호일 = 직전에 닫힌 UTC 일봉 = 어제(UTC). 라이브 크론이 09:05 KST(=00:05 UTC)에
    도는 시점에 마지막으로 닫힌 봉이 그것이다.
    """
    import pandas as pd
    if 'BTC' not in bars or len(bars['BTC']) == 0:
        raise ParityDataStale('BTC 일봉을 못 읽었다')
    allow_stale_days = max(0, int(allow_stale_days))   # 음수 허용치는 의미가 없다
    last_day = datetime.strptime(str(bars['BTC'].index[-1])[:10], '%Y-%m-%d').date()
    expected_day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    lag_days = (expected_day - last_day).days
    last = pd.Timestamp(last_day)
    expected = pd.Timestamp(expected_day)
    if lag_days > allow_stale_days:
        raise ParityDataStale(
            f'선물 일봉이 {lag_days}일 밀렸다 (마지막 {last.date()}, 기대 {expected.date()}). '
            f'데이터를 먼저 갱신해라: python3 strategies/cap_defend/refresh_backtest_data.py --target futures')
    return last, expected, lag_days


def _run_bt(bars, funding, universe, end_date):
    """채택 백테스트 엔진의 매일 신선 선정. 스냅 기계는 끄고 선정 로직만 본다."""
    import unified_backtest as ub
    ub.get_mcap = lambda _date, _u=tuple(universe): list(_u)
    trace = []
    ub.run(bars, funding, interval='D', asset_type='fut', leverage=3.0,
           vol_days=90,
           n_snapshots=1, snap_interval_bars=1,            # 스냅 기계 무력화
           universe_size=3, cap=1 / 3, tx_cost=0.0004, maint_rate=0.004,
           drift_threshold=0.03,
           **BT_FUT_PARAMS,
           dd_lookback=0, bl_drop=0.0, crash_threshold=-10.0,   # 가드 없음 (라이브 정합)
           start_date=BT_START, end_date=end_date, _trace=trace)

    # trace 의 date 는 체결봉이다. 그 직전 봉이 신호일이므로 한 칸 당겨 키를 맞춘다.
    btc = bars['BTC']
    dates = [d for d in btc.index if str(d)[:10] >= BT_START and str(d)[:10] <= end_date]
    pos = {d: i for i, d in enumerate(dates)}
    out = {}
    for entry in trace:
        i = pos.get(entry['date'])
        if i is None or i == 0:
            continue
        out[dates[i - 1]] = _norm(entry['target'])
    return out


def _run_live(bars, universe, signal_dates):
    """라이브 선정함수를 읽기 전용으로 다시 돌린다. 주문도 상태 저장도 없다."""
    import auto_trade_binance as atb
    atb.UNIVERSE = [c + 'USDT' for c in universe]
    atb.UNIVERSE_SIZE = 3
    atb.CAP = 1 / 3
    params = dict(atb.STRATEGIES['D_SMA42'])
    params['n_snapshots'] = 1
    params['snap_interval_bars'] = 1

    full_dates = list(bars['BTC'].index)
    fpos = {d: i for i, d in enumerate(full_dates)}

    out = {}
    for sig_date in signal_dates:
        idx = fpos.get(sig_date)
        if idx is None or idx + 1 >= len(full_dates):
            # 라이브는 마지막 봉을 진행중으로 보므로 신호일 다음 봉이 있어야 한다.
            continue
        next_bar = full_dates[idx + 1]
        bslice = {coin: df[df.index <= next_bar] for coin, df in bars.items()}
        state = {}
        combined = atb.compute_strategy_target('D_SMA42', params, {'D': bslice}, state)
        out[sig_date] = _norm(combined)
    return out


def check_fut(days: int, allow_stale_days: int) -> int:
    import unified_backtest as ub

    print('== 선물(V25) 신호 대조 ==')
    bars, funding = ub.load_data('D')
    last, expected, lag = _check_freshness(bars, allow_stale_days)
    print(f'데이터 마지막 봉 {last.date()} · 기대 신호일 {expected.date()} · 지연 {lag}일')

    universe = [c for c in REF_UNIVERSE if c in bars]
    print(f'고정 유니버스 {len(universe)}종')

    end_date = str(bars['BTC'].index[-1])[:10]
    bt = _run_bt(bars, funding, universe, end_date)

    # 비교 대상 = 양쪽 다 낼 수 있는 가장 최근 신호일 N 개.
    candidates = sorted(d for d in bt if d <= last)
    signal_dates = candidates[-days:]
    if not signal_dates:
        raise ParityDataStale('비교할 신호일이 없다 — 백테스트가 궤적을 못 냈다')

    live = _run_live(bars, universe, signal_dates)

    compared, mismatched = [], []
    for sig_date in signal_dates:
        if sig_date not in live:
            continue
        compared.append(sig_date)
        if bt[sig_date] != live[sig_date]:
            mismatched.append(sig_date)

    if not compared:
        raise ParityDataStale('라이브 선정함수가 비교 가능한 날을 하나도 못 냈다')

    for sig_date in compared:
        if sig_date in mismatched:
            print(f'  {str(sig_date)[:10]}  다름')
            print(f'      백테스트 {bt[sig_date]}')
            print(f'      라이브   {live[sig_date]}')
        else:
            # 같을 때도 값을 찍는다 — 서버 실행 로그의 그날 선정과 눈으로 맞춰볼 수 있어야
            # 이 하니스가 라이브를 실제로 반영하는지 확인된다.
            print(f'  {str(sig_date)[:10]}  같음  {live[sig_date]}')

    print(f'비교 {len(compared)}일 · 불일치 {len(mismatched)}일')
    if mismatched:
        print('판정: 불일치 — 라이브 선정과 채택 백테스트가 갈렸다. 사람이 봐야 한다.')
        return 1
    print('판정: 일치')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='라이브 선정함수와 채택 백테스트의 일별 대조')
    parser.add_argument('--days', type=int, default=3, help='비교할 최근 신호일 수 (기본 3)')
    parser.add_argument('--allow-stale-days', type=int, default=1,
                        help='허용할 데이터 지연 일수 (기본 1)')
    args = parser.parse_args()

    try:
        return check_fut(args.days, args.allow_stale_days)
    except ParityDataStale as exc:
        print(f'판정: 비교 못 함 — {exc}')
        return 2
    except Exception as exc:                      # noqa: BLE001 — 감시는 죽어도 사유를 남긴다
        print(f'판정: 비교 못 함 — 예기치 못한 실패: {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
