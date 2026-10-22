#!/usr/bin/env python3
"""신호 수준 일별 대조 — 라이브 선정함수와 채택 백테스트 엔진이 같은 답을 내는지 매일 본다.

폐기한 월간 replay 자리를 메우는 감시다. 라이브 실행 로그를 읽는 게 아니라,
라이브가 쓰는 선정함수 자체를 읽기 전용으로 다시 돌려 백테스트 엔진과 맞춰 본다.
매매도 상태 저장도 하지 않는다.

덮는 것 셋.
  선물(V25) = 라이브 선정함수를 집에서 다시 돌려 최근 신호일 몇 개를 대조한다.
  주식(V25) = 서버가 그날 실제로 낸 신호(signal_state.json)를 읽어 와 대조한다.
              주식은 다시 돌리지 않고 실제로 나온 답을 본다 — 서버에서 도는 코드와
              서버가 쓴 가격으로 계산된 결과라 배포·데이터 어긋남까지 같이 걸린다.
  선물 스냅 상태 = 서버의 binance_state.json 이 스스로 모순되지 않는지 본다.
              앞의 둘은 "오늘 선정이 맞나" 를 보는데, 스냅 슬롯이 사라지거나 저장된
              합산이 스냅 평균과 다르면 선정이 맞아도 체결 목표가 틀린다. 그 축이다.
현물은 아직 이 감시에 없다. v24_shadow_today.py 는 손으로 돌리는 단발 도구이고
어느 crontab 에도 없다 — 그것을 "현물은 이미 본다" 로 읽으면 안 된다.

판정 셋:
  0 = 비교한 모든 신호일에서 종목·비중이 같다
  1 = 하나라도 다르다 (사람이 봐야 한다)
  2 = 비교 자체를 못 했다

2 를 0 처럼 넘기지 않는 게 요점이다. 데이터가 밀린 채 옛 날짜만 비교하면
감시는 매일 통과를 찍으면서 아무것도 안 보게 된다. 2 로 올리는 것 넷 —
데이터가 기대 신호일에 안 닿음, 집 사본과 서버 배포본의 판본 차이,
자기시험 실패(감시가 불일치를 못 잡는 상태), 예기치 못한 예외.

자기시험은 그날 정상 대조를 먼저 내고 기록한 뒤에 돈다. 순서가 반대면 자기시험이
실패한 날에 그날 선정이 맞았는지 자체가 계산되지 않는다. 선물 쪽 자기시험은 실데이터가
아니라 시장과 무관한 합성 일봉을 쓴다 — 실데이터를 흔드는 방식은 그날 뽑힌 종목이
그대로면 아무것도 증명하지 못한다(180 신호일 중 130일, 2026-10-03 실측).

쓰기:
  python3 trade/ops/signal_parity_daily.py                  # 넷 다, 최근 3 신호일
  python3 trade/ops/signal_parity_daily.py --sleeve stock
  python3 trade/ops/signal_parity_daily.py --sleeve snap      # 선물 스냅 상태만
  python3 trade/ops/signal_parity_daily.py --sleeve spotsnap  # 현물 스냅 상태만
  python3 trade/ops/signal_parity_daily.py --days 7
  python3 trade/ops/signal_parity_daily.py --allow-stale-days 2
  python3 trade/ops/signal_parity_daily.py --skip-server-check --skip-self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'strategies', 'cap_defend'))
sys.path.insert(0, os.path.join(_ROOT, 'trade'))
# 주식 채택 백테스트 함수는 research/ 에 있다(bt_stock_coin_v3 · bt_stock_mom3 ·
# bt_stock_single_snap). V25 주식을 채택할 때 쓴 것이 그 셋이므로, 여기서 다시 옮겨
# 적지 않고 그대로 부른다 — 옮겨 적으면 채택 근거와 감시가 서로 다른 코드가 된다.
sys.path.insert(0, os.path.join(_ROOT, 'strategies', 'cap_defend', 'research'))

SERVER = 'ubuntu@<서버주소>'
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

# 주식 3-모멘텀 기간도 같은 표에서 가져온다. 라이브 모듈(stock_strategy_v25)에서 읽으면
# 라이브를 고칠 때 백테스트 쪽도 같이 움직여서 대조가 늘 통과한다 — 2026-06-06 사고가
# 정확히 이 축이었다(라이브 Z랭킹 모멘텀이 순수 252일로 들어가 선정이 16.8% 날 어긋남).
_STK = _CANON['stock']['strategy']
BT_STOCK_MOM = (_STK['MOM_SHORT'], _STK['MOM_MID'], _STK['MOM_LONG'])
BT_STOCK_SHARPE_LB = _STK['SHARPE_LB']

# 방어 국면 비중식은 균등(1/n)으로 본다. 2026-09-25 판정이 "라이브(균등)를 그대로 두고
# 다음 주식 백테스트를 균등 기본으로 돌린다" 였다(history.md). cap 1/3+현금으로 보면
# 방어일 792일 중 310일이 비중만 다르게 나와서, 실제 결정과 어긋난 값으로 매일 경보가 된다.
BT_STOCK_DEF_MODE = 'ew'

# 주식 신선도 허용치. 서버 신호 크론은 매일(주말 포함) 09:15 KST 에 도므로 갱신 지연은
# 0일만 통과시킨다. 가격 기준일은 1세션까지 받는다 — 서버가 09:15 에 받는 야후 수정종가가
# 직전 세션을 못 받는 일이 알려져 있다(2026-09-29 실측: 표기 신호일 09-28, 실제 종가 09-25).
STOCK_UPDATE_LAG_ALLOW = 0
STOCK_BASIS_GAP_ALLOW = 1
STOCK_SIGNAL_HOUR, STOCK_SIGNAL_MIN = 9, 15   # 서버 신호 크론 시각 (KST)

# 고정 유니버스 — 라이브는 CoinGecko 실시간 상위 40 종, 백테스트는 시점별 시가총액을
# 쓴다. 그 차이까지 대조하면 선정 로직이 아니라 입력 차이를 매일 불일치로 신고하게 된다
# (2026-06-06 통제 패리티에서 5.69% 갈림을 확인했고 의도된 갭으로 판정했다).
# 그래서 여기서는 양쪽에 같은 목록을 주입하고 선정 로직만 본다.
REF_UNIVERSE = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC',
                'TRX', 'LINK', 'LTC', 'UNI', 'ATOM', 'BCH', 'XLM', 'VET', 'FIL', 'THETA',
                'EOS', 'AAVE', 'ALGO', 'GRT', 'NEAR', 'FTM', 'SAND', 'MANA', 'AXS', 'GALA',
                'CHZ', 'APE', 'FLOW', 'ICP', 'APT', 'ARB', 'OP', 'SUI']


# 선물 스냅 상태 검사 (2026-09-30). 신호 대조가 못 보는 축 하나를 메운다 — 대조는
# "오늘 선정이 맞나" 를 보지만 스냅 상태 자체가 어긋나면(슬롯이 사라지거나, 합산이
# 스냅 평균과 다르거나, 비중 합이 1 이 아니거나) 선정이 맞아도 체결 목표가 틀린다.
FUT_STATE_REMOTE = '/home/ubuntu/binance_state.json'
FUT_STATE_MEMBER = 'D_SMA42'
SPOT_STATE_REMOTE = '/home/ubuntu/trade_state.json'
SPOT_STATE_MEMBER = 'D_SMA42'
SNAP_TOL = 1e-9

# 현물 스냅 상태의 허용 지연은 0일이다. 현물·선물 크론은 매일 09:05 KST(00:05 UTC)에
# 돌고 그때 마지막으로 닫힌 일봉은 어제 UTC 자정이라, 집 감시가 도는 10:00 KST 에는
# 지연이 0 인 게 정상이다. 1일을 허용하면 그날 크론이 통째로 죽어도 통과한다.
SPOT_SNAP_STALE_ALLOW = 0

# 현금 키 경계. 엔진 안(스냅·멤버 합산)은 대문자 CASH, 체결기 쪽 목표는 저장 시점에
# 어느 표기를 쓰는지가 자산군마다 다르다 — 현물 last_target_snapshot 은 Cash,
# 선물 last_target 은 CASH 다(2026-10-15 서버 상태 실측). 그래서 표기를 한쪽으로
# 통일하라고 요구하지 않고, 딕셔너리마다 "여기는 이 표기" 를 못박아 검사한다.
CASH_ENGINE, CASH_EXEC_SPOT, CASH_EXEC_FUT = 'CASH', 'Cash', 'CASH'


# 선물 자기시험용 합성 일봉. 시장과 무관하게 고정된 값이라 그날 국면에 따라 둔감해지지
# 않는다 — 라이브가 뽑은 종목이 무엇이든, 전액 현금인 날이든 똑같이 돈다.
# 두 벌을 만든다: 비트코인만 일정 복리로 오르는 벌과 내리는 벌. 나머지 후보는 전부
# 아주 느리게 내려서 어느 벌에서도 건강 판정을 통과하지 못한다.
# 날짜도 고정이다. 오늘 날짜를 쓰면 자기시험 결과가 날짜에 따라 달라질 수 있는데,
# 자기시험은 그런 축이 하나도 없어야 매일 같은 것을 증명한다.
SYN_START = '2025-01-01'
SYN_BARS = 300            # 워밍업(장기 모멘텀 127봉)과 백테스트 시작 지연(43봉)을 다 덮는다
SYN_BASE = 100.0
SYN_RATE = 0.004          # 비트코인 하루 복리 변화율 (±0.4%)
SYN_FLAT_RATE = 0.0005    # 나머지 후보 하루 복리 하락률 (-0.05%)
SYN_NAME = {True: '상승', False: '하락'}

# 미리 정한 기대 목표. 합성 입력이 무엇을 내야 하는지를 엔진에 묻지 않고 먼저 적는다 —
# 양쪽이 같은 답을 내는 것만 보면 둘이 같이 틀린 경우를 통과시킨다.
#   상승 = 카나리 ON, 건강한 코인은 비트코인 하나 → 상한 1/3 + 나머지 현금
#   하락 = 카나리 OFF → 전액 현금
SYN_EXPECT = {
    True: {'BTC': round(1 / 3, 6), 'CASH': round(2 / 3, 6)},
    False: {'CASH': 1.0},
}

# 보조 대조(판정에 안 쓴다)에서 실데이터 보유 코인을 떨구는 세기와 길이.
AUX_TAIL = 128
AUX_RATE = 0.03


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


def _run_live(bars, universe, keep_dates):
    """라이브 선정함수를 읽기 전용으로 다시 돌린다. 주문도 상태 저장도 없다.

    상태는 이력 시작부터 날짜 간 이어서 재생한다 — 카나리에 히스테리시스가 있어서
    앞 날짜의 판정을 물려받아야 한다. 비교할 며칠만 잘라 빈 상태로 시작하면
    사잇구간(죽은 띠)에서 라이브와 다른 답이 나오고, 그러면 일치가 나와도 우연이다.
    돌려주는 것은 keep_dates 에 든 날짜뿐이지만 계산은 처음부터 한다.
    """
    import auto_trade_binance as atb
    atb.UNIVERSE = [c + 'USDT' for c in universe]
    atb.UNIVERSE_SIZE = 3
    atb.CAP = 1 / 3
    params = dict(atb.STRATEGIES['D_SMA42'])
    params['n_snapshots'] = 1
    params['snap_interval_bars'] = 1

    full_dates = list(bars['BTC'].index)
    start_i = next((i for i, d in enumerate(full_dates) if str(d)[:10] >= BT_START), None)
    if start_i is None:
        raise ParityDataStale(f'{BT_START} 이후 봉이 없다')

    keep = set(keep_dates)
    state: dict = {}
    out = {}
    # 마지막 봉은 라이브가 진행중으로 보므로 신호일이 될 수 없다 → len-1 까지.
    for idx in range(start_i, len(full_dates) - 1):
        sig_date = full_dates[idx]
        next_bar = full_dates[idx + 1]
        bslice = {coin: df[df.index <= next_bar] for coin, df in bars.items()}
        combined = atb.compute_strategy_target('D_SMA42', params, {'D': bslice}, state)
        if sig_date in keep:
            out[sig_date] = _norm(combined)
    return out


def _compare_fut(bt_bars, bt_funding, universe, required_day, days,
                 live_bars=None, live=None):
    """엔진 둘을 돌려 한 번 대조한다. 정상 실행과 자기시험이 같이 쓰는 길이다.

    같은 길을 쓰는 게 요점이다 — 자기시험이 별도 경로로 판정하면 "자기시험은 통과하는데
    정상 실행은 다른 코드로 돈다" 가 되어 증명이 성립하지 않는다.

    live 를 주면 라이브 재생을 생략한다. 백테스트 쪽 입력만 흔드는 교란에서 라이브 답은
    안 바뀌므로 같은 계산을 두 번 하지 않는다.
    """
    end_date = str(bt_bars['BTC'].index[-1])[:10]
    bt = _run_bt(bt_bars, bt_funding, universe, end_date)
    if live is None:
        # 기대 신호일은 반드시 비교한다. 그 앞으로 N-1 개를 더 본다.
        wanted = sorted(d for d in bt if d <= required_day)[-max(1, days):]
        if required_day not in wanted:
            wanted.append(required_day)
        live = _run_live(live_bars if live_bars is not None else bt_bars, universe, wanted)
    rc, compared, mismatched, why = decide(bt, live, required_day, days)
    return {'rc': rc, 'compared': compared, 'mismatched': mismatched,
            'why': why, 'bt': bt, 'live': live}


def synthetic_bars(universe, rising: bool):
    """시장과 무관한 합성 일봉 한 벌. 실제 입력 형식(일봉 OHLCV)과 실제 유니버스를 쓴다."""
    import numpy as np
    import pandas as pd

    idx = pd.date_range(SYN_START, periods=SYN_BARS, freq='D')
    steps = np.arange(SYN_BARS)
    btc = SYN_BASE * np.power(1.0 + (SYN_RATE if rising else -SYN_RATE), steps)
    flat = SYN_BASE * np.power(1.0 - SYN_FLAT_RATE, steps)
    out = {}
    for coin in universe:
        px = btc if coin == 'BTC' else flat
        out[coin] = pd.DataFrame({'Open': px, 'High': px, 'Low': px, 'Close': px,
                                  'Volume': np.full(SYN_BARS, 1e9)}, index=idx)
    return out


def synthetic_facts(syn, rising: bool):
    """합성 일봉이 의도한 조건을 실제로 만드는지 생성 뒤에 직접 잰다.

    세기를 눈대중으로 박아두면 다음에 파라미터가 바뀌었을 때 조용히 둔감해진다.
    그래서 카나리 문턱·모멘텀 부호·변동성 상한을 매 실행에서 다시 재고,
    하나라도 안 맞으면 통과가 아니라 실패로 돌린다.

    반환 (만족하나, 설명).
    """
    import numpy as np

    sma_p = _FUT['sma_bars']
    hyst = _FUT['canary_hyst']
    mom_s, mom_l = _FUT['mom_short_bars'], _FUT['mom_long_bars']
    vol_th = _FUT['vol_threshold']

    btc = syn['BTC']['Close'].values[:-1]          # 마지막 봉은 진행중 — 신호일까지만
    ratio = float(btc[-1] / np.mean(btc[-sma_p:]))
    m_short = float(btc[-1] / btc[-mom_s - 1] - 1)
    m_long = float(btc[-1] / btc[-mom_l - 1] - 1)
    vol = float(np.std(np.diff(np.log(btc[-90:]))))

    other = next((c for c in syn if c != 'BTC'), None)
    if other is None:
        return False, '합성 유니버스에 비트코인 말고 다른 후보가 없다'
    flat = syn[other]['Close'].values[:-1]
    flat_short = float(flat[-1] / flat[-mom_s - 1] - 1)
    flat_long = float(flat[-1] / flat[-mom_l - 1] - 1)

    detail = (f'{SYN_NAME[rising]} 합성: 비트코인 종가/평균선 {ratio:.4f} '
              f'(문턱 {1 + hyst:.3f}·{1 - hyst:.3f}) · 단기모멘텀 {m_short:+.3f} · '
              f'장기모멘텀 {m_long:+.3f} · 변동성 {vol:.4f} (상한 {vol_th}) · '
              f'나머지 후보 단기 {flat_short:+.3f} 장기 {flat_long:+.3f}')

    if flat_short >= 0 or flat_long >= 0:
        return False, detail + ' — 나머지 후보가 건강 판정을 통과해 버린다'
    if rising:
        if ratio <= 1 + hyst:
            return False, detail + ' — 카나리를 켤 만큼 안 올랐다'
        if m_short <= 0 or m_long <= 0:
            return False, detail + ' — 비트코인 모멘텀이 양수가 아니다'
        if vol > vol_th:
            return False, detail + ' — 비트코인 변동성이 상한을 넘는다'
    else:
        if ratio >= 1 - hyst:
            return False, detail + ' — 카나리를 끌 만큼 안 내렸다'
        if m_short >= 0 or m_long >= 0:
            return False, detail + ' — 비트코인 모멘텀이 음수가 아니다'
    return True, detail


def decide(bt: dict, live: dict, required_day, days: int):
    """비교 결과를 판정한다. 순수 함수 — 시험에서 그대로 부른다.

    반환 (rc, 비교한 날짜들, 다른 날짜들, 사유).
      rc 0 = 기대 신호일을 포함해 비교한 모든 날이 같다
      rc 1 = 하나라도 다르다
      rc 2 = 기대 신호일을 비교하지 못했다 (양쪽 중 한쪽이라도 그 날을 못 냈다)

    기대 신호일을 강제하는 게 요점이다. 마지막 데이터 날짜만 보고 "있는 것 중
    최근 N개" 를 비교하면, 어제 것이 없을 때 그 전 날짜만 조용히 비교하고 통과를 찍는다.
    """
    if required_day is None:
        return 2, [], [], '기대 신호일을 정하지 못했다'
    if required_day not in bt:
        return 2, [], [], f'백테스트가 기대 신호일 {str(required_day)[:10]} 의 목표를 안 냈다'
    if required_day not in live:
        return 2, [], [], f'라이브 선정이 기대 신호일 {str(required_day)[:10]} 의 목표를 안 냈다'

    candidates = sorted(d for d in bt if d <= required_day and d in live)
    compared = candidates[-max(1, days):]
    if required_day not in compared:
        compared.append(required_day)
    mismatched = [d for d in compared if bt[d] != live[d]]
    if mismatched:
        return 1, compared, mismatched, '라이브 선정과 채택 백테스트가 갈렸다'
    return 0, compared, mismatched, ''


FUT_BUILD = [(os.path.join(_ROOT, 'trade', 'auto_trade_binance.py'),
              '/home/ubuntu/auto_trade_binance.py')]

# 주식은 신호를 만드는 쪽(recommend_personal → stock_strategy_v25)과 주문을 내는 쪽
# (executor_stock)이 갈라져 있다. 셋 다 본다 — 대조는 신호 쪽만 보므로, 주문 쪽이 따로
# 밀려 있으면 대조가 통과하는데 실제 매매는 다른 전략으로 도는 상태가 된다.
STOCK_BUILD = [
    (os.path.join(_ROOT, 'strategies', 'cap_defend', 'stock_strategy_v25.py'),
     '/home/ubuntu/stock_strategy_v25.py'),
    (os.path.join(_ROOT, 'strategies', 'cap_defend', 'recommend_personal.py'),
     '/home/ubuntu/recommend_personal.py'),
    (os.path.join(_ROOT, 'trade', 'executor_stock.py'),
     '/home/ubuntu/executor_stock.py'),
]


def _ssh_text(remote_cmd: str, timeout_sec: int):
    """서버에서 한 줄 명령을 돌려 표준출력을 가져온다. 읽기 전용으로만 쓴다.

    반환 (출력, 사유). 출력이 None 이면 사유가 왜 못 읽었는지다.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ['ssh', '-i', os.path.expanduser('~/.ssh/id_rsa'),
             '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
             SERVER, remote_cmd],
            capture_output=True, text=True, timeout=timeout_sec)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return None, f'서버를 못 읽었다: {type(exc).__name__}'
    if proc.returncode != 0:
        return None, f'서버 조회 실패 (rc={proc.returncode}) {proc.stderr.strip()[:120]}'
    return proc.stdout, ''


def server_build_matches(timeout_sec: int = 30, files=None):
    """집에서 대조에 쓰는 코드가 서버에 실제로 깔린 것과 같은 판인지 본다.

    반환 (같은가, 설명). 같지 않거나 확인 자체를 못 하면 비교불능으로 올린다 —
    서버에 다른 판이 깔려 있는데 집 사본으로 대조해 통과를 찍으면 잘못된 안심이 된다.
    """
    import hashlib
    import subprocess

    pairs = list(files or FUT_BUILD)
    local = []
    for local_path, remote_path in pairs:
        try:
            with open(local_path, 'rb') as fh:
                local.append(hashlib.md5(fh.read()).hexdigest())
        except OSError as exc:
            return False, f'집 사본을 못 읽었다: {exc}'

    try:
        proc = subprocess.run(
            ['ssh', '-i', os.path.expanduser('~/.ssh/id_rsa'),
             '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
             SERVER, 'md5sum ' + ' '.join(rp for _, rp in pairs)],
            capture_output=True, text=True, timeout=timeout_sec)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f'서버 판본을 못 읽었다: {type(exc).__name__}'
    if proc.returncode != 0:
        return False, f'서버 판본 조회 실패 (rc={proc.returncode}) {proc.stderr.strip()[:120]}'

    # md5sum 은 준 순서대로 한 줄씩 낸다. 줄 수가 모자라면 무엇이 빠졌는지 모르므로 막는다.
    lines = [ln for ln in proc.stdout.strip().split('\n') if ln.strip()]
    if len(lines) != len(pairs):
        return False, f'서버 판본을 못 읽었다: 응답 {len(lines)}줄 (기대 {len(pairs)}줄)'

    shown = []
    for (local_path, remote_path), local_md5, line in zip(pairs, local, lines):
        parts = line.split()
        server_md5 = parts[0] if parts else ''
        name = os.path.basename(remote_path)
        if server_md5 != local_md5:
            return False, (f'집 사본과 서버 배포본이 다르다 ({name}: '
                           f'집 {local_md5[:12]} · 서버 {server_md5[:12]})')
        shown.append(f'{name} {local_md5[:12]}')
    return True, '집·서버 같은 판 (' + ' · '.join(shown) + ')'


def self_check(universe):
    """자기시험 — 시장과 무관한 합성 일봉을 두 엔진에 넣어 이 감시가 실제로 도는지 본다.

    감시가 고장나는 가장 흔한 방식은 경보가 뜨는 게 아니라 무엇을 비교하든 늘 "같다" 를
    찍게 되는 것이고, 그건 정상 실행만 보고는 구분이 안 된다. 그래서 매일 같이 돌린다.

    실데이터를 흔드는 방식은 안 쓴다. 파라미터든 가격이든, 그날 뽑힌 종목이 그대로면
    목표가 안 바뀌어서 "감시 고장" 과 "그날의 둔감함" 이 구분되지 않는다 — 단기 모멘텀
    기간을 흔드는 방식은 180 신호일 중 130일(72.2%)에서 아무것도 증명하지 못했다.
    합성 입력은 그 축이 없다. 시장 국면도 보유 종목도 날짜도 결과를 바꾸지 않는다.

    넷을 다 요구한다 — 둘은 교란이 없을 때 거짓 경보가 안 나는지(음성 대조), 둘은
    한쪽만 흔들었을 때 반드시 잡히는지(양성 대조)다. 방향도 양쪽 다 본다.
    기대 목표는 엔진에 묻지 않고 미리 정해 둔 값과 맞춘다(SYN_EXPECT) — 양쪽이 같은
    답을 내는 것만 보면 둘이 같이 틀린 경우가 통과한다.

    반환 (통과했나, 설명).
    """
    syn = {}
    facts = {}
    for rising in (True, False):
        syn[rising] = synthetic_bars(universe, rising)
        ok, detail = synthetic_facts(syn[rising], rising)
        facts[rising] = detail
        if not ok:
            return False, f'합성 일봉이 의도한 조건을 못 만든다 — {detail}'

    required = syn[True]['BTC'].index[-2]      # 마지막 봉은 진행중, 그 앞이 신호일
    end_date = str(syn[True]['BTC'].index[-1])[:10]

    bt, live = {}, {}
    for rising in (True, False):
        bt[rising] = _run_bt(syn[rising], {}, universe, end_date)
        live[rising] = _run_live(syn[rising], universe, [required])
        for side, got in (('백테스트', bt[rising]), ('라이브', live[rising])):
            if required not in got:
                return False, (f'{SYN_NAME[rising]} 합성에서 {side} 가 기대 신호일 '
                               f'{str(required)[:10]} 의 목표를 안 냈다')
            if got[required] != SYN_EXPECT[rising]:
                return False, (f'{SYN_NAME[rising]} 합성에서 {side} 목표가 '
                               f'{got[required]} 로 나왔다 — 미리 정한 '
                               f'{SYN_EXPECT[rising]} 과 다르다. {facts[rising]}')

    shown = []
    for live_r, bt_r, want in ((True, True, 0), (False, False, 0),
                               (True, False, 1), (False, True, 1)):
        rc, _compared, _bad, _why = decide(bt[bt_r], live[live_r], required, 1)
        tag = f'라이브 {SYN_NAME[live_r]}·백테스트 {SYN_NAME[bt_r]}'
        if rc != want:
            kind = '교란이 없는데 불일치가 났다' if want == 0 else '이 감시는 불일치를 못 잡는다'
            return False, f'{tag} 에서 판정 {rc} 가 나왔다 (기대 {want}) — {kind}'
        shown.append(f'{tag}→{rc}')
    return True, ('합성 일봉 네 갈래가 전부 기대대로 (' + ' · '.join(shown) + '). '
                  + facts[True] + ' / ' + facts[False])


def aux_check(bars, funding, universe, required_day, days, live_target, live):
    """보조 대조 — 실데이터에서 그날 보유한 코인 하나의 종가를 떨궈 본다. 판정에 안 쓴다.

    합성 입력이 못 보는 것을 본다: 실제 가격 배열, 실제 날짜, 그날 실제로 뽑힌 종목.
    대신 그날 선정이 이 축에 둔감하면 안 잡히는데 그건 감시 고장이 아니다 — 그래서
    결과를 기록만 하고 판정에는 쓰지 않는다. 필수 자기시험은 위의 합성 쪽이다.

    정상 대조가 일치일 때만 부른다. 이미 불일치인 날에 교란을 얹어 나온 불일치를
    "교란이 잡혔다" 로 읽으면 안 되기 때문이다.
    """
    import numpy as np

    held = [c for c in sorted(live_target) if c != 'CASH']
    if not held:
        return '해당 없음 — 그날 목표가 전액 현금이라 떨굴 코인이 없다'
    victim = held[0]
    df = bars[victim].copy()                   # 흔드는 것 하나만 사본을 뜬다
    tail = min(AUX_TAIL, len(df))
    ramp = np.concatenate([np.ones(len(df) - tail),
                           np.power(1.0 - AUX_RATE, np.arange(1, tail + 1))])
    for col in ('Open', 'High', 'Low', 'Close'):
        if col in df.columns:
            df[col] = df[col].values.astype(float) * ramp
    fake = dict(bars)                          # 겉 사전만 얕은 사본 — 원본은 안 건드린다
    fake[victim] = df
    res = _compare_fut(fake, funding, universe, required_day, days, live=live)
    if res['rc'] == 1:
        return (f'백테스트 쪽 {victim} 최근 {tail}봉을 하루 {AUX_RATE:.0%} 복리로 '
                '떨구자 불일치로 잡혔다')
    return (f'백테스트 쪽 {victim} 최근 {tail}봉을 하루 {AUX_RATE:.0%} 복리로 떨궜는데 '
            f'판정 {res["rc"]} 가 나왔다 — 그날 선정이 이 축에 둔감하다 (판정에 안 쓴다)')


def _norm_risky(weights: dict) -> dict:
    """현금 칸을 빼고 위험자산 비중만 남긴다. 소수점 6자리에서 끊는다.

    현금을 빼는 이유는 양쪽 정의가 같아서다 — 라이브는 executor 가 버퍼를 따로 곱하고
    백테스트는 버퍼 0 으로 돌린다. 남는 현금 칸은 고른 종목 수에서 결정되므로,
    위험자산 비중이 같으면 현금도 같다.
    """
    out = {}
    for key, val in weights.items():
        if key in ('CASH', 'Cash'):
            continue
        if val and abs(val) > 1e-9:
            out[key] = round(float(val), 6)
    return out


STOCK_TICKERS = ['SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ', 'IEF', 'BIL', 'BNDX']

# 검사가 "무엇을 보고 무엇을 뺐나" 를 적어 두는 자리. 영수증에 그대로 실린다 —
# rc 0 하나만 남기면 비교 범위가 조용히 줄어든 날과 멀쩡한 날이 구분되지 않는다.
SCOPE_NOTES: dict = {}

# 서버 주식 가격 파일의 최소 줄 수. 전략이 보는 가장 긴 창은 카나리 SMA200 과 12개월
# 모멘텀(252일)이라 그 둘을 채우고도 남아야 한다. 잘린 파일로 계산된 신호는 조용히
# 다른 답을 내므로 비교하지 않고 멈춘다. 현재 서버 파일은 2018-01-02 부터 2,211줄이다.
STOCK_MIN_CSV_ROWS = 400


def server_stock_signal(timeout_sec: int = 30):
    """서버가 그날 실제로 낸 주식 신호와 그 신호를 만든 가격 기준일을 읽어 온다. 읽기 전용.

    라이브를 집에서 다시 돌리지 않고 실제로 나온 답을 본다. 서버에 깔린 코드와 서버가
    받은 가격으로 계산된 결과라, 선정 로직뿐 아니라 배포·데이터 어긋남까지 같이 걸린다.

    가격 기준일을 따로 읽는 이유가 핵심이다. signal_state.json 의 signal_date 는
    비트코인 시계열의 마지막 날짜라 주식이 실제로 쓴 종가의 날짜가 아니다 — 주식 쪽은
    야후 수정종가가 장 마감 몇 시간 뒤에 채워져서 09:15 KST 다운로드가 직전 세션을
    못 받는 일이 있다(2026-09-29 실측: 신호일 표기는 09-28, 실제 계산은 09-25 종가).
    표기된 날짜로 대조하면 매일 불일치가 나는데 원인은 선정 로직이 아니다. 그래서
    서버가 실제로 쓴 CSV 의 마지막 날짜를 기준일로 잡고, 그 날짜로 양쪽을 맞춘다.
    """
    import json

    probe = '; '.join(
        f'printf "%s %s %s\\n" {t} "$(tail -1 /home/ubuntu/data/{t}.csv)" '
        f'"$(wc -l < /home/ubuntu/data/{t}.csv)"'
        for t in STOCK_TICKERS)
    out, why = _ssh_text(f'cat /home/ubuntu/signal_state.json; echo "@@@"; {probe}', timeout_sec)
    if out is None:
        raise ParityDataStale(f'서버 주식 신호를 못 읽었다: {why}')
    if '@@@' not in out:
        raise ParityDataStale('서버 응답이 잘렸다 (가격 기준일 부분이 없다)')
    head, tail = out.split('@@@', 1)

    try:
        data = json.loads(head)
    except ValueError as exc:
        raise ParityDataStale(f'서버 주식 신호가 JSON 이 아니다: {exc}') from exc
    stock = data.get('stock') or {}
    meta = data.get('meta') or {}
    # 서버가 그날 주식 전략 계산에 실패하면 목표를 새로 쓰지 않고 직전 값을 보존한다
    # (recommend_personal.build_stock_signal_block). 그런 회차에는 비교할 오늘 값이
    # 아예 없으므로, 판본이 낡았다거나 신호가 깨졌다고 적지 않고 사유를 그대로 옮긴다.
    if stock.get('compute_failed'):
        raise ParityDataStale(
            f"서버 주식 전략 계산이 실패한 회차다 (사유 {stock.get('fail_reason')!r}, "
            f"{stock.get('failed_at')}). 목표는 직전 값이라 비교하지 않는다.")
    for key in ('risk_on', 'offense_weights', 'defense_weights'):
        if key not in stock:
            raise ParityDataStale(f'서버 주식 신호에 {key} 가 없다')
    # 카나리는 참·거짓이어야 한다. 문자열 "False" 나 0/1 이 들어오면 bool() 로 감싸는
    # 순간 뜻이 바뀌어(빈 문자열이 아닌 모든 문자열은 참) 양쪽이 다른 국면을 보면서도
    # 같다고 찍힌다. 값을 고쳐 쓰지 말고 그 자리에서 멈춘다.
    if not isinstance(stock['risk_on'], bool):
        raise ParityDataStale(
            f"서버 주식 카나리가 참·거짓이 아니다 ({stock['risk_on']!r})")
    if not meta.get('signal_date') or not meta.get('updated_at'):
        raise ParityDataStale('서버 주식 신호에 신호일·갱신시각이 없다')

    # 비중은 정규화하기 전에 본다. _norm_risky 는 0 과 현금 칸을 지우고 반올림하므로
    # 음수·NaN·터무니없는 값이 섞여 있어도 지나간 뒤에는 흔적이 남지 않는다.
    for name, key in (('공격', 'offense_weights'), ('방어', 'defense_weights')):
        raw = stock[key]
        if not isinstance(raw, dict) or not raw:
            raise ParityDataStale(f'서버 주식 {name} 비중이 비어 있거나 딕셔너리가 아니다')
        total = 0.0
        for tkr, val in raw.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)) \
                    or not math.isfinite(val):
                raise ParityDataStale(f'서버 주식 {name} 비중 {tkr} 이 수가 아니다 ({val!r})')
            if val < -1e-9 or val > 1 + 1e-9:
                raise ParityDataStale(f'서버 주식 {name} 비중 {tkr} 이 0~1 밖이다 ({val})')
            total += float(val)
        if total > 1 + 1e-6:
            raise ParityDataStale(f'서버 주식 {name} 비중 합이 {total:.6f} 로 1 을 넘는다')

    # 라이브는 그날의 국면 한 쪽만 새로 계산하고 반대편은 직전 파일의 값을 그대로
    # 복사한다(recommend_personal.py). 어느 쪽이 새 값인지는 파일에 적혀 있다.
    # 이 둘이 없거나 어긋나면 어느 쪽이 오늘 값인지 알 수 없으므로 그 자리에서 멈춘다 —
    # 모르는 채로 건너뛰면 이 감시가 조용히 아무것도 안 보는 상태가 된다.
    computed = stock.get('computed_side')
    stale = stock.get('stale_side')
    if computed not in ('offense', 'defense') or stale not in ('offense', 'defense'):
        raise ParityDataStale(
            f'서버 주식 신호에 계산된 쪽·낡은 쪽 표시가 없다 (computed_side={computed!r} '
            f'stale_side={stale!r}). 리포트 판본이 낡았다.')
    if computed == stale:
        raise ParityDataStale(f'계산된 쪽과 낡은 쪽이 같다고 적혀 있다 ({computed})')
    want = 'offense' if stock['risk_on'] else 'defense'
    if computed != want:
        raise ParityDataStale(
            f'카나리({stock["risk_on"]})와 계산된 쪽({computed})이 안 맞는다 — '
            f'{want} 가 새로 계산된 쪽이어야 한다')

    # 기준일은 신호 파일에 적힌 값이 정본이다. 조회 시점(10:00)의 CSV 꼬리로 대신하면
    # 09:15 신호가 난 뒤에 가격이 한 세션 더 받아진 날 엉뚱한 날짜로 대조해 거짓
    # 불일치가 난다. CSV 는 그 값을 교차확인하고 서버 가격 자체가 성한지 보는 데만 쓴다.
    if stock.get('basis_mixed'):
        raise ParityDataStale(
            f"서버 주식 신호가 섞인 날짜로 계산됐다 (가장 오래된 종가 "
            f"{stock.get('basis_oldest')}). 라이브도 같은 조건을 경고로 적는다.")
    if not stock.get('basis_date'):
        raise ParityDataStale(
            '서버 주식 신호에 가격 기준일(basis_date)이 없다. 리포트 판본이 낡았다.')
    basis_date = str(stock['basis_date'])[:10]

    csv_meta = {}
    for line in tail.strip().split('\n'):
        parts = line.split()
        if len(parts) == 3 and ',' in parts[1]:
            day, close = parts[1].split(',', 1)
            csv_meta[parts[0]] = (day[:10], close, parts[2])
    missing = [t for t in STOCK_TICKERS if t not in csv_meta]
    if missing:
        raise ParityDataStale(f'서버 가격 파일을 못 읽었다: {", ".join(missing)}')
    sick = []
    for tkr in STOCK_TICKERS:
        day, close, rows = csv_meta[tkr]
        try:
            px, nrows = float(close), int(rows)
        except ValueError:
            sick.append(f'{tkr} 마지막 줄이 숫자가 아니다 ({close!r} {rows!r})')
            continue
        if not math.isfinite(px) or px <= 0:
            sick.append(f'{tkr} 마지막 종가가 {px}')
        if nrows < STOCK_MIN_CSV_ROWS:
            sick.append(f'{tkr} 가격이 {nrows}줄뿐이다 (최소 {STOCK_MIN_CSV_ROWS})')
    if sick:
        raise ParityDataStale('서버 주식 가격이 성하지 않다: ' + ' · '.join(sick))

    csv_days = sorted({v[0] for v in csv_meta.values()})
    if basis_date > csv_days[-1]:
        raise ParityDataStale(
            f'신호가 적은 기준일 {basis_date} 이 서버 가격 마지막 날 {csv_days[-1]} 보다 '
            '뒤다 — 신호와 가격이 안 맞는다')

    return {
        'risk_on': stock['risk_on'],
        'off_w': _norm_risky(stock['offense_weights']),
        'def_w': _norm_risky(stock['defense_weights']),
        'signal_date': str(meta['signal_date'])[:10],
        'updated_at': str(meta['updated_at']),
        'basis_date': basis_date,
        'csv_last': csv_days[-1],
        'csv_mixed': len(csv_days) != 1,
        'computed_side': computed,
        'stale_side': stale,
    }


def _stock_prices():
    """주식 가격 시계열. 채택 백테스트와 같은 로더를 쓴다."""
    import pandas as pd

    from bt_stock_coin_v3 import DEF_TICKERS, OFF_R7
    from stock_engine import load_prices

    tickers = sorted(set(OFF_R7) | set(DEF_TICKERS))
    pdf = pd.DataFrame(load_prices(tickers, start='2005-01-01'))
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    idx = pd.DatetimeIndex(pdf.index)
    pdf = pdf[idx.normalize() == idx]      # 시각이 붙은 행은 버린다 (일봉만 쓴다)
    return pdf.dropna(how='all')


def _stock_bt(pdf, sig_ts, mom=None):
    """채택 백테스트 쪽의 그 날 선정. 스냅·드리프트·체결은 안 본다 (선정만 대조한다)."""
    from bt_stock_coin_v3 import precompute
    from bt_stock_mom3 import fresh_pick_3mom
    from bt_stock_single_snap import picks_to_target, select_off

    ms, mid, ml = mom or BT_STOCK_MOM
    # 백테스트 쪽은 샤프와 방어 모멘텀 기간을 126 으로 박아 두고 있다. 정본 표가 그와
    # 달라지면 이 감시가 조용히 옛 정의로 대조하게 되므로 그 자리에서 멈춘다.
    if BT_STOCK_SHARPE_LB != 126:
        raise ParityDataStale(
            f'정본 표의 주식 샤프 기간이 {BT_STOCK_SHARPE_LB} 인데 채택 백테스트는 126 으로 박혀 있다')

    ranked, mom_off, mom_def, canary = precompute(pdf, [ms, mid, ml], [126])
    on = bool(canary.at[sig_ts])
    pos = canary.index.get_loc(sig_ts)
    off_picks = fresh_pick_3mom(ranked.at[sig_ts],
                                mom_off[ms].loc[sig_ts],
                                mom_off[mid].loc[sig_ts],
                                mom_off[ml].loc[sig_ts])
    return {
        'risk_on': on,
        # 직전 세션의 카나리. 라이브 모듈은 히스테리시스 죽은 구간에서 이전 상태를
        # 그대로 쓰므로, 집 대조에서 같은 조건을 만들려면 이 값이 필요하다.
        'canary_prev': bool(canary.iloc[pos - 1]) if pos > 0 else on,
        'off_w': _norm_risky(picks_to_target(off_picks, 0.0, 'cap')),
        'def_w': _norm_risky(select_off(sig_ts, mom_def, 0.0, BT_STOCK_DEF_MODE)),
    }


def decide_stock(live: dict, bt: dict):
    """주식 대조 판정. 순수 함수 — 시험에서 그대로 부른다.

    반환 (rc, 어긋난 항목들, 안 본 항목들). rc 0 = 본 것이 다 같다, 1 = 하나라도 다르다.

    보는 것은 카나리와 그날 라이브가 실제로 계산한 국면 목표 둘이다. 반대편 국면은
    라이브가 직전 파일에서 복사해 온 값이라 오늘 가격으로 계산한 백테스트와 비교할
    대상이 아니다 — 비교하면 국면이 바뀐 다음 날부터 매일 거짓 불일치가 난다
    (2026-10-18 실측: 카나리가 꺼진 날 공격 목표가 전날 기준일 값이라 어긋났고,
    그날 밤 executor 는 공격·방어를 둘 다 새로 계산해 매매는 안 갈렸다).

    복사된 쪽을 안 봐도 매매는 덮인다. executor_stock 은 신호 파일을 쓰지 않고 매
    실행마다 compute_offense·compute_defense 를 둘 다 그 자리에서 계산하고, 국면이
    바뀌면 그 다음 09:15 에 그 쪽이 새로 계산돼 여기서 비교 대상이 된다.
    """
    fresh = live['computed_side']
    bad, skipped = [], []
    if live['risk_on'] != bt['risk_on']:
        bad.append(f"카나리: 라이브 {live['risk_on']} · 백테스트 {bt['risk_on']}")
    for side, name, key in (('offense', '공격', 'off_w'), ('defense', '방어', 'def_w')):
        if side != fresh:
            skipped.append(f'{name} 목표: 라이브가 낡은 쪽으로 표시했다(직전 신호의 복사본)')
            continue
        if live[key] != bt[key]:
            bad.append(f"{name} 목표: 라이브 {live[key]} · 백테스트 {bt[key]}")
    return (1 if bad else 0), bad, skipped


def local_stock_check(pdf, sig_ts, bt: dict):
    """집 라이브 전략 모듈과 채택 백테스트를 그 날짜로 양쪽 국면 다 대조한다.

    서버 대조는 그날 라이브가 실제로 계산한 한 쪽만 본다 — 반대편은 직전 신호의
    복사본이라 오늘 가격으로 잰 백테스트와 비교할 대상이 아니다. 그러면 지금 안 쓰이는
    쪽 코드가 갈라져도 국면이 바뀌기 전까지 아무도 모른다. 이 검사가 그 구멍을 덮는다:
    서버에 깔린 것과 같은 모듈(stock_strategy_v25)을 집 가격으로 두 쪽 다 돌려 본다.
    서버 배포본이 집 사본과 같은 판본인지는 server_build_matches 가 따로 본다.

    카나리는 직전 세션의 백테스트 값을 이전 상태로 넣어 라이브와 같은 조건을 만든다
    (히스테리시스 죽은 구간에서는 이전 상태가 답을 가른다).

    반환 (rc, 어긋난 항목들).
    """
    import stock_strategy_v25 as sv

    cut = {t: pdf[t].loc[:sig_ts].dropna() for t in pdf.columns}
    _, off_w, _ = sv.compute_offense(cut)
    _, def_w, _ = sv.compute_defense(cut)
    prev_on = bool(bt['canary_prev'])
    on, _meta = sv.eem_canary(cut.get('EEM'), prev_on)
    bad = local_stock_diffs(on, off_w, def_w, bt)
    return (1 if bad else 0), bad


def local_stock_diffs(on: bool, off_w: dict, def_w: dict, bt: dict):
    """집 라이브 모듈의 답과 백테스트의 답을 줄로 만든다. 순수 함수 — 시험에서 그대로 부른다."""
    bad = []
    if bool(on) != bt['risk_on']:
        bad.append(f"카나리: 집 라이브 {bool(on)} · 백테스트 {bt['risk_on']} "
                   f"(이전 상태 {bt['canary_prev']})")
    for name, got, want in (('공격', _norm_risky(off_w), bt['off_w']),
                            ('방어', _norm_risky(def_w), bt['def_w'])):
        if got != want:
            bad.append(f'{name} 목표: 집 라이브 {got} · 백테스트 {want}')
    return bad


def stock_victims(pool, held, available):
    """양성 대조에서 부풀릴 종목 순서. 순수 함수 — 시험에서 그대로 부른다.

    EEM 은 뺀다. 카나리 종목이라 흔들면 국면 자체가 뒤집혀서, 불일치가 나도 그건 목표가
    달라져서가 아니다. 국채·단기채(IEF·BIL·BNDX)를 앞에 세우는 것은 GLD·PDBC 가
    공격·방어 유니버스에 둘 다 들어 있어 흔들면 두 국면이 같이 움직이기 때문이다.
    """
    first = ('IEF', 'BIL', 'BNDX')
    cand = [t for t in pool if t not in held and t in available and t != 'EEM']
    return [t for t in first if t in cand] + [t for t in cand if t not in first]


def self_check_stock(pdf, live: dict, sig_ts, bt: dict):
    """주식 양성 대조 — 백테스트 쪽 입력을 일부러 틀리게 해서 불일치가 잡히는지 본다.

    흔드는 자리는 그날 실제로 비교하는 국면이어야 한다. 비교하지 않는 쪽을 흔들면
    판정이 안 바뀌는 게 당연한데 그게 "감시가 불일치를 못 잡는다" 로 읽힌다.

    고르는 방식이 중요하다. 라이브가 안 고른 종목 하나의 최근 1년 가격을 하루 2%씩
    복리로 밀어 올린다 — 공격 국면이면 그 종목이 Z랭킹 1위가 되고 3-모멘텀(30·72·230일)
    필터도 전부 통과하고, 방어 국면이면 6개월 수익률 1위가 되어 상위 셋에 반드시 든다.
    그래서 어떤 시장 국면에서도 그날의 목표가 달라진다. 기간 파라미터를 흔드는
    방식은 못 쓴다: 그날 상위 종목이 그대로면 안 잡혀도 감시 고장이 아니라 그날의
    둔감함이 되어 판정이 흐려진다(선물 쪽에서 2026-09-29 실측). 12년에 걸친 직선 기울기도
    못 쓴다 — 하루치 기울기가 0.02% 라 30일 모멘텀 부호를 못 뒤집는다(같은 날 실측).

    흔든 뒤에 "불일치가 났다" 만 보지 않는다. 그 불일치가 목표가 달라져서 난 것인지를
    같이 본다 — 카나리가 같이 뒤집히면 목표 비교가 죽어 있어도 불일치는 나므로, 그걸
    통과로 읽으면 양성 대조가 자기 자신을 속인다. 그래서 EEM(카나리 종목)은 흔들지 않고,
    흔든 뒤 카나리가 그대로이고 그 종목이 실제로 그날 목표에 들어왔을 때만 센다.

    반환 (통과했나, 설명).
    """
    import numpy as np

    from bt_stock_coin_v3 import DEF_TICKERS, OFF_R7

    if live['computed_side'] == 'offense':
        pool, held, what, key = OFF_R7, set(live['off_w']), '공격', 'off_w'
    else:
        pool, held, what, key = DEF_TICKERS, set(live['def_w']), '방어', 'def_w'
    cand = stock_victims(pool, held, pdf.columns)
    if not cand:
        return False, f'부풀릴 {what} 종목이 없다 — 양성 대조를 만들 수 없다'

    tried = []
    for victim in cand:
        ramp = np.ones(len(pdf))
        tail = min(260, len(pdf))
        ramp[-tail:] = 1.02 ** np.arange(1, tail + 1)
        fake = pdf.copy()
        fake[victim] = fake[victim] * ramp
        fbt = _stock_bt(fake, sig_ts)
        if fbt['risk_on'] != bt['risk_on']:
            tried.append(f'{victim}(카나리가 같이 뒤집혀 못 씀)')
            continue
        if victim not in fbt[key]:
            tried.append(f'{victim}(부풀려도 {what} 목표에 안 들어옴)')
            continue
        rc, bad, _skip = decide_stock(live, fbt)
        if rc == 1 and any(line.startswith(what) for line in bad):
            return True, (f'백테스트 쪽 {victim} 가격을 부풀리자 {what} 목표가 '
                          '달라져 불일치로 잡혔다')
        tried.append(victim)
    return False, (f'백테스트 쪽 {", ".join(tried)} 가격을 차례로 부풀렸는데 '
                   f'{what} 목표 불일치가 안 났다 — 이 감시는 불일치를 못 잡는다')


def expected_stock_signal_day(now: datetime):
    """지금 시점에서 서버가 이미 냈어야 하는 주식 신호의 날짜.

    달력 날짜로 재면 자정부터 09:15 사이에 돌릴 때마다 "어제 신호" 를 1일 지연으로
    읽어 비교를 거절한다. 정규 감시는 10:00 이라 안 걸리지만, 손으로 그 창에서 돌리면
    멀쩡한 신호를 두고 비교불능이 나고 그 판정이 그대로 영수증에 적혀 서버로 올라간다
    (2026-10-15 00:59 실제 발생). 크론이 아직 안 돈 시간대에는 어제 것이 최신이다.

    얼어붙은 신호를 잡는 능력은 그대로다 — 09:15 이 지나면 기대일이 오늘로 올라가므로
    그날 크론이 죽으면 지연 1일로 걸린다.
    """
    cutoff = now.replace(hour=STOCK_SIGNAL_HOUR, minute=STOCK_SIGNAL_MIN,
                         second=0, microsecond=0)
    return now.date() if now >= cutoff else (now.date() - timedelta(days=1))


def check_stock(check_server: bool = True, self_test: bool = True) -> int:
    """주식 신호 대조. 신선도 허용치는 선물 쪽 --days 와 묶지 않는다.

    선물의 허용치는 "몇 개 신호일을 비교하나" 에서 나오는데 주식은 서버가 낸 답 하나를
    보는 것이라 축이 다르다. 서버 신호 크론은 매일(주말 포함) 09:15 KST 에 돌므로
    갱신 지연은 0일만 통과시킨다 — 얼어붙은 신호와 대조해 매일 통과를 찍는 것이
    이 감시의 가장 흔한 고장 방식이다. 가격 기준일 차이는 1세션까지 받는다(서버의
    야후 수정종가 지연이 알려진 값이다).
    """
    import pandas as pd

    update_allow = STOCK_UPDATE_LAG_ALLOW
    allow_gap = STOCK_BASIS_GAP_ALLOW

    print('== 주식(V25) 신호 대조 ==')

    if check_server:
        same, why = server_build_matches(files=STOCK_BUILD)
        print(f'판본 확인: {why}')
        if not same:
            raise ParityDataStale(why)

    live = server_stock_signal()
    # 서버의 신호 갱신이 가장 최근 크론 시각까지 왔는지 본다. 09:15 KST 크론이 죽으면
    # 신호가 그 자리에 얼어붙는데, 그걸 모르고 옛 신호와 대조하면 매일 통과를 찍는다.
    expected_day = expected_stock_signal_day(datetime.now())
    updated_day = datetime.strptime(live['updated_at'][:10], '%Y-%m-%d').date()
    update_lag = (expected_day - updated_day).days
    print(f"서버 신호 갱신 {live['updated_at']} · 표기 신호일 {live['signal_date']} "
          f"· 기대 갱신일 {expected_day} · 갱신 지연 {update_lag}일 (허용 {update_allow})")
    if update_lag > update_allow:
        raise ParityDataStale(
            f"서버 주식 신호가 {update_lag}일 전({live['updated_at']}) 것이다. "
            '09:15 KST 신호 크론이 돌았는지 봐라.')

    pdf = _stock_prices()
    sig_ts = pd.Timestamp(live['basis_date'])
    if sig_ts not in pdf.index:
        raise ParityDataStale(
            f"집 주식 가격에 서버 기준일 {live['basis_date']} 이 없다 (마지막 "
            f'{str(pdf.index[-1])[:10]}). 데이터를 먼저 갱신해라: '
            'python3 strategies/cap_defend/refresh_backtest_data.py --target stock')

    gap = len(pdf.index) - 1 - pdf.index.get_loc(sig_ts)
    print(f"서버 가격 기준일 {live['basis_date']} · 집 가격 마지막 "
          f'{str(pdf.index[-1])[:10]} · {gap}세션 차이 (허용 {allow_gap})')
    if gap > allow_gap:
        raise ParityDataStale(
            f"서버 주식 가격이 {gap}세션 밀렸다 (기준일 {live['basis_date']}, "
            f'집 마지막 {str(pdf.index[-1])[:10]}). 서버 쪽 다운로드를 봐라.')

    if live['csv_mixed'] or live['basis_date'] != live['csv_last']:
        print(f"  참고: 서버 가격 파일의 마지막 날은 {live['csv_last']} 다 "
              f"(신호가 쓴 기준일 {live['basis_date']}). 대조는 신호가 적은 날짜로 한다.")

    bt = _stock_bt(pdf, sig_ts)
    rc, bad, skipped = decide_stock(live, bt)
    print(f"계산된 쪽 {live['computed_side']} · 낡은 쪽 {live['stale_side']} "
          '(라이브는 그날의 국면 한 쪽만 새로 계산한다)')
    print(f"  카나리   라이브 {live['risk_on']} · 백테스트 {bt['risk_on']}")
    print(f"  공격     라이브 {live['off_w']}")
    print(f"           백테스트 {bt['off_w']}")
    print(f"  방어     라이브 {live['def_w']}")
    print(f"           백테스트 {bt['def_w']}")
    for line in skipped:
        print(f'  비교 안 함 — {line}')

    # 서버 대조에서 뺀 쪽을 여기서 덮는다. 집 라이브 모듈로 두 국면을 다 돌려 백테스트와
    # 맞춘다 — 지금 안 쓰이는 쪽 코드가 갈라지면 국면이 바뀌기 전에 여기서 걸린다.
    lrc, lbad = local_stock_check(pdf, sig_ts, bt)
    if lrc == 0:
        print('  집 라이브 모듈 대조: 공격·방어·카나리 셋 다 백테스트와 같다')
    else:
        for line in lbad:
            print(f'  집 라이브 모듈이 다름 — {line}')

    # 양성 대조는 기준선이 깨끗할 때만 뜻이 있다. 이미 불일치가 난 날에 흔들어 봐야
    # 무엇 때문에 잡혔는지 못 가른다.
    if self_test and rc == 0 and lrc == 0:
        passed, why = self_check_stock(pdf, live, sig_ts, bt)
        print(f'양성 대조: {why}')
        if not passed:
            raise ParityDataStale(why)
    elif self_test:
        print('양성 대조: 기준선이 이미 불일치라 건너뛴다')

    scope = (f"기준일 {live['basis_date']} · 비교한 쪽 {live['computed_side']}"
             + (f" · 뺀 쪽 {live['stale_side']}(직전 신호의 복사본)" if skipped else '')
             + ' · 집 라이브 모듈은 양쪽 다 대조')
    SCOPE_NOTES['주식'] = scope
    print(f'  대조 범위: {scope}')

    if rc == 1 or lrc == 1:
        for line in bad + lbad:
            print(f'  다름 — {line}')
        print('판정: 불일치 — 라이브 신호와 채택 백테스트가 갈렸다. 사람이 봐야 한다.')
        return 1
    print('판정: 일치')
    return 0


def check_fut(days: int, allow_stale_days: int, check_server: bool = True,
              self_test: bool = True) -> int:
    import unified_backtest as ub

    print('== 선물(V25) 신호 대조 ==')

    if check_server:
        same, why = server_build_matches()
        print(f'판본 확인: {why}')
        if not same:
            raise ParityDataStale(why)

    bars, funding = ub.load_data('D')
    last, expected, lag = _check_freshness(bars, allow_stale_days)
    print(f'데이터 마지막 봉 {last.date()} · 기대 신호일 {expected.date()} · 지연 {lag}일')

    universe = [c for c in REF_UNIVERSE if c in bars]
    print(f'고정 유니버스 {len(universe)}종')

    # 오늘의 정상 대조를 먼저 내고 기록한다. 자기시험을 앞에 두면 그게 실패한 날에는
    # 그날 선정이 맞았는지 자체가 계산되지 않아, 사람이 볼 수 있었던 사실이 사라진다.
    res = _compare_fut(bars, funding, universe, expected, days)
    bt, live = res['bt'], res['live']
    for sig_date in res['compared']:
        if sig_date in res['mismatched']:
            print(f'  {str(sig_date)[:10]}  다름')
            print(f'      백테스트 {bt[sig_date]}')
            print(f'      라이브   {live[sig_date]}')
        else:
            # 같을 때도 값을 찍는다 — 서버 실행 로그의 그날 선정과 눈으로 맞춰볼 수 있어야
            # 이 하니스가 라이브를 실제로 반영하는지 확인된다.
            print(f'  {str(sig_date)[:10]}  같음  {live[sig_date]}')
    if res['rc'] == 2:
        print(f"정상 대조: 비교 못 함 — {res['why']}")
    else:
        print(f"정상 대조: {'불일치' if res['rc'] == 1 else '일치'} · 비교 "
              f"{len(res['compared'])}일 · 불일치 {len(res['mismatched'])}일 · "
              f'기대 신호일 {str(expected)[:10]} 포함')

    self_ok, self_why = True, '건너뜀 (--skip-self-test)'
    if self_test:
        self_ok, self_why = self_check(universe)
    print(f'자기시험: {self_why}')

    if not self_test:
        aux_why = '건너뜀 (--skip-self-test)'
    elif res['rc'] != 0:
        aux_why = '해당 없음 — 정상 대조가 일치일 때만 본다'
    else:
        aux_why = aux_check(bars, funding, universe, expected, days,
                            live[expected], live)
    print(f'보조 대조: {aux_why}')

    # 최종 판정. 자기시험 실패는 정상 대조의 결과를 지우지 않는다 — 위에 이미 찍었다.
    if res['rc'] == 2 or not self_ok:
        reasons = []
        if res['rc'] == 2:
            reasons.append(res['why'])
        if res['rc'] == 1:
            reasons.append('그날 정상 대조는 불일치였다 (위 출력을 봐라)')
        if not self_ok:
            reasons.append(f'자기시험 실패: {self_why}')
        raise ParityDataStale(' · '.join(reasons))
    if res['rc'] == 1:
        print(f"판정: 불일치 — {res['why']}. 사람이 봐야 한다.")
        return 1
    print('판정: 일치')
    return 0


def expected_last_closed_daily(now_utc: datetime) -> datetime:
    """지금 시점에 마지막으로 닫힌 일봉의 여는 시각(UTC 자정 기준).

    바이낸스 일봉은 여는 시각으로 이름이 붙고 24시간 뒤에 닫힌다. 그래서 지금 닫혀
    있는 마지막 봉은 "오늘 UTC 자정" 이 아니라 그 하루 앞이다.
    """
    midnight = now_utc.astimezone(timezone.utc).replace(hour=0, minute=0, second=0,
                                                        microsecond=0)
    return midnight - timedelta(days=1)


def _cash_folded(d: dict, label: str, diffs: list, expect: str = CASH_ENGINE) -> dict:
    """현금 키를 대문자 CASH 하나로 접는다. 확인된 메타 키 `_ts` 만 뺀다.

    같은 값이 자리마다 다른 표기로 저장되므로(엔진 안 CASH, 현물 체결 목표 Cash)
    표기가 다르다는 이유만으로 불일치를 신고하면 매일 거짓 경보가 난다. 그래서
    비교는 접어서 하되, 그 자리에 허용된 표기인지는 따로 본다. 보는 것 둘.
      · 한 딕셔너리에 CASH 와 Cash 가 같이 있다 — 2026-08-20 헛주문 사고의 모양이다
      · 그 자리에 안 맞는 표기 하나만 있다 — 위를 안 거치고 조용히 흘러드는 길이다
    현금 비중이 아예 없는 딕셔너리에 현금 키를 요구하지는 않는다.

    밑줄 접두사만으로 모르는 키를 버리지 않는다. `_ts` 외의 키는 비중으로 세고,
    숫자가 아니거나 NaN·무한대면 그 자리에서 신고한다 — 버리면 합 검사가 눈감는다.
    """
    seen = [k for k in d if k.upper() == 'CASH']
    if len(seen) > 1:
        diffs.append(f'{label} 에 현금 키가 둘 이상이다({sorted(seen)}) — '
                     f'표기가 섞이면 체결 쪽이 현금을 코인으로 읽는다')
    for k in seen:
        if k != expect:
            diffs.append(f'{label} 의 현금 키가 {k!r} 다 — 여기는 {expect!r} 만 쓴다')
    out = {}
    for k, v in d.items():
        if k == '_ts':
            continue
        key = 'CASH' if k.upper() == 'CASH' else k
        # 숫자로 읽히는 문자열('0.33')도 받지 않는다. JSON 이 수로 쓴 값만 비중이다 —
        # 문자열이 흘러들면 다른 데서 조용히 자료형이 바뀌었다는 뜻이라 그 자체가 사건이다.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            diffs.append(f'{label} {k} 비중이 숫자가 아니다({v!r})')
            continue
        wf = float(v)
        if not math.isfinite(wf):
            diffs.append(f'{label} {k} 비중이 유한한 수가 아니다({v!r})')
            continue
        out[key] = out.get(key, 0.0) + wf
    return out


def cash_boundary_diffs(items: list) -> list:
    """현금 키 경계만 따로 본다. items = [(딕셔너리, 이름, 허용 표기)].

    비교에 안 쓰는 저장값(멤버 합산, 멤버별 목표)도 표기 경계는 같이 본다. 값이
    어긋나는 건 리필 때문에 정상일 수 있지만, 표기가 어긋나는 건 어느 자리에서도
    정상이 아니다.
    """
    diffs = []
    for d, label, expect in items:
        if isinstance(d, dict):
            _cash_folded(d, label, diffs, expect)
    return diffs


def snap_state_diffs(member: dict, n_expected: int, cap: float,
                     tol: float = SNAP_TOL, combined: dict = None,
                     combined_label: str = 'last_combined',
                     exec_cash: str = CASH_ENGINE, cap_note: str = '') -> list:
    """스냅 상태가 스스로 모순되지 않는가. 어긋난 것들을 문장으로 돌려준다.

    순수 함수다 — 서버도 시계도 안 본다. 보는 것 넷.
      1. 스냅 개수가 파라미터 대조표의 n_snapshots 와 같은가
      2. 스냅마다 비중 합이 1 이고, 음수가 없고, 종목당 cap 을 안 넘는가
      3. 체결에 쓰인 합산이 스냅 평균과 정확히 같은가
      4. 합산의 비중 합이 1 인가
    3번이 이 검사의 핵심이다. 선정이 맞아도 합산이 어긋나면 체결 목표가 틀린다.

    `combined` 를 주면 그걸 3번의 비교 대상으로 쓴다. 안 주면 member['last_combined'].
    리필은 스냅만 고치고 저장된 last_combined 는 안 고치므로(현물·선물 공통), 리필이
    실제로 코인을 바꾼 날에는 last_combined 가 스냅 평균과 어긋난 채로 남는다. 그래서
    호출자는 그날 체결에 쓰인 목표(현물 last_target_snapshot / 선물 last_target)를
    넘겨야 한다 — 그 값은 리필 뒤 스냅에서 다시 계산된 것이라 항상 평균과 같다.
    """
    diffs = []
    snaps = member.get('snapshots')
    if not isinstance(snaps, list):
        return [f'snapshots 가 목록이 아니다({type(snaps).__name__})']
    if len(snaps) != n_expected:
        diffs.append(f'스냅 개수 {len(snaps)} (기대 {n_expected})')
    if not snaps:
        return diffs

    agg = {}
    for i, snap in enumerate(snaps):
        if not isinstance(snap, dict):
            diffs.append(f'스냅 {i} 이 딕셔너리가 아니다({type(snap).__name__})')
            continue
        folded = _cash_folded(snap, f'스냅 {i}', diffs, CASH_ENGINE)
        total = 0.0
        for coin, wf in folded.items():
            if wf < -tol:
                diffs.append(f'스냅 {i} {coin} 비중 음수({wf:.9f})')
            # 현금은 상한을 안 받는다. 코인만 cap 1/3 이다.
            if coin != 'CASH' and wf > cap + 1e-6:
                diffs.append(f'스냅 {i} {coin} 비중 {wf:.9f} 이 상한 {cap:.9f} 초과'
                             + (f' — {cap_note}' if cap_note else ''))
            total += wf
            agg[coin] = agg.get(coin, 0.0) + wf / len(snaps)
        if abs(total - 1.0) > 1e-6:
            diffs.append(f'스냅 {i} 비중 합 {total:.9f} (기대 1.0)')

    if combined is None:
        combined = member.get('last_combined')
    if not isinstance(combined, dict):
        diffs.append(f'{combined_label} 가 딕셔너리가 아니다({type(combined).__name__})')
        return diffs

    comb = _cash_folded(combined, combined_label, diffs, exec_cash)
    for coin in sorted(set(agg) | set(comb)):
        gap = abs(agg.get(coin, 0.0) - comb.get(coin, 0.0))
        if gap > tol:
            diffs.append(f'{coin} 합산 {comb.get(coin, 0.0):.9f} ≠ 스냅 평균 '
                         f'{agg.get(coin, 0.0):.9f} (차 {gap:.2e})')
    csum = sum(comb.values())
    if abs(csum - 1.0) > 1e-6:
        diffs.append(f'{combined_label} 비중 합 {csum:.9f} (기대 1.0)')
    return diffs


def synthetic_snap_state(n_expected: int, exec_cash: str) -> tuple:
    """깨끗한 합성 스냅 상태 한 벌. 자기시험은 서버 상태가 아니라 이걸로 한다.

    서버 상태로 자기시험을 하면 그날 상태가 어떤 모양이냐에 따라 대조가 둔해진다 —
    전액 현금인 날에는 흔들 종목이 없고, 이미 어긋난 날에는 무엇을 흔들어도 잡힌다.
    """
    snap = {'AAA': 1 / 3, 'BBB': 1 / 3, CASH_ENGINE: 1 / 3}
    member = {'snapshots': [dict(snap) for _ in range(n_expected)]}
    target = {'AAA': 1 / 3, 'BBB': 1 / 3, exec_cash: 1 / 3}
    return member, target


def _ctl_move(m, t, c):
    t['AAA'] += 0.01
    t['BBB'] -= 0.01


def _ctl_cash_split(m, t, c):
    half = t.pop(c) / 2
    t['CASH'], t['Cash'] = half, half


def _ctl_cash_spelling(m, t, c):
    t['Cash' if c == 'CASH' else 'CASH'] = t.pop(c)


def _ctl_count(m, t, c):
    m['snapshots'].pop()


def _ctl_cap(m, t, c):
    m['snapshots'][0].update({'AAA': 2 / 3, 'BBB': 0.0})


def _ctl_sum(m, t, c):
    m['snapshots'][0]['AAA'] += 0.01


def _ctl_negative(m, t, c):
    s = m['snapshots'][0]
    s['AAA'], s[CASH_ENGINE] = -0.01, s[CASH_ENGINE] + 1 / 3 + 0.01
    s['BBB'] = 1 / 3


def _ctl_nan(m, t, c):
    m['snapshots'][0]['AAA'] = float('nan')


# 대조마다 "어느 규칙이 물어야 하는가" 를 같이 못박는다. 다른 경보가 대신 떴다는
# 이유로 통과를 인정하면, 정작 그 규칙이 고장나도 자기시험이 초록으로 남는다.
SNAP_CONTROLS = [
    ('합을 보존한 목표 비중 이동', _ctl_move, '합산'),
    ('합·평균을 보존한 현금 키 분할', _ctl_cash_split, '현금 키가 둘 이상'),
    ('체결 목표 현금 표기 뒤집기', _ctl_cash_spelling, '현금 키가'),
    ('스냅 개수', _ctl_count, '스냅 개수'),
    ('종목당 상한', _ctl_cap, '상한'),
    ('스냅 비중 합', _ctl_sum, '비중 합'),
    ('음수 비중', _ctl_negative, '음수'),
    ('유한하지 않은 값', _ctl_nan, '유한한 수가 아니다'),
]


def self_check_snap(n_expected: int, cap: float, exec_cash: str,
                    combined_label: str = 'last_target') -> list:
    """대조 묶음 — 깨끗한 합성 상태를 규칙별로 하나씩 망가뜨려 그 규칙이 무는지 본다.

    검사가 고장나는 흔한 방식은 경보가 뜨는 게 아니라 무엇을 넣든 "이상 없음" 을
    찍게 되는 것이다. 정상 실행만 보고는 그 상태를 구분할 수 없다.
    """
    import copy

    base_m, base_t = synthetic_snap_state(n_expected, exec_cash)
    results = []
    clean = snap_state_diffs(base_m, n_expected, cap, combined=base_t,
                             combined_label=combined_label, exec_cash=exec_cash)
    if clean:
        results.append(('깨끗한 합성 상태', False,
                        f'정상 대조가 먼저 걸렸다: {clean[0]}'))
        return results
    results.append(('깨끗한 합성 상태', True, '이상 없음으로 통과'))

    for name, mutate, expect in SNAP_CONTROLS:
        m, t = copy.deepcopy(base_m), copy.deepcopy(base_t)
        mutate(m, t, exec_cash)
        hit = snap_state_diffs(m, n_expected, cap, combined=t,
                               combined_label=combined_label, exec_cash=exec_cash)
        matched = [d for d in hit if expect in d]
        if matched:
            results.append((name, True, matched[0]))
        elif hit:
            results.append((name, False,
                            f'다른 규칙만 물었다({hit[0]}) — 기대 사유 {expect!r}'))
        else:
            results.append((name, False, f'안 잡힌다 — 기대 사유 {expect!r}'))
    return results


def _snap_state_check(title: str, remote: str, pick_member, pick_combined,
                      n_expected: int, cap: float,
                      allow_stale_days: int, self_test: bool,
                      exec_cash: str = CASH_ENGINE, cap_note: str = '',
                      extra_boundary=None) -> int:
    """서버가 쓴 상태 파일 하나를 읽어 스냅이 스스로 모순이 없는지 본다. 읽기 전용이다.

    판본 대조를 안 한다 — 이 검사는 집 사본 코드를 한 줄도 안 돌리고 서버가 쓴
    상태 파일만 본다. 여기에 판본 대조를 붙이면 상관없는 이유로 비교불능이 된다.

    현물·선물이 상태 파일 모양만 다르고 보는 것은 같아서 한 몸으로 둔다. 다른 것은
    넷뿐이다 — 파일 경로, 멤버가 어디 있나, 체결 목표가 어디 있나, 스냅 개수.
    """
    import json

    print(f'== {title} ==')

    raw, why = _ssh_text(f'cat {remote}', 30)
    if raw is None:
        raise ParityDataStale(f'{title}: 상태 파일을 못 읽었다: {why}')
    try:
        state = json.loads(raw)
    except ValueError as exc:
        raise ParityDataStale(f'{title}: 상태 파일이 JSON 이 아니다: {exc}')
    member = pick_member(state)
    if not isinstance(member, dict):
        raise ParityDataStale(f'{title}: 상태에서 멤버를 못 찾았다')

    # 상태가 기대 봉까지 왔는가. 안 왔으면 스냅 내용이 맞아도 그건 어제 답이다.
    expected_bar = expected_last_closed_daily(datetime.now(timezone.utc))
    raw_ts = str(member.get('last_bar_ts') or '')
    try:
        actual_bar = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
    except ValueError:
        raise ParityDataStale(f'{title}: last_bar_ts 를 못 읽었다({raw_ts!r})')
    if actual_bar.tzinfo is None:
        actual_bar = actual_bar.replace(tzinfo=timezone.utc)
    lag_days = (expected_bar - actual_bar).days
    print(f'상태 마지막 봉 {str(actual_bar)[:10]} · 기대 봉 {str(expected_bar)[:10]} '
          f'· 지연 {lag_days}일 · bar_counter {member.get("bar_counter")} '
          f'· 카나리 {"ON" if member.get("canary_on") else "OFF"}')
    if lag_days < 0:
        print('판정: 불일치 — 상태의 마지막 봉이 미래다. 사람이 봐야 한다.')
        return 1
    if (actual_bar.astimezone(timezone.utc).hour,
            actual_bar.minute, actual_bar.second) != (0, 0, 0):
        print(f'판정: 불일치 — 마지막 봉 시각이 UTC 자정이 아니다({actual_bar}). '
              f'일봉이 아닌 값이 들어왔다.')
        return 1
    if lag_days > allow_stale_days:
        raise ParityDataStale(f'{title}: 상태가 기대 봉까지 안 왔다 (지연 {lag_days}일)')
    if lag_days > 0:
        print(f'주의: 지연 {lag_days}일이다. 허용치({allow_stale_days}일) 안이라 '
              f'판정은 이어가지만 이상 없음과 같은 상태가 아니다.')

    combined, label = pick_combined(state, member)
    if combined is None:
        raise ParityDataStale(f'{title}: 체결 목표({label})가 상태에 없다')

    diffs = snap_state_diffs(member, n_expected, cap, combined=combined,
                             combined_label=label, exec_cash=exec_cash,
                             cap_note=cap_note)
    if extra_boundary:
        diffs += cash_boundary_diffs(extra_boundary(state, member))
    snaps = member.get('snapshots') or []
    print(f'스냅 {len(snaps)}개 (기대 {n_expected}) · 상한 {cap:.6f}')
    for i, snap in enumerate(snaps):
        if isinstance(snap, dict):
            print(f'  스냅 {i}: {dict(sorted(snap.items(), key=lambda x: -x[1]))}')
    shown = {k: v for k, v in combined.items() if k != '_ts'}
    print(f'  체결 목표({label}): {dict(sorted(shown.items(), key=lambda x: -x[1]))}')

    # 참고 — 저장된 멤버 합산은 리필 날 스냅 평균과 다른 게 정상이다(리필이 스냅만
    # 고치고 이 값은 안 고친다). 판정에 넣지 않고 사람이 볼 수 있게만 적는다.
    stored = member.get('last_combined')
    if isinstance(stored, dict):
        ref = _cash_folded(stored, 'last_combined', [], CASH_ENGINE)
        mean = {}
        for snap in snaps:
            if isinstance(snap, dict):
                for k, v in _cash_folded(snap, '', [], CASH_ENGINE).items():
                    mean[k] = mean.get(k, 0.0) + v / len(snaps)
        gap = max([abs(ref.get(k, 0.0) - mean.get(k, 0.0))
                   for k in set(ref) | set(mean)] or [0.0])
        print(f'  참고: 저장된 last_combined 와 스냅 평균의 최대 차 {gap:.9f} '
              f'(리필 날은 0 이 아닌 게 정상, 판정에 안 넣는다)')

    if diffs:
        print(f'판정: 불일치 — {len(diffs)}건. 사람이 봐야 한다.')
        for d in diffs:
            print(f'    {d}')
        rc = 1
    else:
        print('판정: 이상 없음')
        rc = 0

    # 실제 상태 판정을 먼저 남긴 뒤에 자기시험을 한다. 순서를 뒤집으면 자기시험이
    # 걸린 날 그날 상태가 어땠는지가 기록에 아예 안 남는다.
    if self_test:
        failed = []
        for name, passed, why in self_check_snap(n_expected, cap, exec_cash, label):
            print(f'대조 {"통과" if passed else "실패"} — {name}: {why}')
            if not passed:
                failed.append(f'{name}: {why}')
        if failed:
            raise ParityDataStale(f'{title}: 자기시험 {len(failed)}건 실패 — '
                                  f'{failed[0]}')
    return rc


STOCK_LOG_GLOB = 'executor_stock.log executor_stock.log.*'
STOCK_STATE_REMOTE = '/home/ubuntu/kis_trade_state.json'
_RE_EXEC_DAY = re.compile(r'^\[(\d{4}-\d{2}-\d{2})')
_RE_EXEC_ORDER = re.compile(r'주문 (buy|sell) ([A-Z]+) (\d+)주')
_RE_EXEC_BAL = re.compile(r"잔고: (\{[^}]*\})")
_RE_EXEC_TARGET = re.compile(r"보고 \(silent\): target=(\{[^}]*\})")


def parse_stock_exec_log(raw: str) -> dict:
    """주식 executor 로그를 날짜별로 접는다. 순수 함수다.

    모으는 것 셋 — 그날 낸 주문(부호 있는 주식 수), 그날 끝 잔고, 찍혔으면 목표.
    같은 날 두 번 돌면 주문은 합치고 잔고는 뒤엣것이 끝이다.
    """
    days = {}
    for line in raw.splitlines():
        m = _RE_EXEC_DAY.match(line)
        if not m:
            continue
        rec = days.setdefault(m.group(1),
                              {'orders': [], 'balance': None, 'target': None})
        mo = _RE_EXEC_ORDER.search(line)
        if mo:
            qty = int(mo.group(3))
            rec['orders'].append((mo.group(2), qty if mo.group(1) == 'buy' else -qty))
        mt = _RE_EXEC_TARGET.search(line)
        if mt:
            rec['target'] = sorted(
                t for t, w in re.findall(r"'([A-Za-z]+)': ([0-9.e-]+)", mt.group(1))
                if t != 'Cash' and float(w) > 1e-9)
        mb = _RE_EXEC_BAL.search(line)
        if mb:
            rec['balance'] = {k: int(v) for k, v in
                              re.findall(r"'([A-Z]+)': (\d+)", mb.group(1))}
    return days


def stock_exec_diffs(before: dict, after: dict, orders: list,
                     target_set: list = None) -> list:
    """실행 결과가 주문과 목표로 설명되는가. 어긋난 것들을 문장으로 돌려준다.

    주식 수로만 본다. 금액으로 재면 로그에 안 남는 체결 시점 환율·현재가 때문에
    멀쩡한 날도 어긋나 보인다. 주식 수에는 그 미지수가 없고, 부분체결·조용한
    거절·계좌 외부 변경이 전부 이 항등식에서 드러난다.

      그날 끝 잔고 = 전날 끝 잔고 + 그날 낸 주문의 부호 합
    """
    diffs = []
    expected = dict(before)
    for ticker, qty in orders:
        expected[ticker] = expected.get(ticker, 0) + qty
    expected = {k: v for k, v in expected.items() if v != 0}
    actual = {k: v for k, v in after.items() if v != 0}
    for ticker in sorted(set(expected) | set(actual)):
        want, got = expected.get(ticker, 0), actual.get(ticker, 0)
        if want != got:
            diffs.append(f'{ticker} 주식 수 {got} (전날 {before.get(ticker, 0)} + '
                         f'주문 {want - before.get(ticker, 0):+d} = {want} 이어야 한다)')
    if target_set is not None:
        extra = sorted(set(actual) - set(target_set))
        if extra:
            diffs.append(f'목표에 없는 종목을 들고 있다 {extra} '
                         f'(목표 {sorted(target_set)}) — 매도가 안 나갔거나 거절됐다')
    return diffs


def stock_target_gap(held: dict, target_set: list) -> list:
    """목표에 있는데 안 들고 있는 종목. 판정이 아니라 참고다.

    비중이 작은 종목은 주문 금액이 최소주문에 못 미쳐 합법적으로 안 사진다. 그걸
    불일치로 올리면 소액 계좌에서 매일 경보가 난다. 반대 방향(목표에 없는데 들고
    있다)은 그런 합법 경로가 없어서 판정에 넣는다.
    """
    return sorted(set(target_set) - {k for k, v in held.items() if v})


def self_check_stock_exec() -> list:
    """대조 묶음 — 깨끗한 합성 실행을 규칙별로 망가뜨려 그 규칙이 무는지 본다.

    서버 실제 기록이 아니라 합성으로 하는 이유는 스냅 검사와 같다. 그날 주문이
    없거나 잔고가 비면 실제 기록으로는 흔들 자리가 없어서 대조가 조용히 둔해진다.
    """
    before = {'SPY': 10, 'EEM': 10}
    orders = [('SPY', -3), ('VNQ', 5)]
    after = {'SPY': 7, 'EEM': 10, 'VNQ': 5}
    target = ['EEM', 'SPY', 'VNQ']

    results = []
    clean = stock_exec_diffs(before, after, orders, target)
    if clean:
        return [('깨끗한 합성 실행', False, f'정상 대조가 먼저 걸렸다: {clean[0]}')]
    results.append(('깨끗한 합성 실행', True, '이상 없음으로 통과'))

    controls = [
        ('잔고 한 주 어긋남',
         (before, dict(after, SPY=after['SPY'] + 1), orders, target), '주식 수'),
        ('주문이 로그에 안 찍힘',
         (before, after, [('SPY', -3)], target), '주식 수'),
        ('목표 밖 종목 보유',
         (before, dict(after, TLT=4), orders + [('TLT', 4)], target),
         '목표에 없는 종목'),
    ]
    for name, args, expect in controls:
        hit = stock_exec_diffs(*args)
        matched = [d for d in hit if expect in d]
        if matched:
            results.append((name, True, matched[0]))
        elif hit:
            results.append((name, False,
                            f'다른 규칙만 물었다({hit[0]}) — 기대 사유 {expect!r}'))
        else:
            results.append((name, False, f'안 잡힌다 — 기대 사유 {expect!r}'))
    return results


def merged_stock_target(state: dict) -> list:
    """상태파일의 스냅 셋을 병합해 그날 목표 종목 집합을 낸다.

    executor 가 체결에 쓰는 목표와 같은 집합이다. 비중이 아니라 집합만 쓰는 이유는
    비중 비교에는 평가금액이 필요하고 그 축에 결제기준 배율이 숨어 있기 때문이다.
    """
    snaps = (state.get('snapshots') or {})
    picks = set()
    for snap in snaps.values():
        for t in (snap.get('picks') or []):
            if t != 'Cash':
                picks.add(t)
    return sorted(picks)


def check_stock_exec(self_test: bool = True) -> int:
    """주식 실행 결과를 서버 로그·상태로 독립 대조한다. 읽기 전용이다.

    목표를 복원하지 못하면 통과가 아니라 판정 불가로 낸다 — 복원 못 한 날을 조용히
    넘기면 이 감시는 "아무 일도 안 일어난 날" 과 "확인을 못 한 날" 을 같은 값으로
    찍는다.
    """
    import json

    print('== 주식 실행 결과 대조 ==')

    raw, why = _ssh_text(
        f'cd /home/ubuntu && for f in {STOCK_LOG_GLOB}; do [ -f "$f" ] && cat "$f"; done',
        60)
    if raw is None:
        raise ParityDataStale(f'주식 실행 로그를 못 읽었다: {why}')
    days = parse_stock_exec_log(raw)
    known = sorted(d for d in days if days[d]['balance'] is not None)
    if len(known) < 2:
        raise ParityDataStale(f'잔고가 찍힌 실행일이 {len(known)}일뿐이라 대조할 짝이 없다')

    prev, cur = known[-2], known[-1]
    before, after = days[prev]['balance'], days[cur]['balance']
    orders = days[cur]['orders']

    # 목표는 그날 로그가 적은 것을 먼저 쓴다. 상태파일은 오늘 값이라 검사하는 날과
    # 어긋날 수 있어서 로그가 없을 때만 쓰고, 썼으면 썼다고 적는다.
    target_set, src = days[cur]['target'], f'{cur} 로그'
    if not target_set:
        raw_state, why = _ssh_text(f'cat {STOCK_STATE_REMOTE}', 30)
        if raw_state is None:
            print(f'목표 복원 실패: {why}')
        else:
            try:
                target_set = merged_stock_target(json.loads(raw_state))
                src = '상태파일 스냅 병합(오늘 값)'
            except ValueError as exc:
                print(f'목표 복원 실패: 상태 파일이 JSON 이 아니다: {exc}')
    if not target_set:
        raise ParityDataStale('주식 목표를 복원하지 못했다 — 통과로 적지 않는다')

    print(f'직전 실행 {prev} 잔고 {before}')
    print(f'마지막 실행 {cur} 잔고 {after} · 주문 {orders or "없음"}')
    print(f'목표 종목 {target_set} ({src})')
    missing = stock_target_gap(after, target_set)
    if missing:
        print(f'참고: 목표에 있는데 안 들고 있다 {missing} — 최소주문 미만이면 정상이다. '
              f'판정에 안 넣는다.')

    diffs = stock_exec_diffs(before, after, orders, target_set)
    if diffs:
        print(f'판정: 불일치 — {len(diffs)}건. 사람이 봐야 한다.')
        for d in diffs:
            print(f'    {d}')
        rc = 1
    else:
        print('판정: 이상 없음')
        rc = 0

    if self_test:
        failed = []
        for name, passed, why in self_check_stock_exec():
            print(f'대조 {"통과" if passed else "실패"} — {name}: {why}')
            if not passed:
                failed.append(f'{name}: {why}')
        if failed:
            raise ParityDataStale(f'주식 실행 결과 대조: 자기시험 {len(failed)}건 실패 — '
                                  f'{failed[0]}')
    return rc


def _single_member(holder: dict, want: str, title: str, field: str) -> dict:
    """멤버가 그 하나뿐임을 확인하고 돌려준다. 아니면 비교불능이다.

    스냅 평균을 전체 체결 목표와 맞댈 수 있는 것은 그 목표가 이 멤버 것일 때뿐이다.
    멤버가 둘 이상이면 목표는 앙상블 합산이라 평균과 다른 게 정상인데, 그때 비교
    대상을 멤버 자기 합산으로 조용히 바꾸면 검사 이름은 그대로인 채 보는 것이 약해진다
    (리필 날 거짓 경보가 나는 바로 그 값이다). 그래서 대체하지 않고 비교불능으로 적는다.
    """
    members = holder or {}
    if list(members) != [want]:
        raise ParityDataStale(
            f'{title}: {field} 구성이 단일 {want} 가 아니다({sorted(members)}) — '
            f'이 비교는 단일 멤버일 때만 뜻이 있다')
    return members[want]


def _fut_combined(state: dict, member: dict):
    """선물이 그날 체결에 쓴 목표. 리필이 스냅을 고쳐도 같이 따라오는 값이다."""
    target = state.get('last_target')
    if not isinstance(target, dict):
        raise ParityDataStale('선물 스냅 상태 검사: last_target 이 상태에 없다 — '
                              'last_combined 로 대체하지 않는다')
    return target, 'last_target'


def _spot_combined(state: dict, member: dict):
    """현물이 그날 체결에 쓴 목표. 멤버별 목표와 같은 구성인지도 같이 본다."""
    target = state.get('last_target_snapshot')
    if not isinstance(target, dict):
        raise ParityDataStale('현물 스냅 상태 검사: last_target_snapshot 이 상태에 '
                              '없다 — last_combined 로 대체하지 않는다')
    mine = (state.get('last_member_targets') or {}).get(SPOT_STATE_MEMBER)
    if isinstance(mine, dict):
        a = _cash_folded(target, '', [], CASH_EXEC_SPOT)
        b = _cash_folded(mine, '', [], CASH_EXEC_SPOT)
        gap = max([abs(a.get(k, 0.0) - b.get(k, 0.0))
                   for k in set(a) | set(b)] or [0.0])
        if gap > 1e-9:
            raise ParityDataStale(
                f'현물 스냅 상태 검사: 멤버 목표와 전체 목표가 다르다(최대 차 {gap:.9f}) — '
                f'전체 목표가 이 멤버 것이라고 볼 수 없다')
    return target, 'last_target_snapshot'


def check_snap_state(allow_stale_days: int, self_test: bool = True) -> int:
    """선물 스냅 상태."""
    title = '선물 스냅 상태 검사'
    return _snap_state_check(
        title, FUT_STATE_REMOTE,
        lambda st: _single_member(st.get('strategies'), FUT_STATE_MEMBER,
                                  title, 'strategies'),
        _fut_combined,
        int(_CANON['fut']['params']['n_snapshots']),
        float(_CANON['spot']['params']['cap']),   # 종목당 상한은 현물·선물 같은 1/3
        allow_stale_days, self_test,
        exec_cash=CASH_EXEC_FUT,
        extra_boundary=lambda st, m: [
            (m.get('last_combined'), 'last_combined', CASH_ENGINE)])


def check_spot_snap_state(allow_stale_days: int = SPOT_SNAP_STALE_ALLOW,
                          self_test: bool = True) -> int:
    """현물 스냅 상태. 선물과 보는 것은 같고 상태 파일 모양만 다르다."""
    title = '현물 스냅 상태 검사'
    return _snap_state_check(
        title, SPOT_STATE_REMOTE,
        lambda st: _single_member(st.get('members'), SPOT_STATE_MEMBER,
                                  title, 'members'),
        _spot_combined,
        int(_CANON['spot']['params']['n_snapshots']),
        float(_CANON['spot']['params']['cap']),
        allow_stale_days, self_test,
        exec_cash=CASH_EXEC_SPOT,
        # 현물 리필에는 종목당 상한이 안 걸린다(선물만 2026-10-02 에 걸었다).
        # 상한 위반을 봤을 때 먼저 의심할 자리라 같이 적되, 그걸 이유로 통과시키지 않는다.
        cap_note='현물 리필에는 상한이 안 걸린다. 리필 날이면 그 경로를 먼저 본다',
        extra_boundary=lambda st, m: [
            (m.get('last_combined'), 'last_combined', CASH_ENGINE),
            ((st.get('last_member_targets') or {}).get(SPOT_STATE_MEMBER),
             f'last_member_targets.{SPOT_STATE_MEMBER}', CASH_EXEC_SPOT)])


def main() -> int:
    parser = argparse.ArgumentParser(description='라이브 선정함수와 채택 백테스트의 일별 대조')
    parser.add_argument('--sleeve',
                        choices=['fut', 'stock', 'snap', 'spotsnap', 'stockexec', 'all'],
                        default='all',
                        help='무엇을 볼지 (기본 all — 선물 대조·주식 대조·선물 스냅 상태·'
                             '현물 스냅 상태·주식 실행 결과 다섯)')
    parser.add_argument('--days', type=int, default=3, help='비교할 최근 신호일 수 (기본 3)')
    parser.add_argument('--allow-stale-days', type=int, default=1,
                        help='허용할 데이터 지연 일수 (기본 1)')
    parser.add_argument('--skip-server-check', action='store_true',
                        help='집 사본과 서버 배포본의 판본 대조를 건너뛴다 (손으로 볼 때만)')
    parser.add_argument('--skip-self-test', action='store_true',
                        help='자기시험(양성·음성 대조)을 건너뛴다 (손으로 빨리 볼 때만)')
    parser.add_argument('--receipt',
                        help='영수증 파일 경로. 각 검사의 내부 판정을 원자적으로 적는다 '
                             '(서버의 침묵 감시가 이걸 읽는다)')
    args = parser.parse_args()

    jobs = []
    if args.sleeve in ('fut', 'all'):
        jobs.append(('선물', lambda: check_fut(
            args.days, args.allow_stale_days,
            check_server=not args.skip_server_check,
            self_test=not args.skip_self_test)))
    if args.sleeve in ('stock', 'all'):
        jobs.append(('주식', lambda: check_stock(
            check_server=not args.skip_server_check,
            self_test=not args.skip_self_test)))
    if args.sleeve in ('snap', 'all'):
        jobs.append(('선물 스냅 상태', lambda: check_snap_state(
            args.allow_stale_days, self_test=not args.skip_self_test)))
    if args.sleeve in ('spotsnap', 'all'):
        # 현물은 허용 지연을 CLI 값이 아니라 0 으로 못박는다 — 아래 상수 주석 참조.
        jobs.append(('현물 스냅 상태', lambda: check_spot_snap_state(
            SPOT_SNAP_STALE_ALLOW, self_test=not args.skip_self_test)))
    if args.sleeve in ('stockexec', 'all'):
        jobs.append(('주식 실행 결과', lambda: check_stock_exec(
            self_test=not args.skip_self_test)))

    # 한쪽이 비교불능이어도 다른 쪽은 끝까지 본다. 먼저 실패한 것 때문에 나머지를
    # 건너뛰면 그날 감시 범위가 조용히 줄어든다. 최종 판정은 나쁜 쪽을 따른다(2 > 1 > 0).
    results = []
    for name, run in jobs:
        try:
            rc = run()
        except ParityDataStale as exc:
            print(f'판정: 비교 못 함 — {exc}')
            rc = 2
        except Exception as exc:                  # noqa: BLE001 — 감시는 죽어도 사유를 남긴다
            print(f'판정: 비교 못 함 — 예기치 못한 실패: {type(exc).__name__}: {exc}')
            rc = 2
        results.append((name, rc))
        print()

    worst = max(rc for _, rc in results)
    if len(results) > 1:
        label = {0: '일치', 1: '불일치', 2: '비교 못 함'}
        summary = ' · '.join(f'{name} {label[rc]}' for name, rc in results)
        print(f'전체 판정: {summary}')

    if args.receipt:
        write_receipt(args.receipt, 'parity', dict(results), worst)
    return worst


def write_receipt(path: str, mode: str, checks: dict, worst: int):
    """이 실행이 실제로 무엇을 어떻게 판정했는지 한 파일로 남긴다.

    껍데기 종료값이 아니라 검사별 내부 판정을 적는 이유 — 껍데기가 0 으로 끝나도
    안쪽에서 검사 하나가 비교불능이었을 수 있고, 반대로 껍데기가 죽으면 판정 자체가
    없다. 서버의 침묵 감시(parity_watch.py)는 이 파일이 오늘 날짜로 있는지,
    그 안의 판정이 깨끗한지 둘 다 본다.

    실행 식별자는 껍데기가 환경변수로 준다. 두 실행(갱신·대조)에 서로 다른 값을 줘야
    옛 영수증이 다시 올라온 것을 다음 날 잡을 수 있다.
    """
    now = datetime.now()
    body = {
        'mode': mode,
        'date': now.strftime('%Y-%m-%d'),
        'run_id': os.environ.get('PARITY_RUN_ID')
                  or f"{mode}-{now.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}",
        'checks': checks,
        'scope': dict(SCOPE_NOTES),
        'worst': worst,
        'finished_at': now.strftime('%Y-%m-%d %H:%M:%S KST'),
        'host': os.uname().nodename,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(body, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    print(f'영수증: {path} (run_id={body["run_id"]})')


if __name__ == '__main__':
    sys.exit(main())
