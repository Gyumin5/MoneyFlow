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
양성 대조 실패(감시가 불일치를 못 잡는 상태), 예기치 못한 예외.

쓰기:
  python3 trade/ops/signal_parity_daily.py                  # 셋 다, 최근 3 신호일
  python3 trade/ops/signal_parity_daily.py --sleeve stock
  python3 trade/ops/signal_parity_daily.py --sleeve snap     # 선물 스냅 상태만
  python3 trade/ops/signal_parity_daily.py --days 7
  python3 trade/ops/signal_parity_daily.py --allow-stale-days 2
  python3 trade/ops/signal_parity_daily.py --skip-server-check --skip-self-test
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
SNAP_TOL = 1e-9


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


def self_check(allow_stale_days: int):
    """양성 대조 — 한쪽을 일부러 틀리게 해서 이 감시가 실제로 불일치를 잡는지 본다.

    매일 같이 돌린다. 감시가 고장나는 가장 흔한 방식은 경보가 뜨는 게 아니라
    무엇을 비교하든 늘 "같다" 를 찍게 되는 것이고, 그건 정상 실행만 보고는 구분이 안 된다.

    무엇을 틀리게 하느냐가 중요하다. 라이브 쪽 단기 모멘텀 기간을 쓴다 — 헬스 판정에
    바로 들어가서 보유 종목을 바꾸므로, 어떤 시장 국면에서도 목표가 달라진다.
    카나리 SMA 기간이나 장기 모멘텀 기간은 못 쓴다 — 비트코인이 SMA 에서 멀리 떨어진
    날에는 기간을 바꿔도 카나리가 그대로 켜져 있고 상위 두 종목도 그대로라,
    안 잡혀도 감시 고장이 아니라 그날의 둔감함이다(2026-09-29 실측).

    반환 (통과했나, 설명).
    """
    import contextlib
    import io

    import auto_trade_binance as atb

    key = 'mom_short_bars'
    original = atb.STRATEGIES['D_SMA42'].get(key)
    atb.STRATEGIES['D_SMA42'][key] = 40
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_fut(days=1, allow_stale_days=allow_stale_days,
                           check_server=False, self_test=False)
    finally:
        if original is None:
            atb.STRATEGIES['D_SMA42'].pop(key, None)
        else:
            atb.STRATEGIES['D_SMA42'][key] = original

    if rc == 1:
        return True, f'라이브 {key} 를 {original}→40 으로 틀리자 불일치로 잡혔다'
    return False, (f'라이브 {key} 를 {original}→40 으로 틀렸는데 rc={rc} 가 나왔다 — '
                   '이 감시는 불일치를 못 잡는다')


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

    probe = '; '.join(f'printf "%s %s\\n" {t} "$(tail -1 /home/ubuntu/data/{t}.csv | cut -d, -f1)"'
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
    for key in ('risk_on', 'offense_weights', 'defense_weights'):
        if key not in stock:
            raise ParityDataStale(f'서버 주식 신호에 {key} 가 없다')
    if not meta.get('signal_date') or not meta.get('updated_at'):
        raise ParityDataStale('서버 주식 신호에 신호일·갱신시각이 없다')

    basis = {}
    for line in tail.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            basis[parts[0]] = parts[1][:10]
    missing = [t for t in STOCK_TICKERS if t not in basis]
    if missing:
        raise ParityDataStale(f'서버 가격 기준일을 못 읽었다: {", ".join(missing)}')
    # 종목마다 기준일이 다르면 그 신호는 섞인 날짜로 계산된 것이다. 라이브도 같은 조건을
    # 검사해 건너뛰게 되어 있으니, 여기서도 대조하지 않는다.
    dates = sorted(set(basis.values()))
    if len(dates) != 1:
        raise ParityDataStale(f'서버 주식 가격 기준일이 종목마다 다르다: {basis}')

    return {
        'risk_on': bool(stock['risk_on']),
        'off_w': _norm_risky(stock['offense_weights']),
        'def_w': _norm_risky(stock['defense_weights']),
        'signal_date': str(meta['signal_date'])[:10],
        'updated_at': str(meta['updated_at']),
        'basis_date': dates[0],
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
    off_picks = fresh_pick_3mom(ranked.at[sig_ts],
                                mom_off[ms].loc[sig_ts],
                                mom_off[mid].loc[sig_ts],
                                mom_off[ml].loc[sig_ts])
    return {
        'risk_on': on,
        'off_w': _norm_risky(picks_to_target(off_picks, 0.0, 'cap')),
        'def_w': _norm_risky(select_off(sig_ts, mom_def, 0.0, BT_STOCK_DEF_MODE)),
    }


def decide_stock(live: dict, bt: dict):
    """주식 대조 판정. 순수 함수 — 시험에서 그대로 부른다.

    반환 (rc, 어긋난 항목들). rc 0 = 셋 다 같다, 1 = 하나라도 다르다.
    셋을 다 본다 — 카나리(공격·방어 국면 판정), 공격 국면 목표, 방어 국면 목표.
    지금 쓰지 않는 국면도 같이 보는 이유는 국면이 바뀌는 날 처음 어긋남을 보게 되면
    그날 바로 매매가 갈리기 때문이다.
    """
    bad = []
    if live['risk_on'] != bt['risk_on']:
        bad.append(f"카나리: 라이브 {live['risk_on']} · 백테스트 {bt['risk_on']}")
    if live['off_w'] != bt['off_w']:
        bad.append(f"공격 목표: 라이브 {live['off_w']} · 백테스트 {bt['off_w']}")
    if live['def_w'] != bt['def_w']:
        bad.append(f"방어 목표: 라이브 {live['def_w']} · 백테스트 {bt['def_w']}")
    return (1 if bad else 0), bad


def self_check_stock(pdf, live: dict, sig_ts):
    """주식 양성 대조 — 백테스트 쪽 입력을 일부러 틀리게 해서 불일치가 잡히는지 본다.

    고르는 방식이 중요하다. 라이브가 안 고른 공격 종목 하나의 최근 1년 가격을 하루 2%씩
    복리로 밀어 올린다 — 그 종목은 Z랭킹 1위가 되고 3-모멘텀(30·72·230일) 필터도 전부
    통과하므로 어떤 시장 국면에서도 반드시 공격 목표가 달라진다. 기간 파라미터를 흔드는
    방식은 못 쓴다: 그날 상위 종목이 그대로면 안 잡혀도 감시 고장이 아니라 그날의
    둔감함이 되어 판정이 흐려진다(선물 쪽에서 2026-09-29 실측). 12년에 걸친 직선 기울기도
    못 쓴다 — 하루치 기울기가 0.02% 라 30일 모멘텀 부호를 못 뒤집는다(같은 날 실측).

    반환 (통과했나, 설명).
    """
    import numpy as np

    from bt_stock_coin_v3 import OFF_R7

    held = set(live['off_w'])
    cand = [t for t in OFF_R7 if t not in held and t in pdf.columns]
    if not cand:
        return False, '부풀릴 공격 종목이 없다 — 양성 대조를 만들 수 없다'

    tried = []
    for victim in cand:
        ramp = np.ones(len(pdf))
        tail = min(260, len(pdf))
        ramp[-tail:] = 1.02 ** np.arange(1, tail + 1)
        fake = pdf.copy()
        fake[victim] = fake[victim] * ramp
        rc, _bad = decide_stock(live, _stock_bt(fake, sig_ts))
        if rc == 1:
            return True, f'백테스트 쪽 {victim} 가격을 부풀리자 불일치로 잡혔다'
        tried.append(victim)
    return False, (f'백테스트 쪽 {", ".join(tried)} 가격을 차례로 부풀렸는데 전부 '
                   '같다고 나왔다 — 이 감시는 불일치를 못 잡는다')


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
    # 서버의 신호 갱신이 오늘 돌았는지 본다. 09:15 KST 크론이 죽으면 신호가 그 자리에
    # 얼어붙는데, 그걸 모르고 옛 신호와 대조하면 매일 통과를 찍는다.
    today = datetime.now().date()
    updated_day = datetime.strptime(live['updated_at'][:10], '%Y-%m-%d').date()
    update_lag = (today - updated_day).days
    print(f"서버 신호 갱신 {live['updated_at']} · 표기 신호일 {live['signal_date']} "
          f"· 갱신 지연 {update_lag}일 (허용 {update_allow})")
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

    if self_test:
        passed, why = self_check_stock(pdf, live, sig_ts)
        print(f'양성 대조: {why}')
        if not passed:
            raise ParityDataStale(why)

    bt = _stock_bt(pdf, sig_ts)
    rc, bad = decide_stock(live, bt)
    print(f"  카나리   라이브 {live['risk_on']} · 백테스트 {bt['risk_on']}")
    print(f"  공격     라이브 {live['off_w']}")
    print(f"           백테스트 {bt['off_w']}")
    print(f"  방어     라이브 {live['def_w']}")
    print(f"           백테스트 {bt['def_w']}")
    if rc == 1:
        for line in bad:
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

    if self_test:
        passed, why = self_check(allow_stale_days)
        print(f'양성 대조: {why}')
        if not passed:
            raise ParityDataStale(why)

    bars, funding = ub.load_data('D')
    last, expected, lag = _check_freshness(bars, allow_stale_days)
    print(f'데이터 마지막 봉 {last.date()} · 기대 신호일 {expected.date()} · 지연 {lag}일')

    universe = [c for c in REF_UNIVERSE if c in bars]
    print(f'고정 유니버스 {len(universe)}종')

    end_date = str(bars['BTC'].index[-1])[:10]
    bt = _run_bt(bars, funding, universe, end_date)

    # 기대 신호일은 반드시 비교한다. 그 앞으로 N-1 개를 더 본다.
    wanted = sorted(d for d in bt if d <= expected)[-max(1, days):]
    if expected not in wanted:
        wanted.append(expected)
    live = _run_live(bars, universe, wanted)

    rc, compared, mismatched, why = decide(bt, live, expected, days)

    for sig_date in compared:
        if sig_date in mismatched:
            print(f'  {str(sig_date)[:10]}  다름')
            print(f'      백테스트 {bt[sig_date]}')
            print(f'      라이브   {live[sig_date]}')
        else:
            # 같을 때도 값을 찍는다 — 서버 실행 로그의 그날 선정과 눈으로 맞춰볼 수 있어야
            # 이 하니스가 라이브를 실제로 반영하는지 확인된다.
            print(f'  {str(sig_date)[:10]}  같음  {live[sig_date]}')

    if rc == 2:
        raise ParityDataStale(why)
    print(f'비교 {len(compared)}일 · 불일치 {len(mismatched)}일 · 기대 신호일 {str(expected)[:10]} 포함')
    if rc == 1:
        print(f'판정: 불일치 — {why}. 사람이 봐야 한다.')
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


def snap_state_diffs(member: dict, n_expected: int, cap: float,
                     tol: float = SNAP_TOL) -> list:
    """선물 스냅 상태가 스스로 모순되지 않는가. 어긋난 것들을 문장으로 돌려준다.

    순수 함수다 — 서버도 시계도 안 본다. 보는 것 넷.
      1. 스냅 개수가 파라미터 대조표의 n_snapshots 와 같은가
      2. 스냅마다 비중 합이 1 이고, 음수가 없고, 종목당 cap 을 안 넘는가
      3. 저장된 합산(last_combined)이 스냅 평균과 정확히 같은가
      4. 합산의 비중 합이 1 인가
    3번이 이 검사의 핵심이다. 선정이 맞아도 합산이 어긋나면 체결 목표가 틀린다.
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
        total = 0.0
        for coin, w in snap.items():
            try:
                wf = float(w)
            except (TypeError, ValueError):
                diffs.append(f'스냅 {i} {coin} 비중이 숫자가 아니다({w!r})')
                continue
            if wf < -tol:
                diffs.append(f'스냅 {i} {coin} 비중 음수({wf:.9f})')
            # 현금은 상한을 안 받는다. 코인만 cap 1/3 이다.
            if coin.upper() != 'CASH' and wf > cap + 1e-6:
                diffs.append(f'스냅 {i} {coin} 비중 {wf:.9f} 이 상한 {cap:.9f} 초과')
            total += wf
            agg[coin] = agg.get(coin, 0.0) + wf / len(snaps)
        if abs(total - 1.0) > 1e-6:
            diffs.append(f'스냅 {i} 비중 합 {total:.9f} (기대 1.0)')

    combined = member.get('last_combined')
    if not isinstance(combined, dict):
        diffs.append(f'last_combined 가 딕셔너리가 아니다({type(combined).__name__})')
        return diffs

    comb = {}
    for coin, w in combined.items():
        try:
            comb[coin] = float(w)
        except (TypeError, ValueError):
            diffs.append(f'last_combined {coin} 비중이 숫자가 아니다({w!r})')
    for coin in sorted(set(agg) | set(comb)):
        gap = abs(agg.get(coin, 0.0) - comb.get(coin, 0.0))
        if gap > tol:
            diffs.append(f'{coin} 합산 {comb.get(coin, 0.0):.9f} ≠ 스냅 평균 '
                         f'{agg.get(coin, 0.0):.9f} (차 {gap:.2e})')
    csum = sum(comb.values())
    if abs(csum - 1.0) > 1e-6:
        diffs.append(f'last_combined 비중 합 {csum:.9f} (기대 1.0)')
    return diffs


def self_check_snap(member: dict, n_expected: int, cap: float):
    """양성 대조 — 합산을 일부러 흔들어 이 검사가 실제로 잡는지 본다.

    검사가 고장나는 흔한 방식은 경보가 뜨는 게 아니라 무엇을 넣든 "이상 없음" 을
    찍게 되는 것이다. 정상 실행만 보고는 그 상태를 구분할 수 없다.
    """
    import copy
    if not snap_state_diffs(member, n_expected, cap):
        pass  # 현재 상태는 정상이어야 한다. 정상 여부 판정은 호출자가 한다.
    poisoned = copy.deepcopy(member)
    comb = poisoned.get('last_combined')
    if not isinstance(comb, dict) or not comb:
        return False, '양성 대조를 못 했다: last_combined 가 비었다'
    key = sorted(comb)[0]
    comb[key] = float(comb[key]) + 0.01
    if not snap_state_diffs(poisoned, n_expected, cap):
        return False, f'양성 대조 실패: {key} 합산을 0.01 흔들었는데 안 잡힌다'
    return True, f'양성 대조 통과 ({key} 합산 0.01 흔들기를 잡는다)'


def check_snap_state(allow_stale_days: int, self_test: bool = True) -> int:
    """서버의 선물 스냅 상태를 읽어 스스로 모순이 없는지 본다. 읽기 전용이다.

    판본 대조를 안 한다 — 이 검사는 집 사본 코드를 한 줄도 안 돌리고 서버가 쓴
    상태 파일만 본다. 여기에 판본 대조를 붙이면 상관없는 이유로 비교불능이 된다.
    """
    import json

    print('== 선물 스냅 상태 검사 ==')

    raw, why = _ssh_text(f'cat {FUT_STATE_REMOTE}', 30)
    if raw is None:
        raise ParityDataStale(f'선물 상태 파일을 못 읽었다: {why}')
    try:
        state = json.loads(raw)
    except ValueError as exc:
        raise ParityDataStale(f'선물 상태 파일이 JSON 이 아니다: {exc}')
    member = (state.get('strategies') or {}).get(FUT_STATE_MEMBER)
    if not isinstance(member, dict):
        raise ParityDataStale(f'상태에 strategies.{FUT_STATE_MEMBER} 가 없다')

    # 상태가 기대 봉까지 왔는가. 안 왔으면 스냅 내용이 맞아도 그건 어제 답이다.
    expected_bar = expected_last_closed_daily(datetime.now(timezone.utc))
    raw_ts = str(member.get('last_bar_ts') or '')
    try:
        actual_bar = datetime.fromisoformat(raw_ts.replace('Z', '+00:00'))
    except ValueError:
        raise ParityDataStale(f'last_bar_ts 를 못 읽었다({raw_ts!r})')
    if actual_bar.tzinfo is None:
        actual_bar = actual_bar.replace(tzinfo=timezone.utc)
    lag_days = (expected_bar - actual_bar).days
    print(f'상태 마지막 봉 {str(actual_bar)[:10]} · 기대 봉 {str(expected_bar)[:10]} '
          f'· 지연 {lag_days}일 · bar_counter {member.get("bar_counter")} '
          f'· 카나리 {"ON" if member.get("canary_on") else "OFF"}')
    if lag_days < 0:
        print('판정: 불일치 — 상태의 마지막 봉이 미래다. 사람이 봐야 한다.')
        return 1
    if lag_days > allow_stale_days:
        raise ParityDataStale(f'선물 상태가 기대 봉까지 안 왔다 (지연 {lag_days}일)')

    n_expected = int(_CANON['fut']['params']['n_snapshots'])
    cap = float(_CANON['spot']['params']['cap'])   # 종목당 상한은 현물·선물 같은 1/3

    if self_test:
        passed, why = self_check_snap(member, n_expected, cap)
        print(f'양성 대조: {why}')
        if not passed:
            raise ParityDataStale(why)

    diffs = snap_state_diffs(member, n_expected, cap)
    snaps = member.get('snapshots') or []
    print(f'스냅 {len(snaps)}개 (기대 {n_expected}) · 상한 {cap:.6f}')
    for i, snap in enumerate(snaps):
        if isinstance(snap, dict):
            print(f'  스냅 {i}: {dict(sorted(snap.items(), key=lambda x: -x[1]))}')
    print(f'  합산: {dict(sorted((member.get("last_combined") or {}).items(), key=lambda x: -x[1]))}')

    if diffs:
        print(f'판정: 불일치 — {len(diffs)}건. 사람이 봐야 한다.')
        for d in diffs:
            print(f'    {d}')
        return 1
    print('판정: 이상 없음')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='라이브 선정함수와 채택 백테스트의 일별 대조')
    parser.add_argument('--sleeve', choices=['fut', 'stock', 'snap', 'all'], default='all',
                        help='무엇을 볼지 (기본 all — 선물 대조·주식 대조·선물 스냅 상태 셋)')
    parser.add_argument('--days', type=int, default=3, help='비교할 최근 신호일 수 (기본 3)')
    parser.add_argument('--allow-stale-days', type=int, default=1,
                        help='허용할 데이터 지연 일수 (기본 1)')
    parser.add_argument('--skip-server-check', action='store_true',
                        help='집 사본과 서버 배포본의 판본 대조를 건너뛴다 (손으로 볼 때만)')
    parser.add_argument('--skip-self-test', action='store_true',
                        help='양성 대조를 건너뛴다 (손으로 빨리 볼 때만)')
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
    return worst


if __name__ == '__main__':
    sys.exit(main())
