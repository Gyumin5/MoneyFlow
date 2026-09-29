#!/usr/bin/env python3
"""신호 수준 일별 대조 — 라이브 선정함수와 채택 백테스트 엔진이 같은 답을 내는지 매일 본다.

폐기한 월간 replay 자리를 메우는 감시다. 라이브 실행 로그를 읽는 게 아니라,
라이브가 쓰는 선정함수 자체를 읽기 전용으로 다시 돌려 백테스트 엔진과 맞춰 본다.
매매도 상태 저장도 하지 않는다.

지금 덮는 것 = 선물(V25). 현물은 v24_shadow_today.py 가 이미 있고, 주식은 다음 층.

판정 셋:
  0 = 비교한 모든 신호일에서 종목·비중이 같다
  1 = 하나라도 다르다 (사람이 봐야 한다)
  2 = 비교 자체를 못 했다

2 를 0 처럼 넘기지 않는 게 요점이다. 데이터가 밀린 채 옛 날짜만 비교하면
감시는 매일 통과를 찍으면서 아무것도 안 보게 된다. 2 로 올리는 것 넷 —
데이터가 기대 신호일에 안 닿음, 집 사본과 서버 배포본의 판본 차이,
양성 대조 실패(감시가 불일치를 못 잡는 상태), 예기치 못한 예외.

쓰기:
  python3 trade/ops/signal_parity_daily.py            # 최근 3 신호일
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


def server_build_matches(timeout_sec: int = 30):
    """집에서 다시 돌리는 선정함수가 서버에 실제로 깔린 것과 같은 판인지 본다.

    반환 (같은가, 설명). 같지 않거나 확인 자체를 못 하면 비교불능으로 올린다 —
    서버에 다른 판이 깔려 있는데 집 사본으로 대조해 통과를 찍으면 잘못된 안심이 된다.
    """
    import hashlib
    import subprocess

    local_path = os.path.join(_ROOT, 'trade', 'auto_trade_binance.py')
    try:
        with open(local_path, 'rb') as fh:
            local_md5 = hashlib.md5(fh.read()).hexdigest()
    except OSError as exc:
        return False, f'집 사본을 못 읽었다: {exc}'

    try:
        proc = subprocess.run(
            ['ssh', '-i', os.path.expanduser('~/.ssh/id_rsa'),
             '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
             'ubuntu@<서버주소>', 'md5sum /home/ubuntu/auto_trade_binance.py'],
            capture_output=True, text=True, timeout=timeout_sec)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f'서버 판본을 못 읽었다: {type(exc).__name__}'
    if proc.returncode != 0:
        return False, f'서버 판본 조회 실패 (rc={proc.returncode}) {proc.stderr.strip()[:120]}'

    server_md5 = proc.stdout.split()[0] if proc.stdout.split() else ''
    if server_md5 != local_md5:
        return False, f'집 사본과 서버 배포본이 다르다 (집 {local_md5[:12]} · 서버 {server_md5[:12]})'
    return True, f'집·서버 같은 판 ({local_md5[:12]})'


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


def main() -> int:
    parser = argparse.ArgumentParser(description='라이브 선정함수와 채택 백테스트의 일별 대조')
    parser.add_argument('--days', type=int, default=3, help='비교할 최근 신호일 수 (기본 3)')
    parser.add_argument('--allow-stale-days', type=int, default=1,
                        help='허용할 데이터 지연 일수 (기본 1)')
    parser.add_argument('--skip-server-check', action='store_true',
                        help='집 사본과 서버 배포본의 판본 대조를 건너뛴다 (손으로 볼 때만)')
    parser.add_argument('--skip-self-test', action='store_true',
                        help='양성 대조를 건너뛴다 (손으로 빨리 볼 때만)')
    args = parser.parse_args()

    try:
        return check_fut(args.days, args.allow_stale_days,
                         check_server=not args.skip_server_check,
                         self_test=not args.skip_self_test)
    except ParityDataStale as exc:
        print(f'판정: 비교 못 함 — {exc}')
        return 2
    except Exception as exc:                      # noqa: BLE001 — 감시는 죽어도 사유를 남긴다
        print(f'판정: 비교 못 함 — 예기치 못한 실패: {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
