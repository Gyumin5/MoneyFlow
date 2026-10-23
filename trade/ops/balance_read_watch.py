#!/usr/bin/env python3
"""밤 동안 주식 계좌 조회가 멀쩡한지 주기로 본다 — 읽기 전용, 주문·상태쓰기 없음.

왜. 23:35 집행 회차 한 번에만 조회가 들어가서, 그 한 번이 실패하면 그 회차가 쉬고
우리는 다음 날에야 안다. 조회 실패가 그 시각에만 나는 것인지 밤 내내 나는 것인지도
모른다(2026-10-12·10-21 두 회차에서 실패, 원인 미상). 미리 재 두면 둘 다 풀린다.

무엇을. executor_stock.KISAPI.get_balance 를 그대로 부른다 — 집행기가 쓰는 바로 그
경로라야 같은 실패를 잡는다. 성공하면 한 줄 적고 끝낸다. 실패하면 그 줄에 사유를
적고, 연속 실패가 FAIL_ALERT_STREAK 회에 닿는 첫 회차에만 텔레그램으로 알린다
(회차마다 보내면 밤새 쌓인다). 다시 성공하면 해소됐다고 한 번 알리고 센 것을 지운다.

안 하는 것. 주문, 상태파일 쓰기, 토큰 새로 발급. 전략 판단에 쓰이는 값을 아무것도
안 남긴다 — 이 파일이 쓰는 것은 자기 로그와 연속실패 횟수 하나뿐이다.

설치(승인 뒤). 서버 /home/ubuntu/balance_read_watch.py 로 복사하고 crontab 에
  0 20-23,0-6 * * * cd /home/ubuntu && python3 balance_read_watch.py >> /home/ubuntu/balance_watch.log 2>&1
한 줄. 매시 정각, 20시부터 다음 날 6시까지 열한 번이다.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/ubuntu')

STREAK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'balance_watch_streak.json')
if os.path.isdir('/home/ubuntu'):
    STREAK_FILE = '/home/ubuntu/balance_watch_streak.json'
FAIL_ALERT_STREAK = 2   # 연속 몇 번 실패하면 사람에게 알리나 (한 시간 간격이면 두 시간)


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _read_streak() -> int:
    try:
        with open(STREAK_FILE) as f:
            return int(json.load(f).get('fail_streak', 0))
    except Exception:
        return 0


def _write_streak(n: int) -> None:
    tmp = STREAK_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump({'fail_streak': n, 'updated_at': _now()}, f)
        os.replace(tmp, STREAK_FILE)
    except Exception as e:  # noqa: BLE001
        print(f'{_now()}  연속실패 기록 실패: {type(e).__name__}')


def main() -> int:
    import executor_stock as ex

    # 집행기 로그 파일에 섞어 쓰지 않는다 — 그 파일은 회차 기록이고 감시·대조 도구가 읽는다.
    # 이 감시의 줄은 자기 로그로만 간다.
    ex.log = lambda msg='': print(f'{_now()}  | {str(msg).strip()}')

    prev = _read_streak()
    try:
        holdings, total_usd, exrt = ex.KISAPI(dry_run=True).get_balance()
    except Exception as e:  # noqa: BLE001  — BalanceUnavailable 과 그 밖의 모두
        n = prev + 1
        _write_streak(n)
        print(f'{_now()}  실패({n}회 연속): {type(e).__name__}: {e}')
        if n == FAIL_ALERT_STREAK:
            ex.send_telegram(
                f'주식 계좌 조회가 {n}회 연속 실패했다 ({e}). '
                f'23시 35분 회차가 이 상태면 그 회차는 쉰다.')
        return 1

    print(f'{_now()}  정상  보유 {len(holdings)}종  총자산 ${total_usd:,.0f}  환율 {exrt:,.2f}')
    if prev >= FAIL_ALERT_STREAK:
        ex.send_telegram(f'주식 계좌 조회가 다시 된다 (연속 실패 {prev}회 뒤 해소).')
    if prev:
        _write_streak(0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
