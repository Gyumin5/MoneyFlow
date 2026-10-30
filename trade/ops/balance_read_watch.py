#!/usr/bin/env python3
"""밤 동안 주식 계좌 조회가 멀쩡한지 주기로 본다 — 읽기 전용, 주문·상태쓰기 없음.

왜. 23:35 집행 회차 한 번에만 조회가 들어가서, 그 한 번이 실패하면 그 회차가 쉬고
우리는 다음 날에야 안다. 조회 실패가 그 시각에만 나는 것인지 밤 내내 나는 것인지도
모른다(2026-10-12·10-21 두 회차에서 실패, 원인 미상). 미리 재 두면 둘 다 풀린다.

무엇을. executor_stock.KISAPI.get_balance 를 그대로 부른다 — 집행기가 쓰는 바로 그
경로라야 같은 실패를 잡는다. 성공하면 한 줄 적고 끝낸다. 실패하면 그 줄에 사유를
적고, 연속 실패가 FAIL_ALERT_STREAK 회에 닿는 첫 회차에만 텔레그램으로 알린다
(회차마다 보내면 밤새 쌓인다). 다시 성공하면 해소됐다고 한 번 알리고 센 것을 지운다.

안 하는 것. 주문, 전략 상태파일 쓰기. 전략 판단에 쓰이는 값을 아무것도 안 남긴다 —
이 파일이 쓰는 것은 자기 로그와 연속실패 기록 하나뿐이다.

토큰은 집행기와 같은 캐시(~/.kis_token.json)를 쓴다. 살아 있으면 그대로 쓰고,
만료됐으면 이 감시가 새로 받아 캐시에 넣는다 — 그래서 집행기와 같은 파일을 건드린다.
그 쓰기는 집행기 쪽 _get_token 이 잠금 + 임시파일 rename 으로 처리한다.

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
GAP_RESET_H = 3.0       # 회차 간격이 이보다 벌어졌으면 '연속' 이 아니다 (평소 간격 1시간)
TIME_FMT = '%Y-%m-%d %H:%M:%S'


def _now() -> str:
    return datetime.now().strftime(TIME_FMT)


def _read_state() -> dict:
    """{'fail_streak': int, 'alerted': bool} — 못 읽으면 처음부터.

    연속 실패 횟수와 '사람에게 알렸나' 를 따로 들고 있는다. 둘을 한 숫자로 묶으면
    알림 전송이 실패한 회차가 알린 것으로 세어져서 그 장애는 영영 안 알려진다.
    """
    blank = {'fail_streak': 0, 'alerted': False}
    try:
        with open(STREAK_FILE) as f:
            raw = json.load(f)
        state = {'fail_streak': int(raw.get('fail_streak', 0)),
                 'alerted': bool(raw.get('alerted', False))}
    except Exception:  # noqa: BLE001 — 깨진 기록 하나가 감시를 멈추게 두지 않는다
        return blank
    # 관측이 오래 끊겼으면 옛 실패를 오늘 실패에 이어 세지 않는다.
    try:
        gap_h = (datetime.now()
                 - datetime.strptime(str(raw.get('updated_at', '')), TIME_FMT)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return blank
    if gap_h > GAP_RESET_H:
        if state['fail_streak'] or state['alerted']:
            print(f'{_now()}  앞 회차와 {gap_h:.1f}시간 떨어져 있어 연속 횟수를 처음부터 센다')
        return blank
    return state


def _write_state(fail_streak: int, alerted: bool) -> bool:
    """저장 성공 여부를 돌려준다 — 실패를 정상 회차로 덮지 않으려는 것."""
    tmp = STREAK_FILE + f'.tmp{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump({'fail_streak': fail_streak, 'alerted': alerted,
                       'updated_at': _now()}, f)
        os.replace(tmp, STREAK_FILE)
        return True
    except Exception as e:  # noqa: BLE001
        print(f'{_now()}  상태 기록 실패: {type(e).__name__}: {e}')
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def _log_alert(text: str, sent: bool) -> None:
    """집 중계기가 읽을 자리에 같은 글을 남긴다. 실패해도 알림 자체는 건드리지 않는다."""
    try:
        import server_alert_log
        server_alert_log.record('balance_read_watch', text, sent=sent)
    except Exception:  # noqa: BLE001
        pass


def _notify(ex, text: str) -> bool:
    """보냈으면 True. 전송 자체가 실패하면 그 사실을 적고 False.

    보낸 글은 server_alerts.jsonl 에도 적는다 — 집 중계기가 세션에 주입한다(2026-10-30).
    """
    try:
        ex.send_telegram(text)
        _log_alert(text, True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f'{_now()}  알림 전송 실패: {type(e).__name__}: {e}')
        _log_alert(text, False)
        return False


def main() -> int:
    import executor_stock as ex

    # 집행기 로그 파일에 섞어 쓰지 않는다 — 그 파일은 회차 기록이고 감시·대조 도구가 읽는다.
    # 이 감시의 줄은 자기 로그로만 간다.
    ex.log = lambda msg='': print(f'{_now()}  | {str(msg).strip()}')

    prev = _read_state()
    try:
        holdings, total_usd, exrt = ex.KISAPI(dry_run=True).get_balance()
    except Exception as e:  # noqa: BLE001  — BalanceUnavailable 과 그 밖의 모두
        n = prev['fail_streak'] + 1
        print(f'{_now()}  실패({n}회 연속): {type(e).__name__}: {e}')
        alerted = prev['alerted']
        # 문턱에 닿았는데 아직 못 알렸으면 회차마다 다시 보낸다 — 전송이 한 번
        # 실패했다고 그 장애를 영영 안 알리면 감시가 있으나 마나다.
        if n >= FAIL_ALERT_STREAK and not alerted:
            alerted = _notify(ex,
                              f'주식 계좌 조회가 {n}회 연속 실패했다 ({e}). '
                              f'평일 23시 35분 주식 회차가 이 상태면 그 회차는 쉰다.')
        saved = _write_state(n, alerted)
        return 1 if saved else 2

    print(f'{_now()}  정상  보유 {len(holdings)}종  총자산 ${total_usd:,.0f}  환율 {exrt:,.2f}')
    rc = 0
    if prev['alerted']:
        # 알린 장애에만 해소를 알린다. 안 알린 실패는 사람이 모르므로 해소도 안 보낸다.
        _notify(ex, f"주식 계좌 조회가 다시 된다 (연속 실패 {prev['fail_streak']}회 뒤 해소).")
    if prev['fail_streak'] or prev['alerted']:
        if not _write_state(0, False):
            rc = 2
    return rc


if __name__ == '__main__':
    sys.exit(main())
