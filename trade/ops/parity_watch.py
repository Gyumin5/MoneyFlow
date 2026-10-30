#!/usr/bin/env python3
"""집 감시의 침묵을 서버에서 잡는다 (2026-09-30, 토론 run-20260930T054956Z).

무엇이 문제였나. 신호 대조 감시는 집 컴퓨터의 크론에서 돈다. 집이 꺼져 있거나 네트워크가
끊기면 감시가 안 도는데, 안 도는 것을 알릴 주체도 집이라 아무 일도 안 일어난다.
경보 통로(유저봇)도 같은 집에 있어서 한 대가 죽으면 감시와 경보가 같이 죽는다.
그동안 "조용하다" 가 "이상 없다" 로 읽힌다. 기존 감시 전체의 목줄이 거기 하나다.

어떻게 막나. 집이 매 실행마다 영수증을 서버에 남긴다. 서버의 독립 크론이 유예시각 뒤에
그 영수증을 보고, 없거나 낡았거나 판정이 나쁘면 서버 자기 통로(텔레그램 봇)로 알린다.
집 유저봇을 안 탄다 — 그래야 집이 통째로 죽어도 알림이 나간다.

영수증에 무엇을 적나. 껍데기 종료값이 아니라 각 검사의 내부 판정이다. 껍데기가 0 으로
끝나도 안쪽에서 검사 하나가 비교불능이었을 수 있다.

안 하는 것. 이 경보는 매매를 잠그지 않는다. 감시의 감시일 뿐이고, 여기서 매매를 막으면
집이 꺼졌다는 이유로 서버 매매가 멈춘다.

서버에서 도는 파일이라 저장소 모듈을 import 하지 않는다 — 서버는 파일이 평평하게 놓인다.
집에서는 judge() 만 불러다 시험한다(망·자격증명 미사용).

실행: python3 parity_watch.py [--now 'YYYY-MM-DD HH:MM'] [--dry-run]
종료값: 0 정상 · 1 경보 · 2 자기 시험 실패(판정을 믿을 수 없다)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

RECEIPT_DIR = os.environ.get('PARITY_RECEIPT_DIR', '/home/ubuntu/parity-receipts')
STATE_PATH = os.path.join(RECEIPT_DIR, '_watch_state.json')
KEEP_DAYS = 30

# 집 크론 시각(KST)과 유예. 유예는 넉넉하게 준다 — 대조 자체가 최대 600초 걸리고,
# 지연을 장애로 오인해 매일 경보가 뜨면 사람이 경보를 안 읽게 된다.
REFRESH_AT_MIN = 9 * 60 + 20      # 09:20
PARITY_AT_MIN = 10 * 60           # 10:00
GRACE_MIN = 90

VERDICT_LABEL = {0: '일치', 1: '불일치', 2: '비교 못 함'}


def _minutes(ts: datetime) -> int:
    return ts.hour * 60 + ts.minute


def judge(now, receipts: dict, state: dict) -> dict:
    """영수증을 보고 무엇을 알릴지 정한다. 순수 함수 — 파일도 망도 안 탄다.

    now       datetime (KST, naive)
    receipts  {'refresh': dict|None, 'parity': dict|None}  — 오늘 날짜 영수증
    state     {'seen_run_ids': {run_id: 'YYYY-MM-DD'}, 'last_alarm': {'YYYY-MM-DD': key}}

    반환 {'alarms': [...], 'info': [...], 'waiting': bool, 'alarm_key': str}
    """
    alarms, info = [], []
    today = now.date().isoformat()
    mins = _minutes(now)

    if mins < PARITY_AT_MIN + GRACE_MIN:
        return {'alarms': [], 'info': ['아직 유예 안이다 — 판정하지 않는다'],
                'waiting': True, 'alarm_key': ''}

    seen = (state or {}).get('seen_run_ids') or {}

    for mode, due in (('refresh', REFRESH_AT_MIN), ('parity', PARITY_AT_MIN)):
        r = (receipts or {}).get(mode)
        label = {'refresh': '데이터 갱신', 'parity': '신호 대조'}[mode]
        if not r:
            alarms.append(f'{label} 영수증이 없다 — 집에서 {due // 60:02d}:'
                          f'{due % 60:02d} 실행이 안 됐거나 서버에 못 올렸다')
            continue
        if r.get('date') != today:
            alarms.append(f"{label} 영수증이 오늘 것이 아니다 (적힌 날짜 {r.get('date')})")
            continue
        rid = r.get('run_id')
        if not rid:
            alarms.append(f'{label} 영수증에 실행 식별자가 없다')
        elif seen.get(rid) not in (None, today):
            alarms.append(f'{label} 실행 식별자 {rid} 가 {seen[rid]} 에 이미 쓰였다 '
                          f'— 옛 영수증이 다시 올라왔다')

    par = (receipts or {}).get('parity')
    if par and par.get('date') == today:
        checks = par.get('checks') or {}
        if not checks:
            alarms.append('신호 대조 영수증에 검사 판정이 하나도 없다')
        bad = {k: v for k, v in checks.items() if v != 0}
        if bad:
            alarms.append('신호 대조 판정: '
                          + ', '.join(f'{k} {VERDICT_LABEL.get(v, v)}'
                                      for k, v in sorted(bad.items())))
        else:
            info.append('신호 대조 ' + ', '.join(f'{k} 일치' for k in sorted(checks)))

    ref = (receipts or {}).get('refresh')
    if ref and ref.get('date') == today:
        rrc = ref.get('worst')
        if rrc not in (0, None):
            alarms.append(f'데이터 갱신이 실패로 끝났다 (rc {rrc})')

    key = ' | '.join(alarms)
    return {'alarms': alarms, 'info': info, 'waiting': False, 'alarm_key': key}


# ── 양성 대조 ────────────────────────────────────────────────────────────────
# 판정기가 살아 있는지 매 실행 확인한다. 세 갈래를 인위로 넣어 본다 — 건강한 오늘 영수증은
# 조용해야 하고, 사흘 전 날짜와 실패 판정은 반드시 경보가 돼야 한다. 하나라도 어긋나면
# 그날 판정 전체를 믿을 수 없다고 말한다. 통과만 말하는 감시는 감시가 아니다.
def self_check(now) -> list:
    # 판정 시각은 유예 뒤로 고정한다. 자기 시험이 보는 것은 판정 논리이지 지금 몇 시냐가
    # 아니다 — 실제 now 를 그대로 쓰면 유예 안(예: 10:30)에 돌 때 judge 가 대기를 반환해서
    # "경보가 안 났다" 로 전부 실패한다. 실측으로 이 구멍을 봤다(2026-09-30 서버 시험).
    probe = now.replace(hour=23, minute=59, second=0, microsecond=0)
    today = probe.date().isoformat()
    old = (probe - timedelta(days=3)).date().isoformat()
    now = probe
    healthy = {
        'refresh': {'date': today, 'run_id': 'sc-r', 'worst': 0},
        'parity': {'date': today, 'run_id': 'sc-p', 'worst': 0,
                   'checks': {'선물': 0, '주식': 0, '선물 스냅 상태': 0}},
    }
    fails = []
    if judge(now, healthy, {})['alarms']:
        fails.append('건강한 영수증인데 경보가 났다')

    stale = json.loads(json.dumps(healthy))
    stale['parity']['date'] = old
    if not judge(now, stale, {})['alarms']:
        fails.append('사흘 전 영수증인데 경보가 안 났다')

    broken = json.loads(json.dumps(healthy))
    broken['parity']['checks']['주식'] = 2
    if not judge(now, broken, {})['alarms']:
        fails.append('검사 판정이 비교불능인데 경보가 안 났다')

    missing = {'refresh': healthy['refresh'], 'parity': None}
    if not judge(now, missing, {})['alarms']:
        fails.append('대조 영수증이 없는데 경보가 안 났다')

    repeat = json.loads(json.dumps(healthy))
    if not judge(now, repeat, {'seen_run_ids': {'sc-p': old}})['alarms']:
        fails.append('옛 실행 식별자가 되풀이되는데 경보가 안 났다')

    return fails


# ── 파일·통보 ────────────────────────────────────────────────────────────────
def read_receipts(day: str) -> dict:
    out = {}
    for mode in ('refresh', 'parity'):
        p = os.path.join(RECEIPT_DIR, f'{day}.{mode}.json')
        try:
            with open(p, encoding='utf-8') as fh:
                out[mode] = json.load(fh)
        except FileNotFoundError:
            out[mode] = None
        except Exception as exc:                      # noqa: BLE001
            out[mode] = {'date': None, 'broken': f'{type(exc).__name__}: {exc}'}
    return out


def read_state() -> dict:
    try:
        with open(STATE_PATH, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:                                 # noqa: BLE001
        return {'seen_run_ids': {}, 'last_alarm': {}}


def write_state(state: dict):
    os.makedirs(RECEIPT_DIR, exist_ok=True)
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def prune(now):
    """오래된 영수증을 덜어낸다. 남겨 둬도 값이 없고 디스크만 먹는다."""
    cut = (now - timedelta(days=KEEP_DAYS)).date().isoformat()
    try:
        for name in os.listdir(RECEIPT_DIR):
            if name.startswith('_'):
                continue
            if name[:10] < cut and name.endswith('.json'):
                os.remove(os.path.join(RECEIPT_DIR, name))
    except FileNotFoundError:
        pass


def _log_alert(text: str, sent: bool) -> None:
    """집 중계기가 읽을 자리에 같은 글을 남긴다. 실패해도 알림 자체는 건드리지 않는다."""
    try:
        sys.path.insert(0, '/home/ubuntu')
        import server_alert_log
        server_alert_log.record('parity_watch', text, sent=sent)
    except Exception:                                 # noqa: BLE001
        pass


def notify(text: str) -> bool:
    """서버 자기 통로로 보낸다. 집 유저봇을 안 탄다 — 그게 이 감시의 요점이다.

    보낸 글은 server_alerts.jsonl 에도 한 줄 적는다. 집 중계기가 그걸 읽어 세션에
    주입한다 — 통로를 바꾸는 게 아니라 하나 더 두는 것이다(2026-10-30).
    """
    try:
        import requests
        sys.path.insert(0, '/home/ubuntu')
        import config
        token = getattr(config, 'TELEGRAM_BOT_TOKEN', None)
        chat = getattr(config, 'TELEGRAM_CHAT_ID', None)
        if not token or not chat:
            print('통보 못 함 — 서버 config 에 텔레그램 값이 없다')
            _log_alert(text, False)
            return False
        requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                      data={'chat_id': chat, 'text': text}, timeout=10)
        _log_alert(text, True)
        return True
    except Exception as exc:                          # noqa: BLE001
        print(f'통보 실패 — {type(exc).__name__}: {exc}')
        _log_alert(text, False)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description='집 신호대조 감시의 침묵을 서버에서 잡는다')
    ap.add_argument('--now', help="판정 기준 시각 'YYYY-MM-DD HH:MM' (시험용)")
    ap.add_argument('--dry-run', action='store_true', help='통보하지 않고 판정만 낸다')
    ap.add_argument('--skip-self-test', action='store_true', help='양성 대조 생략 (손으로 볼 때만)')
    args = ap.parse_args()

    now = (datetime.strptime(args.now, '%Y-%m-%d %H:%M') if args.now
           else datetime.now())
    print(f"=== {now.strftime('%Y-%m-%d %H:%M KST')} 집 감시 침묵 점검")

    if not args.skip_self_test:
        fails = self_check(now)
        if fails:
            msg = ('감시의 감시 자기 시험 실패 — 오늘 판정을 믿을 수 없다: '
                   + '; '.join(fails))
            print(msg)
            if not args.dry_run:
                notify('[감시] ' + msg)
            return 2
        print('양성 대조 통과 (건강·낡음·실패판정·누락·식별자되풀이 5갈래)')

    today = now.date().isoformat()
    receipts = read_receipts(today)
    state = read_state()
    res = judge(now, receipts, state)

    for line in res['info']:
        print('  ' + line)

    if res['waiting']:
        print('판정: 유예 안 — 아직 안 본다')
        return 0

    # 본 실행 식별자를 기록한다. 옛 영수증이 다시 올라오는 것을 다음 날 잡기 위해서다.
    seen = state.setdefault('seen_run_ids', {})
    for mode in ('refresh', 'parity'):
        r = receipts.get(mode) or {}
        rid = r.get('run_id')
        if rid and r.get('date') == today:
            seen.setdefault(rid, today)
    if len(seen) > 400:
        for k in sorted(seen, key=lambda x: seen[x])[:100]:
            seen.pop(k, None)

    rc = 0
    if res['alarms']:
        rc = 1
        body = ('[감시] 집 신호대조 감시에 이상이 있다\n'
                + '\n'.join('· ' + a for a in res['alarms'])
                + '\n\n이 알림은 서버가 보냈다. 집 유저봇과 통로가 다르다.'
                  ' 매매는 이 알림으로 멈추지 않는다.')
        last = (state.setdefault('last_alarm', {})).get(today)
        if last == res['alarm_key']:
            print('판정: 경보 — 오늘 같은 내용으로 이미 보냈다(재전송 안 함)')
        else:
            print('판정: 경보\n' + body)
            if not args.dry_run and notify(body):
                state['last_alarm'][today] = res['alarm_key']
            elif args.dry_run:
                state['last_alarm'][today] = res['alarm_key']
    else:
        print('판정: 정상 — 오늘 집 감시가 돌았고 판정도 깨끗하다')

    if not args.dry_run:
        write_state(state)
        prune(now)
    return rc


if __name__ == '__main__':
    sys.exit(main())
