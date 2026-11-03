#!/usr/bin/env python3
"""집으로 받아 둔 상태파일 백업이 실제로 들어오고 있는지 본다 (2026-11-03).

왜 따로 있나. 수신기(pull_state_backups.py)가 죽으면 그 기록도 같이 멈춘다. 자기 기록만
보고 판정하면 "아무 일도 안 일어난 것" 이 "정상" 으로 읽힌다 — 침묵을 신호로 바꾸는 자리가
따로 있어야 한다. 그래서 이 검사기는 수신기의 성공 기록을 믿지 않고 실제 디렉토리를 센다.

무엇을 보나 (하나라도 걸리면 경보)
  1. 15:40 KST 를 넘겼는데 오늘 묶음이 없다 (서버는 09:10 에 만들고 집 크론은 매시 30분)
  2. 가장 최신 묶음이 36시간보다 오래됐다
  3. 최신 묶음에 상태파일 넷이 다 있지 않다
  4. 수신기가 2시간 넘게 안 돌았다 (크론·잠금·기계 문제)
  5. 지난 회차에 수신 실패나 내용 불일치가 있었다
  6. 보관처 파티션 여유가 1GB 미만이다

알림은 집 세션 주입 경로(claude-userbot-send)를 쓴다 — 서버 경보 중계기가 쓰는 길과 같다.
억제하지 않는다. 전송에 실패하면 실패로 적는다(성공으로 적지 않는다).

실행: python3 check_state_backup_health.py [--dry-run]
종료값: 0 정상 · 1 경보 있음 · 2 알림 전송 실패
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

DEST = '/home/gmoh/backups/251229-state'
RECEIPT = os.path.join(DEST, 'receipt.json')
SEND = '/home/gmoh/dotfiles/bin/claude-userbot-send'
SESSION = '251229'

DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
REQUIRED = ('signal_state.json', 'trade_state.json',
            'kis_trade_state.json', 'binance_state.json')

TODAY_DUE_HHMM = 1540      # 이 시각을 넘기면 오늘 묶음이 있어야 한다
MAX_AGE_H = 36             # 최신 묶음이 이보다 오래되면 경보
RUN_STALE_H = 2            # 수신기가 이보다 오래 안 돌면 경보
MIN_FREE_GB = 1.0


def cohorts() -> list:
    try:
        names = os.listdir(DEST)
    except FileNotFoundError:
        return []
    return sorted(d for d in names
                  if DATE_RE.match(d) and os.path.isdir(os.path.join(DEST, d)))


def read_receipt() -> dict:
    try:
        with open(RECEIPT, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def parse_ts(text):
    """'YYYY-MM-DD HH:MM:SS' 를 datetime 으로. 못 읽으면 None."""
    try:
        return datetime.strptime(str(text), '%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return None


def free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
    except OSError:
        return -1.0
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def diagnose(now: datetime) -> list:
    """경보 사유 목록. 빈 목록이면 정상."""
    bad = []
    days = cohorts()
    if not days:
        return ['집에 받아 둔 묶음이 하나도 없다 (수신기가 한 번도 성공하지 못했다)']

    latest = days[-1]
    today = now.strftime('%Y-%m-%d')
    hhmm = now.hour * 100 + now.minute
    if hhmm >= TODAY_DUE_HHMM and latest < today:
        bad.append(f'오늘({today}) 묶음이 없다 — 최신은 {latest}')

    latest_dt = parse_ts(latest + ' 09:10:00')
    if latest_dt and now - latest_dt > timedelta(hours=MAX_AGE_H):
        age = (now - latest_dt).total_seconds() / 3600
        bad.append(f'최신 묶음 {latest} 이 {age:.0f}시간 지났다 (상한 {MAX_AGE_H})')

    missing = [n for n in REQUIRED
               if not os.path.isfile(os.path.join(DEST, latest, n))]
    if missing:
        bad.append(f'최신 묶음 {latest} 에 빠진 파일 {missing}')

    receipt = read_receipt()
    last_run = parse_ts(receipt.get('last_run'))
    if last_run is None:
        bad.append('수신 기록(receipt.json)을 읽지 못했다')
    elif now - last_run > timedelta(hours=RUN_STALE_H):
        age = (now - last_run).total_seconds() / 3600
        bad.append(f'수신기가 {age:.1f}시간째 안 돌았다 (상한 {RUN_STALE_H})')

    err = (receipt.get('last_error') or '').strip()
    if err:
        bad.append(f'지난 회차 실패가 남아 있다 — {err[:200]}')

    gb = free_gb(DEST)
    if 0 <= gb < MIN_FREE_GB:
        bad.append(f'보관처 여유 {gb:.2f}GB (하한 {MIN_FREE_GB}GB)')
    elif gb < 0:
        bad.append('보관처 용량을 읽지 못했다')

    return bad


def notify(text: str) -> bool:
    if not os.access(SEND, os.X_OK):
        return False
    res = subprocess.run([SEND, '--session', SESSION, text],
                         capture_output=True, text=True, timeout=60)
    return res.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description='상태파일 백업 수신 상태를 본다')
    ap.add_argument('--dry-run', action='store_true', help='알리지 않고 판정만 본다')
    args = ap.parse_args()

    now = datetime.now()
    bad = diagnose(now)
    days = cohorts()
    head = f"=== {now.strftime('%Y-%m-%d %H:%M KST')} 상태 백업 점검"
    print(f'{head} · 보관 {len(days)}개 · 최신 {days[-1] if days else "없음"}')
    if not bad:
        print('판정: 정상')
        return 0

    for line in bad:
        print('경보: ' + line)
    if args.dry_run:
        return 1

    text = ('[상태파일 백업 점검] 서버 상태파일을 집으로 받아 두는 경로에 문제가 있다.\n'
            + '\n'.join('- ' + b for b in bad)
            + f'\n보관 {len(days)}개 · 최신 {days[-1] if days else "없음"} · 보관처 {DEST}'
            + '\n확인: python3 /home/gmoh/mon/251229/trade/ops/pull_state_backups.py --dry-run')
    if notify(text):
        print('세션 주입 완료')
        return 1
    print('세션 주입 실패 — 다음 회차에 다시 시도한다')
    return 2


if __name__ == '__main__':
    sys.exit(main())
