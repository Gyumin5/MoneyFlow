#!/usr/bin/env python3
"""서버가 보낸 경보를 집 세션에 주입한다 (2026-10-30).

왜 있나. 서버 감시(parity_watch·balance_read_watch)는 서버 자기 텔레그램 봇으로 운영자에게
알린다. 그 통로는 집 유저봇을 일부러 안 탄다 — 집이 죽어도 알림이 나가야 하기 때문이다.
그래서 집 세션은 그 경보를 못 본다. 운영자가 손으로 옮겨 적어야 세션이 알게 되는 상태였다
(2026-10-30 텔레그램 9365 "봇에서 왔는데 너한테 주입 안 되고 있는 거 아니야").

어떻게 하나. 서버가 보낸 글을 server_alerts.jsonl 에 한 줄씩 적고(server_alert_log.py),
이 중계기가 10분마다 그 파일을 읽어 아직 안 본 줄만 세션에 주입한다. 본 줄의 식별자는
집에 적어 둔다 — 같은 경보를 다시 넣지 않기 위해서다.

안 하는 것. 서버 통로를 바꾸지 않는다(운영자는 그대로 서버 봇에서 받는다). 매매를 멈추지
않는다. ssh 가 안 되면 조용히 로그만 남긴다 — 서버 통로가 이미 운영자에게 닿아 있어서,
중계 실패를 다시 경보로 만들면 같은 사건이 두 번 시끄러워진다.

실행: python3 server_alert_relay.py [--dry-run]
종료값: 0 정상(주입 여부 무관) · 1 서버를 못 읽음
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEN_PATH = os.path.join(REPO, 'state', 'server-alert-relay', 'seen.json')
SEND = '/home/gmoh/dotfiles/bin/claude-userbot-send'
SESSION = '251229'
SERVER = 'ubuntu@<서버주소>'
SERVER_KEY = '/home/gmoh/.ssh/id_rsa'
REMOTE_PATH = '/home/ubuntu/server_alerts.jsonl'

MAX_AGE_H = 24        # 이보다 낡은 줄은 안 넣는다 — 늦게 켠 중계기가 과거를 쏟지 않게.
MAX_PER_RUN = 5       # 한 회차에 넣는 최대 건수. 나머지는 다음 회차로 밀린다.
KEEP_IDS = 500


def parse_lines(text: str) -> list:
    """jsonl 을 레코드 목록으로. 깨진 줄은 버린다 — 반쯤 쓰인 줄이 있을 수 있다."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('id') and row.get('text'):
            out.append(row)
    return out


def pick_new(records: list, seen_ids, now: datetime) -> list:
    """아직 안 본 것 중 24시간 안쪽만, 오래된 순으로 최대 MAX_PER_RUN 건."""
    cut = now - timedelta(hours=MAX_AGE_H)
    fresh = []
    for row in records:
        if row['id'] in seen_ids:
            continue
        try:
            ts = datetime.strptime(row.get('ts', ''), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
        if ts < cut:
            continue
        fresh.append(row)
    fresh.sort(key=lambda r: r['ts'])
    return fresh[:MAX_PER_RUN]


def compose(row: dict) -> str:
    """세션에 넣을 글. 어디서 왔는지와 운영자가 이미 받았는지를 같이 적는다."""
    head = f"[서버 경보 중계] {row.get('source', '?')} · {row.get('ts', '?')}"
    tail = ('운영자는 서버 봇으로 이미 같은 글을 받았다.' if row.get('sent')
            else '서버 봇 전송은 실패했다 — 운영자는 이 글을 못 봤다.')
    return f"{head}\n{row.get('text', '')}\n\n{tail}"


def read_seen() -> dict:
    try:
        with open(SEEN_PATH, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get('ids'), list):
            return data
    except (FileNotFoundError, ValueError):
        pass
    return {'ids': []}


def write_seen(data: dict) -> None:
    os.makedirs(os.path.dirname(SEEN_PATH), exist_ok=True)
    data['ids'] = data['ids'][-KEEP_IDS:]
    data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tmp = SEEN_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, SEEN_PATH)


def fetch() -> str:
    cmd = ['ssh', '-i', SERVER_KEY, '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes',
           SERVER, f'cat {REMOTE_PATH} 2>/dev/null || true']
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f'ssh rc={res.returncode}: {res.stderr.strip()[:200]}')
    return res.stdout


def inject(text: str) -> bool:
    if not os.access(SEND, os.X_OK):
        return False
    res = subprocess.run([SEND, '--session', SESSION, text],
                         capture_output=True, text=True, timeout=60)
    return res.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description='서버 경보를 집 세션에 중계한다')
    ap.add_argument('--dry-run', action='store_true', help='주입하지 않고 무엇을 넣을지만 본다')
    args = ap.parse_args()

    now = datetime.now()
    print(f"=== {now.strftime('%Y-%m-%d %H:%M KST')} 서버 경보 중계")
    try:
        raw = fetch()
    except Exception as exc:                           # noqa: BLE001
        print(f'서버를 못 읽었다 — {type(exc).__name__}: {exc}')
        return 1

    records = parse_lines(raw)
    seen = read_seen()
    fresh = pick_new(records, set(seen['ids']), now)
    print(f'서버 줄 {len(records)}건 · 이미 본 것 {len(seen["ids"])}건 · 넣을 것 {len(fresh)}건')

    if args.dry_run:
        for row in fresh:
            print('---\n' + compose(row))
        return 0

    for row in fresh:
        if inject(compose(row)):
            seen['ids'].append(row['id'])
            print(f"주입: {row['source']} {row['ts']}")
        else:
            print(f"주입 실패 — 다음 회차에 다시 시도한다: {row['source']} {row['ts']}")
    write_seen(seen)
    return 0


if __name__ == '__main__':
    sys.exit(main())
