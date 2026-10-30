#!/usr/bin/env python3
"""서버가 자기 통로로 보낸 경보를 한 파일에 적는다 (2026-10-30).

무엇이 문제였나. 서버의 감시(parity_watch·balance_read_watch)는 서버 자기 텔레그램 봇으로
운영자에게 알린다. 그 통로는 집 유저봇을 일부러 안 탄다 — 집이 통째로 죽어도 알림이
나가게 하려는 설계다. 그런데 그 때문에 집 세션은 그 경보를 영영 못 본다. 운영자가 경보를
손으로 옮겨 적어야 세션이 알게 된다(2026-10-30 텔레그램 9365).

어떻게 푸나. 보낼 때 같은 글을 이 파일에도 한 줄 적는다. 집의 중계기가 그 파일을 읽어
세션에 주입한다. 통로는 둘로 남는다 — 서버 봇은 그대로 운영자에게 가고, 중계는 덤이다.
적기가 실패해도 알림 자체는 그대로 나간다(이 모듈의 모든 예외는 삼킨다).

안 하는 것. 매매 경로는 건드리지 않는다. 비밀(봇 토큰·계좌번호)은 적지 않는다 —
적는 것은 이미 사람에게 보낸 글 그대로다.
"""
import json
import os
import time

ALERT_PATH = os.environ.get('SERVER_ALERT_LOG', '/home/ubuntu/server_alerts.jsonl')
KEEP_LINES = 300


def make_id(ts: str, source: str, text: str) -> str:
    """같은 글이 같은 초에 두 번 적혀도 한 건으로 세도록 안정된 식별자를 만든다."""
    import hashlib
    raw = f'{ts}|{source}|{text}'.encode('utf-8')
    return hashlib.sha1(raw).hexdigest()[:12]


def _prune(path: str) -> None:
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return
    if len(lines) <= KEEP_LINES:
        return
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.writelines(lines[-KEEP_LINES:])
    os.replace(tmp, path)


def record(source: str, text: str, sent: bool = True, path: str = '') -> bool:
    """경보 한 건을 적는다. 적었으면 True. 어떤 이유로든 실패하면 False 를 내고 삼킨다."""
    target = path or ALERT_PATH
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        row = {'ts': ts, 'source': source, 'sent': bool(sent),
               'id': make_id(ts, source, text), 'text': text}
        with open(target, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        _prune(target)
        return True
    except Exception:                                  # noqa: BLE001
        return False
