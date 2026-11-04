"""운영 서버 주소를 저장소 밖에서 읽는다 (2026-11-04).

이 저장소(github.com/Gyumin5/MoneyFlow)는 공개다. 주소를 추적 파일에 적으면 그대로
공개되므로, 실제 값은 저장소 루트의 server.conf 에 둔다. 그 파일은 .gitignore 에 있다.

찾는 순서는 환경변수, 그 다음 server.conf 다. 둘 다 없으면 멈춘다 — 기본값으로 아무
주소나 넣으면 엉뚱한 곳에 접속하거나, 접속 실패를 설정 누락이 아니라 서버 장애로 읽는다.

server.conf 형식:
    SERVER=ubuntu@<주소>
    HOST=<주소>
"""

import os
from pathlib import Path

_CONF = Path(__file__).resolve().parents[2] / 'server.conf'


def _read_conf(key):
    if not _CONF.exists():
        return None
    for line in _CONF.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith(key + '='):
            return line.split('=', 1)[1].strip().strip('"\'')
    return None


def get_server():
    """ssh/scp 에 넘길 'user@host' 를 돌려준다."""
    v = os.environ.get('MONEYFLOW_SERVER') or _read_conf('SERVER')
    if not v:
        raise RuntimeError(
            '운영 서버 주소를 찾지 못했다. %s 의 SERVER= 줄이나 '
            '환경변수 MONEYFLOW_SERVER 를 둔다.' % _CONF)
    return v


def get_host():
    """주소만 돌려준다(http 같은 데 쓴다)."""
    v = os.environ.get('MONEYFLOW_HOST') or _read_conf('HOST')
    if v:
        return v
    return get_server().split('@', 1)[-1]
