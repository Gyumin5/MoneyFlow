#!/usr/bin/env python3
"""서버의 상태파일 백업 묶음을 집으로 받아 둔다 (2026-11-03).

왜 있나. 서버 watchdog_serve.sh 가 매일 09:10 KST 이후 첫 회차에 상태파일 넷을
/home/ubuntu/state_backups/<날짜>/ 로 복사한다. 설계는 꼼꼼하지만 그 백업이 전부 같은
기계 안에 있어서, 인스턴스가 통째로 사라지면 원본과 백업이 같이 사라진다. 상태파일은
단순 캐시가 아니라 전략 상태다 — 카나리 히스테리시스 직전 상태, 스냅샷 앵커 위상
(bar_counter), 전환 후 재평가 예약, 미체결 복구 목록이 들어 있다. 없으면 전 트랜치가
현재 신호로 초기화되고 앵커 위상이 리셋된다. 즉 전략이 다른 전략이 된다.

무엇을 하나. 서버에서 "완성된 날짜 디렉토리" 만 골라, 집에 없는 것을 모두 받는다.
받는 자리는 저장소 밖 /home/gmoh/backups/251229-state/ 다(디렉토리 700, 파일 600).
임시 디렉토리에 받아 해시·구조를 다 확인한 뒤에야 날짜 이름으로 바꾼다 — 서버의
"완성된 것만 이름을 갖는다" 규약을 집에서도 그대로 지킨다.

안 하는 것. 서버는 한 줄도 안 바꾼다(읽기 전용 복사). 매매 경로를 건드리지 않는다.
rsync --delete 를 쓰지 않는다 — 서버가 14일 뒤 지운 묶음을 집에서 따라 지우면 보관
기간을 늘린 의미가 없다. 이미 집에 있는 완성본은 덮어쓰지 않는다. 내용이 다르면
덮어쓰는 대신 경보한다.

이 스크립트는 수신만 한다. "요즘 안 들어오고 있다" 를 알아채는 것은 별도 검사기
check_state_backup_health.py 가 한다 — 수신기가 죽으면 수신기의 기록도 같이 멈추므로,
자기 기록만 보고 판정하면 침묵이 정상으로 읽힌다.

실행: python3 pull_state_backups.py [--dry-run] [--max N]
종료값: 0 정상 · 1 서버를 못 읽음 · 2 받은 묶음 중 검증 실패가 있음 · 3 중복 실행
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from server_target import get_server  # 주소는 저장소 밖 server.conf 에 있다
SERVER = get_server()
SERVER_KEY = '/home/gmoh/.ssh/id_rsa'
REMOTE_ROOT = '/home/ubuntu/state_backups'

DEST = '/home/gmoh/backups/251229-state'
RECEIPT = os.path.join(DEST, 'receipt.json')
LOCK = os.path.join(DEST, '.lock')

# 묶음이 갖춰야 하는 파일과, 그 파일이 실제 상태파일인지 보는 최소 구조.
# 파일마다 키가 다르다 — 넷에 같은 키 목록을 강제하면 엉뚱한 파일도 통과한다.
REQUIRED = {
    'signal_state.json': ('meta',),
    'trade_state.json': ('members', 'schema_version'),
    'kis_trade_state.json': ('snapshots', 'schema_version'),
    'binance_state.json': ('strategies', 'schema_version'),
}
OPTIONAL = ('MANIFEST',)          # 있으면 같이 보존한다. 체크섬 파일이 아니라 ls 출력이다.

KEEP_DAYS = 90                    # 집 보관 기간 (묶음 날짜 기준. 파일 mtime 이 아니다)
KEEP_MIN = 7                      # 기간과 무관하게 지키는 최근 묶음 수
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
SSH_TIMEOUT = 120
_LOCK_FH = None                   # 잠금 파일 핸들 (회차가 끝날 때까지 산다)


def run_ssh(script: str) -> str:
    cmd = ['ssh', '-i', SERVER_KEY, '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes',
           SERVER, script]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT)
    if res.returncode != 0:
        raise RuntimeError(f'ssh rc={res.returncode}: {res.stderr.strip()[:200]}')
    return res.stdout


def remote_index() -> dict:
    """서버의 완성 묶음 목록. {날짜: {파일명: sha256}}

    -type d 라서 심볼릭 링크는 빠지고, 이름 규칙이 .staging_<날짜> 를 거른다.
    """
    script = (
        f"cd {REMOTE_ROOT} 2>/dev/null || exit 9; "
        "for d in $(find . -maxdepth 1 -mindepth 1 -type d "
        "-name '20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]' -printf '%f\\n' | sort); do "
        "for f in \"$d\"/*; do [ -f \"$f\" ] && [ ! -L \"$f\" ] && sha256sum \"$f\"; done; "
        "done"
    )
    out = run_ssh(script)
    index: dict = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        digest, path = parts[0], parts[1].strip()
        path = path[2:] if path.startswith('./') else path
        if '/' not in path:
            continue
        day, name = path.split('/', 1)
        if not DATE_RE.match(day) or '/' in name:
            continue
        index.setdefault(day, {})[name] = digest
    return index


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(path: str, expect: dict) -> list:
    """검증 실패 사유 목록. 빈 목록이면 통과."""
    problems = []
    for name, keys in REQUIRED.items():
        fp = os.path.join(path, name)
        if os.path.islink(fp) or not os.path.isfile(fp):
            problems.append(f'{name} 없음')
            continue
        if name in expect and sha256_of(fp) != expect[name]:
            problems.append(f'{name} 해시 불일치')
            continue
        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (ValueError, UnicodeDecodeError) as exc:
            problems.append(f'{name} JSON 깨짐 ({type(exc).__name__})')
            continue
        if not isinstance(data, dict) or not data:
            problems.append(f'{name} 최상위가 빈 객체이거나 사전이 아님')
            continue
        missing = [k for k in keys if k not in data]
        if missing:
            problems.append(f'{name} 필수 키 없음 {missing}')
    for name in OPTIONAL:
        fp = os.path.join(path, name)
        if os.path.exists(fp) and name in expect and sha256_of(fp) != expect[name]:
            problems.append(f'{name} 해시 불일치')
    return problems


def fetch_cohort(day: str, files: dict, dry: bool) -> tuple:
    """한 묶음을 받아 검증하고 원자적으로 공개한다. (성공여부, 메모)"""
    final = os.path.join(DEST, day)
    tmp = os.path.join(DEST, f'.incoming-{day}-{os.getpid()}')
    if dry:
        return True, f'받을 것 {sorted(files)}'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, mode=0o700)
    try:
        for name in sorted(files):
            if '/' in name or name.startswith('.'):
                continue
            # scp 는 여러 파일을 중괄호·공백으로 한 번에 받으면 protocol error 가 난다.
            # 하나씩 절대경로로 받는다.
            cmd = ['scp', '-i', SERVER_KEY, '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes',
                   f'{SERVER}:{REMOTE_ROOT}/{day}/{name}', os.path.join(tmp, name)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT)
            if res.returncode != 0:
                return False, f'scp 실패 {name}: {res.stderr.strip()[:120]}'
        problems = verify_dir(tmp, files)
        if problems:
            return False, '; '.join(problems)
        for name in os.listdir(tmp):
            os.chmod(os.path.join(tmp, name), 0o600)
        os.chmod(tmp, 0o700)
        os.rename(tmp, final)        # 검증을 통과한 것만 날짜 이름을 갖는다
        return True, 'ok'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare_existing(day: str, files: dict) -> list:
    """이미 집에 있는 묶음이 서버 것과 다른지 본다. 다르면 사유 목록."""
    local = os.path.join(DEST, day)
    diff = []
    for name, digest in sorted(files.items()):
        fp = os.path.join(local, name)
        if not os.path.isfile(fp):
            diff.append(f'{name} 집에 없음')
        elif sha256_of(fp) != digest:
            diff.append(f'{name} 내용 다름')
    return diff


def read_receipt() -> dict:
    try:
        with open(RECEIPT, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data.setdefault('cohorts', {})
            return data
    except (FileNotFoundError, ValueError):
        pass
    return {'cohorts': {}}


def write_receipt(data: dict) -> None:
    tmp = RECEIPT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=True)
    os.chmod(tmp, 0o600)
    os.replace(tmp, RECEIPT)


def prune(receipt: dict, healthy: bool) -> list:
    """묶음 날짜 기준 KEEP_DAYS 초과분 정리. 최근 KEEP_MIN 개는 언제나 남긴다."""
    if not healthy:
        return []                   # 최신 수신이 실패 중이면 과거를 지우지 않는다
    days = sorted(d for d in os.listdir(DEST) if DATE_RE.match(d)
                  and os.path.isdir(os.path.join(DEST, d)))
    cut = (datetime.now() - timedelta(days=KEEP_DAYS)).strftime('%Y-%m-%d')
    protect = set(days[-KEEP_MIN:])
    verified = receipt.get('last_restore_check', {}).get('cohort')
    if verified:
        protect.add(verified)
    removed = []
    for day in days:
        if day in protect or day >= cut:
            continue
        shutil.rmtree(os.path.join(DEST, day), ignore_errors=True)
        receipt['cohorts'].pop(day, None)
        removed.append(day)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description='서버 상태파일 백업을 집으로 받는다')
    ap.add_argument('--dry-run', action='store_true', help='받지 않고 무엇을 받을지만 본다')
    ap.add_argument('--max', type=int, default=0, help='이번 회차에 받을 묶음 수 상한 (0=제한 없음)')
    args = ap.parse_args()

    now = datetime.now()
    print(f"=== {now.strftime('%Y-%m-%d %H:%M KST')} 상태파일 백업 수신")
    os.makedirs(DEST, mode=0o700, exist_ok=True)
    os.chmod(DEST, 0o700)

    # 전역에 들고 있는다 — 지역 변수로 두면 회차가 끝나기 전에 수거돼 잠금이 풀릴 수 있다.
    global _LOCK_FH
    _LOCK_FH = open(LOCK, 'w')
    try:
        fcntl.flock(_LOCK_FH, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('이미 도는 회차가 있다 — 건너뛴다')
        return 3

    receipt = read_receipt()
    receipt['last_run'] = now.strftime('%Y-%m-%d %H:%M:%S')
    try:
        index = remote_index()
    except Exception as exc:                            # noqa: BLE001
        receipt['last_error'] = f'{type(exc).__name__}: {exc}'
        receipt['last_error_at'] = receipt['last_run']
        if not args.dry_run:
            write_receipt(receipt)
        print(f'서버를 못 읽었다 — {receipt["last_error"]}')
        return 1

    have = {d for d in os.listdir(DEST) if DATE_RE.match(d)
            and os.path.isdir(os.path.join(DEST, d))}
    todo = sorted(d for d in index if d not in have)
    if args.max > 0:
        todo = todo[-args.max:]
    print(f'서버 묶음 {len(index)}개 · 집 {len(have)}개 · 받을 것 {len(todo)}개')

    failed, mismatched, got = [], [], []
    for day in sorted(set(index) & have):
        diff = compare_existing(day, index[day])
        if diff:
            mismatched.append(f'{day}: {"; ".join(diff)}')

    for day in todo:
        ok, memo = fetch_cohort(day, index[day], args.dry_run)
        print(f'  {day} {"받음" if ok else "실패"} — {memo}')
        if ok and not args.dry_run:
            got.append(day)
            receipt['cohorts'][day] = {
                'received_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sha256': index[day],
            }
        elif not ok:
            failed.append(f'{day}: {memo}')

    if mismatched:
        print('이미 있는 묶음이 서버 것과 다르다 (덮어쓰지 않았다):')
        for line in mismatched:
            print('  ' + line)

    if args.dry_run:
        return 2 if failed else 0

    receipt['last_result'] = {
        'received': got, 'failed': failed, 'mismatched': mismatched,
    }
    if failed or mismatched:
        receipt['last_error'] = '; '.join(failed + mismatched)[:500]
        receipt['last_error_at'] = receipt['last_run']
    else:
        receipt['last_error'] = ''
        receipt['last_ok'] = receipt['last_run']
    removed = prune(receipt, healthy=not (failed or mismatched))
    if removed:
        print(f'보관 기간({KEEP_DAYS}일) 넘은 묶음 정리: {removed}')
    write_receipt(receipt)

    cohorts = sorted(d for d in os.listdir(DEST) if DATE_RE.match(d))
    print(f'집 보관 {len(cohorts)}개 (최신 {cohorts[-1] if cohorts else "없음"})')
    return 2 if (failed or mismatched) else 0


if __name__ == '__main__':
    sys.exit(main())
