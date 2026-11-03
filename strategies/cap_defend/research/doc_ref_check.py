"""문서가 가리키는 파일 경로와 코드 심볼이 실제로 있는지 전수 확인한다.

doc_drift_scan.py 와 다른 축이다. 그쪽은 "값이 맞나" 를 보고 이쪽은 "가리키는
것이 있나" 를 본다. 값 불일치는 문맥을 읽어야 가려지지만 존재 여부는 기계가
끝까지 판정할 수 있다 — 없는 파일을 가리키는 줄은 예외 없이 고칠 자리다.

계기는 2026-11-03 조사다. 세션 메모리 색인이 V25_OPERATION_MANUAL.md 를
가리키고 있었는데 그 파일이 저장소에 없었다. 같은 자리가 더 있는지 세어 본다.

보는 것 둘.
  1. 문서 안의 저장소 상대경로 — 그 파일이 있나.
  2. 문서가 "<파일>:<줄번호>" 또는 "<파일> 의 <심볼>" 로 코드를 가리킬 때
     그 심볼이 그 파일에 있나.

세지 않는 것 — 외부 URL, 서버 쪽 경로(/home/ubuntu/...), 상태파일처럼 실행 중에
생기는 것, git 미추적 산출물 디렉토리(state/ logs/ data/).

출력
  state/<job>/ref_check.txt
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

SKIP_DIRS = ('/state/', '/logs/', '/data_research/', '/.git/', '/node_modules/')

# 실행 중에 생기거나 저장소 밖에 있는 것 — 없다고 해서 결함이 아니다.
RUNTIME = re.compile(
    r'^(/|~|\.\./)'                       # 절대경로·홈·상위
    r'|^(state|logs|data|data_research)/'  # 미추적 산출물
    r'|\.(json|log|csv|bak|pid|lock)$'     # 실행 산출물
    r'|\*',                                # 글로브
    re.I)

# 이름 자리에 모양만 적어 둔 것 — 가리키는 파일이 애초에 하나가 아니다.
PLACEHOLDER = re.compile(r'YYYY|MM-DD|날짜|주제|<|>|\{', re.I)

# 문서 안의 경로처럼 보이는 토막. 확장자가 있는 것만 본다.
# 앞에 글로브(*)나 경로 글자가 붙어 있으면 더 긴 이름에서 잘려 나온 토막이라 버린다
# (`run_current_*_backtest.py` 에서 `_backtest.py` 만 떼어 오던 자리).
PATH_RE = re.compile(r'(?<![\w/.*-])((?:[\w.-]+/)*[\w.-]+\.(?:py|md|sh|txt|html))')

# "<파일>.py 의 <심볼>" / "<파일>.py:<심볼>" 둘만 본다. 공백 구분까지 받으면
# 한국어 설명문의 바로 뒤 낱말("API 키 템플릿" 의 API)이 전부 심볼로 잡힌다.
SYM_RE = re.compile(
    r'([\w./-]+\.py)\s*(?:의\s|:)\s*([A-Za-z_][A-Za-z0-9_]{2,})\s*\(?\)?')


def tracked(pattern: str) -> list[str]:
    out = subprocess.run(['git', '-C', ROOT, 'ls-files', pattern],
                         capture_output=True, text=True, check=True).stdout
    return [r for r in out.splitlines() if r.strip()]


def resolve(ref: str, from_doc: str) -> str | None:
    """문서가 가리키는 경로를 저장소 기준 실제 경로로 푼다. 못 찾으면 None.

    존재 판정은 git 추적 목록이 아니라 실제 디스크로 한다 — `config/settings.py`
    처럼 미추적이지만 실재하는 파일을 "없다" 고 잡던 자리다.
    """
    cands = [ref, os.path.join(os.path.dirname(from_doc), ref)]
    for c in cands:
        c = os.path.normpath(c)
        if os.path.exists(os.path.join(ROOT, c)):
            return c
    # 파일명만 적은 경우 저장소 어디든 같은 이름이 있으면 맞은 것으로 본다.
    return DISK_BY_NAME.get(os.path.basename(ref))


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'state/20261103-doc-drift'
    os.makedirs(os.path.join(ROOT, outdir), exist_ok=True)
    path = os.path.join(ROOT, outdir, 'ref_check.txt')

    docs = [d for d in tracked('*.md')
            if not any(s in '/' + d for s in SKIP_DIRS)
            and not d.startswith('history/')
            and os.path.basename(d) != 'history.md']

    lines: list[str] = [f'대상 문서 {len(docs)}개', '']

    missing_path: list[tuple[str, int, str, str]] = []
    missing_sym: list[tuple[str, int, str, str, str]] = []
    sym_cache: dict[str, str] = {}

    for rel in docs:
        full = os.path.join(ROOT, rel)
        try:
            text = open(full, encoding='utf-8').read().splitlines()
        except Exception as exc:
            lines.append(f'!! {rel} 읽기 실패: {exc}')
            continue
        for i, ln in enumerate(text, 1):
            s = ln.strip()
            if s.startswith('http') or '://' in s:
                continue
            # 그 줄이 스스로 "저장소 밖·미추적·폐기" 라고 밝히면 없는 게 정상이다.
            if re.search(r'미추적|서버|폐기|템플릿|예시', s):
                continue
            for ref in set(PATH_RE.findall(s)):
                if RUNTIME.search(ref) or PLACEHOLDER.search(ref):
                    continue
                if resolve(ref, rel) is None:
                    missing_path.append((rel, i, ref, s[:150]))
            for pyfile, sym in SYM_RE.findall(s):
                if RUNTIME.search(pyfile):
                    continue
                target = resolve(pyfile, rel)
                if target is None or not target.endswith('.py'):
                    continue
                if target not in sym_cache:
                    try:
                        sym_cache[target] = open(
                            os.path.join(ROOT, target), encoding='utf-8').read()
                    except Exception:
                        sym_cache[target] = ''
                body = sym_cache[target]
                if not body:
                    continue
                if not re.search(r'(?<![\w.])' + re.escape(sym) + r'(?![\w])', body):
                    missing_sym.append((rel, i, pyfile, sym, s[:150]))

    lines.append('=' * 100)
    lines.append(f'[없는 파일을 가리키는 줄]  {len(missing_path)}건')
    lines.append('=' * 100)
    if not missing_path:
        lines.append('  없음')
    for rel, i, ref, s in missing_path:
        lines.append(f'  {rel}:{i}  →  {ref}')
        lines.append(f'      {s}')

    lines.append('')
    lines.append('=' * 100)
    lines.append(f'[그 파일에 없는 심볼을 가리키는 줄]  {len(missing_sym)}건')
    lines.append('=' * 100)
    if not missing_sym:
        lines.append('  없음')
    for rel, i, pyfile, sym, s in missing_sym:
        lines.append(f'  {rel}:{i}  →  {pyfile} 의 {sym}')
        lines.append(f'      {s}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n저장: {path}')
    return 0


def disk_index() -> dict[str, str]:
    """저장소 안 실제 파일의 이름 → 상대경로. 같은 이름이 여럿이면 첫 것."""
    idx: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        if any(s.strip('/') in rel.split(os.sep)
               for s in SKIP_DIRS) or rel.startswith('.git'):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if d not in ('.git', 'node_modules', 'state', 'logs')]
        for fn in filenames:
            idx.setdefault(fn, os.path.normpath(os.path.join(rel, fn)))
    return idx


DISK_BY_NAME = disk_index()

if __name__ == '__main__':
    sys.exit(main())
