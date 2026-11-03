"""코드 안 주석·독스트링·화면 문구가 코드와 어긋나는 자리를 모은다.

doc_drift_scan.py 는 .md 만 본다. 하지만 CLAUDE.md 가 직접 경고한다 —
"docstring, 로그 배너, HTML 문구, 클래스명, 매뉴얼의 버전 표기가 stale일 수
있다. 버전 표기보다 실제 파라미터와 실행 순서를 기준으로 판단한다."
실행되는 값은 trade/ops/check_params_drift.py 가 매일 09:15 에 대조하지만
사람이 읽는 글은 아무도 안 센다. 그 자리가 여기다.

실행되는 코드 줄은 보지 않는다 — 그쪽은 매일 도는 대조기의 몫이고, 정본
정의가 다른 축의 "어긋남" 패턴에 걸려 거짓 신호가 된다. 보는 것은 셋이다.
  - `#` 뒤의 주석
  - 삼중따옴표 독스트링 안
  - 운영자·화면에 나가는 문자열 리터럴

축 열둘은 doc_drift_scan.py 에서 그대로 가져온다 — 두 벌로 나누면 한쪽만
고치는 날이 온다.

출력
  state/<job>/comment_drift.txt
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doc_drift_scan import AXES, ROOT  # noqa: E402  # 같은 디렉토리

# 시험 자료는 제외한다 — 더미 입력({'isolated': False, 'leverage': 3} 같은 것)이
# 사람이 읽는 글이 아닌데 마진모드·레버리지 축에 통째로 걸린다. 시험이 든 값은
# 그 시험이 재현하려는 상황이지 지금 참이라는 주장이 아니다.
SKIP_DIRS = ('/state/', '/logs/', '/data_research/', '/.git/', '/legacy/',
             '/node_modules/', '/research/', '/tests/')
SKIP_NAME = re.compile(r'(^|/)test_[^/]*\.py$')

# 문자열 리터럴과 주석만 떼어낸다. 실행되는 식은 버린다.
COMMENT = re.compile(r'#(.*)$')
STRING = re.compile(r'''(?:"""|\'\'\'|"|\')([^"\']{4,})''')


def readable(line: str, in_doc: bool) -> str:
    """그 줄에서 사람이 읽는 부분만 남긴다. 없으면 빈 문자열."""
    if in_doc:
        return line
    parts = []
    m = COMMENT.search(line)
    if m:
        parts.append(m.group(1))
    for s in STRING.findall(line):
        parts.append(s)
    return ' '.join(parts)


def tracked_code() -> list[str]:
    out = subprocess.run(['git', '-C', ROOT, 'ls-files', '*.py'],
                         capture_output=True, text=True, check=True).stdout
    files = []
    for rel in out.splitlines():
        if not rel.strip():
            continue
        if any(s in '/' + rel for s in SKIP_DIRS) or SKIP_NAME.search(rel):
            continue
        files.append(rel)
    return sorted(files)


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'state/20261103-doc-drift'
    os.makedirs(os.path.join(ROOT, outdir), exist_ok=True)
    path = os.path.join(ROOT, outdir, 'comment_drift.txt')

    files = tracked_code()
    lines: list[str] = [f'대상 코드 {len(files)}개 (추적 중인 py, '
                        'legacy·research 제외)', '',
                        '실행되는 식은 안 본다 — 주석·독스트링·문자열만.', '']

    total = 0
    for name, canon, topic_re, bad_re in AXES:
        topic = re.compile(topic_re, re.I)
        bad = re.compile(bad_re, re.I)
        hits: list[tuple[str, int, str]] = []
        for rel in files:
            try:
                raw = open(os.path.join(ROOT, rel), encoding='utf-8').read()
            except Exception as exc:
                lines.append(f'  !! {rel} 읽기 실패: {exc}')
                continue
            in_doc = False
            for i, ln in enumerate(raw.splitlines(), 1):
                fences = ln.count('"""') + ln.count("'''")
                text = readable(ln, in_doc)
                if fences % 2 == 1:
                    in_doc = not in_doc
                    text = ln
                if not text:
                    continue
                if topic.search(text) and bad.search(text):
                    hits.append((rel, i, ln.strip()))
        lines.append('=' * 100)
        lines.append(f'[{name}]  코드 정본 = {canon}')
        lines.append('=' * 100)
        if not hits:
            lines.append('  어긋나는 줄 없음')
        else:
            total += len(hits)
            for rel, i, s in hits:
                if len(s) > 200:
                    s = s[:200] + ' …'
                lines.append(f'  {rel}:{i}')
                lines.append(f'      {s}')
        lines.append('')

    lines.append('=' * 100)
    lines.append(f'합계 {total}줄. 후보이지 판정이 아니다 — 변경 경위를 적은 '
                 '주석은 옛 값을 담는 것이 정상이다.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n저장: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
