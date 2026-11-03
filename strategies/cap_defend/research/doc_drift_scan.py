"""문서와 코드의 정보 불일치를 기계로 모은다 — 판정은 사람과 토론이 한다.

왜 만드나. 전략 파라미터는 trade/ops/check_params_drift.py 가 매일 대조하지만
그 표에는 자산배분·현금버퍼·성과 수치·파일 경로 칸이 없다. 문서 쪽 표기는
아무도 안 센다. 실제로 메모리 색인이 자산배분을 60/25/15 로 들고 있는 것이
2026-11-03 에 드러났다(라이브는 11-01 에 60/20/20 으로 바뀌었다).

무엇을 하나. 축마다 ⑴코드에서 정본값을 읽고 ⑵문서 전체에서 그 축을 말하는
줄을 모아 ⑶값이 다른 줄만 남긴다. 판정은 안 한다 — 역사 서술이라 달라도
되는 줄과 지금 참이라고 주장하는 줄을 가르는 것은 문맥이 필요하고, 그게
이 조사를 토론으로 넘기는 이유다.

append-only 규약 파일은 대상에서 뺀다(history.md, history/*.md). 과거 기록은
그 시점에 참이던 것을 적은 것이라 지금 값과 달라야 정상이다.

출력
  state/<job>/drift_scan.txt   축별 코드 정본값과 어긋나는 문서 줄
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# 대상에서 빼는 것 — append-only 감사 로그와 미추적 작업물.
SKIP_DIRS = ('/state/', '/logs/', '/data_research/', '/.git/', '/legacy/',
             '/history/', '/node_modules/')
SKIP_FILES = ('history.md',)

# 축 = (이름, 코드 정본 설명, 그 축을 말하는 줄을 찾는 정규식,
#        그 줄에 있으면 어긋남인 패턴)
AXES = [
    ('자산배분 비중',
     'recommend.py:188 STOCK_RATIO, COIN_RATIO, FUTURES_RATIO = 0.60, 0.20, 0.20',
     r'자산\s*배분|자산배분|alloc|배분',
     r'60/25/15|60/30/10|60/20/15|60/35/5|60/40/0'),

    ('반회전율 문턱 T1',
     'recommend_personal.py:582 REBAL_HT_THRESHOLD = 0.25',
     r'T1|반회전율|half_turnover|HT_THRESHOLD',
     r'T1[ =:]*1[0-9]pp|T1[ =:]*20pp|ht\s*>=\s*0\.1[0-9]|13pp'),

    ('상대부족 문턱 T3U',
     'recommend_personal.py:583 REBAL_T3U_REL = 0.20',
     r'T3U',
     r'T3U[^0-9]*25%|T3U_can[^0-9]*25'),

    ('현물 드리프트 문턱',
     'coin_live_engine.py:90 DRIFT_THRESHOLD_SPOT = 0.10',
     r'현물.*drift|spot.*drift|drift.*현물',
     r'drift_threshold\s*=?\s*0\.0[0-9]|현물.*drift.*0\.0[0-9]'),

    ('선물 드리프트 문턱',
     'auto_trade_binance.py DRIFT_THRESHOLD = 0.015',
     r'선물.*drift|fut.*drift|drift.*선물',
     r'선물.*drift[^0-9]*0\.1|fut.*drift[^0-9]*0\.1'),

    ('현물 스냅 구조',
     'coin_live_engine.py snap_interval_bars=217, n_snapshots=7',
     r'snap_interval|스냅.*현물|현물.*스냅|sn=|snap=',
     r'현물.*snap[^0-9]*(60|95|69)|snap_interval_bars\s*=\s*(60|95|69)'),

    ('선물 레버리지',
     'auto_trade_binance.py LEVERAGE_FLOOR/MID/CEILING = 2/3/4',
     r'레버리지|LEVERAGE|동적 L|L3|L4',
     r'레버리지.*5배|L3 고정|ISOLATED'),

    ('선물 마진모드',
     'futures_live_config.py MARGIN_TYPE = CROSSED',
     r'마진모드|MARGIN_TYPE|마진 모드',
     r'ISOLATED'),

    ('현금버퍼',
     'stock 7% / spot 1% / fut 1%',
     r'현금\s*버퍼|cash\s*buffer|cash_buffer|버퍼',
     r'6/1/1|2%\s*USDT|fut.*2%'),

    ('주식 유니버스',
     'stock_strategy_v25.py R7 = SPY QQQ VEA EEM GLD PDBC VNQ',
     r'R7|주식.*유니버스|universe.*stock',
     r'R7B|EWJ'),

    ('현물 단독 성과',
     'CLAUDE.md 정본: Cal 1.69 / CAGR +51.6% / MDD -30.7% (6.1yr, 앵커 7곳 평균)',
     r'현물.*(Cal|칼마)|spot.*Cal|코인 현물.*성과',
     r'Cal 4\.35|Cal 4\.63|CAGR \+?77\.6|CAGR \+?82|Cal 2\.17'),

    ('합성 성과',
     'CLAUDE.md 정본: 원화 기준 Cal 3.95 / CAGR 81.0% / MDD -20.6% / Sharpe 1.76 '
     '(달러 기준 3.82 / 72.4% / -19.1% / 1.63 은 같이 적어도 된다)',
     r'합성|alloc 60|3자산',
     r'Cal 5\.72|Cal 3\.88|Cal 4\.68|CAGR 71\.2|Cal 4\.07|CAGR 76\.4'),
]


def tracked_docs() -> list[str]:
    out = subprocess.run(['git', '-C', ROOT, 'ls-files', '*.md'],
                         capture_output=True, text=True, check=True).stdout
    docs = []
    for rel in out.splitlines():
        if not rel.strip():
            continue
        full = '/' + rel
        if any(s in full for s in SKIP_DIRS):
            continue
        if os.path.basename(rel) in SKIP_FILES:
            continue
        docs.append(rel)
    return sorted(docs)


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'state/20261103-doc-drift'
    os.makedirs(os.path.join(ROOT, outdir), exist_ok=True)
    path = os.path.join(ROOT, outdir, 'drift_scan.txt')

    docs = tracked_docs()
    lines: list[str] = []
    lines.append(f'대상 문서 {len(docs)}개 (추적 중인 md, history 계열 제외)')
    lines.append('')
    for d in docs:
        lines.append(f'  {d}')
    lines.append('')

    total = 0
    for name, canon, topic_re, bad_re in AXES:
        topic = re.compile(topic_re, re.I)
        bad = re.compile(bad_re, re.I)
        hits: list[tuple[str, int, str]] = []
        for rel in docs:
            full = os.path.join(ROOT, rel)
            try:
                with open(full, encoding='utf-8') as f:
                    for i, ln in enumerate(f, 1):
                        s = ln.rstrip()
                        if topic.search(s) and bad.search(s):
                            hits.append((rel, i, s.strip()))
            except Exception as exc:
                lines.append(f'  !! {rel} 읽기 실패: {exc}')
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
    lines.append(f'합계 {total}줄. 이 목록은 후보이지 판정이 아니다 — '
                 '역사 서술이라 달라도 되는 줄이 섞여 있다.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n저장: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
