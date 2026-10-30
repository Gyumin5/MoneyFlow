#!/usr/bin/env python3
"""서버 경보 중계 검증 (2026-10-30).

확인하는 것 다섯.
  1. 적기 — 한 줄이 jsonl 로 쌓이고 식별자·보낸 여부·글이 그대로 남는다.
  2. 솎기 — 보관 줄 수를 넘기면 오래된 것부터 버리고 최근 것을 남긴다.
  3. 고르기 — 이미 본 것, 24시간보다 낡은 것, 시각이 깨진 것은 안 넣는다.
  4. 차례와 상한 — 오래된 순으로 넣고 한 회차 최대 건수를 넘기지 않는다.
  5. 깨진 줄 — 반쯤 쓰인 줄이 섞여도 나머지는 읽는다.
  6. 글 — 운영자가 이미 받았는지가 주입문에 드러난다.

망도 ssh 도 타지 않는다. 파일과 순수 함수만 본다.
"""
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta

OPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(OPS, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sal = _load('server_alert_log')
rel = _load('server_alert_relay')

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


# 1. 적기
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'alerts.jsonl')
    ok = sal.record('parity_watch', '주식 불일치다', sent=True, path=path)
    check(ok, '1: 적기가 실패했다')
    rows = [json.loads(x) for x in open(path, encoding='utf-8').read().splitlines()]
    check(len(rows) == 1, f'1: 줄이 1개가 아니다 ({len(rows)})')
    check(rows[0]['text'] == '주식 불일치다', '1: 글이 그대로 안 남았다')
    check(rows[0]['source'] == 'parity_watch', '1: 출처가 안 남았다')
    check(rows[0]['sent'] is True, '1: 보낸 여부가 안 남았다')
    check(len(rows[0]['id']) == 12, '1: 식별자 길이가 12가 아니다')

    # 전송 실패도 적힌다 — 운영자가 못 본 경보일수록 세션이 알아야 한다.
    sal.record('parity_watch', '못 보낸 글', sent=False, path=path)
    rows = [json.loads(x) for x in open(path, encoding='utf-8').read().splitlines()]
    check(rows[1]['sent'] is False, '1: 전송 실패가 sent=False 로 안 남았다')

# 2. 솎기
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'alerts.jsonl')
    keep = sal.KEEP_LINES
    for i in range(keep + 20):
        sal.record('t', f'줄{i}', path=path)
    lines = open(path, encoding='utf-8').read().splitlines()
    check(len(lines) == keep, f'2: 솎은 뒤 줄 수가 {len(lines)} 다 (기대 {keep})')
    check(json.loads(lines[-1])['text'] == f'줄{keep + 19}', '2: 최근 줄이 안 남았다')
    check(json.loads(lines[0])['text'] == '줄20', '2: 오래된 것부터 안 버렸다')

# 3·4. 고르기와 차례
now = datetime(2026, 10, 30, 12, 0, 0)


def row(rid, ts, text='글', source='parity_watch', sent=True):
    return {'id': rid, 'ts': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'text': text, 'source': source, 'sent': sent}


recs = [
    row('old1', now - timedelta(hours=30)),          # 너무 낡음
    row('seen1', now - timedelta(hours=1)),          # 이미 봄
    row('b', now - timedelta(minutes=10)),
    row('a', now - timedelta(minutes=50)),
    {'id': 'bad', 'ts': 'not-a-time', 'text': '글'},  # 시각 깨짐
]
picked = rel.pick_new(recs, {'seen1'}, now)
ids = [r['id'] for r in picked]
check(ids == ['a', 'b'], f'3·4: 고른 것이 {ids} 다 (기대 a,b — 오래된 순)')

many = [row(f'r{i}', now - timedelta(minutes=i)) for i in range(20)]
check(len(rel.pick_new(many, set(), now)) == rel.MAX_PER_RUN,
      f'4: 한 회차 상한 {rel.MAX_PER_RUN} 을 안 지켰다')

# 5. 깨진 줄
raw = ('{"id":"x","ts":"2026-10-30 11:00:00","text":"정상","source":"s"}\n'
       '{"id":"y","ts":"2026-10-30 11:0\n'          # 반쯤 쓰인 줄
       '\n'
       '{"id":"z","ts":"2026-10-30 11:30:00","text":"정상2","source":"s"}\n')
parsed = rel.parse_lines(raw)
check([r['id'] for r in parsed] == ['x', 'z'],
      f'5: 깨진 줄을 거르고 남은 것이 {[r["id"] for r in parsed]} 다')

# 6. 주입문
sent_text = rel.compose(row('a', now, '주식 불일치', sent=True))
unsent_text = rel.compose(row('a', now, '주식 불일치', sent=False))
check('주식 불일치' in sent_text, '6: 본문이 안 들어갔다')
check('parity_watch' in sent_text, '6: 출처가 안 들어갔다')
check('이미 같은 글을 받았다' in sent_text, '6: 운영자가 받았다는 표시가 없다')
check('못 봤다' in unsent_text, '6: 운영자가 못 봤다는 표시가 없다')

if fails:
    print(f'실패 {len(fails)}건')
    for f in fails:
        print('  - ' + f)
    sys.exit(1)
print('서버 경보 중계 검증 통과 — 적기·솎기·고르기·차례와 상한·깨진 줄·주입문 6갈래')
