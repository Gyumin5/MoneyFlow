#!/usr/bin/env python3
"""공개 전략 페이지가 정본과 어긋나지 않는지 검사한다 (2026-11-03).

무엇을 막나. strategies/cap_defend/strategy.html 은 서버 8080 으로 외부에 열려 있고
손으로 쓴 정적 HTML 이라 전략을 바꿔도 저절로 안 따라온다. 실제로 2026-11-03 에
서버본이 V15·V17, 집 사본이 V20 인 채 공개되고 있었다 — 카나리 SMA50, 배분 60/25/15,
주식 V17, 선물 d005 앙상블 5배, 드리프트 8pp 밴드가 전부 한 세대 이전 값이었다.

어떻게 보나. 페이지에 적힌 숫자를 코드의 정본 상수와 대조한다. 페이지 안에서
자기들끼리 맞는지가 아니라, 라이브가 읽는 값과 맞는지를 본다. 그래서 상수를 과거
값으로 되돌리면 이 검사가 깨진다.

확인하는 것 여덟.
  T1 자산배분 셋이 recommend.py 의 STOCK_RATIO/COIN_RATIO/FUTURES_RATIO 와 같다.
  T2 송금 트리거 둘이 recommend_personal.py 의 REBAL_HT_THRESHOLD/REBAL_T3U_REL 과 같다.
  T3 선물 레버리지 범위가 futures_live_config.py 의 LEVERAGE_MIN/MAX 와 같다.
  T4 세 갈래의 스냅샷 수·어긋나는 간격·교체 주기가 다 적혀 있고 서로소 설계가 유지된다.
  T5 폐기된 세대 표기가 본문에 없다 (V15~V20 배지, 60/25/15, SMA50, d005, 8pp 밴드).
  T6 주식 단독 성과를 싣지 않는다 — 재현 진입점이 없어 측정 조건을 못 댄다.
  T7 안내 페이지는 수치를 안 들고 본문으로 보낸다.
  T8 두 페이지 다 한글이 깨지지 않을 머리글을 갖춘다 (charset 첫 줄·viewport·CJK 폰트).

네트워크·실계좌를 안 탄다. 파일을 읽기만 한다.
"""
import math
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, 'strategies', 'cap_defend')
PAGE = os.path.join(CD, 'strategy.html')
GUIDE = os.path.join(CD, 'strategy_guide.html')


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def const(path, name):
    """소스에서 `name = <숫자>` 를 읽는다. import 하지 않는다(부작용 차단)."""
    src = read(path)
    m = re.search(r'^%s\s*=\s*([0-9.]+)' % re.escape(name), src, re.M)
    if m:
        return float(m.group(1))
    raise AssertionError('%s 에서 %s 를 못 찾았다' % (os.path.basename(path), name))


def ratios(path):
    """`STOCK_RATIO, COIN_RATIO, FUTURES_RATIO = a, b, c` 한 줄에서 셋을 읽는다."""
    src = read(path)
    m = re.search(
        r'^STOCK_RATIO,\s*COIN_RATIO,\s*FUTURES_RATIO\s*=\s*'
        r'([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)', src, re.M)
    if not m:
        raise AssertionError('recommend.py 에서 배분 세 상수를 못 찾았다')
    return tuple(float(g) for g in m.groups())


class TestPublicStrategyPage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.page = read(PAGE)
        cls.guide = read(GUIDE)
        # 표시용 본문만 본다 — style 블록의 색상·크기 숫자가 섞이지 않게 뺀다.
        cls.body = re.sub(r'<style.*?</style>', '', cls.page, flags=re.S)

    # T1
    def test_자산배분이_정본과_같다(self):
        s, c, f = ratios(os.path.join(CD, 'recommend.py'))
        for name, v in (('주식', s), ('코인현물', c), ('코인선물', f)):
            pct = '%d%%' % round(v * 100)
            self.assertIn(pct, self.body,
                          '%s 목표 비중 %s 가 페이지에 없다' % (name, pct))
        # 배분 표의 세 칸이 실제로 그 값들이다
        rows = re.findall(r'<tr><td>(주식|코인현물|코인선물)</td><td>(\d+)%</td>', self.body)
        self.assertEqual(len(rows), 3, '배분 표 세 줄을 못 찾았다')
        got = {k: int(v) for k, v in rows}
        self.assertEqual(got['주식'], round(s * 100))
        self.assertEqual(got['코인현물'], round(c * 100))
        self.assertEqual(got['코인선물'], round(f * 100))

    # T2
    def test_송금_트리거가_정본과_같다(self):
        rp = os.path.join(CD, 'recommend_personal.py')
        ht = const(rp, 'REBAL_HT_THRESHOLD')
        t3u = const(rp, 'REBAL_T3U_REL')
        self.assertIn('%d%%포인트' % round(ht * 100), self.body,
                      '반회전율 문턱이 페이지와 어긋난다')
        self.assertIn('%d%% 이상 모자라고' % round(t3u * 100), self.body,
                      'T3U 상대부족 문턱이 페이지와 어긋난다')

    # T3
    def test_선물_레버리지_범위가_정본과_같다(self):
        fc = os.path.join(CD, 'futures_live_config.py')
        lo = int(const(fc, 'LEVERAGE_MIN'))
        hi = int(const(fc, 'LEVERAGE_MAX'))
        self.assertIn('%d배에서 %d배' % (lo, hi), self.body,
                      '레버리지 범위가 페이지와 어긋난다')

    # T4
    def test_스냅샷_설계가_다_적혀있다(self):
        want = [
            ('코인현물', 7, 31, 217),
            ('코인선물', 5, 19, 95),
            ('주식', 3, 23, 69),
        ]
        for name, n, stag, interval in want:
            row = re.search(
                r'<tr><td>%s</td><td>(\d+)</td><td>(\d+)[^<]*</td><td>(\d+)[^<]*</td>'
                % name, self.body)
            if row is None:
                self.fail('%s 스냅샷 표 줄이 없다' % name)
            self.assertEqual(
                (int(row.group(1)), int(row.group(2)), int(row.group(3))),
                (n, stag, interval), '%s 스냅샷 수치가 어긋난다' % name)
        # 서로소 설계가 깨지지 않았는지 — 스냅샷 수 셋의 최대공약수가 1
        self.assertEqual(math.gcd(math.gcd(7, 5), 3), 1)

    # T5
    def test_폐기된_세대_표기가_없다(self):
        banned = [
            ('60/25/15', '옛 자산배분'),
            ('60:40', '옛 자산배분'),
            ('SMA50', '옛 카나리 이동평균'),
            ('d005', '옛 선물 앙상블 이름'),
            ('8pp', '옛 드리프트 밴드'),
            ('3-tranche Day', '달력 앵커 표기'),
        ]
        for token, why in banned:
            self.assertNotIn(token, self.body, '%s (%s) 가 아직 남아 있다' % (token, why))
        for gen in ('V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23'):
            self.assertNotIn(gen, self.body, '폐기된 세대 표기 %s 가 남아 있다' % gen)
        for gen in ('V24', 'V25'):
            self.assertIn(gen, self.body, '현행 세대 표기 %s 가 없다' % gen)

    # T6
    def test_주식_단독_성과를_싣지_않는다(self):
        m = re.search(r'<h2>백테스트 성과</h2>(.*?)<h2>', self.body, re.S)
        if m is None:
            self.fail('성과 절을 못 찾았다')
        table = m.group(1)
        self.assertNotIn('<td>주식 단독</td>', table,
                         '주식 단독 성과는 재현 진입점이 없어 싣지 않는다')
        self.assertIn('주식 단독 성과는 싣지 않는다', table,
                      '안 싣는 이유가 적혀 있어야 한다')
        for label in ('세 갈래 합성', '코인현물 단독', '코인선물 단독'):
            self.assertIn(label, table, '%s 줄이 없다' % label)
        # 측정 조건 칸이 비어 있지 않다
        self.assertIn('5.8년', table)
        self.assertIn('6.1년', table)

    # T7
    def test_안내_페이지는_본문으로_보낸다(self):
        self.assertIn('href="/strategy.html"', self.guide,
                      '안내 페이지에 본문 링크가 없다')
        gbody = re.sub(r'<style.*?</style>', '', self.guide, flags=re.S)
        # 수치를 들지 않는다 — 퍼센트 표기와 세대 표기가 없어야 한다
        self.assertNotRegex(gbody, r'\d+%', '안내 페이지가 수치를 들고 있다')
        for gen in ('V17', 'V20', 'V24', 'V25'):
            self.assertNotIn(gen, gbody, '안내 페이지에 세대 표기 %s 가 있다' % gen)

    # T8
    def test_한글_머리글을_갖춘다(self):
        for path, src in ((PAGE, self.page), (GUIDE, self.guide)):
            name = os.path.basename(path)
            head = src[:1024]
            self.assertIn('<meta charset="utf-8">', head,
                          '%s: charset 이 앞 1024바이트 안에 없다' % name)
            self.assertIn('<html lang="ko">', src[:200], '%s: lang 이 없다' % name)
            self.assertIn('width=device-width', src, '%s: viewport 가 없다' % name)
            self.assertIn("'Noto Sans CJK KR'", src,
                          '%s: 설치된 한글 폰트 이름이 없다' % name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
