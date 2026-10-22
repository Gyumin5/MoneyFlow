"""주식 실행 결과 독립 대조 — 주문과 잔고가 서로를 설명하는가.

금액으로는 못 본다. 로그의 총자산과 목표액은 체결 시점 환율·현재가로 잰 값이라
바깥에서 금액을 다시 세면 멀쩡한 날도 어긋나 보인다(2026-10-13 실측: 목표
<금액> 대 총자산 <금액> = 0.984, 버퍼 7% 가 뜻하는 0.93 이 아니다).
주식 수에는 그 미지수가 없다.

그래서 보는 축 둘.
  1. 그날 끝 잔고 = 전날 끝 잔고 + 그날 낸 주문의 부호 합
  2. 보유 종목이 목표 종목 안에 있다 (반대 방향은 최소주문 때문에 정상일 수 있다)

1번은 서버 로그 전체(2026-09-24~10-14)에서 14/14일 성립을 먼저 재고 넣었다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops'))

from signal_parity_daily import (merged_stock_target,  # noqa: E402
                                 parse_stock_exec_log, self_check_stock_exec,
                                 stock_exec_diffs, stock_target_gap)

LOG = """[2026-10-13 23:35:01] [aaaa] 주식 executor 시작
[2026-10-13 23:35:03] [aaaa] 주식 보고 (silent): target={'SPY': 0.3333, 'EEM': 0.3333, 'VNQ': 0.3333, 'Cash': 0.0} max_diff=0.1 mode=LIVE
[2026-10-13 23:35:04] [aaaa] 주문 sell SPY 3주
[2026-10-13 23:35:05] [aaaa] 주문 buy VNQ 5주
[2026-10-13 23:35:06] [aaaa] 주식 executor 완료 (5.0s) | 총자산: <금액> | 잔고: {'SPY': 7, 'EEM': 10, 'VNQ': 5}
[2026-10-14 23:35:01] [bbbb] 주식 executor 시작
[2026-10-14 23:35:06] [bbbb] 주식 executor 완료 (5.0s) | 총자산: <금액> | 잔고: {'SPY': 7, 'EEM': 10, 'VNQ': 5}
"""


class TestParse(unittest.TestCase):

    def setUp(self):
        self.days = parse_stock_exec_log(LOG)

    def test_날짜별로_접는다(self):
        self.assertEqual(sorted(self.days), ['2026-10-13', '2026-10-14'])

    def test_주문은_부호가_붙는다(self):
        self.assertEqual(self.days['2026-10-13']['orders'],
                         [('SPY', -3), ('VNQ', 5)])

    def test_끝_잔고를_읽는다(self):
        self.assertEqual(self.days['2026-10-14']['balance'],
                         {'SPY': 7, 'EEM': 10, 'VNQ': 5})

    def test_목표에서_현금은_빼고_읽는다(self):
        self.assertEqual(self.days['2026-10-13']['target'], ['EEM', 'SPY', 'VNQ'])

    def test_주문이_없는_날은_빈_목록이다(self):
        self.assertEqual(self.days['2026-10-14']['orders'], [])

    def test_같은_날_두_번_돌면_주문을_합치고_잔고는_뒤엣것이_끝이다(self):
        twice = LOG + ("[2026-10-14 23:40:01] [cccc] 주문 buy EEM 2주\n"
                       "[2026-10-14 23:40:02] [cccc] 주식 executor 완료 (1.0s) | "
                       "잔고: {'SPY': 7, 'EEM': 12, 'VNQ': 5}\n")
        day = parse_stock_exec_log(twice)['2026-10-14']
        self.assertEqual(day['orders'], [('EEM', 2)])
        self.assertEqual(day['balance']['EEM'], 12)


class TestIdentity(unittest.TestCase):
    """그날 끝 잔고 = 전날 끝 잔고 + 그날 낸 주문의 부호 합."""

    def test_맞으면_조용하다(self):
        self.assertEqual(stock_exec_diffs({'SPY': 10, 'EEM': 10},
                                          {'SPY': 7, 'EEM': 10, 'VNQ': 5},
                                          [('SPY', -3), ('VNQ', 5)]), [])

    def test_부분체결은_어긋남으로_드러난다(self):
        diffs = stock_exec_diffs({'SPY': 10}, {'SPY': 9}, [('SPY', -3)])
        self.assertTrue(any('SPY' in d for d in diffs), diffs)

    def test_주문_없이_잔고가_바뀌면_잡는다(self):
        self.assertTrue(stock_exec_diffs({'SPY': 10}, {'SPY': 11}, []))

    def test_전량매도로_0_이_된_종목은_사라진_것으로_본다(self):
        self.assertEqual(stock_exec_diffs({'SPY': 10, 'EEM': 5},
                                          {'EEM': 5}, [('SPY', -10)]), [])

    def test_주문도_잔고도_그대로면_조용하다(self):
        self.assertEqual(stock_exec_diffs({'SPY': 7}, {'SPY': 7}, []), [])


class TestTargetSet(unittest.TestCase):
    """방향이 다르다 — 목표 밖 보유는 판정, 목표인데 미보유는 참고."""

    def test_목표_밖_종목을_들고_있으면_잡는다(self):
        diffs = stock_exec_diffs({'SPY': 10}, {'SPY': 10, 'TLT': 4},
                                 [('TLT', 4)], ['SPY', 'EEM'])
        self.assertTrue(any('목표에 없는 종목' in d for d in diffs), diffs)

    def test_목표인데_안_들고_있으면_판정에_안_넣는다(self):
        self.assertEqual(stock_exec_diffs({'SPY': 10}, {'SPY': 10}, [],
                                          ['SPY', 'EEM', 'VNQ']), [])

    def test_그_차이는_참고로_돌려준다(self):
        self.assertEqual(stock_target_gap({'SPY': 10}, ['SPY', 'EEM', 'VNQ']),
                         ['EEM', 'VNQ'])

    def test_0주_보유는_안_들고_있는_것으로_센다(self):
        self.assertEqual(stock_target_gap({'SPY': 0, 'EEM': 3}, ['SPY', 'EEM']),
                         ['SPY'])


class TestMergedTarget(unittest.TestCase):

    def test_스냅_셋의_픽을_합친다(self):
        state = {'snapshots': {'0': {'picks': ['SPY', 'EEM']},
                               '1': {'picks': ['SPY', 'VNQ']},
                               '2': {'picks': ['Cash']}}}
        self.assertEqual(merged_stock_target(state), ['EEM', 'SPY', 'VNQ'])

    def test_스냅이_없으면_빈_목록이다(self):
        self.assertEqual(merged_stock_target({}), [])


class TestSelfCheckSuite(unittest.TestCase):

    def test_모든_대조가_통과한다(self):
        for name, ok, why in self_check_stock_exec():
            self.assertTrue(ok, f'{name}: {why}')

    def test_깨끗한_합성_실행이_먼저_통과한다(self):
        name, ok, why = self_check_stock_exec()[0]
        self.assertEqual(name, '깨끗한 합성 실행')
        self.assertTrue(ok, why)

    def test_집합_규칙을_끄면_그_대조가_실패한다(self):
        import signal_parity_daily as mod
        original = mod.stock_exec_diffs

        def blind(*a, **kw):
            return [d for d in original(*a, **kw) if '목표에 없는 종목' not in d]

        mod.stock_exec_diffs = blind
        try:
            results = mod.self_check_stock_exec()
        finally:
            mod.stock_exec_diffs = original
        self.assertIn('목표 밖 종목 보유', [n for n, ok, _ in results if not ok])


if __name__ == '__main__':
    unittest.main(verbosity=2)
