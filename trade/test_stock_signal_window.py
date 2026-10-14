"""주식 신호 신선도를 재는 기준 — 달력 날짜가 아니라 그 시각의 기대 신호일.

계기는 2026-10-15 00:59 KST 에 손으로 신호 대조를 돌린 일이다. 서버 신호는 매일
09:15 에 쓰이므로 자정~09:15 사이에는 어제 것이 최신인데, 지연을 달력 날짜로 재던
코드가 이걸 1일 지연으로 보고 비교를 거절했다. 거절 판정은 영수증에 그대로 적혀
서버로 올라가고, 서버 감시는 그걸 이상으로 읽는다.

여기서 보는 것 둘.
  1. 크론 시각 전에는 어제가 기대일이다 (거짓 거절을 안 만든다)
  2. 크론 시각이 지나면 오늘이 기대일이다 (얼어붙은 신호를 그대로 잡는다)
"""
import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops'))

from signal_parity_daily import (STOCK_SIGNAL_HOUR, STOCK_SIGNAL_MIN,  # noqa: E402
                                 STOCK_UPDATE_LAG_ALLOW,
                                 expected_stock_signal_day)

DAY = datetime(2026, 10, 15)


def at(hour, minute=0):
    return DAY.replace(hour=hour, minute=minute)


def lag(now, updated_day):
    return (expected_stock_signal_day(now) - updated_day).days


class TestExpectedDay(unittest.TestCase):

    def test_크론_시각_전에는_어제가_기대일이다(self):
        for now in (at(0, 0), at(0, 59), at(5), at(9, 14)):
            self.assertEqual(expected_stock_signal_day(now),
                             DAY.date() - timedelta(days=1), now)

    def test_크론_시각부터는_오늘이_기대일이다(self):
        for now in (at(STOCK_SIGNAL_HOUR, STOCK_SIGNAL_MIN), at(10), at(23, 59)):
            self.assertEqual(expected_stock_signal_day(now), DAY.date(), now)


class TestLagVerdict(unittest.TestCase):
    """지연값이 허용치를 넘는가 — 실제 판정과 같은 식으로 본다."""

    def test_새벽에_어제_신호면_통과한다(self):
        # 2026-10-15 00:59 에 어제(10-14) 09:15 신호. 사고가 난 바로 그 자리다.
        self.assertLessEqual(lag(at(0, 59), DAY.date() - timedelta(days=1)),
                             STOCK_UPDATE_LAG_ALLOW)

    def test_크론_시각_뒤에_어제_신호면_걸린다(self):
        # 같은 신호라도 09:15 이 지나면 오늘 크론이 죽은 것이다.
        self.assertGreater(lag(at(10, 0), DAY.date() - timedelta(days=1)),
                           STOCK_UPDATE_LAG_ALLOW)

    def test_이틀_전_신호는_새벽에도_걸린다(self):
        self.assertGreater(lag(at(0, 59), DAY.date() - timedelta(days=2)),
                           STOCK_UPDATE_LAG_ALLOW)

    def test_오늘_신호는_언제_봐도_통과한다(self):
        for now in (at(9, 16), at(10), at(23, 59)):
            self.assertLessEqual(lag(now, DAY.date()), STOCK_UPDATE_LAG_ALLOW, now)


if __name__ == '__main__':
    unittest.main(verbosity=2)
