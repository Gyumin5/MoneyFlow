"""현물 스냅 상태 검사 — 무엇을 비교 대상으로 삼아야 하는가.

선물 검사와 구조는 같고(trade/test_fut_snap_state.py 가 구조를 본다) 여기서는 현물에만
있는 두 축을 본다.
  1. 현금 키 표기가 CASH / Cash 로 갈리는 경계
  2. 리필이 스냅만 고치고 저장된 합산은 안 고치는 날

2번이 이 파일의 이유다. 리필은 스냅샷을 고친 뒤 재계산한 합산을 executor 에게
돌려주지만, 멤버 상태의 last_combined 는 리필 전 값 그대로 남긴다(현물·선물 공통).
그래서 저장된 last_combined 를 비교 대상으로 삼으면 리필이 실제로 코인을 바꾼 날마다
거짓 경보가 난다. 체결에 쓰인 목표를 비교 대상으로 삼아야 그 날도 조용하다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops'))

from signal_parity_daily import (_fut_combined, self_check_snap,  # noqa: E402
                                 snap_state_diffs)

CAP = 1.0 / 3.0
N = 3


def member(snaps, combined):
    return {'snapshots': snaps, 'last_combined': combined}


def even(*coins):
    """코인들에 1/3 씩, 남는 몫은 CASH."""
    snap = {c: CAP for c in coins}
    snap['CASH'] = 1.0 - CAP * len(coins)
    return snap


def average(snaps):
    agg = {}
    for snap in snaps:
        for coin, w in snap.items():
            key = 'CASH' if coin.upper() == 'CASH' else coin
            agg[key] = agg.get(key, 0.0) + w / len(snaps)
    return agg


class TestCashKeyBoundary(unittest.TestCase):
    """같은 값이면 표기가 달라도 안 잡고, 한 딕셔너리에 섞이면 잡는다."""

    def setUp(self):
        self.snaps = [even('BTC', 'ETH')] * N
        self.avg = average(self.snaps)

    def test_소문자_표기여도_값이_같으면_조용하다(self):
        target = {('Cash' if k == 'CASH' else k): v for k, v in self.avg.items()}
        self.assertEqual(
            snap_state_diffs(member(self.snaps, {}), N, CAP,
                             combined=target, combined_label='체결 목표'), [])

    def test_ts_메타키는_비교에서_빠진다(self):
        target = dict(self.avg)
        target['_ts'] = '2026-10-15T00:00:00Z'
        self.assertEqual(
            snap_state_diffs(member(self.snaps, {}), N, CAP,
                             combined=target, combined_label='체결 목표'), [])

    def test_한_딕셔너리에_CASH_와_Cash_가_같이_있으면_잡는다(self):
        target = dict(self.avg)
        half = target.pop('CASH') / 2
        target['CASH'] = half
        target['Cash'] = half
        diffs = snap_state_diffs(member(self.snaps, {}), N, CAP,
                                 combined=target, combined_label='체결 목표')
        self.assertTrue(any('현금 키가 둘 이상' in d for d in diffs), diffs)

    def test_표기만_다르고_값이_틀리면_그대로_잡는다(self):
        target = {('Cash' if k == 'CASH' else k): v for k, v in self.avg.items()}
        target['BTC'] += 0.05
        self.assertTrue(snap_state_diffs(member(self.snaps, {}), N, CAP,
                                         combined=target, combined_label='체결 목표'))


class TestRefillDay(unittest.TestCase):
    """리필이 돈 날 — 무엇을 비교하느냐로 판정이 갈린다."""

    def setUp(self):
        before = [even('BTC', 'ETH')] * N
        self.before_combined = average(before)
        # 리필: 스냅 0 의 ETH 를 FRESH 로 갈았다. 스냅 자체는 멀쩡하다.
        self.after_snaps = [even('BTC', 'FRESH')] + [even('BTC', 'ETH')] * (N - 1)
        self.executed = average(self.after_snaps)

    def test_저장된_옛_합산으로_비교하면_거짓_경보가_난다(self):
        diffs = snap_state_diffs(member(self.after_snaps, self.before_combined), N, CAP)
        self.assertTrue(diffs, '리필 전 합산은 리필 뒤 스냅과 어긋나야 한다')

    def test_체결에_쓰인_목표로_비교하면_조용하다(self):
        diffs = snap_state_diffs(member(self.after_snaps, self.before_combined), N, CAP,
                                 combined=self.executed, combined_label='체결 목표')
        self.assertEqual(diffs, [])

    def test_체결_목표가_정말_틀리면_그날도_잡는다(self):
        wrong = dict(self.executed)
        wrong['FRESH'] = wrong.get('FRESH', 0.0) + 0.02
        wrong['BTC'] -= 0.02
        self.assertTrue(snap_state_diffs(member(self.after_snaps, self.before_combined),
                                         N, CAP, combined=wrong, combined_label='체결 목표'))


class TestSelfCheck(unittest.TestCase):
    """양성 대조는 비교 대상을 밖에서 줬을 때도 물어야 한다."""

    def test_비교대상을_준_경우에도_흔들기를_잡는다(self):
        snaps = [even('BTC', 'ETH')] * N
        ok, why = self_check_snap(member(snaps, {}), N, CAP,
                                  combined=average(snaps), combined_label='체결 목표')
        self.assertTrue(ok, why)

    def test_비교대상이_비면_양성_대조를_못_했다고_말한다(self):
        snaps = [even('BTC', 'ETH')] * N
        ok, why = self_check_snap(member(snaps, {}), N, CAP,
                                  combined={}, combined_label='체결 목표')
        self.assertFalse(ok)
        self.assertIn('비었다', why)

    def test_ts_만_있는_목표를_흔들려_들지_않는다(self):
        snaps = [even('BTC', 'ETH')] * N
        target = dict(average(snaps))
        target['_ts'] = '2026-10-15T00:00:00Z'
        ok, why = self_check_snap(member(snaps, {}), N, CAP,
                                  combined=target, combined_label='체결 목표')
        self.assertTrue(ok, why)
        self.assertNotIn('_ts', why)


class TestFutCombinedPick(unittest.TestCase):
    """선물은 멤버가 하나일 때만 앙상블 합산을 비교 대상으로 쓴다."""

    def test_전략이_하나면_체결_목표를_쓴다(self):
        state = {'strategies': {'D_SMA42': {}}, 'last_target': {'BTC': 1.0}}
        combined, label = _fut_combined(state, {'last_combined': {'ETH': 1.0}})
        self.assertEqual(label, 'last_target')
        self.assertEqual(combined, {'BTC': 1.0})

    def test_전략이_둘이면_멤버_자기_합산으로_되돌린다(self):
        state = {'strategies': {'A': {}, 'B': {}}, 'last_target': {'BTC': 1.0}}
        combined, label = _fut_combined(state, {'last_combined': {'ETH': 1.0}})
        self.assertEqual(label, 'last_combined')
        self.assertEqual(combined, {'ETH': 1.0})

    def test_체결_목표가_없으면_멤버_자기_합산으로_되돌린다(self):
        state = {'strategies': {'D_SMA42': {}}}
        combined, label = _fut_combined(state, {'last_combined': {'ETH': 1.0}})
        self.assertEqual(label, 'last_combined')
        self.assertEqual(combined, {'ETH': 1.0})


if __name__ == '__main__':
    unittest.main(verbosity=2)
