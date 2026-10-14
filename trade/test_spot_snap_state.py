"""현물 스냅 상태 검사 — 무엇을 비교 대상으로 삼고, 어디까지를 비교불능으로 보나.

선물 검사와 구조는 같고(trade/test_fut_snap_state.py 가 구조를 본다) 여기서는 현물에만
있는 축을 본다.
  1. 현금 키 표기 경계 — 엔진 안은 CASH, 현물 체결 목표는 Cash
  2. 리필이 스냅만 고치고 저장된 합산은 안 고치는 날
  3. 멤버가 하나가 아니거나 체결 목표가 없는 구성 — 조용히 대체하지 않고 비교불능

2번이 이 파일의 이유다. 리필은 스냅샷을 고친 뒤 재계산한 합산을 executor 에게
돌려주지만, 멤버 상태의 last_combined 는 리필 전 값 그대로 남긴다(현물·선물 공통).
그래서 저장된 last_combined 를 비교 대상으로 삼으면 리필이 실제로 코인을 바꾼 날마다
거짓 경보가 난다. 체결에 쓰인 목표를 비교 대상으로 삼아야 그 날도 조용하다.

3번은 2026-10-14 토론(run-20261014T154635Z)의 필수 수정이다. 멤버가 둘인데 비교
대상을 멤버 자기 합산으로 되돌리면, 검사 이름은 그대로인 채 보는 것이 2번의 거짓
경보 값으로 약해진다. 그건 통과도 불일치도 아니라 비교불능이다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops'))

from signal_parity_daily import (CASH_ENGINE, CASH_EXEC_SPOT,  # noqa: E402
                                 ParityDataStale, _fut_combined, _single_member,
                                 _spot_combined, cash_boundary_diffs,
                                 self_check_snap, snap_state_diffs)

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


def lower_cash(d):
    return {('Cash' if k == 'CASH' else k): v for k, v in d.items()}


def diffs_of(snaps, target, exec_cash=CASH_EXEC_SPOT, combined=None):
    return snap_state_diffs(member(snaps, combined or {}), N, CAP,
                            combined=target, combined_label='체결 목표',
                            exec_cash=exec_cash)


class TestCashKeyBoundary(unittest.TestCase):
    """표기가 그 자리에 맞으면 접어서 비교하고, 안 맞으면 그 자체를 잡는다."""

    def setUp(self):
        self.snaps = [even('BTC', 'ETH')] * N
        self.avg = average(self.snaps)

    def test_현물_체결_목표의_소문자_표기는_정상이다(self):
        self.assertEqual(diffs_of(self.snaps, lower_cash(self.avg)), [])

    def test_그_자리에_안_맞는_표기_하나만_있어도_잡는다(self):
        # 엔진 쪽 자리(CASH 만 쓰는 곳)에 대문자가 아닌 표기가 들어온 경우.
        diffs = diffs_of(self.snaps, lower_cash(self.avg), exec_cash=CASH_ENGINE)
        self.assertTrue(any('현금 키가' in d for d in diffs), diffs)

    def test_ts_메타키는_비교에서_빠진다(self):
        target = lower_cash(self.avg)
        target['_ts'] = '2026-10-15T00:00:00Z'
        self.assertEqual(diffs_of(self.snaps, target), [])

    def test_ts_가_아닌_모르는_키는_안_버린다(self):
        # 밑줄 접두사만으로 버리면 합 검사가 눈감는다.
        target = lower_cash(self.avg)
        target['_foo'] = 0.05
        self.assertTrue(diffs_of(self.snaps, target))

    def test_한_딕셔너리에_CASH_와_Cash_가_같이_있으면_잡는다(self):
        target = dict(self.avg)
        half = target.pop('CASH') / 2
        target['CASH'], target['Cash'] = half, half
        diffs = diffs_of(self.snaps, target)
        self.assertTrue(any('현금 키가 둘 이상' in d for d in diffs), diffs)

    def test_표기만_다르고_값이_틀리면_그대로_잡는다(self):
        target = lower_cash(self.avg)
        target['BTC'] += 0.05
        self.assertTrue(diffs_of(self.snaps, target))

    def test_비교에_안_쓰는_저장값도_표기_경계는_본다(self):
        diffs = cash_boundary_diffs([
            ({'BTC': 0.5, 'Cash': 0.5}, 'last_combined', CASH_ENGINE)])
        self.assertTrue(any('현금 키가' in d for d in diffs), diffs)

    def test_현금이_아예_없으면_현금_키를_요구하지_않는다(self):
        self.assertEqual(cash_boundary_diffs([
            ({'BTC': 0.5, 'ETH': 0.5}, 'last_combined', CASH_ENGINE)]), [])


class TestFiniteValues(unittest.TestCase):
    """NaN·무한대·잘못된 자료형이 정상 판정을 받지 않는다."""

    def setUp(self):
        self.snaps = [even('BTC', 'ETH')] * N
        self.avg = average(self.snaps)

    def test_NaN_은_유한하지_않다고_잡는다(self):
        bad = [dict(s) for s in self.snaps]
        bad[0]['BTC'] = float('nan')
        diffs = diffs_of(bad, lower_cash(self.avg))
        self.assertTrue(any('유한한 수가 아니다' in d for d in diffs), diffs)

    def test_무한대도_같이_잡는다(self):
        target = lower_cash(self.avg)
        target['BTC'] = float('inf')
        diffs = diffs_of(self.snaps, target)
        self.assertTrue(any('유한한 수가 아니다' in d for d in diffs), diffs)

    def test_문자열_비중은_숫자가_아니라고_잡는다(self):
        target = lower_cash(self.avg)
        target['BTC'] = '0.33'
        diffs = diffs_of(self.snaps, target)
        self.assertTrue(any('숫자가 아니다' in d for d in diffs), diffs)


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
        self.assertEqual(diffs_of(self.after_snaps, lower_cash(self.executed),
                                  combined=self.before_combined), [])

    def test_체결_목표가_정말_틀리면_그날도_잡는다(self):
        wrong = lower_cash(self.executed)
        wrong['FRESH'] = wrong.get('FRESH', 0.0) + 0.02
        wrong['BTC'] -= 0.02
        self.assertTrue(diffs_of(self.after_snaps, wrong,
                                 combined=self.before_combined))

    def test_상한_위반에는_리필_경로를_같이_적는다(self):
        bad = [dict(s) for s in self.after_snaps]
        bad[0].update({'BTC': 2 / 3, 'FRESH': 0.0})
        diffs = snap_state_diffs(member(bad, {}), N, CAP,
                                 combined=average(bad), combined_label='체결 목표',
                                 exec_cash=CASH_ENGINE,
                                 cap_note='현물 리필에는 상한이 안 걸린다')
        hit = [d for d in diffs if '상한' in d]
        self.assertTrue(hit, diffs)
        self.assertIn('리필', hit[0])


class TestSelfCheckSuite(unittest.TestCase):
    """자기시험은 깨끗한 합성 상태를 규칙별로 망가뜨려 그 규칙이 무는지 본다."""

    def test_모든_대조가_통과한다(self):
        for cash in (CASH_ENGINE, CASH_EXEC_SPOT):
            for name, ok, why in self_check_snap(7, CAP, cash):  # noqa
                self.assertTrue(ok, f'{cash} {name}: {why}')

    def test_깨끗한_합성_상태가_먼저_통과한다(self):
        name, ok, why = self_check_snap(7, CAP, CASH_EXEC_SPOT)[0]
        self.assertEqual(name, '깨끗한 합성 상태')
        self.assertTrue(ok, why)

    def test_규칙별로_기대한_사유가_붙는다(self):
        got = {name: why for name, ok, why in self_check_snap(7, CAP, CASH_EXEC_SPOT)}
        self.assertIn('합산', got['합을 보존한 목표 비중 이동'])
        self.assertIn('현금 키가 둘 이상', got['합·평균을 보존한 현금 키 분할'])
        self.assertIn('스냅 개수', got['스냅 개수'])
        self.assertIn('음수', got['음수 비중'])

    def test_평균_비교를_끄면_자기시험이_실패한다(self):
        """대조가 실제로 그 규칙에 걸려 있는지 — 규칙을 빼 보고 확인한다."""
        import signal_parity_daily as mod
        original = mod.snap_state_diffs

        def blind(*a, **kw):
            # 합산 비교만 빼고 나머지는 그대로 둔 변형.
            return [d for d in original(*a, **kw) if '합산' not in d]

        mod.snap_state_diffs = blind
        try:
            results = mod.self_check_snap(7, CAP, CASH_EXEC_SPOT)
        finally:
            mod.snap_state_diffs = original
        failed = [n for n, ok, _ in results if not ok]
        self.assertIn('합을 보존한 목표 비중 이동', failed)


class TestCompositionGuard(unittest.TestCase):
    """단일 멤버·체결 목표가 확인될 때만 비교한다. 아니면 비교불능이다."""

    def test_멤버가_하나면_그_멤버를_돌려준다(self):
        got = _single_member({'D_SMA42': {'x': 1}}, 'D_SMA42', 't', 'members')
        self.assertEqual(got, {'x': 1})

    def test_멤버가_둘이면_비교불능이다(self):
        with self.assertRaises(ParityDataStale):
            _single_member({'A': {}, 'B': {}}, 'A', 't', 'members')

    def test_멤버_이름이_다르면_비교불능이다(self):
        with self.assertRaises(ParityDataStale):
            _single_member({'OTHER': {}}, 'D_SMA42', 't', 'members')

    def test_선물_체결_목표가_없으면_비교불능이다(self):
        with self.assertRaises(ParityDataStale):
            _fut_combined({'strategies': {'D_SMA42': {}}},
                          {'last_combined': {'ETH': 1.0}})

    def test_선물은_체결_목표가_있으면_그걸_쓴다(self):
        combined, label = _fut_combined({'last_target': {'BTC': 1.0}}, {})
        self.assertEqual((label, combined), ('last_target', {'BTC': 1.0}))

    def test_현물_체결_목표가_없으면_비교불능이다(self):
        with self.assertRaises(ParityDataStale):
            _spot_combined({}, {'last_combined': {'ETH': 1.0}})

    def test_멤버_목표와_전체_목표가_다르면_비교불능이다(self):
        state = {'last_target_snapshot': {'BTC': 0.5, 'Cash': 0.5},
                 'last_member_targets': {'D_SMA42': {'BTC': 0.6, 'Cash': 0.4}}}
        with self.assertRaises(ParityDataStale):
            _spot_combined(state, {})

    def test_멤버_목표와_전체_목표가_같으면_통과한다(self):
        same = {'BTC': 0.5, 'Cash': 0.5}
        state = {'last_target_snapshot': dict(same, _ts='2026-10-15T00:00:00Z'),
                 'last_member_targets': {'D_SMA42': dict(same)}}
        combined, label = _spot_combined(state, {})
        self.assertEqual(label, 'last_target_snapshot')
        self.assertEqual(combined['BTC'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
