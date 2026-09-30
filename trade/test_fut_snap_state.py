#!/usr/bin/env python3
"""선물 스냅 상태 검사의 순수 판정부 시험.

보는 것은 `snap_state_diffs` 하나다. 서버도 시계도 안 탄다 — 망가진 상태를 손으로
만들어 넣고 그 어긋남을 실제로 집어내는지만 본다.

이 검사가 고장나는 흔한 방식은 경보가 뜨는 게 아니라 무엇을 넣든 "이상 없음" 을
찍게 되는 것이다. 그래서 정상 상태 한 건보다 망가진 상태 여러 건이 중요하다.
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'ops'))

from signal_parity_daily import (SNAP_TOL, expected_last_closed_daily,   # noqa: E402
                                 self_check_snap, snap_state_diffs)

N = 5
CAP = 1.0 / 3.0


def healthy(n=N):
    """스냅 n 개가 전부 같은 보유를 가리키는 정상 상태.

    실제 서버 상태를 본떴다 (2026-09-30 실측: 스냅 5개, 합산이 스냅 평균과 오차 0,
    각 스냅·합산 비중 합 1.0).
    """
    snap = {'BTC': CAP, 'ETH': CAP, 'CASH': 1.0 - 2 * CAP}
    snaps = [dict(snap) for _ in range(n)]
    combined = {}
    for s in snaps:
        for k, v in s.items():
            combined[k] = combined.get(k, 0.0) + v / n
    return {'snapshots': snaps, 'last_combined': combined}


class TestSnapStateDiffs(unittest.TestCase):

    def test_healthy_state_has_no_diffs(self):
        self.assertEqual(snap_state_diffs(healthy(), N, CAP), [])

    def test_mixed_snapshots_average_correctly(self):
        """스냅마다 보유가 달라도 합산이 평균이면 통과해야 한다.

        스냅 엇갈림(stagger)이 이 전략의 본질이라, 스냅이 서로 다르다는 것만으로
        경보가 뜨면 이 검사는 매일 울린다.
        """
        snaps = [
            {'BTC': CAP, 'ETH': CAP, 'CASH': 1.0 - 2 * CAP},
            {'BTC': CAP, 'SOL': CAP, 'CASH': 1.0 - 2 * CAP},
            {'CASH': 1.0},
        ]
        combined = {}
        for s in snaps:
            for k, v in s.items():
                combined[k] = combined.get(k, 0.0) + v / len(snaps)
        state = {'snapshots': snaps, 'last_combined': combined}
        self.assertEqual(snap_state_diffs(state, 3, CAP), [])

    def test_missing_snapshot_slot(self):
        diffs = snap_state_diffs(healthy(N - 1), N, CAP)
        self.assertTrue(any('스냅 개수 4' in d for d in diffs), diffs)

    def test_combined_drifted_from_snapshot_mean(self):
        """이 검사의 핵심 — 스냅은 멀쩡한데 저장된 합산만 어긋난 경우."""
        state = healthy()
        state['last_combined']['BTC'] += 0.01
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('BTC' in d and '스냅 평균' in d for d in diffs), diffs)

    def test_combined_drift_just_above_tolerance_is_caught(self):
        state = healthy()
        state['last_combined']['ETH'] += SNAP_TOL * 10
        self.assertTrue(snap_state_diffs(state, N, CAP))

    def test_float_noise_below_tolerance_passes(self):
        """부동소수 잡음으로 매일 울리면 안 된다."""
        state = healthy()
        state['last_combined']['ETH'] += SNAP_TOL / 10
        self.assertEqual(snap_state_diffs(state, N, CAP), [])

    def test_coin_above_cap(self):
        state = healthy()
        state['snapshots'][0] = {'BTC': 0.5, 'CASH': 0.5}
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('상한' in d for d in diffs), diffs)

    def test_cash_above_cap_is_fine(self):
        """현금은 상한을 안 받는다. 전액 현금이 정상 국면이다."""
        snaps = [{'CASH': 1.0} for _ in range(N)]
        state = {'snapshots': snaps, 'last_combined': {'CASH': 1.0}}
        self.assertEqual(snap_state_diffs(state, N, CAP), [])

    def test_negative_weight(self):
        state = healthy()
        state['snapshots'][1] = {'BTC': -0.1, 'ETH': CAP, 'CASH': 1.1 - CAP}
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('음수' in d for d in diffs), diffs)

    def test_snapshot_weights_do_not_sum_to_one(self):
        state = healthy()
        state['snapshots'][2] = {'BTC': CAP, 'ETH': CAP}
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('비중 합' in d for d in diffs), diffs)

    def test_combined_weights_do_not_sum_to_one(self):
        snaps = [{'BTC': CAP, 'CASH': 1.0 - CAP} for _ in range(N)]
        state = {'snapshots': snaps, 'last_combined': {'BTC': CAP}}
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('last_combined 비중 합' in d for d in diffs), diffs)

    def test_snapshots_not_a_list(self):
        diffs = snap_state_diffs({'snapshots': {}, 'last_combined': {}}, N, CAP)
        self.assertTrue(any('목록이 아니다' in d for d in diffs), diffs)

    def test_non_numeric_weight(self):
        state = healthy()
        state['snapshots'][0]['BTC'] = 'x'
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('숫자가 아니다' in d for d in diffs), diffs)

    def test_coin_only_in_combined(self):
        """스냅에 없는 코인이 합산에만 있는 경우도 잡아야 한다."""
        state = healthy()
        state['last_combined']['SOL'] = 0.0001
        diffs = snap_state_diffs(state, N, CAP)
        self.assertTrue(any('SOL' in d for d in diffs), diffs)


class TestSelfCheck(unittest.TestCase):

    def test_positive_control_passes_on_healthy_state(self):
        ok, why = self_check_snap(healthy(), N, CAP)
        self.assertTrue(ok, why)

    def test_positive_control_reports_when_combined_missing(self):
        ok, why = self_check_snap({'snapshots': [], 'last_combined': {}}, N, CAP)
        self.assertFalse(ok)
        self.assertIn('양성 대조', why)


class TestExpectedBar(unittest.TestCase):

    def test_expected_bar_is_previous_utc_midnight(self):
        from datetime import datetime, timezone
        now = datetime(2026, 9, 30, 4, 30, tzinfo=timezone.utc)
        self.assertEqual(str(expected_last_closed_daily(now)),
                         '2026-09-29 00:00:00+00:00')

    def test_expected_bar_right_after_utc_midnight(self):
        """09:05 KST 크론 직후(=00:05 UTC)에도 기대 봉은 전날이어야 한다."""
        from datetime import datetime, timezone
        now = datetime(2026, 9, 30, 0, 5, tzinfo=timezone.utc)
        self.assertEqual(str(expected_last_closed_daily(now)),
                         '2026-09-29 00:00:00+00:00')


if __name__ == '__main__':
    unittest.main(verbosity=2)
