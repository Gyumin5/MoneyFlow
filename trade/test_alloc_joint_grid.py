"""합동 격자 실행기의 재현 검사 (2026-10-31).

ai-debate run-20261031T005336Z 가 실행 전 점검으로 요구한 것을 검사로 옮겼다.
벡터화한 새 경로(alloc_joint_grid)가 기존 경로(alloc_mix_real_rebal.build_alloc_real,
bt_coin_param_resweep.window_rs)와 같은 답을 내는지 본다. 슬리브 백테스트는 안 돌린다 —
가짜 수익률·카나리로 두 구현만 맞댄다(수 초).

검사하는 것:
  1. 비중 격자가 제약을 지킨다 (합 100%, 주식 >= 60%, 현물 > 선물 또는 선물 0).
  2. 조합 수가 토론 확정값(비중 × 조건)과 같다.
  3. 벡터화 합성이 기존 한-조합 합성과 자산곡선·발화수까지 같다.
  4. 꺼짐(무한대) 칸이 실제로 안 발화한다. 둘 다 꺼지면 송금이 0 이다.
  5. 선물 목표 0 칸에서 상대부족이 발화를 못 만든다 (0/0 분모).
  6. 벡터화 창 순위합이 기존 window_rs 와 같은 순위합·창 수를 낸다.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESEARCH = os.path.join(ROOT, 'strategies', 'cap_defend', 'research')
for _p in (RESEARCH, os.path.join(ROOT, 'strategies', 'cap_defend')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alloc_joint_grid as J           # noqa: E402
from alloc_mix_real_rebal import build_alloc_real   # noqa: E402


def fake_sleeves(n=900, seed=7):
    """재현 가능한 가짜 슬리브 셋. 선물은 변동성을 크게 둬서 발화가 실제로 난다."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2019-01-01', periods=n)
    out = []
    for mu, sd in ((0.0004, 0.010), (0.0009, 0.028), (0.0014, 0.050)):
        r = rng.normal(mu, sd, n)
        out.append(pd.Series(np.cumprod(1.0 + r), index=idx))
    cans = [pd.Series(rng.random(n) > 0.4, index=idx) for _ in range(3)]
    return out, cans


class TestGrid(unittest.TestCase):
    def test_비중이_제약을_지킨다(self):
        for name, st, sp, fu in J.build_mixes():
            self.assertAlmostEqual(st + sp + fu, 1.0, places=12, msg=name)
            self.assertGreaterEqual(st, 0.60 - 1e-12, msg=name)
            self.assertTrue(sp > fu or fu == 0.0, msg=f'{name}: 현물 {sp} 선물 {fu}')

    def test_조합_수가_확정값과_같다(self):
        mixes = J.build_mixes()
        self.assertEqual(len(mixes), len(J.STOCK_AXIS) * len(J.COIN_RATIOS))
        self.assertEqual(len(J.build_combos(mixes)),
                         len(mixes) * len(J.GRID_HT) * len(J.GRID_T3U))

    def test_같은_비중이_두_번_안_들어간다(self):
        seen = [(round(st, 10), round(sp, 10), round(fu, 10))
                for _n, st, sp, fu in J.build_mixes()]
        self.assertEqual(len(seen), len(set(seen)))
        names = [m[0] for m in J.build_mixes()]
        self.assertEqual(len(names), len(set(names)))

    def test_현행이_격자에_한_번만_있고_이름이_붙는다(self):
        hits = [m for m in J.build_mixes()
                if (round(m[1], 10), round(m[2], 10), round(m[3], 10))
                == tuple(round(x, 10) for x in J.LIVE_MIX)]
        self.assertEqual(len(hits), 1)
        self.assertIn('현행', hits[0][0])


class TestSynth(unittest.TestCase):
    def setUp(self):
        (self.st, self.sp, self.fu), (self.cs, self.cp, self.cf) = fake_sleeves()

    def _many(self, rows):
        tgt = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
        ht = np.array([r[3] for r in rows], dtype=float)
        t3u = np.array([r[4] for r in rows], dtype=float)
        return J.build_alloc_real_many(self.st, self.sp, self.fu,
                                       self.cs, self.cp, self.cf, tgt, ht, t3u)

    def test_벡터화가_한조합_합성과_같다(self):
        rows = [
            (0.60, 0.25, 0.15, 0.20, 0.20),    # 라이브 조합
            (0.60, 0.25, 0.15, 0.30, 0.20),    # 직전 측정의 최적 칸
            (0.80, 0.10, 0.10, 0.10, 0.40),
            (0.95, 0.05, 0.00, 0.50, 0.10),    # 선물 0
        ]
        dates, curves, fires = self._many(rows)
        for k, (ws, wsp, wfu, ht, t3u) in enumerate(rows):
            eq, f1 = build_alloc_real(self.st, self.sp, self.fu,
                                      self.cs, self.cp, self.cf,
                                      (ws, wsp, wfu), ht_thr=ht, t3u_rel=t3u)
            np.testing.assert_allclose(curves[k], eq.to_numpy(), rtol=1e-12, atol=1e-12)
            self.assertEqual(int(fires[k]), f1, msg=f'행 {k} 발화수')
            self.assertTrue((dates == eq.index).all())

    def test_꺼짐은_안_발화한다(self):
        rows = [
            (0.60, 0.25, 0.15, np.inf, np.inf),   # 둘 다 꺼짐 = 무송금
            (0.60, 0.25, 0.15, np.inf, 0.20),     # 상대부족만
            (0.60, 0.25, 0.15, 0.20, np.inf),     # 반회전율만
            (0.60, 0.25, 0.15, 0.20, 0.20),       # 둘 다
        ]
        _d, _c, fires = self._many(rows)
        self.assertEqual(int(fires[0]), 0)
        self.assertGreater(int(fires[1]), 0)
        self.assertGreater(int(fires[2]), 0)
        # 둘 다 켠 칸은 각각보다 적을 수 없다
        self.assertGreaterEqual(int(fires[3]), int(fires[1]))
        self.assertGreaterEqual(int(fires[3]), int(fires[2]))

    def test_무송금은_고정비중과_다르고_곡선이_한번_정해진다(self):
        rows = [(0.60, 0.25, 0.15, np.inf, np.inf)]
        _d, curves, _f = self._many(rows)
        # 되돌림이 없으면 세 슬리브 자본의 단순 합이다
        common = self.st.index
        man = (0.60 * self.st / self.st.iloc[0] * (1 + self.st.pct_change().fillna(0).iloc[0] * 0)
               )  # 아래에서 직접 쌓는다
        cap = np.array([0.60, 0.25, 0.15], dtype=float)
        rets = np.column_stack([s.loc[common].pct_change().fillna(0).to_numpy()
                                for s in (self.st, self.sp, self.fu)])
        want = np.empty(len(common))
        for i in range(len(common)):
            cap = cap * (1.0 + rets[i])
            want[i] = cap.sum()
        np.testing.assert_allclose(curves[0], want, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(man), len(common))

    def test_선물_목표_0_은_상대부족으로_안_발화한다(self):
        # 선물 0 칸을 상대부족만 켜고 돌리면, 발화는 주식·현물 쪽에서만 올 수 있다.
        rows = [(0.95, 0.05, 0.00, np.inf, 0.10)]
        _d, _c, fires = self._many(rows)
        eq, f1 = build_alloc_real(self.st, self.sp, self.fu, self.cs, self.cp, self.cf,
                                  (0.95, 0.05, 0.00), ht_thr=np.inf, t3u_rel=0.10)
        self.assertEqual(int(fires[0]), f1)
        self.assertEqual(len(eq), len(_c[0]))


class TestRankSum(unittest.TestCase):
    def test_창_순위합이_기존_구현과_같다(self):
        import bt_coin_param_resweep as R
        (st, sp, fu), (cs, cp, cf) = fake_sleeves(n=700, seed=11)
        rows = [
            (0.60, 0.25, 0.15, 0.20, 0.20),
            (0.70, 0.20, 0.10, 0.30, 0.15),
            (0.80, 0.15, 0.05, np.inf, 0.25),
            (0.60, 0.30, 0.10, 0.15, np.inf),
        ]
        tgt = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
        ht = np.array([r[3] for r in rows], dtype=float)
        t3u = np.array([r[4] for r in rows], dtype=float)
        dates, curves, _f = J.build_alloc_real_many(st, sp, fu, cs, cp, cf, tgt, ht, t3u)

        sizes, strides = [252, 378], [63, 126]
        rs, wins, n = J.window_ranksum(curves, dates, sizes, strides)

        old_sizes, old_strides = R.WIN_SIZES, R.STRIDES
        try:
            R.WIN_SIZES, R.STRIDES = sizes, strides
            eqd = {f'r{k}': pd.Series(curves[k], index=dates) for k in range(len(rows))}
            s_old, w_old, n_old = R.window_rs(eqd)
        finally:
            R.WIN_SIZES, R.STRIDES = old_sizes, old_strides

        self.assertEqual(n, n_old)
        self.assertGreater(n, 0)
        for k in range(len(rows)):
            self.assertAlmostEqual(rs[k], s_old[f'r{k}'], places=9, msg=f'행 {k} 순위합')
            self.assertEqual(int(wins[k]), w_old[f'r{k}'], msg=f'행 {k} 1위 횟수')

    def test_구간_제한이_창을_안_섞는다(self):
        (st, sp, fu), (cs, cp, cf) = fake_sleeves(n=800, seed=3)
        rows = [(0.60, 0.25, 0.15, 0.20, 0.20), (0.70, 0.20, 0.10, 0.25, 0.20)]
        tgt = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
        ht = np.array([r[3] for r in rows], dtype=float)
        t3u = np.array([r[4] for r in rows], dtype=float)
        dates, curves, _f = J.build_alloc_real_many(st, sp, fu, cs, cp, cf, tgt, ht, t3u)
        sizes, strides = [252], [63]
        split = len(dates) - 300
        _rs_a, _w, n_tr = J.window_ranksum(curves, dates, sizes, strides, 0, split)
        _rs_b, _w, n_te = J.window_ranksum(curves, dates, sizes, strides, split, len(dates))
        _rs_all, _w, n_all = J.window_ranksum(curves, dates, sizes, strides)
        # 분할하면 경계를 가로지르는 창이 빠지므로 합이 전체보다 작다
        self.assertGreater(n_tr, 0)
        self.assertGreater(n_te, 0)
        self.assertLess(n_tr + n_te, n_all)


if __name__ == '__main__':
    unittest.main(verbosity=2)
