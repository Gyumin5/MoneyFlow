"""토론이 동결한 후보 문법(E·결합·대기 W·휴지 R·최소편차·송금범위)을 배치로 돈다.

alloc_sim_batch 가 기존 하니스와 바이트까지 같음을 보인 뒤, 그 위에 문법을 얹은 것이다.
팔 축 벡터화는 그대로다 — 날짜 루프 1,447번 안에서 모든 후보의 상태를 한 배열로 민다.

발화 판정 (토론 action_steps 3):
  U = T3U_can (상대 미달 >= 20% 이고 그 슬리브 배분 게이트 ON)
  C = T1 OR U                              ← 현행
  결합 6종 = C OR E · T1 OR E · U OR E · E 단독 · T1 OR (U AND E) · C AND E

대기 W (action_steps 4): 발화 후 실행 요청을 미루는 기간. 범위가 '전이 경로만' 이면
E 없이는 발화하지 않았을 발화만 미룬다 — T1 이 혼자 켜진 날은 그대로 즉시 간다.
휴지 R: 재발화를 막는 기간. 범위가 '전이 경로만' 이면 그 기간 동안 E 항만 끈다.
시작점은 전이 시점 또는 송금 완료 시점.

최소편차: E 로 발화할 때 half_turnover 가 이 값 미만이면 안 민다(0·5·10pp).
송금범위: 전체 목표 복귀, 또는 부분 — 신호에 관련된 미달 슬리브의 부족액만 채우고
목표 초과 계좌에서 초과액 비례로 조달한다. 자금이 모자라면 비례 축소.
T1 만으로 발화한 날은 두 방식 모두 전체 복귀다(토론이 그렇게 정의했다).

달력: 공통 날짜축은 주식 거래일이라 주말이 없다. 코인 카나리는 주말에도 전이하므로
그 전이를 버리지 않고 다음 공통일로 옮겨 싣는다(토론 action_steps 10). 지연 단위는
공통 날짜축의 칸이고, 그 축이 거래일이라 영업일 근사다 — 미국 장 휴일만큼 어긋난다.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alloc_sim_batch import TGT, T1, T3U  # noqa: E402

# 결합 6종
CB_C_OR_E, CB_T1_OR_E, CB_U_OR_E, CB_E_ONLY, CB_T1_OR_UE, CB_C_AND_E = range(6)
COMBO_NAMES = ("C OR E", "T1 OR E", "U OR E", "E 단독", "T1 OR (U AND E)", "C AND E")


def align_to_axis(s: pd.Series, axis: pd.DatetimeIndex) -> np.ndarray:
    """카나리 계열을 공통 날짜축에 싣는다 — 축에 없는 날의 상태는 버리지 않는다.

    직전 값으로 채운 뒤 축 날짜만 뽑는다. 주말에 켜졌다가 주말에 꺼지는 왕복은
    이 방식으로도 사라진다 — 그건 전이 날짜를 옮기는 문제가 아니라 축의 해상도
    문제라 여기서 못 푼다. 그런 왕복이 몇 번인지는 event_matrix 가 같이 센다.
    """
    return s.reindex(s.index.union(axis)).ffill().reindex(axis).fillna(False).values.astype(bool)


def event_matrix(E: dict, axis: pd.DatetimeIndex) -> tuple:
    """E 명세별 발화일을 공통 축 위의 (n, k) 불리언 행렬로.

    전이는 원래 축(코인 일봉)에서 판정한 뒤 다음 공통일로 옮긴다. 공통일 사이에
    전이가 두 번 이상 있으면 그 날 하나로 접힌다 — 접힌 건수를 같이 돌려준다.
    """
    names = sorted(E.keys())
    n = len(axis)
    out = np.zeros((n, len(names)), dtype=bool)
    folded = 0
    src = E[names[0]].index
    pos = axis.searchsorted(src, side='left')
    # 축보다 앞선 전이는 버린다 — 안 버리면 searchsorted 가 전부 0 을 줘서
    # 축 첫날 하나에 쌓인다(2019-10 ~ 2020-11 치가 통째로 첫날 발화가 된다).
    inside = (src >= axis[0]) & (pos < n)
    for j, nm in enumerate(names):
        hit = E[nm].values & inside
        tgt = pos[hit.nonzero()[0]]
        if len(tgt):
            folded += len(tgt) - len(np.unique(tgt))
            out[tgt, j] = True
    return out, names, folded


def simulate_grid(R: np.ndarray, gate: np.ndarray, EV: np.ndarray,
                  spec: dict) -> tuple:
    """후보 A 개를 동시에 돈다.

    R    (n,3)  슬리브 일별 수익
    gate (n,3)  그 날 그 슬리브의 배분 게이트 ON 여부
    EV   (n,k)  E 명세별 발화일
    spec 팔별 축 — 전부 길이 A 의 정수 배열
         e_idx(-1=E 없음) combo w_len w_all r_len r_all r_from_done band_i tscope
    """
    n, A = R.shape[0], spec['combo'].shape[0]
    BANDS = np.array([0.0, 0.05, 0.10])

    e_idx = spec['e_idx']
    has_e = e_idx >= 0
    e_col = np.where(has_e, e_idx, 0)
    combo = spec['combo']
    w_len, w_all = spec['w_len'], spec['w_all'].astype(bool)
    r_len, r_all = spec['r_len'], spec['r_all'].astype(bool)
    r_done = spec['r_from_done'].astype(bool)
    band = BANDS[spec['band_i']]
    partial = spec['tscope'].astype(bool)

    v = np.repeat(TGT[None, :], A, axis=0)
    pv = np.empty((n, A))
    pend = np.full(A, -1, dtype=np.int64)
    pend_partial = np.zeros(A, dtype=bool)
    pend_rel = np.zeros((A, 3), dtype=bool)      # 예약 당시 '관련 미달 슬리브'
    block_until = np.full(A, -1, dtype=np.int64)
    fires = np.zeros(A, dtype=np.int64)
    moves = np.zeros(A, dtype=np.int64)
    max_ht = np.zeros(A)
    moved_amt = np.zeros(A)

    def apply_move(sel, tot, w, rel_mask, do_partial):
        """sel 인 팔에 송금을 반영한다. 전체 복귀 또는 부분 채움."""
        tgt_v = tot[:, None] * TGT[None, :]
        full = sel & ~do_partial
        if full.any():
            amt = np.abs(v[full] - tgt_v[full]).sum(axis=1) / 2
            moved_amt[full] += amt
            v[full] = tgt_v[full]
        par = sel & do_partial
        if not par.any():
            return
        short = np.maximum(tgt_v - v, 0.0) * rel_mask          # 채울 몫
        need = short.sum(axis=1)
        over = np.maximum(v - tgt_v, 0.0)                      # 댈 수 있는 몫
        avail = over.sum(axis=1)
        take = np.minimum(need, avail)
        scale_in = np.divide(take, need, out=np.zeros_like(need),
                             where=need > 0)
        scale_out = np.divide(take, avail, out=np.zeros_like(avail),
                              where=avail > 0)
        delta = short * scale_in[:, None] - over * scale_out[:, None]
        v[par] += delta[par]
        moved_amt[par] += take[par]

    for i in range(n):
        v *= (1.0 + R[i])[None, :]
        tot = v.sum(axis=1)
        pv[i] = tot
        ok = tot > 0
        w = np.where(ok[:, None], v / np.where(ok, tot, 1.0)[:, None], TGT[None, :])
        ht = np.abs(w - TGT).sum(axis=1) / 2
        np.maximum(max_ht, ht, out=max_ht)

        # ── 예약된 송금 반영일
        due = pend == i
        if due.any():
            apply_move(due, tot, w, pend_rel, pend_partial)
            moves += due
            pend[due] = -1
            fin = due & r_done & (r_len > 0)
            if fin.any():
                block_until[fin] = i + r_len[fin]
            # 반영 뒤 비중이 바뀌었으니 같은 날 재평가는 안 한다(기존 하니스와 같다)

        free = (pend < 0) & ~due
        if not free.any():
            continue

        # ── 발화 항
        t1 = ht >= T1
        under = (TGT[None, :] - w) / TGT[None, :]
        u_s = (under >= T3U) & gate[i][None, :]
        u = u_s.any(axis=1)
        e_raw = EV[i][e_col] & has_e
        e = e_raw & (ht >= band)

        blocked = block_until > i
        e_eff = e & ~(blocked & ~r_all)          # 전이 경로만 막는 휴지
        hard = blocked & r_all                   # 모든 경로를 막는 휴지

        def eval_combo(ev):
            f = np.empty(A, dtype=bool)
            f[:] = False
            m = combo == CB_C_OR_E
            f[m] = (t1 | u | ev)[m]
            m = combo == CB_T1_OR_E
            f[m] = (t1 | ev)[m]
            m = combo == CB_U_OR_E
            f[m] = (u | ev)[m]
            m = combo == CB_E_ONLY
            f[m] = ev[m]
            m = combo == CB_T1_OR_UE
            f[m] = (t1 | (u & ev))[m]
            m = combo == CB_C_AND_E
            f[m] = ((t1 | u) & ev)[m]
            return f

        fire = eval_combo(e_eff) & free & ~hard
        if not fire.any():
            continue
        fires += fire

        # E 없이도 났을 발화인가 — 대기 범위가 '전이 경로만' 일 때 쓴다
        no_e = eval_combo(np.zeros(A, dtype=bool))
        dep_e = fire & ~no_e

        delay = fire & (w_len > 0) & (w_all | dep_e)
        now = fire & ~delay

        # 관련 미달 슬리브 — 부분 송금의 수신처
        rel_mask = u_s | (e_eff[:, None] & (under > 0))
        rel_mask &= (under > 0)
        # T1 만으로 난 발화는 두 방식 모두 전체 복귀
        t1_only = fire & t1 & ~u & ~e_eff
        do_par = partial & ~t1_only & rel_mask.any(axis=1)

        if now.any():
            apply_move(now, tot, w, rel_mask, do_par)
            moves += now
        if delay.any():
            pend[delay] = np.minimum(i + w_len[delay], n - 1)
            pend_partial[delay] = do_par[delay]
            pend_rel[delay] = rel_mask[delay]

        start_now = fire & ~r_done & (r_len > 0)
        if start_now.any():
            block_until[start_now] = i + r_len[start_now]

    return pv, dict(fires=fires, moves=moves, max_ht=max_ht, moved=moved_amt)


def main() -> int:
    import alloc_sim_batch as B
    import alloc_trigger_space as S
    import bt_alloc_transfer_lag as H

    eq_sp = H.run_spot_live()
    eq_st = H.run_stock_live()
    eq_fu = H.run_fut('margin', 0.015)['_equity']
    axis = pd.DatetimeIndex(sorted(
        eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)))
    Rm = np.column_stack([
        eq_st.loc[axis].pct_change().fillna(0).values,
        eq_sp.loc[axis].pct_change().fillna(0).values,
        eq_fu.loc[axis].pct_change().fillna(0).values,
    ])

    g = S.gate_series()
    gate = np.column_stack([align_to_axis(g[s], axis) for s in S.SLEEVES])
    E = S.build_E(g)
    EV, names, folded = event_matrix(E, axis)
    print(f"공통 축 {axis[0].date()} ~ {axis[-1].date()} ({len(axis)}일)")
    print(f"  배분 게이트 ON 비율 주식 {gate[:, 0].mean():.2f}"
          f" · 현물 {gate[:, 1].mean():.2f} · 선물 {gate[:, 2].mean():.2f}")
    print(f"  E {len(names)}종을 축에 실음 — 공통일 사이에 접힌 전이 {folded}건")
    print(f"  E 발화일 합계 {int(EV.sum())}  (명세당 평균 {EV.sum() / len(names):.1f})")

    # ── 정합 ① E 없는 후보(C 결합)는 기존 배치 시뮬의 '트리거' 팔과 같아야 한다
    A = 4
    spec = dict(
        e_idx=np.full(A, -1, dtype=np.int64),
        combo=np.full(A, CB_C_OR_E, dtype=np.int64),
        w_len=np.array([0, 3, 5, 10], dtype=np.int64),
        w_all=np.ones(A, dtype=np.int64),
        r_len=np.zeros(A, dtype=np.int64),
        r_all=np.ones(A, dtype=np.int64),
        r_from_done=np.zeros(A, dtype=np.int64),
        band_i=np.zeros(A, dtype=np.int64),
        tscope=np.zeros(A, dtype=np.int64),
    )
    pv, info = simulate_grid(Rm, gate, EV, spec)

    base_modes = np.full(A, B.MODE_TRIGGER, dtype=np.int64)
    base_lags = np.array([0, 3, 5, 10], dtype=np.int64)
    pv_b, info_b = B.simulate_batch(Rm, base_modes, base_lags)

    print("\n[정합 ①] E 없는 C 결합 = 기존 배치 시뮬의 트리거 팔")
    print("  (게이트가 다르다 — 기존은 20일 평균수익 근사, 여기는 라이브 배분 게이트.")
    print("   그래서 수치가 같을 필요는 없고, 구조가 같은지만 본다.)")
    bad = 0
    for j, lag in enumerate(base_lags):
        d = float(np.abs(pv[:, j] - pv_b[:, j]).max())
        print(f"  지연 {lag:>2}칸  발화 {info['fires'][j]:>3}/{info_b['fires'][j]:<3}"
              f" 송금 {info['moves'][j]:>3}/{info_b['resets'][j]:<3}"
              f" 최대이탈 {info['max_ht'][j]:.4f}/{info_b['max_ht'][j]:.4f}"
              f"  곡선 최대차 {d:,.1f}")
        if info['fires'][j] != info['moves'][j]:
            bad += 1
    print(f"  발화=송금 (지연은 늦출 뿐 떨구지 않는다): {'성립' if not bad else '어긋남'}")

    # ── 정합 ② 사전점검 — 축이 차이를 볼 수 있나 (history #52)
    A2 = 3
    sp2 = dict(
        e_idx=np.array([-1, 0, 0], dtype=np.int64),
        combo=np.array([CB_C_OR_E, CB_C_OR_E, CB_E_ONLY], dtype=np.int64),
        w_len=np.zeros(A2, dtype=np.int64), w_all=np.ones(A2, dtype=np.int64),
        r_len=np.zeros(A2, dtype=np.int64), r_all=np.ones(A2, dtype=np.int64),
        r_from_done=np.zeros(A2, dtype=np.int64),
        band_i=np.zeros(A2, dtype=np.int64), tscope=np.zeros(A2, dtype=np.int64),
    )
    pv2, i2 = simulate_grid(Rm, gate, EV, sp2)
    print(f"\n[정합 ②] 사전점검 — E 를 더하면 달라지는가 ({names[0]})")
    print(f"  현행 C         발화 {i2['fires'][0]:>3} 송금 {i2['moves'][0]:>3}"
          f" 최종 {pv2[-1, 0]:,.1f}")
    print(f"  C OR E        발화 {i2['fires'][1]:>3} 송금 {i2['moves'][1]:>3}"
          f" 최종 {pv2[-1, 1]:,.1f}")
    print(f"  E 단독        발화 {i2['fires'][2]:>3} 송금 {i2['moves'][2]:>3}"
          f" 최종 {pv2[-1, 2]:,.1f}")
    d01 = float(np.abs(pv2[:, 0] - pv2[:, 1]).max())
    print(f"  현행과 C OR E 의 곡선 최대차 {d01:,.1f}"
          f"  ({'0 이 아니다 — 축이 비어 있지 않다' if d01 > 0 else '!! 0 — 축이 무의미'})")

    # ── 정합 ③ 휴지·대기·최소편차·부분송금이 각각 발화·송금을 움직이는가
    def one(**kw):
        d = dict(e_idx=[0], combo=[CB_C_OR_E], w_len=[0], w_all=[1], r_len=[0],
                 r_all=[1], r_from_done=[0], band_i=[0], tscope=[0])
        d.update({k: [v] for k, v in kw.items()})
        return {k: np.array(v, dtype=np.int64) for k, v in d.items()}

    print("\n[정합 ③] 축마다 실제로 움직이는가 (E 고정, C OR E)")
    rows = [("기준", {}),
            ("대기 21칸 전부", dict(w_len=21, w_all=1)),
            ("대기 21칸 전이만", dict(w_len=21, w_all=0)),
            ("휴지 42칸 전부", dict(r_len=42, r_all=1)),
            ("휴지 42칸 전이만", dict(r_len=42, r_all=0)),
            ("휴지 42칸 송금후", dict(r_len=42, r_all=1, r_from_done=1)),
            ("최소편차 10pp", dict(band_i=2)),
            ("부분 송금", dict(tscope=1))]
    for label, kw in rows:
        p, inf = simulate_grid(Rm, gate, EV, one(**kw))
        print(f"  {label:<16} 발화 {int(inf['fires'][0]):>3}"
              f" 송금 {int(inf['moves'][0]):>3}"
              f" 최대이탈 {float(inf['max_ht'][0]):.4f}"
              f" 옮긴양 {float(inf['moved'][0]):>8,.2f}"
              f" 최종 {p[-1, 0]:>10,.1f}")

    # ── 속도
    import time
    A3 = 20000
    rng = np.random.default_rng(0)
    sp3 = dict(
        e_idx=rng.integers(-1, len(names), A3),
        combo=rng.integers(0, 6, A3),
        w_len=rng.choice([0, 10, 21, 42, 63], A3),
        w_all=rng.integers(0, 2, A3),
        r_len=rng.choice([0, 10, 21, 42, 63], A3),
        r_all=rng.integers(0, 2, A3),
        r_from_done=rng.integers(0, 2, A3),
        band_i=rng.integers(0, 3, A3),
        tscope=rng.integers(0, 2, A3),
    )
    t0 = time.time()
    simulate_grid(Rm, gate, EV, sp3)
    dt = time.time() - t0
    print(f"\n속도: 무작위 후보 {A3:,}개 {dt:.2f}s  →  팔 하나 {dt / A3 * 1000:.3f}ms")
    print(f"  고유 후보 708,563 × 위상 7 × 비용 2 = "
          f"{708563 * 14 * dt / A3 / 3600:.1f} 시간 (한 코어)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
