"""후보 수십만 개를 한꺼번에 도는 합성 시뮬레이터 — 토론이 요구한 연구 러너의 계산 부분.

왜 새로 짜나. 기존 bt_alloc_transfer_lag.simulate 는 팔 하나를 날짜 루프로 돈다. 실측
27밀리초다(probe_grid_budget.py). 토론이 동결한 격자는 명목 117만 5천 개이고 위상 7 ·
비용 2 를 곱하면 1645만 회, 123시간이다. 위약 200벌까지 곱하면 3년이 된다.

그래서 축을 바꾼다. 날짜를 벡터화할 수는 없다 — 오늘 비중이 어제 송금에 달려 있어서다.
대신 팔을 벡터화한다. 하루를 한 번 돌 때 모든 팔의 상태를 한 배열로 같이 민다. 날짜
루프는 1,447번 그대로 돌지만 그 안이 팔 수만큼의 배열 연산 하나가 된다.

정합은 하나로 본다 — 기존 simulate 와 같은 입력에서 자산곡선이 바이트까지 같아야 한다.
다르면 빠른 게 아니라 틀린 것이다. main() 이 그 대조를 돌린다.

여기서 다루는 발화 규칙은 기존 하니스의 것(T1 · T3U_can · 매일 · 안 함)이다. 토론이
동결한 전이 신호 E·대기 W·휴지 R·최소편차·송금범위는 이 위에 얹는다 — 그건 이 파일의
정합이 선 다음이다.

아직 안 한 것: 지연을 실제 영업일 달력이 아니라 교집합 가격 인덱스의 칸 수로 센다.
기존 하니스와 같은 셈이라 대조가 서고, 토론이 지적한 달력 문제는 E 후보를 얹을 때 같이 푼다.
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

TGT = np.array([0.60, 0.25, 0.15])
T1 = 0.20
T3U = 0.20

MODE_DAILY, MODE_TRIGGER, MODE_NONE = 0, 1, 2


def canary_matrix(R: np.ndarray, lookback: int = 20) -> np.ndarray:
    """기존 하니스의 근사 카나리를 날짜×슬리브 행렬로 미리 만든다.

    기존 _canary_on(rets, i) 는 rets[i-20:i].mean() > 0 이다. 같은 값을 누적합으로 낸다.
    앞 20칸은 False — 기존 함수가 i < lookback 에서 False 를 돌려주는 것과 같다.
    """
    n = R.shape[0]
    out = np.zeros_like(R, dtype=bool)
    if n <= lookback:
        return out
    cs = np.cumsum(R, axis=0)
    # rets[i-20:i] 의 합 = cs[i-1] - cs[i-21]
    tot = cs[lookback - 1:n - 1] - np.vstack([np.zeros((1, R.shape[1])),
                                              cs[:n - lookback - 1]])
    out[lookback:] = tot > 0
    return out


def simulate_batch(R: np.ndarray, modes: np.ndarray, lags: np.ndarray,
                   can: np.ndarray | None = None) -> tuple:
    """팔 A 개를 동시에 돈다.

    R     (n,3)  슬리브 일별 수익
    modes (A,)   0=매일 1=트리거 2=안 함
    lags  (A,)   트리거 발화 후 송금 반영까지 미루는 칸 수
    반환  pv (n,A) 합성 자산곡선, 그리고 팔별 발화·송금·최대이탈
    """
    n = R.shape[0]
    A = modes.shape[0]
    if can is None:
        can = canary_matrix(R)

    v = np.repeat(TGT[None, :], A, axis=0)          # (A,3)
    pv = np.empty((n, A))
    pending = np.full(A, -1, dtype=np.int64)
    fires = np.zeros(A, dtype=np.int64)
    resets = np.zeros(A, dtype=np.int64)
    max_ht = np.zeros(A)

    is_daily = modes == MODE_DAILY
    is_trig = modes == MODE_TRIGGER
    lag0 = is_trig & (lags <= 0)
    lagN = is_trig & (lags > 0)

    for i in range(n):
        v *= (1.0 + R[i])[None, :]
        tot = v.sum(axis=1)                          # (A,)
        pv[i] = tot
        safe = tot > 0
        w = np.where(safe[:, None], v / np.where(safe, tot, 1.0)[:, None],
                     TGT[None, :])
        ht = np.abs(w - TGT).sum(axis=1) / 2
        np.maximum(max_ht, ht, out=max_ht)

        # 매일 재조정
        if is_daily.any():
            v[is_daily] = tot[is_daily, None] * TGT[None, :]

        # 오늘이 예약된 송금 반영일
        due = is_trig & (pending == i)
        if due.any():
            v[due] = tot[due, None] * TGT[None, :]
            resets += due
            pending[due] = -1

        # 이미 예약돼 있으면 새로 안 잡는다
        free = is_trig & (pending < 0) & ~due
        if not free.any():
            continue

        fire = ht >= T1
        rel = (TGT[None, :] - w) / TGT[None, :]
        fire |= ((rel >= T3U) & can[i][None, :]).any(axis=1)
        fire &= free
        if not fire.any():
            continue
        fires += fire

        now = fire & lag0
        if now.any():
            v[now] = tot[now, None] * TGT[None, :]
            resets += now
        later = fire & lagN
        if later.any():
            pending[later] = np.minimum(i + lags[later], n - 1)

    end_w = np.where(v.sum(axis=1)[:, None] > 0,
                     v / np.maximum(v.sum(axis=1), 1e-300)[:, None], TGT[None, :])
    return pv, dict(fires=fires, resets=resets, max_ht=max_ht, end_w=end_w)


def main() -> int:
    import bt_alloc_transfer_lag as H

    t_sp = H.run_spot_live()
    t_st = H.run_stock_live()
    t_fu = H.run_fut('margin', 0.015)['_equity']

    common = sorted(t_st.index.intersection(t_sp.index).intersection(t_fu.index))
    R = np.column_stack([
        t_st.loc[common].pct_change().fillna(0).values,
        t_sp.loc[common].pct_change().fillna(0).values,
        t_fu.loc[common].pct_change().fillna(0).values,
    ])
    print(f"공통 {len(common)}일  팔 {len(H.ARMS)}개 대조")

    name2mode = {'daily': MODE_DAILY, 'trigger': MODE_TRIGGER, 'none': MODE_NONE}
    modes = np.array([name2mode[m] for _, m, _ in H.ARMS], dtype=np.int64)
    lags = np.array([lg for _, _, lg in H.ARMS], dtype=np.int64)

    pv, info = simulate_batch(R, modes, lags)

    bad = 0
    for j, (label, mode, lag) in enumerate(H.ARMS):
        eq, st = H.simulate(t_st, t_sp, t_fu, mode, lag)
        d = float(np.abs(eq.values - pv[:, j]).max())
        same = (d == 0.0)
        ok_f = int(st['fires']) == int(info['fires'][j])
        ok_r = int(st['resets']) == int(info['resets'][j])
        ok_h = abs(float(st['max_ht']) - float(info['max_ht'][j])) < 1e-15
        flag = "일치" if (same and ok_f and ok_r and ok_h) else "어긋남"
        if flag == "어긋남":
            bad += 1
        print(f"  {label:<16} 곡선 최대차 {d:.3e}  발화 {st['fires']}/{info['fires'][j]}"
              f"  송금 {st['resets']}/{info['resets'][j]}"
              f"  최대이탈 {st['max_ht']:.6f}/{info['max_ht'][j]:.6f}  {flag}")

    # 사전점검 — 재는 장치가 차이를 볼 수 있나 (history #52)
    spread = float(np.abs(pv[:, 0] - pv[:, 5]).max())
    print(f"\n사전점검: 매일 재조정과 송금 없음의 곡선 최대차 {spread:,.1f}"
          f"  ({'0 이 아니다 — 대조가 비어 있지 않다' if spread > 0 else '!! 0 이다 — 대조가 무의미'})")

    # 속도
    import time
    for A in (6, 1000, 20000):
        m = np.resize(modes, A)
        lg = np.resize(lags, A)
        t0 = time.time()
        simulate_batch(R, m, lg)
        dt = time.time() - t0
        print(f"  팔 {A:>6,}개  {dt:6.2f}s  →  팔 하나 {dt / A * 1000:7.3f}ms")

    if bad:
        print(f"\n!! {bad}개 팔이 어긋났다 — 빠른 게 아니라 틀린 것이다")
    else:
        print("\n통과: 여섯 팔 모두 기존 하니스와 바이트까지 같다.")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
