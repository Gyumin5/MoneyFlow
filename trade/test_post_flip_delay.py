#!/usr/bin/env python3
"""전환 후 5봉 재평가(post_flip_delay)가 라이브 두 엔진에서 채택 BT 와 같게 도는지 본다.

고치는 결함: 카나리가 켜지는 봉에 헬스 통과 코인이 하나도 없으면 전 스냅샷이 현금으로
채워지고 다음 스태거 자리까지 슬리브가 잠긴다(현물 최장 31일, 선물 19일). 드리프트는
현금 100%→현금 100% 라 반회전율이 0 이어서 못 깨운다. 채택 BT 둘에는 전환 후 5봉 뒤
전 스냅샷을 한 번 다시 고르는 분기가 있고 라이브에는 없었다.

여기서 재는 것
  ⑴ 두 엔진의 replay_canary_pfd 가 같은 입력에 같은 답을 낸다(사본이 갈리는 것을 막는다).
  ⑵ 전환 봉을 0 으로 세어 다섯 봉이 지난 첫 회차에 한 번만 발화한다. 1~4봉에는 안 하고
     6봉에도 다시 안 한다.
  ⑶ 중간에 꺼지면 예약이 사라지고, 다시 켜지면 0 부터 다시 센다.
  ⑷ 옛 상태파일(칸 없음)·최초 초기화는 전환으로 세지 않는다.
  ⑸ 크론을 걸러 봉이 밀리면 밀린 사이의 전환·경과를 재생해 이번 회차에 한 번 수행한다.
  ⑹ 자료가 밀린 회차(freshness 미달)에서는 예약을 소비하지 않는다.
  ⑺ 현물·선물 엔진을 봉 단위로 돌려, 전환일에 통과 코인이 0 이던 슬리브가 다섯 봉 뒤에
     실제로 다시 채워진다. 같은 자료를 post_flip_delay=0 으로 돌리면 잠긴 채 남는다
     (앵커가 대신 채워 준 게 아님을 그 대조가 증명한다).

실행: cd /home/gmoh/mon/251229 && python3 trade/test_post_flip_delay.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import coin_live_engine as cle  # noqa: E402
import auto_trade_binance as atb  # noqa: E402

SMA_P = 42
HYST = 0.015
PFD = 5


# ─── 합성 가격 ───
def _btc_series(n: int, flip_at: int) -> np.ndarray:
    """flip_at 봉 근처에서 SMA42 를 위로 뚫는 BTC 종가.

    앞 구간은 완만한 하락이라 카나리 OFF, flip_at 직전부터 가파르게 올라 켜진다.
    """
    r = np.full(n, -0.002)
    r[flip_at - 4:] = 0.02
    return 100.0 * np.cumprod(1.0 + r)


def _coin_series(n: int, drop_at: int) -> np.ndarray:
    """drop_at 봉에 한 번 크게 빠지는, 그 밖에는 완만히 오르는 코인 종가.

    20봉 모멘텀 창에 그 낙폭이 들어 있는 동안은 헬스 탈락이고, 창 밖으로 빠지면 통과한다.
    낙폭이 하루치 한 번뿐이라 90일 일간 변동성은 상한(5%) 아래로 남는다.
    """
    r = np.full(n, 0.001)
    r[drop_at] = -0.05
    return 100.0 * np.cumprod(1.0 + r)


def _find_flip(btc: np.ndarray) -> int:
    """합성 BTC 계열에서 카나리가 처음 켜지는 봉."""
    canary = False
    for i in range(SMA_P + 1, len(btc)):
        sma = float(np.mean(btc[i - SMA_P + 1:i + 1]))
        canary = (not (btc[i] < sma * (1 - HYST))) if canary else (btc[i] > sma * (1 + HYST))
        if canary:
            return i
    raise AssertionError("합성 계열에서 카나리 전환이 안 났다")


def _bars_for_sim(n: int, flip_at: int, idx: pd.DatetimeIndex):
    """전환 봉에는 통과 코인이 0, 다섯 봉 뒤에는 통과하는 합성 자료.

    낙폭 봉을 전환-15 에 둔다. 20봉·18봉 모멘텀 창 둘 다 전환 봉에서는 낙폭을 품고
    전환+5 봉에서는 창 밖으로 내보낸다.
    """
    btc = _btc_series(n, flip_at)
    flip = _find_flip(btc)
    bars = {'BTC': _frame(btc, idx)}
    for c in ('AAA', 'BBB', 'CCC'):
        bars[c] = _frame(_coin_series(n, flip - 15), idx)
    return bars, flip


def _frame(close: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({'Open': close, 'High': close, 'Low': close,
                         'Close': close, 'Volume': 1.0}, index=idx)


def _ok(msg: str) -> None:
    print(f"  통과 — {msg}")


# ─── ⑴~⑸ replay_canary_pfd 직접 ───
def _call_both(**kw):
    a = cle.replay_canary_pfd(**kw)
    b = atb.replay_canary_pfd(**kw)
    assert a == b, f"두 엔진의 재생 판정이 다르다:\n  현물 {a}\n  선물 {b}"
    return a


def test_replay() -> None:
    n = 200
    flip_at = 150
    closes = _btc_series(n, flip_at)
    ts = [f"bar{i:04d}" for i in range(n)]

    # 전환 봉 찾기 — 한 봉씩 진행하며 상태를 이어 받는다.
    canary = False
    flip_bar = None
    done = True
    fired_at = []
    for i in range(SMA_P + 1, n):
        canary, flipped, fire, flip_bar, done, _ = _call_both(
            closes=closes[:i + 1], bar_ts=ts[:i + 1], last_bar_ts=ts[i - 1],
            prev_canary=canary, sma_p=SMA_P, hyst=HYST,
            flip_bar_ts=flip_bar, pfd_done=done, pfd_bars=PFD, allow_arm=True)
        if flipped and canary:
            flip_idx = i
        if fire:
            fired_at.append(i)

    assert fired_at, "전환이 있었는데 재평가가 한 번도 발화하지 않았다"
    assert len(fired_at) == 1, f"재평가가 여러 번 발화했다: {fired_at}"
    assert fired_at[0] - flip_idx == PFD, (
        f"발화 봉이 전환 봉+{PFD} 가 아니다: 전환 {flip_idx}, 발화 {fired_at[0]}")
    _ok(f"전환 봉 {flip_idx} → {PFD}봉 뒤 {fired_at[0]} 에 한 번만 발화")

    # pfd_bars=0 이면 한 번도 안 한다 (되돌림 경로)
    canary, flip_bar, done, fired = False, None, True, 0
    for i in range(SMA_P + 1, n):
        canary, _, fire, flip_bar, done, _ = _call_both(
            closes=closes[:i + 1], bar_ts=ts[:i + 1], last_bar_ts=ts[i - 1],
            prev_canary=canary, sma_p=SMA_P, hyst=HYST,
            flip_bar_ts=flip_bar, pfd_done=done, pfd_bars=0, allow_arm=True)
        fired += int(fire)
    assert fired == 0, "post_flip_delay=0 인데 발화했다"
    _ok("post_flip_delay=0 이면 발화 없음 (되돌림)")

    # 옛 상태파일 / 최초 회차 — 예약을 걸지 않는다
    canary, flip_bar, done, fired = False, None, True, 0
    for i in range(SMA_P + 1, n):
        canary, _, fire, flip_bar, done, _ = _call_both(
            closes=closes[:i + 1], bar_ts=ts[:i + 1], last_bar_ts=None,
            prev_canary=canary, sma_p=SMA_P, hyst=HYST,
            flip_bar_ts=flip_bar, pfd_done=done, pfd_bars=PFD, allow_arm=True)
        fired += int(fire)
    assert fired == 0, "직전 봉을 모르는 회차에서 예약이 걸렸다"
    _ok("직전 상태를 모르는 회차는 전환으로 세지 않는다")

    # 중간에 꺼지면 예약이 사라진다 — 전환 2봉 뒤에 OFF 를 넣고 1봉 뒤 다시 ON
    c2 = closes.copy()
    off_at = flip_idx + 2
    c2[off_at] = float(np.mean(c2[off_at - SMA_P:off_at])) * 0.90   # SMA 아래로
    canary, flip_bar, done = False, None, True
    fired_at2, flips2 = [], []
    for i in range(SMA_P + 1, n):
        canary, flipped, fire, flip_bar, done, _ = _call_both(
            closes=c2[:i + 1], bar_ts=ts[:i + 1], last_bar_ts=ts[i - 1],
            prev_canary=canary, sma_p=SMA_P, hyst=HYST,
            flip_bar_ts=flip_bar, pfd_done=done, pfd_bars=PFD, allow_arm=True)
        if flipped and canary:
            flips2.append(i)
        if fire:
            fired_at2.append(i)
    assert len(flips2) >= 2, f"OFF 를 끼운 계열에서 재전환이 안 났다: {flips2}"
    assert fired_at2 == [flips2[-1] + PFD], (
        f"재전환 뒤 0 부터 다시 세지 않았다: 전환 {flips2}, 발화 {fired_at2}")
    _ok(f"중간 OFF 가 예약을 지우고 재전환 {flips2[-1]} 에서 다시 센다")

    # 누락봉 재생 — 전환 직후부터 7봉을 한 번에 넘긴다
    gap_from = flip_idx - 1
    gap_to = flip_idx + 7
    canary, flipped, fire, flip_bar, done, _ = _call_both(
        closes=closes[:gap_to + 1], bar_ts=ts[:gap_to + 1], last_bar_ts=ts[gap_from],
        prev_canary=False, sma_p=SMA_P, hyst=HYST,
        flip_bar_ts=None, pfd_done=True, pfd_bars=PFD, allow_arm=True)
    assert canary and flipped, "밀린 구간에 전환이 있었는데 전환으로 안 잡혔다"
    assert fire, "밀린 사이에 재평가가 예정돼 있었는데 이번 회차에 안 했다"
    assert done, "발화했는데 예약이 안 닫혔다"
    _ok("크론을 걸러도 밀린 사이의 전환·경과를 재생해 한 번 수행한다")


# ─── ⑹⑺ 현물 엔진 봉 단위 ───
def _spot_sim(pfd: int, stale_at_fire: bool = False):
    n, flip_at = 320, 260
    idx = pd.date_range(end=datetime(2026, 1, 1), periods=n, freq='D')
    bars_all, _ = _bars_for_sim(n, flip_at, idx)
    universe = ['AAA', 'BBB', 'CCC']

    cfg = dict(cle.MEMBER_D_SMA42)
    cfg['post_flip_delay'] = pfd

    state = cle.MemberState()
    trace = []
    flip_run = None
    for k in range(200, n):
        sliced = {c: df.iloc[:k + 1] for c, df in bars_all.items()}
        bar_dt = idx[k].to_pydatetime().replace(tzinfo=timezone.utc)
        now = bar_dt + timedelta(days=1, hours=9)
        if stale_at_fire and flip_run is not None and (k - flip_run) == PFD:
            now = now + timedelta(days=3)   # 자료가 밀린 회차
        res = cle.compute_member_target('D_SMA42', cfg, sliced, universe, state, now)
        state = res.new_state
        if res.canary_flipped and state.canary_on:
            flip_run = k
        trace.append((k, dict(res.target), res.pfd_fired, state.pfd_done,
                      state.bar_counter))
    return flip_run, trace


def _is_cash_only(t) -> bool:
    return sum(v for key, v in t.items() if str(key).lower() != 'cash') < 1e-9


def test_spot_engine() -> None:
    flip_run, trace = _spot_sim(PFD)
    assert flip_run is not None, "현물 합성 계열에서 카나리 전환이 안 났다"
    by_run = {k: (t, fired, done, bc) for k, t, fired, done, bc in trace}

    assert _is_cash_only(by_run[flip_run][0]), (
        f"전환 봉에 이미 코인이 잡혔다 — 잠김 상황을 재현 못 했다: {by_run[flip_run][0]}")
    for d in range(1, PFD):
        assert not by_run[flip_run + d][1], f"전환 +{d}봉에 재평가가 발화했다"
        assert _is_cash_only(by_run[flip_run + d][0]), f"전환 +{d}봉에 목표가 바뀌었다"
    fire_run = flip_run + PFD
    assert by_run[fire_run][1], f"전환 +{PFD}봉에 재평가가 발화하지 않았다"
    assert not _is_cash_only(by_run[fire_run][0]), (
        f"재평가가 돌았는데 목표가 그대로 현금이다: {by_run[fire_run][0]}")
    assert not by_run[fire_run + 1][1], "다음 봉에 또 발화했다"
    assert by_run[fire_run][2], "발화 뒤 예약이 안 닫혔다"
    # 앵커 위상은 회차 수로만 전진한다 — 재평가가 건드리지 않는다
    assert by_run[fire_run][3] == by_run[fire_run - 1][3] + 1, "bar_counter 가 어긋났다"
    _ok(f"현물: 전환 {flip_run} 에 현금 잠김 → +{PFD}봉 {fire_run} 에서 {by_run[fire_run][0]}")

    flip0, trace0 = _spot_sim(0)
    by0 = {k: t for k, t, _, _, _ in trace0}
    assert flip0 == flip_run
    for d in range(1, PFD + 6):
        assert _is_cash_only(by0[flip0 + d]), (
            f"post_flip_delay=0 인데 전환 +{d}봉에 코인이 잡혔다 — 앵커가 채운 것이라면"
            f" 이 시험은 재평가를 재지 못한다: {by0[flip0 + d]}")
    _ok("현물: 같은 자료를 post_flip_delay=0 으로 돌리면 잠긴 채 남는다 (대조군)")

    flip_s, trace_s = _spot_sim(PFD, stale_at_fire=True)
    by_s = {k: (t, fired, done) for k, t, fired, done, _ in trace_s}
    assert not by_s[flip_s + PFD][1], "자료가 밀린 회차에서 발화했다"
    assert not by_s[flip_s + PFD][2], "자료가 밀린 회차에서 예약을 소비했다"
    assert by_s[flip_s + PFD + 1][1], "다음 정상 회차에서 발화하지 않았다"
    _ok("현물: 자료가 밀린 회차는 건너뛰고 다음 정상 회차에 발화한다")


# ─── ⑺ 선물 엔진 봉 단위 ───
def test_futures_engine() -> None:
    n, flip_at = 320, 260
    idx = pd.date_range(end=datetime(2026, 1, 1), periods=n, freq='D')
    bars_all, _ = _bars_for_sim(n, flip_at, idx)

    orig_universe = atb.UNIVERSE
    orig_fresh = atb._is_completed_bar_fresh
    atb.UNIVERSE = ['AAAUSDT', 'BBBUSDT', 'CCCUSDT']
    atb._is_completed_bar_fresh = lambda df, now_utc=None: True
    try:
        def sim(pfd):
            params = dict(atb.STRATEGIES['D_SMA42'])
            params['post_flip_delay'] = pfd
            state = {}
            flip_run, out = None, {}
            for k in range(200, n - 1):
                # [-1] 은 진행중 봉이므로 완성봉 k 까지 쓰려면 k+1 까지 넘긴다
                sliced = {c: df.iloc[:k + 2] for c, df in bars_all.items()}
                tgt = atb.compute_strategy_target('D_SMA42', params, {'D': sliced}, state)
                ss = state['strategies']['D_SMA42']
                if ss['canary_info']['flipped'] and ss['canary_on']:
                    flip_run = k
                out[k] = (dict(tgt), bool(ss.get('pfd_fired')), bool(ss.get('pfd_done')))
            return flip_run, out

        flip_run, out = sim(PFD)
        assert flip_run is not None, "선물 합성 계열에서 카나리 전환이 안 났다"
        assert _is_cash_only(out[flip_run][0]), (
            f"전환 봉에 이미 코인이 잡혔다: {out[flip_run][0]}")
        for d in range(1, PFD):
            assert not out[flip_run + d][1], f"전환 +{d}봉에 재평가가 발화했다"
        fire_run = flip_run + PFD
        assert out[fire_run][1], f"전환 +{PFD}봉에 재평가가 발화하지 않았다"
        assert not _is_cash_only(out[fire_run][0]), (
            f"재평가가 돌았는데 목표가 그대로 현금이다: {out[fire_run][0]}")
        assert not out[fire_run + 1][1], "다음 봉에 또 발화했다"
        _ok(f"선물: 전환 {flip_run} 에 현금 잠김 → +{PFD}봉 {fire_run} 에서 {out[fire_run][0]}")

        flip0, out0 = sim(0)
        assert flip0 == flip_run
        for d in range(1, PFD + 6):
            assert _is_cash_only(out0[flip0 + d][0]), (
                f"post_flip_delay=0 인데 전환 +{d}봉에 코인이 잡혔다: {out0[flip0 + d][0]}")
        _ok("선물: 같은 자료를 post_flip_delay=0 으로 돌리면 잠긴 채 남는다 (대조군)")
    finally:
        atb.UNIVERSE = orig_universe
        atb._is_completed_bar_fresh = orig_fresh


def test_futures_freshness_guard() -> None:
    now = datetime(2026, 3, 10, 0, 5)      # UTC, tz 없음
    idx_ok = pd.to_datetime(['2026-03-07', '2026-03-08', '2026-03-09', '2026-03-10'])
    assert atb._is_completed_bar_fresh(_frame(np.arange(4.0) + 1, idx_ok), now)
    idx_stale = pd.to_datetime(['2026-03-05', '2026-03-06', '2026-03-07', '2026-03-08'])
    assert not atb._is_completed_bar_fresh(_frame(np.arange(4.0) + 1, idx_stale), now)
    _ok("선물: 완성봉 신선도 판정 (어제 봉이면 참, 밀렸으면 거짓)")


def main() -> int:
    print("전환 후 5봉 재평가 검사")
    test_replay()
    test_spot_engine()
    test_futures_engine()
    test_futures_freshness_guard()
    print("전부 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())
