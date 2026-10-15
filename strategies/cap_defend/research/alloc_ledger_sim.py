"""자산간 송금을 '사건 장부' 위에서 돈다 — 가격 인덱스 칸이 아니라 실제 날짜와 영업일.

왜 다시 쓰나. alloc_grid_sim.py 는 세 슬리브 가격 인덱스의 교집합을 축으로 썼다.
그 축은 주말이 통째로 빠져서 코인 쪽 주말 전이가 사라지고, "5칸" 이 "5영업일" 이라는
보장도 없다. 2026-10-15 토론(run-20261015T005408Z) action_steps 10 이 바로 그걸
금지했다 — "교집합 가격 인덱스의 다섯 칸을 영업일로 대신 쓰지 않는다".

그래서 여기서는 셋을 바꾼다.

  ① 축 = 달력의 모든 날. 주식 휴장일에는 주식 평가값을 그대로 들고 가고(수익 0)
     가짜 주식 신호를 만들지 않는다. 코인 주말 전이는 그대로 남는다.
  ② 영업일 = 주식 슬리브의 실제 거래일. 사전 명시한 규칙이고 성과를 보고 안 바꾼다.
     근거는 송금 경로의 병목이 증권계좌 매도·결제라는 것이다. 한국 공휴일은 이 호스트에
     자료가 없어 반영하지 않았다 — 실제보다 지연을 짧게 보는 방향이고, 아래 '안 한 것'.
  ③ 돈이 공중에 뜬다. 요청이 수락되면 그 자리에서 초과 계좌를 팔고(그 금액은 거기서
     빠져 더는 수익을 못 낸다), 5영업일 뒤에 받는 계좌의 현금으로 도착한다. 그 사이
     금액은 아무 수익도 얻지 않는다. 비중 이탈을 잴 때는 떠 있는 돈도 분모에 넣는다 —
     목표에 도달하지 않은 상태로 세야 한다.

한 계좌에 요청은 한 번에 하나다. 떠 있는 동안 새로 발화해도 무시한다(사람이 송금
하나를 보내놓고 또 보내지는 않는다). 토론의 '중복 요청' 검증 항목이 이것이다.

실행: cd strategies/cap_defend/research && python3 alloc_ledger_sim.py
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

import alloc_trigger_space as S  # noqa: E402

TGT = np.array([0.60, 0.25, 0.15])
T1 = 0.20
T3U = 0.20
LAG_BD = 5                 # 요청 수락 → 입금, 영업일 (토론이 고정)
FRIC = 0.003               # 옮긴 금액에 붙는 마찰(매매+환전+출금), 비용 스트레스에 같이 곱한다
BANDS = np.array([0.0, 0.05, 0.10])

CB_C_OR_E, CB_T1_OR_E, CB_U_OR_E, CB_E_ONLY, CB_T1_OR_UE, CB_C_AND_E = range(6)
COMBO_NAMES = ("C OR E", "T1 OR E", "U OR E", "E 단독", "T1 OR (U AND E)", "C AND E")

# 대기 W / 휴지 R 의 기간 — 0 과 네 영업일수와 달력 한 달
SPANS = (10, 21, 42, 63)
SPAN_MONTH = -1            # 달력상 한 달 (영업일수와 같은 것으로 취급하지 않는다)


# ────────────────────────────────────────────────────────────── 장부 만들기
def build_ledger(eq_st, eq_sp, eq_fu):
    """달력의 모든 날을 축으로 세 슬리브 일수익과 영업일 표식을 만든다."""
    lo = max(eq_st.index[0], eq_sp.index[0], eq_fu.index[0])
    hi = min(eq_st.index[-1], eq_sp.index[-1], eq_fu.index[-1])
    axis = pd.DatetimeIndex(pd.date_range(lo.normalize(), hi.normalize(), freq='D'))

    cols = []
    for s in (eq_st, eq_sp, eq_fu):
        s = s[~s.index.duplicated(keep='first')].sort_index()
        # 그 슬리브가 거래하지 않은 날은 평가값 유지 = 수익 0
        v = s.reindex(axis.union(s.index)).ffill().reindex(axis)
        cols.append(v.pct_change().fillna(0.0).values.astype(float))
    R = np.column_stack(cols)

    # 영업일 = 주식 슬리브가 실제로 거래한 날 (사전 명시 규칙)
    st_days = pd.DatetimeIndex(eq_st.index.normalize().unique())
    is_bd = axis.isin(st_days)
    return axis, R, np.asarray(is_bd, dtype=bool)


def bd_tables(axis, is_bd, spans=(3, 5, 10, 21, 42, 63)):
    """i 일에 수락한 요청이 n영업일 뒤 / 한 달 뒤 언제 집행되는지 미리 표로."""
    n = len(axis)
    bpos = np.flatnonzero(is_bd)
    # 각 i 에 대해 i 보다 '뒤에' 오는 영업일들의 시작 위치
    nxt = np.searchsorted(bpos, np.arange(n), side='right')

    tab = {0: np.arange(n, dtype=np.int64)}
    for k in spans:
        j = nxt + (k - 1)
        out = np.where(j < len(bpos), bpos[np.minimum(j, len(bpos) - 1)], n)
        out = np.where(j < len(bpos), out, n)          # 축 밖 = 영영 안 온다
        tab[k] = out.astype(np.int64)

    # 달력 한 달 — 같은 날짜의 다음 달, 축 안에서 그 날 이상인 첫 날
    nxt_m = (axis + pd.DateOffset(months=1)).normalize()
    pos = axis.searchsorted(nxt_m, side='left')
    tab[SPAN_MONTH] = np.where(pos < n, pos, n).astype(np.int64)
    return tab


def align_level(s, axis):
    """게이트 '수준' — 마지막 관측을 들고 간다(휴장일에 가짜 신호를 만들지 않는다)."""
    s = s[~s.index.duplicated(keep='first')].sort_index()
    v = s.reindex(axis.union(s.index)).ffill().reindex(axis)
    return v.fillna(False).values.astype(bool)


def gate_series_ledger(axis):
    """세 배분 게이트를 장부 축 위에 올린다 — 주말을 안 지우고.

    교집합으로 맞추면(alloc_trigger_space.gate_series) 주식 거래일만 남아서 코인 쪽
    주말 전이가 통째로 사라진다. 여기서는 각 계열을 축에 ffill 로 얹는다. 그러면
    주식 수준은 거래일에만 바뀌고(= 휴장일에 가짜 신호가 안 생긴다), 코인 수준은
    주말에도 바뀐다(= 주말 전이가 보존된다).
    """
    import alloc_canary as A
    from unified_backtest import load_data
    bars, _ = load_data("D")
    raw = {'주식': A.stock_canary(), '현물': A.spot_alloc_gate(bars),
           '선물': A.coin_canary(bars)}
    out = {}
    for k, s in raw.items():
        s = s[~s.index.duplicated(keep='first')].sort_index()
        v = s.reindex(pd.DatetimeIndex(axis).union(s.index)).ffill()
        out[k] = v.reindex(axis).fillna(False).astype(bool)
    return out


def event_matrix(E, axis):
    """전이 '사건' — 각 신호가 실제로 관측된 날에만 찍는다(ffill 금지)."""
    names = sorted(E)
    n = len(axis)
    out = np.zeros((n, len(names)), dtype=bool)
    src = pd.DatetimeIndex(E[names[0]].index).normalize()
    pos = axis.searchsorted(src, side='left')
    exact = (pos < n) & (src >= axis[0]) & (axis[np.minimum(pos, n - 1)] == src)
    dropped = int((~exact).sum())
    for j, nm in enumerate(names):
        hit = E[nm].values & exact
        tgt = pos[hit.nonzero()[0]]
        if len(tgt):
            out[tgt, j] = True
    return out, names, dropped


# ────────────────────────────────────────────────────────────── 시뮬레이터
def simulate_ledger(R, is_bd, gate, EV, spec, tab, lag_bd=LAG_BD, fric=FRIC,
                    month_id=None):
    """팔(후보) 축으로 벡터화한 사건 장부 시뮬.

    spec 축 (각각 길이 A 의 정수 배열):
      e_idx       전이 신호 번호, -1 이면 E 를 안 쓴다
      combo       현행과의 결합 여섯 (CB_*)
      w_key       대기 기간 키 (0 / 10 / 21 / 42 / 63 / SPAN_MONTH)
      w_all       1 이면 모든 발화를 미룬다, 0 이면 전이가 끼어 난 발화만
      r_key       휴지 기간 키 (같은 눈금)
      r_all       1 이면 모든 경로를 막는다, 0 이면 전이 경로만
      r_from_done 1 이면 휴지를 송금 완료 시점부터, 0 이면 전이 시점부터
      band_i      전이 발화의 최소 편차 (BANDS 색인)
      tscope      0 전체 목표 복귀 / 1 관련 부족 슬리브만
      block_mode  0 보통, 1 전이를 '금지 신호' 로 쓴다(휴지 동안 발화를 막기만)
    """
    n, A = R.shape[0], len(spec['e_idx'])
    e_idx = spec['e_idx'].astype(np.int64)
    combo = spec['combo'].astype(np.int64)
    w_all = spec['w_all'].astype(bool)
    r_all = spec['r_all'].astype(bool)
    r_done = spec['r_from_done'].astype(bool)
    band = BANDS[spec['band_i'].astype(np.int64)]
    partial = spec['tscope'].astype(bool)
    bmode = spec.get('block_mode', np.zeros(A, dtype=np.int64)).astype(bool)
    has_e = e_idx >= 0
    e_col = np.maximum(e_idx, 0)

    # 기간 키 → 미리 만든 표의 열
    def span_rows(keys):
        out = np.empty((n, A), dtype=np.int64)
        for k in np.unique(keys):
            out[:, keys == k] = tab[int(k)][:, None]
        return out
    w_tab = span_rows(spec['w_key'].astype(np.int64))
    r_tab = span_rows(spec['r_key'].astype(np.int64))
    w_on = spec['w_key'].astype(np.int64) != 0
    r_on = spec['r_key'].astype(np.int64) != 0
    lag_tab = tab[lag_bd]

    v = np.repeat(TGT[None, :], A, axis=0).copy()
    transit = np.zeros((A, 3))
    arrive = np.full(A, -1, dtype=np.int64)
    wait_due = np.full(A, -1, dtype=np.int64)
    pend_e = np.zeros(A, dtype=bool)       # 대기 중인 요청이 전이 때문에 났나
    block_until = np.zeros(A, dtype=np.int64)

    pv = np.empty((n, A))
    fires = np.zeros(A, dtype=np.int64)
    reqs = np.zeros(A, dtype=np.int64)
    dones = np.zeros(A, dtype=np.int64)
    moved = np.zeros(A)
    max_ht = np.zeros(A)
    ht_hist = np.empty((n, A))
    cancels = np.zeros(A, dtype=np.int64)
    # 달력월에 두 번 이상 입금된 달의 수 (토론 통과 기준)
    if month_id is None:
        month_id = np.zeros(n, dtype=np.int64)
    mo_cnt = np.zeros(A, dtype=np.int64)
    multi_mo = np.zeros(A, dtype=np.int64)
    cur_mo = int(month_id[0]) if n else 0

    fr = fric

    def do_request(sel, i, rel_mask, do_par):
        """sel 인 팔에서 지금 송금 요청을 집행한다 — 팔고, 띄우고, 도착일을 적는다."""
        if not sel.any():
            return
        tot = v.sum(axis=1) + transit.sum(axis=1)
        tgt_v = tot[:, None] * TGT[None, :]
        over = np.maximum(v - tgt_v, 0.0)
        short = np.maximum(tgt_v - v, 0.0)
        recv = np.where(do_par[:, None], short * rel_mask, short)
        need = recv.sum(axis=1)
        avail = over.sum(axis=1)
        amt = np.minimum(need, avail)
        live = sel & (amt > 1e-12)
        dead = sel & ~live
        if dead.any():                      # 만기에 다시 보니 보낼 게 없다 = 취소
            cancels[dead] += 1
        if not live.any():
            return
        # 조달: 초과 계좌에서 초과액 비례
        take = np.zeros_like(v)
        a = avail.copy()
        a[a <= 0] = 1.0
        take[live] = over[live] * (amt[live] / a[live])[:, None]
        v[live] -= take[live]
        # 배분: 받는 쪽 부족액 비례, 마찰 차감
        rr = recv.sum(axis=1).copy()
        rr[rr <= 0] = 1.0
        share = np.zeros_like(v)
        share[live] = recv[live] * (amt[live] / rr[live])[:, None] * (1.0 - fr)
        transit[live] += share[live]
        arrive[live] = lag_tab[i]
        moved[live] += amt[live]
        reqs[live] += 1
        # 지연 0 진단 팔 — 도착 검사는 하루 앞에서 이미 지나갔으므로 여기서 바로 넣는다
        inst = live & (arrive == i)
        if inst.any():
            v[inst] += transit[inst]
            transit[inst] = 0.0
            arrive[inst] = -1
            dones[inst] += 1

    for i in range(n):
        if int(month_id[i]) != cur_mo:
            multi_mo += (mo_cnt >= 2)
            mo_cnt[:] = 0
            cur_mo = int(month_id[i])
        v *= (1.0 + R[i])[None, :]

        hit = arrive == i
        if hit.any():
            v[hit] += transit[hit]
            transit[hit] = 0.0
            arrive[hit] = -1
            dones[hit] += 1
            mo_cnt[hit] += 1
            upd = hit & r_on & r_done & (r_all | pend_e)
            if upd.any():
                block_until[upd] = r_tab[i][upd]
            pend_e[hit] = False

        tot = v.sum(axis=1) + transit.sum(axis=1)
        pv[i] = tot
        w = v / tot[:, None]
        ht = np.abs(w - TGT).sum(axis=1) / 2
        ht_hist[i] = ht
        max_ht = np.maximum(max_ht, ht)

        due = wait_due == i
        if due.any():
            under = (TGT[None, :] - w) / TGT[None, :]
            u_s = (under >= T3U) & gate[i][None, :]
            rel = (u_s | pend_e[:, None]) & (under > 0)
            do_request(due, i, rel, partial & due)
            wait_due[due] = -1

        free = (wait_due < 0) & (arrive < 0)
        if not free.any():
            continue

        under = (TGT[None, :] - w) / TGT[None, :]
        u_s = (under >= T3U) & gate[i][None, :]
        u = u_s.any(axis=1)
        t1 = ht >= T1
        e_raw = EV[i][e_col] & has_e
        e = e_raw & (ht >= band)

        blocked = block_until > i
        # 전이 경로만 막는 휴지 = E 쪽만 죽인다. 전부 막는 휴지 = 발화 자체를 막는다.
        e_eff = e & ~(blocked & ~r_all)
        hard = blocked & r_all

        def eval_combo(ev):
            c = t1 | u
            out = np.zeros(A, dtype=bool)
            out |= (combo == CB_C_OR_E) & (c | ev)
            out |= (combo == CB_T1_OR_E) & (t1 | ev)
            out |= (combo == CB_U_OR_E) & (u | ev)
            out |= (combo == CB_E_ONLY) & ev
            out |= (combo == CB_T1_OR_UE) & (t1 | (u & ev))
            out |= (combo == CB_C_AND_E) & (c & ev)
            return out

        # 금지 신호 갈래: 결합은 현행 C 고정이고 E 는 휴지를 여는 데만 쓴다.
        # r_all=0 이면 그 휴지 동안 U 만 막고 T1 은 살린다, 1 이면 C 전체를 막는다.
        u_b = u & ~(blocked & bmode & ~r_all)
        cand = np.where(bmode, t1 | u_b, eval_combo(e_eff))
        no_e = np.where(bmode, t1 | u_b, eval_combo(np.zeros(A, dtype=bool)))
        fire = cand & free & ~hard
        dep_e = fire & ~no_e

        if bmode.any():
            open_b = bmode & e_raw & r_on
            if open_b.any():
                sel = open_b & ~r_done
                if sel.any():
                    block_until[sel] = r_tab[i][sel]

        if fire.any():
            fires += fire
            delay = fire & w_on & (w_all | dep_e)
            now = fire & ~delay
            if delay.any():
                wait_due[delay] = w_tab[i][delay]
                pend_e[delay] = dep_e[delay]
            if now.any():
                rel = (u_s | (dep_e[:, None] & (under > 0))) & (under > 0)
                do_request(now, i, rel, partial & now)
                pend_e[now] = dep_e[now]
            opened = fire & r_on & ~r_done & (r_all | dep_e)
            if opened.any():
                block_until[opened] = r_tab[i][opened]

    multi_mo += (mo_cnt >= 2)
    info = dict(fires=fires, reqs=reqs, dones=dones, moved=moved,
                max_ht=max_ht, cancels=cancels, multi_months=multi_mo,
                ht_p95=np.percentile(ht_hist, 95, axis=0))
    return pv, info


# ────────────────────────────────────────────────────────────── 자기 점검
def _spec(A, **kw):
    d = dict(e_idx=-1, combo=CB_C_OR_E, w_key=0, w_all=1, r_key=0, r_all=1,
             r_from_done=0, band_i=0, tscope=0, block_mode=0)
    d.update(kw)
    return {k: np.full(A, v, dtype=np.int64) for k, v in d.items()}


def main() -> int:
    import time
    import bt_alloc_transfer_lag as H

    t0 = time.time()
    eq_st = H.run_stock_live()
    eq_sp = H.run_spot_live()
    eq_fu = H.run_fut('margin', 0.015)['_equity']
    print(f"슬리브 로드 {time.time() - t0:.0f}s")

    axis, R, is_bd = build_ledger(eq_st, eq_sp, eq_fu)
    tab = bd_tables(axis, is_bd)
    g = gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    E = S.build_E(g)
    EV, names, dropped = event_matrix(E, axis)

    n = len(axis)
    print(f"\n장부 축 {axis[0].date()} ~ {axis[-1].date()} ({n}일, 주말 포함)")
    print(f"  영업일 {int(is_bd.sum())}일 ({is_bd.mean() * 100:.1f}%) — 주식 슬리브 실제 거래일")
    dow = np.asarray(pd.DatetimeIndex(axis).dayofweek)
    wk = pd.Series(dow[is_bd]).value_counts().sort_index()
    print(f"  영업일 요일 분포 월~일 {[int(wk.get(d, 0)) for d in range(7)]}")
    wknd = int((dow < 5).sum())
    print(f"  축의 평일 {wknd}일 중 영업일 {int(is_bd.sum())}일"
          f" → 평일 휴장 {wknd - int(is_bd.sum())}일")
    print(f"  전이 사건을 축 밖으로 흘린 날 {dropped}건")
    # 주말 전이가 실제로 살아 있나 (교집합 축에서는 통째로 사라지던 것)
    wk_mask = dow >= 5
    print(f"  E 발화일 합계 {int(EV.sum())} 중 주말 {int(EV[wk_mask].sum())}건"
          f" — 교집합 축이면 전부 유실됐을 분량")
    lag_ok = tab[5]
    sample = [i for i in range(0, n, max(1, n // 4))][:4]
    for i in sample:
        j = lag_tab_i = int(lag_ok[i])
        if j < n:
            bd_between = int(is_bd[i + 1:j + 1].sum())
            print(f"  5영업일 확인 {axis[i].date()}({axis[i].day_name()[:3]})"
                  f" → {axis[j].date()}({axis[j].day_name()[:3]}) 사이 영업일 {bd_between}")

    # ── 정합 ① 송금 없음 = 세 슬리브를 한 번 넣고 방치한 값
    A = 1
    sp_none = _spec(A, combo=CB_E_ONLY)       # E 없음 + E 단독 = 영영 발화 없음
    pv0, i0 = simulate_ledger(R, is_bd, gate, EV, sp_none, tab)
    drift = (TGT * np.cumprod(1 + R, axis=0)).sum(axis=1)
    d = float(np.abs(pv0[:, 0] / drift - 1).max())
    print(f"\n[정합 ①] 무송금 곡선 = 직접 누적한 방치 곡선"
          f"  발화 {int(i0['fires'][0])} 요청 {int(i0['reqs'][0])}"
          f"  상대 최대차 {d:.2e}  {'일치' if d < 1e-10 else '어긋남'}")

    # ── 정합 ② 돈이 떠 있는 동안 수익을 안 낸다
    A = 3
    sp2 = _spec(A)
    pv2, i2 = simulate_ledger(R, is_bd, gate, EV, sp2, tab, lag_bd=0)
    pv2b, i2b = simulate_ledger(R, is_bd, gate, EV, sp2, tab, lag_bd=5)
    print(f"\n[정합 ②] 공중에 뜨는 값 (현행 C, 같은 발화)")
    print(f"  지연 0영업일  요청 {int(i2['reqs'][0])} 입금 {int(i2['dones'][0])}"
          f" 최종 {pv2[-1, 0]:,.1f}")
    print(f"  지연 5영업일  요청 {int(i2b['reqs'][0])} 입금 {int(i2b['dones'][0])}"
          f" 최종 {pv2b[-1, 0]:,.1f}")
    print(f"  차이 {pv2b[-1, 0] - pv2[-1, 0]:+,.1f}"
          f"  ({'떠 있는 동안 수익이 멈춘다' if pv2b[-1, 0] != pv2[-1, 0] else '!! 같다'})")

    # ── 정합 ③ 축마다 움직이나
    eb = int(EV.sum(axis=0).argmax())
    print(f"\n[정합 ③] 축마다 실제로 움직이는가"
          f" (E = {names[eb]}, 발화 {int(EV[:, eb].sum())}일)")
    rows = [("기준", dict(e_idx=eb)),
            ("대기 21영업일 전부", dict(e_idx=eb, w_key=21, w_all=1)),
            ("대기 한 달 전부", dict(e_idx=eb, w_key=SPAN_MONTH, w_all=1)),
            ("휴지 42영업일 전부", dict(e_idx=eb, r_key=42, r_all=1)),
            ("휴지 42영업일 송금후", dict(e_idx=eb, r_key=42, r_all=1, r_from_done=1)),
            ("부분 송금", dict(e_idx=eb, tscope=1)),
            ("E 단독", dict(e_idx=eb, combo=CB_E_ONLY)),
            ("E 단독+최소편차 5pp", dict(e_idx=eb, combo=CB_E_ONLY, band_i=1)),
            ("E 단독+최소편차 10pp", dict(e_idx=eb, combo=CB_E_ONLY, band_i=2)),
            ("금지신호 U만 차단", dict(e_idx=eb, r_key=42, r_all=0, block_mode=1)),
            ("금지신호 C전체 차단", dict(e_idx=eb, r_key=42, r_all=1, block_mode=1))]
    for label, kw in rows:
        p, inf = simulate_ledger(R, is_bd, gate, EV, _spec(1, **kw), tab)
        print(f"  {label:<18} 발화 {int(inf['fires'][0]):>3}"
              f" 요청 {int(inf['reqs'][0]):>3} 입금 {int(inf['dones'][0]):>3}"
              f" 취소 {int(inf['cancels'][0]):>2}"
              f" 최대이탈 {float(inf['max_ht'][0]):.4f}"
              f" 95분위 {float(inf['ht_p95'][0]):.4f}"
              f" 옮긴양 {float(inf['moved'][0]):>7,.2f}"
              f" 최종 {p[-1, 0]:>9,.1f}")

    # ── 속도
    A3 = 20000
    rng = np.random.default_rng(0)
    keys = np.array([0, 10, 21, 42, 63, SPAN_MONTH])
    sp3 = dict(
        e_idx=rng.integers(-1, len(names), A3),
        combo=rng.integers(0, 6, A3),
        w_key=rng.choice(keys, A3), w_all=rng.integers(0, 2, A3),
        r_key=rng.choice(keys, A3), r_all=rng.integers(0, 2, A3),
        r_from_done=rng.integers(0, 2, A3),
        band_i=rng.integers(0, 3, A3), tscope=rng.integers(0, 2, A3),
        block_mode=np.zeros(A3, dtype=np.int64),
    )
    t0 = time.time()
    simulate_ledger(R, is_bd, gate, EV, sp3, tab)
    dt = time.time() - t0
    print(f"\n속도: 무작위 후보 {A3:,}개 {dt:.2f}s  →  팔 하나 {dt / A3 * 1000:.3f}ms")
    print(f"  고유 후보 708,563 × 위상 7 × 비용 2 = "
          f"{708563 * 14 * dt / A3 / 3600:.1f} 시간 (한 코어)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
