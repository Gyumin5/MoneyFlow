"""자산간 문턱을 원화 장부 위에서 다시 채점한다.

묻는 것 하나 — 달러 기준으로 고른 현행 문턱(반회전율 25pp · 상대부족 20%)이
원화 기준에서도 같은 선택인가.

규칙은 state/20261103-krw-ledger/규칙.md 에 먼저 적었고 결과를 보고 안 고친다.
요약하면 슬리브 곡선은 달러 그대로 두고 원화 환산만 제대로 한다. 네 단계를
차례로 재서 무엇이 무엇을 바꾸는지 가른다.

  0 달러          기준선
  1 +환율         셋 다 × 환율. 공통이라 비중에서 약분돼야 한다(검사 ①)
  2 +통화별 현금   현물 현금은 원화 → 환율을 안 받는다
  3 +김프         현물은 코인 김프, 선물은 테더 김프

일수익 수준에서 가른다. 현물 달러 일수익 r 과 전일 현금비중 c 가 주어지면
코인 부분의 달러 수익은 r/(1-c) 이고, 원화 수익은

    r_krw = (1-c)·[(1 + r/(1-c))·(1+환율변동)·(1+김프변동) - 1]

c=0 이면 (1+r)(1+fx)(1+kimp)-1 이고 c=1 이면 0 이다(원화 현금은 가만히 있다).
현물은 전 구간의 절반이 전액 현금이라 이 구분이 실제로 곡선을 바꾼다.

실행: python3 krw_score.py          (전체)
      python3 krw_score.py --quick  (앵커 0, 비용 기본만)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

RES = "/home/gmoh/mon/251229/strategies/cap_defend/research"
PARENT = os.path.dirname(RES)
for _p in (RES, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HERE = "/home/gmoh/mon/251229/state/20261103-krw-ledger"
EQDIR = os.path.join(HERE, "eq")
FXDIR = "/home/gmoh/mon/251229/state/20261103-fx-kimchi"

TGT = np.array([0.60, 0.20, 0.20])          # 현행 정본 (recommend.py)
CUR_T1, CUR_T3U = 0.25, 0.20                # 현행 문턱
T1_GRID = [0.20, 0.25, 0.30]
T3U_GRID = [0.15, 0.20, 0.25]
STAGES = ("달러", "+환율", "+통화별현금", "+김프")

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]


# ─────────────────────────────────────────────── 자재
def load_fx(axis):
    """환율·김프를 장부 축에 올린다. 김프는 퍼센트로 저장돼 있어 비율로 바꾼다."""
    fx = pd.read_csv(os.path.join(FXDIR, "usdkrw.csv"), index_col=0,
                     parse_dates=True).iloc[:, 0]
    kb = pd.read_csv(os.path.join(FXDIR, "kimp_btc.csv"), index_col=0,
                     parse_dates=True)["kimp"] / 100.0
    ku = pd.read_csv(os.path.join(FXDIR, "kimp_usdt.csv"), index_col=0,
                     parse_dates=True)["kimp_usdt"] / 100.0

    def on_axis(s):
        s = s[~s.index.duplicated(keep="first")].sort_index()
        return s.reindex(pd.DatetimeIndex(axis).union(s.index)).ffill().reindex(axis)

    fx, kb, ku = on_axis(fx), on_axis(kb), on_axis(ku)
    # 테더 김프가 없는 앞 구간은 코인 김프로 메운다 (규칙 4절, 겹치는 591일 차이 중앙 -0.01%p)
    ku_filled = ku.where(ku.notna(), kb)
    return (fx.ffill().bfill().values.astype(float),
            kb.ffill().bfill().values.astype(float),
            ku_filled.ffill().bfill().values.astype(float),
            int(ku.notna().sum()))


def spot_cash_frac(eq_sp, axis, anchor):
    """현물 현금 비중 c 를 장부 축 위에 세운다.

    리밸 날은 그날 목표의 CASH 비중으로 리셋하고, 그 사이는 현금이 가만히 있으므로
    c(t) = c(t-1) / (1 + 그날 달러 일수익) 이다 — 전체가 (1+r) 배가 되는 동안
    현금 액수만 그대로라서 비중이 그만큼 줄어든다.
    """
    tg = pd.read_csv(os.path.join(HERE, f"targets_spot_a{anchor}.csv"),
                     parse_dates=["date"])
    cash = tg[tg["coin"] == "CASH"].set_index("date")["w"]
    reb = tg.groupby("date")["rebal"].first()
    reb = reb[reb.astype(str).str.lower().isin(("true", "1"))]
    reb_days = set(pd.DatetimeIndex(reb.index).normalize())

    s = eq_sp[~eq_sp.index.duplicated(keep="first")].sort_index()
    v = s.reindex(pd.DatetimeIndex(axis).union(s.index)).ffill().reindex(axis)
    r = v.pct_change().fillna(0.0).values.astype(float)

    cash = cash[~cash.index.duplicated(keep="first")].sort_index()
    cash.index = pd.DatetimeIndex(cash.index).normalize()
    cmap = cash.to_dict()

    n = len(axis)
    c = np.zeros(n)
    cur = float(cmap.get(pd.Timestamp(axis[0]).normalize(), 0.0))
    for i in range(n):
        d = pd.Timestamp(axis[i]).normalize()
        if d in reb_days and d in cmap:
            cur = float(cmap[d])
        elif i > 0:
            den = 1.0 + r[i]
            cur = cur / den if den > 1e-12 else cur
        c[i] = min(max(cur, 0.0), 1.0)
    return c


def krw_returns(R_usd, c_spot, fx, kb, ku, stage):
    """달러 일수익 행렬을 단계에 맞춰 원화 일수익으로 바꾼다. 열 = 주식·현물·선물."""
    if stage == 0:
        return R_usd.copy()
    g_fx = np.zeros(len(fx)); g_fx[1:] = fx[1:] / fx[:-1] - 1.0
    g_kb = np.zeros(len(kb)); g_kb[1:] = (1 + kb[1:]) / (1 + kb[:-1]) - 1.0
    g_ku = np.zeros(len(ku)); g_ku[1:] = (1 + ku[1:]) / (1 + ku[:-1]) - 1.0

    out = np.empty_like(R_usd)
    one_fx = 1.0 + g_fx
    # 주식 — 전체가 달러(외화예수금 포함). 환율만.
    out[:, 0] = (1.0 + R_usd[:, 0]) * one_fx - 1.0
    # 선물 — 전체가 테더. 환율 × 테더 김프.
    k_fu = (1.0 + g_ku) if stage >= 3 else 1.0
    out[:, 2] = (1.0 + R_usd[:, 2]) * one_fx * k_fu - 1.0
    # 현물 — 코인은 환율 × 코인 김프, 현금은 원화라 가만히 있다.
    k_sp = (1.0 + g_kb) if stage >= 3 else 1.0
    if stage == 1:
        out[:, 1] = (1.0 + R_usd[:, 1]) * one_fx - 1.0
    else:
        cp = np.empty(len(c_spot)); cp[0] = c_spot[0]; cp[1:] = c_spot[:-1]
        coin = 1.0 - cp
        safe = np.where(coin > 1e-9, coin, 1.0)
        r_coin = np.where(coin > 1e-9, R_usd[:, 1] / safe, 0.0)
        grown = (1.0 + r_coin) * one_fx * k_sp
        out[:, 1] = np.where(coin > 1e-9, coin * (grown - 1.0), 0.0)
    return out


# ─────────────────────────────────────────────── 창 채점 (alloc_grid_run 규약 그대로)
def window_bounds(is_bd):
    bpos = np.flatnonzero(is_bd)
    out = []
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(bpos) - size, stride):
                out.append((int(bpos[i]), int(bpos[i + size - 1])))
    return out


def window_cal(pv, axis, bounds):
    out = np.empty((len(bounds), pv.shape[1]))
    for k, (i0, i1) in enumerate(bounds):
        seg = pv[i0:i1 + 1]
        yrs = (axis[i1] - axis[i0]).days / 365.25
        g = (seg[-1] / seg[0]) ** (1.0 / yrs) - 1.0
        dd = (seg / np.maximum.accumulate(seg, axis=0) - 1.0).min(axis=0)
        out[k] = np.where(dd < 0, g / np.abs(np.where(dd < 0, dd, 1)), 0.0)
    return out


def full_stats(pv, axis):
    yrs = (axis[-1] - axis[0]).days / 365.25
    g = (pv[-1] / pv[0]) ** (1.0 / yrs) - 1.0
    dd = (pv / np.maximum.accumulate(pv, axis=0) - 1.0).min(axis=0)
    r = pv[1:] / pv[:-1] - 1.0
    sd = r.std(axis=0)
    sh = np.where(sd > 0, r.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)
    cal = np.where(dd < 0, g / np.abs(np.where(dd < 0, dd, 1)), 0.0)
    return dict(CAGR=g, MDD=dd, Sharpe=sh, Cal=cal)


# ─────────────────────────────────────────────── 한 칸
def run_cell(anchor, stress, L, S):
    tagx = "x5" if stress else "x1"
    eq = {}
    for kind in ("stock", "spot", "fut"):
        p = os.path.join(EQDIR, f"{kind}_a{anchor}_{tagx}.csv")
        eq[kind] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]

    axis, R_usd, is_bd = L.build_ledger(eq["stock"], eq["spot"], eq["fut"])
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    EV = np.zeros((len(axis), 1), dtype=bool)        # 전이 신호는 안 쓴다
    month_id = np.asarray(pd.DatetimeIndex(axis).year * 12
                          + pd.DatetimeIndex(axis).month, dtype=np.int64)
    bounds = window_bounds(is_bd)
    yrs = (axis[-1] - axis[0]).days / 365.25

    fx, kb, ku, n_ku = load_fx(axis)
    c_spot = spot_cash_frac(eq["spot"], axis, anchor)

    def one(Rm, t1, t3u):
        L.TGT, L.T1, L.T3U = TGT, t1, t3u
        spec = {k: np.array([v], dtype=np.int64) for k, v in
                dict(e_idx=-1, combo=L.CB_C_OR_E, w_key=0, w_all=1, r_key=0,
                     r_all=1, r_from_done=0, band_i=0, tscope=0,
                     block_mode=0).items()}
        return L.simulate_ledger(Rm, is_bd, gate, EV, spec, tab, month_id=month_id)

    cell = {}
    for st, sname in enumerate(STAGES):
        Rm = krw_returns(R_usd, c_spot, fx, kb, ku, st)
        pv_b, inf_b = one(Rm, CUR_T1, CUR_T3U)
        cal_b = window_cal(pv_b, axis, bounds)[:, 0]
        rows = {}
        for t1 in T1_GRID:
            for t3u in T3U_GRID:
                pv, inf = one(Rm, t1, t3u)
                cal = window_cal(pv, axis, bounds)[:, 0]
                rk = float(np.where(cal > cal_b, 1.0,
                                    np.where(cal < cal_b, 2.0, 1.5)).mean())
                fs = full_stats(pv, axis)
                rows[f"{int(t1 * 100)}/{int(t3u * 100)}"] = dict(
                    rank=rk, Cal=float(fs["Cal"][0]), MDD=float(fs["MDD"][0]),
                    Sharpe=float(fs["Sharpe"][0]), CAGR=float(fs["CAGR"][0]),
                    per_yr=float(inf["dones"][0]) / yrs,
                    ht95=float(inf["ht_p95"][0]), maxht=float(inf["max_ht"][0]),
                    dones=int(inf["dones"][0]))
        cell[sname] = dict(rows=rows,
                           base=dict(Cal=float(full_stats(pv_b, axis)["Cal"][0]),
                                     MDD=float(full_stats(pv_b, axis)["MDD"][0]),
                                     per_yr=float(inf_b["dones"][0]) / yrs))
    cell["_meta"] = dict(anchor=anchor, stress=bool(stress), 창=len(bounds),
                         연수=round(yrs, 2), 영업일=int(is_bd.sum()),
                         테더김프_실측일=n_ku,
                         현물현금_평균=round(float(c_spot.mean()), 4))
    return cell


def main() -> int:
    quick = "--quick" in sys.argv
    import alloc_ledger_sim as L
    import alloc_trigger_space as S

    anchors = [0] if quick else list(range(7))
    stresses = [False] if quick else [False, True]

    out, t0 = {}, time.time()
    for stress in stresses:
        for a in anchors:
            key = f"a{a}_{'x5' if stress else 'x1'}"
            out[key] = run_cell(a, stress, L, S)
            m = out[key]["_meta"]
            print(f"{key} 완료 · 창 {m['창']} · {time.time() - t0:.0f}초", flush=True)

    p = os.path.join(HERE, "score_quick.json" if quick else "score.json")
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
