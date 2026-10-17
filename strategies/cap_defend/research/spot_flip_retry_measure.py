"""전환일 전부 현금 잠김 — 후보 넷을 사전등록(spot_flip_retry_prereg.md)대로 잰다.

현물만 바꾼다. 선물·주식은 합성 60/25/15 의 고정 자리라 전 팔에서 같은 곡선을 쓴다.
결정층은 합성의 창 기반 순위합이고 슬리브 단독은 진단으로만 적는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 spot_flip_retry_measure.py
중간 곡선은 state/20261018-fliplock/eq/*.csv 에 남고 다시 돌리면 이어 간다.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261018-fliplock"
EQDIR = os.path.join(OUT, "eq")
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

CUR = "현행"
LIVE = "라이브현행"

# (이름, 현물 덮어쓰기, FLIP_ALL_CASH_RETRY, post_flip_delay)
ARMS = [
    (CUR,            {}, "off",   5),
    (LIVE,           {}, "off",   0),
    ("가-다음봉",     {}, "next",  5),
    ("나-생길때까지", {}, "until", 5),
    ("다-단기18일",   dict(mom_short_bars=18), "off", 5),
]


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("±", "pm")


def _p(kind: str, key: str) -> str:
    return os.path.join(EQDIR, f"{kind}__{key}.csv")


def _load(kind: str, key: str):
    p = _p(kind, key)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(_p(kind, key), header=["eq"])


def run_spot(over: dict, retry: str, pfd: int, phase: int, stress: bool,
             trace: list | None = None) -> pd.Series:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["FLIP_ALL_CASH_RETRY"] = retry
    kw = dict(R.SPOT_BASE)
    kw.update(over)
    try:
        m = bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
                   tx_cost=R.TX_SPOT * (R.STRESS if stress else 1.0),
                   start_date=R.START, end_date=R.END,
                   vol_mode="daily", phase_offset_bars=phase,
                   post_flip_delay=pfd, selection="greedy",
                   stop_kind="none", stop_pct=0.0,
                   dd_lookback=60, dd_threshold=-99.0,
                   bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
                   health_mode="mom2vol", _trace=trace, **kw)
    finally:
        os.environ["FLIP_ALL_CASH_RETRY"] = "off"
    return m.get("_equity")


def canary_series() -> pd.Series:
    """BTC 종가 대 SMA42, ±1.5% 히스테리시스. 백테스트와 같은 상태기계."""
    c = R._bars["BTC"]["Close"]
    sma = c.rolling(42).mean()
    on = False
    out = {}
    for d, px, sv in zip(c.index, c.values, sma.values):
        if np.isnan(sv):
            out[d] = False
            continue
        if on:
            on = px > sv * (1 - R.SPOT_BASE["canary_hyst"])
        else:
            on = px > sv * (1 + R.SPOT_BASE["canary_hyst"])
        out[d] = on
    # 백테스트는 t 봉의 카나리를 t-1 까지의 종가로 판정한다 — 한 봉 민다.
    return pd.Series(out).shift(1).fillna(False).astype(bool)


def census(tr: list, can: pd.Series) -> list[dict]:
    """전환 ON 인데 그날 목표가 전부 현금인 사건과 그 잠김 길이."""
    tgt = {pd.Timestamp(r["date"]): r["target"] for r in tr}
    dates = sorted(tgt)
    cs = can.reindex(dates).fillna(False)
    ev = []
    prev = False
    for i, d in enumerate(dates):
        onv = bool(cs.iloc[i])
        if onv and not prev:
            t = tgt[d]
            if set(t.keys()) == {"CASH"}:
                n = 0
                for j in range(i, len(dates)):
                    if not bool(cs.iloc[j]):
                        break
                    if set(tgt[dates[j]].keys()) != {"CASH"}:
                        break
                    n += 1
                ev.append(dict(date=str(d.date()), locked_bars=n))
        prev = onv
    return ev


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    # 주식 — 고정 자리. 9월 스윕 캐시가 같은 구간·같은 앵커라 그대로 쓴다.
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(STOCK_CACHE, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(os.path.join(EQDIR, f"st__a{a}.csv"), header=["eq"])
    print(f"주식 곡선 {len(stock_eq)}개 ({time.time()-t0:.0f}초)", flush=True)

    # 선물 — 고정 자리.
    fut_eq = {}
    for a in R.ANCHORS:
        for st in (False, True):
            k = f"a{a}{'__s5' if st else ''}"
            s = _load("fu", k)
            if s is None:
                s = R.run_fut({}, R.PH_FUT[a], st)
                _save("fu", k, s)
            fut_eq[(a, st)] = s
    print(f"선물 곡선 {len(fut_eq)}개 ({time.time()-t0:.0f}초)", flush=True)

    # 현물 — 팔마다.
    sp_eq = {}
    for name, over, retry, pfd in ARMS:
        for a in R.ANCHORS:
            for st in (False, True):
                k = f"{_safe(name)}__a{a}{'__s5' if st else ''}"
                s = _load("sp", k)
                if s is None:
                    s = run_spot(over, retry, pfd, R.PH_SPOT[a], st)
                    _save("sp", k, s)
                sp_eq[(name, a, st)] = s
        print(f"  현물 {name} 완료 ({time.time()-t0:.0f}초)", flush=True)

    # ── 사건 집계 (현행 / 라이브현행) ──
    can = canary_series()
    cen = {}
    for name, over, retry, pfd in ARMS:
        if name not in (CUR, LIVE):
            continue
        rows = []
        for a in R.ANCHORS:
            tr: list = []
            run_spot(over, retry, pfd, R.PH_SPOT[a], False, trace=tr)
            rows.append(census(tr, can))
        cen[name] = rows
    print(f"사건 집계 완료 ({time.time()-t0:.0f}초)", flush=True)

    # ── 보고 ──
    names = [a[0] for a in ARMS]
    result = {}
    for st in (False, True):
        rank = {n: [] for n in names}
        cal = {n: [] for n in names}
        mdd = {n: [] for n in names}
        shp = {n: [] for n in names}
        cagr = {n: [] for n in names}
        sp_cal = {n: [] for n in names}
        for a in R.ANCHORS:
            comps = {}
            for n in names:
                comps[n] = R.build_alloc(stock_eq[a], sp_eq[(n, a, st)], fut_eq[(a, st)])
                m = R.metrics(comps[n])
                cal[n].append(m["Cal"]); mdd[n].append(m["MDD"])
                shp[n].append(m["Sharpe"]); cagr[n].append(m["CAGR"])
                sp_cal[n].append(R.metrics(sp_eq[(n, a, st)])["Cal"])
            sums, _, nw = R.window_rs(comps)
            for n in names:
                rank[n].append(sums[n] / max(nw, 1))

        tag = "비용 5배" if st else "기본 비용"
        print(f"\n=== 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 ===")
        print(f"{'팔':<16}{'순위평균':>9}{'중앙':>7}{'최악':>7}{'현행대비':>9}"
              f"{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}{'현물칼마':>9}")
        base = float(np.mean(rank[CUR]))
        rows = sorted((float(np.mean(rank[n])), n) for n in names)
        for mr, n in rows:
            mark = "  <- 현행" if n == CUR else ""
            print(f"{n:<16}{mr:>9.3f}{float(np.median(rank[n])):>7.3f}"
                  f"{max(rank[n]):>7.3f}{mr-base:>+9.3f}"
                  f"{float(np.mean(cal[n])):>7.2f}{float(np.mean(shp[n])):>7.2f}"
                  f"{float(np.mean(cagr[n])):>8.2f}{float(np.mean(mdd[n])):>9.2f}"
                  f"{float(np.mean(sp_cal[n])):>9.2f}{mark}")
        print("  앵커별 순위 차이(현행 기준) — 음수면 그 팔이 낫다")
        for mr, n in rows:
            if n == CUR:
                continue
            d = [x - y for x, y in zip(rank[n], rank[CUR])]
            print(f"    {n:<14} " + " ".join(f"{x:+6.2f}" for x in d)
                  + f"   이긴 앵커 {sum(1 for x in d if x < 0)}/{len(d)}")

        # 사전등록 기준 판정
        print("  사전등록 기준 — 1 순위우세 2 앵커 5/7 3 낙폭 -1.0pp 이내 4 칼마 비열위")
        for n in names:
            if n in (CUR, LIVE):
                continue
            d = [x - y for x, y in zip(rank[n], rank[CUR])]
            g1 = float(np.mean(rank[n])) < base
            g2 = sum(1 for x in d if x < 0) >= 5
            g3 = float(np.mean(mdd[n])) >= float(np.mean(mdd[CUR])) - 1.0
            g4 = float(np.mean(cal[n])) >= float(np.mean(cal[CUR]))
            print(f"    {n:<14} 1:{'O' if g1 else 'X'} 2:{'O' if g2 else 'X'} "
                  f"3:{'O' if g3 else 'X'} 4:{'O' if g4 else 'X'}")

        result[tag] = {n: dict(rank=rank[n], cal=cal[n], mdd=mdd[n],
                               sharpe=shp[n], cagr=cagr[n], spot_cal=sp_cal[n])
                       for n in names}

    print("\n=== 전환일 전부 현금 사건 (앵커별) ===")
    for n, rows in cen.items():
        tot = sum(len(r) for r in rows)
        lens = [e["locked_bars"] for r in rows for e in r]
        print(f"  {n:<10} 사건 {tot}건 · 잠김 봉수 평균 "
              f"{float(np.mean(lens)) if lens else 0:.1f} · 최대 {max(lens) if lens else 0}")
        seen = {}
        for r in rows:
            for e in r:
                seen.setdefault(e["date"], []).append(e["locked_bars"])
        for d in sorted(seen):
            print(f"    {d}  잠김 {seen[d]}")

    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(dict(metrics=result, census=cen), f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
