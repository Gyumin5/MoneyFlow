#!/usr/bin/env python3
"""추가 검증 D: Intrabar liquidation 근사 + Funding 반영.

A. 각 fut event에 대해 bars_held 구간의 Low 기준 MAE 계산.
   3x lev × MAE ≤ -0.95 (margin wipeout threshold) → 강제 청산 (pnl = -33.3%/lev 수준).
B. Binance funding CSV 8h 단위 → 이벤트 보유 중 누적 funding cost 차감 (롱은 양수 funding시 손실).
C. Cap 0.12 / 0.25 / 0.30 각각 Full/Train/Holdout 재시뮬.
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_FUT_OPTS, TRAIN_END, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, load_bars_1h, slice_v21,
                     run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

FUTURES_DATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "futures"))


def apply_liq_and_funding(ev: pd.DataFrame, lev: float = 3.0,
                           wipeout_thresh: float = -0.95,
                           include_funding: bool = True) -> pd.DataFrame:
    """각 fut event에 intrabar liquidation + funding 반영."""
    if len(ev) == 0:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"] = pd.to_datetime(ev["exit_ts"])

    # funding 캐시
    fund_cache = {}
    bar_cache = {}

    n_liq = 0
    new_pnls = []
    new_exits = []
    new_reasons = []
    funding_applied = []

    for _, r in ev.iterrows():
        coin = r["coin"]
        # 1h bars
        if coin not in bar_cache:
            bar_cache[coin] = load_bars_1h(coin)
        bars = bar_cache[coin]
        if bars is None:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            funding_applied.append(0.0)
            continue

        # funding
        if coin not in fund_cache:
            p = os.path.join(FUTURES_DATA, f"{coin}USDT_funding.csv")
            if os.path.isfile(p):
                fd = pd.read_csv(p)
                tcol = "fundingTime" if "fundingTime" in fd.columns else fd.columns[0]
                rcol = "fundingRate" if "fundingRate" in fd.columns else fd.columns[1]
                fd[tcol] = pd.to_datetime(fd[tcol])
                fd = fd.set_index(tcol)
                fund_cache[coin] = fd[rcol].astype(float)
            else:
                fund_cache[coin] = None
        funding = fund_cache[coin]

        # intrabar MAE
        mask = (bars.index >= r["entry_ts"]) & (bars.index <= r["exit_ts"])
        sub = bars[mask]
        if len(sub) == 0:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            funding_applied.append(0.0)
            continue

        min_low = float(sub["Low"].min())
        mae_pct = (min_low / float(r["entry_px"])) - 1.0
        lev_mae = mae_pct * lev

        if lev_mae <= wipeout_thresh:
            # margin wipeout. pnl = -95% (레버리지 포함된 raw 손실)
            # 원시 pnl_pct는 레버리지 미적용이므로 -95%/lev 기록
            hit_mask = sub["Low"] <= float(r["entry_px"]) * (1 + wipeout_thresh / lev)
            hit_ts = sub[hit_mask].index[0] if hit_mask.any() else r["exit_ts"]
            new_pnls.append(round(wipeout_thresh / lev * 100, 3))  # raw pnl%
            new_exits.append(hit_ts)
            new_reasons.append("liquidation")
            n_liq += 1
            # funding 생략 (즉시 청산 가정)
            funding_applied.append(0.0)
            continue

        # funding 누적 (entry~exit 동안 8h funding 지불)
        fund_cum = 0.0
        if include_funding and funding is not None:
            period_mask = (funding.index >= r["entry_ts"]) & (funding.index <= r["exit_ts"])
            funding_slice = funding[period_mask]
            # 롱 포지션: funding 양수면 롱이 숏에게 지불 (손실)
            fund_cum = float(funding_slice.sum()) * -1.0  # 롱은 반대 부호
            # lev 반영
            fund_cum *= lev
        # pnl_pct에 funding 반영
        adjusted_pnl = float(r["pnl_pct"]) + fund_cum * 100
        new_pnls.append(round(adjusted_pnl, 3))
        new_exits.append(r["exit_ts"])
        new_reasons.append(r.get("reason", ""))
        funding_applied.append(fund_cum)

    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["reason"] = new_reasons
    ev["funding_applied"] = funding_applied
    print(f"  intrabar liq: {n_liq} events")
    return ev


def run_split(name, v21, ev, cap):
    rows = []
    for span, s, e in [("full", v21.index[0], FULL_END),
                        ("train", v21.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21, s, e)
        evs = ev[(ev["entry_ts"] >= v21s.index[0]) & (ev["entry_ts"] <= v21s.index[-1])].copy()
        _, st = run_fut_combo(evs, kwargs["cd"], v21s.copy(), kwargs["hist"], cap)
        rows.append({"label": name, "span": span, "cap": cap,
                     "Cal": round(st["Cal"], 3),
                     "CAGR": round(st["CAGR"], 4),
                     "MDD": round(st["MDD"], 4),
                     "n_entries": st["n_entries"],
                     "n_liq_engine": st.get("n_liquidations", 0)})
    return rows


def main():
    global kwargs
    v21_s, v21_f, hist, avail, cd = load_all()
    kwargs = {"cd": cd, "hist": hist}

    ev_f = load_cached_events("fut")
    print(f"Base fut events: {len(ev_f)}")

    # 4 시나리오: base / +liq / +funding / +liq+funding
    scenarios = [
        ("base",   False, False),
        ("liq",    True,  False),
        ("fund",   False, True),
        ("liqfund",True,  True),
    ]

    rows = []
    for sname, do_liq, do_fund in scenarios:
        print(f"\n[{sname}] liq={do_liq} funding={do_fund}")
        if not do_liq and not do_fund:
            ev_mod = ev_f.copy()
        else:
            # helper function — liq 모드는 wipeout_thresh -0.95, fund는 include_funding
            if do_liq and do_fund:
                ev_mod = apply_liq_and_funding(ev_f, lev=3.0, wipeout_thresh=-0.95, include_funding=True)
            elif do_liq:
                ev_mod = apply_liq_and_funding(ev_f, lev=3.0, wipeout_thresh=-0.95, include_funding=False)
            else:  # fund only
                ev_mod = apply_liq_and_funding(ev_f, lev=3.0, wipeout_thresh=-99.0, include_funding=True)  # 청산 발생 안 함

        for cap in CAP_FUT_OPTS + [0.12, 0.15, 0.20]:
            cap = round(cap, 3)
            rows += run_split(f"fut_{sname}_cap{cap}", v21_f, ev_mod, cap)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["label", "span"])
    df.to_csv(os.path.join(OUT, "addl_v4_liq_funding.csv"), index=False)

    # 피벗
    print("\n=== FUT Cal by scenario × cap × span ===")
    pv = df.pivot_table(index="label", columns="span", values="Cal", aggfunc="first")
    print(pv.to_string())

    print("\n=== FUT MDD by scenario × cap × span ===")
    pv_mdd = df.pivot_table(index="label", columns="span", values="MDD", aggfunc="first")
    print(pv_mdd.to_string())

    print(f"\n저장: {OUT}/addl_v4_liq_funding.csv")


if __name__ == "__main__":
    main()
