"""주식 카나리가 방어면 코인을 쉰다 — 합성 60/25/15 에서 현행과 머리를 맞댄다.

사전등록: stock_gate_coin_prereg.md (판정 규칙 셋은 거기 적혀 있다).
실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 stock_gate_coin.py
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

EQDIR = "/home/gmoh/mon/251229/state/20261018-aligned3/eq"
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"
OUT = "/home/gmoh/mon/251229/state/20261018-stockgate"
PFD = 5  # 현행 라이브값


def load_eq(kind: str, anchor: int, stress: bool) -> pd.Series:
    key = f"pfd{PFD}__a{anchor}" + ("__s5" if stress else "")
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True)["eq"]


def load_stock(anchor: int) -> pd.Series:
    p = os.path.join(STOCK_CACHE, f"st__a{anchor}.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True)["eq"]


def stock_canary() -> pd.Series:
    """채택 주식 백테스트와 같은 출처의 EEM 카나리(SMA200 ±0.5% 히스테리시스)."""
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    _, _, _, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    return canary.astype(bool)


def gate_series(canary: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """코인 날짜 t 의 게이트 = t 보다 엄격히 이전인 마지막 미국 거래일의 카나리.

    앞날 참조 차단이 목적이다. 하루 밀어 둔 뒤 코인 달력으로 끌어 쓴다.
    구간 맨 앞처럼 아직 값이 없는 자리는 켜짐으로 둔다(현행과 같게).
    """
    lagged = canary.shift(1).dropna()
    return lagged.reindex(lagged.index.union(idx)).ffill().reindex(idx).fillna(True).astype(bool)


def apply_gate(eq: pd.Series, gate: pd.Series, tx: float) -> pd.Series:
    """게이트가 꺼진 날 그 슬리브 수익률을 0 으로 만든다.

    게이트가 뒤집히는 날, 그 직전에 실제로 들고 있었으면(수익률 0 이 아니면)
    전량 회전으로 보고 tx 를 한 번 물린다. 이미 현금이면 안 물린다.
    """
    r = eq.pct_change().fillna(0.0)
    g = gate.reindex(r.index).ffill().fillna(True).astype(bool)
    invested = (r != 0.0)
    gr = r.where(g, 0.0)
    flip = g.ne(g.shift(1).fillna(g.iloc[0]))
    # 끄는 날: 직전까지 들고 있었으면 판다. 켜는 날: 그날 살 것이 있으면 산다.
    sell = flip & (~g) & invested.shift(1).fillna(False)
    buy = flip & g & invested
    cost = (sell | buy).astype(float) * tx
    return (1 + gr - cost).cumprod()


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print("주식 카나리 계산 중...", flush=True)
    can = stock_canary()
    print(f"  카나리 {len(can)}일 ({can.index[0].date()}~{can.index[-1].date()}), "
          f"{time.time()-t0:.0f}초", flush=True)

    arms = ["현행", "둘 다 쉼", "현물만 쉼", "선물만 쉼"]
    res = {}
    desc = {}

    for stress in (False, True):
        tag = "비용 5배" if stress else "기본 비용"
        mult = R.STRESS if stress else 1.0
        rank = {a: [] for a in arms}
        met = {a: [] for a in arms}

        for anc in R.ANCHORS:
            st = load_stock(anc)
            sp = load_eq("sp", anc, stress)
            fu = load_eq("fu", anc, stress)
            idx = sp.index.intersection(fu.index)
            g = gate_series(can, idx)

            sp_g = apply_gate(sp, g, R.TX_SPOT * mult)
            fu_g = apply_gate(fu, g, R.TX_FUT * mult)

            comps = {
                "현행": R.build_alloc(st, sp, fu),
                "둘 다 쉼": R.build_alloc(st, sp_g, fu_g),
                "현물만 쉼": R.build_alloc(st, sp_g, fu),
                "선물만 쉼": R.build_alloc(st, sp, fu_g),
            }
            for a in arms:
                met[a].append(R.metrics(comps[a]))
            sums, _, nw = R.window_rs(comps)
            for a in arms:
                rank[a].append(sums[a] / max(nw, 1))

            if not stress and anc == R.ANCHORS[0]:
                # 서술 수치는 비용·앵커와 무관한 성질이라 한 번만 잰다.
                off = ~g
                r_sp = sp.pct_change().fillna(0.0)
                r_fu = fu.pct_change().fillna(0.0)
                desc = dict(
                    days_total=int(len(g)),
                    days_stock_defense=int(off.sum()),
                    pct_stock_defense=float(off.mean() * 100),
                    spot_cum_during_defense=float((1 + r_sp[off]).prod() - 1) * 100,
                    fut_cum_during_defense=float((1 + r_fu[off]).prod() - 1) * 100,
                    spot_cum_during_offense=float((1 + r_sp[~off]).prod() - 1) * 100,
                    fut_cum_during_offense=float((1 + r_fu[~off]).prod() - 1) * 100,
                    spot_invested_pct_during_defense=float((r_sp[off] != 0).mean() * 100),
                    fut_invested_pct_during_defense=float((r_fu[off] != 0).mean() * 100),
                    spot_invested_pct_overall=float((r_sp != 0).mean() * 100),
                    gate_flips=int(g.ne(g.shift(1).fillna(g.iloc[0])).sum()),
                )

        print(f"\n=== 주식 방어면 코인 쉼 · 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 ===")
        print(f"{'팔':<12}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}{'현행대비승':>11}")
        for a in sorted(arms, key=lambda k: float(np.mean(rank[k]))):
            wins = sum(1 for x, y in zip(rank[a], rank["현행"]) if x < y)
            print(f"{a:<12}{float(np.mean(rank[a])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[a]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[a]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[a]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[a]])):>9.2f}"
                  f"{wins:>8}/{len(R.ANCHORS)}")

        res[tag] = dict(rank={a: rank[a] for a in arms},
                        metrics={a: met[a] for a in arms})

    print("\n=== 서술 수치 (판정과 무관) ===")
    print(f"  전 구간 {desc['days_total']}일 중 주식 방어 {desc['days_stock_defense']}일 "
          f"({desc['pct_stock_defense']:.1f}%), 게이트 전환 {desc['gate_flips']}회")
    print(f"  그 구간 코인 누적 — 현물 {desc['spot_cum_during_defense']:+.1f}%, "
          f"선물 {desc['fut_cum_during_defense']:+.1f}%")
    print(f"  공격 구간 코인 누적 — 현물 {desc['spot_cum_during_offense']:+.1f}%, "
          f"선물 {desc['fut_cum_during_offense']:+.1f}%")
    print(f"  주식 방어 구간에서 코인이 실제로 들고 있던 날 — 현물 "
          f"{desc['spot_invested_pct_during_defense']:.1f}%, "
          f"선물 {desc['fut_invested_pct_during_defense']:.1f}% "
          f"(전 구간 현물 {desc['spot_invested_pct_overall']:.1f}%)")

    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(dict(res=res, desc=desc), f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
