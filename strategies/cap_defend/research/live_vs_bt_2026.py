#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실거래 자산곡선과 백테스트를 맞대 본다 (2026-10-17 운영자 요청).

묻는 것: 선물 계좌에 돈이 들어간 2026-05 말 이후, 실거래가 백테스트처럼 굴러갔나.
차이가 있으면 얼마나인가.

재는 방법
  ⑴ 실거래 — 서버 assets.db 의 일간 스냅샷에서 계좌 셋의 평가액을 꺼낸다. 세 계좌 모두
     달러로 환산해서 본다. 원화로 보면 환율(1507.8 → 1348.5, 10.6% 원화 강세)이 성과를
     통째로 가린다. 주식·선물은 각 계좌가 적어 둔 환율, 업비트는 그날 행 환율을 쓴다.
  ⑵ 계좌간 송금은 수익이 아니다. 한 계좌가 늘고 다른 계좌가 주는 날은 그 계좌의 그날
     수익률을 빼고 센다(합성 수익률에는 영향이 없다 — 합은 보존되므로).
  ⑶ 백테스트 — 채택 엔진 그대로. 현물·선물은 unified_backtest / backtest_futures_v25,
     주식은 bt_stock_mom3. 스냅 위상(앵커)은 라이브와 맞출 방법이 없으므로 일곱 개를
     전부 돌리고 라이브가 그 일곱이 만든 폭 안에 있는지를 본다. 폭 밖이면 그게 차이다.
  ⑷ 합성은 라이브와 같은 방식 — 시작일에 60/25/15 로 맞추고 그 뒤 계좌간 재조정 없음.

안 하는 것: 라이브 체결 하나하나의 재현(주문 기록이 없다), 세금·환전 수수료.

실행: cd .../research && python3 live_vs_bt_2026.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

OUT = "/home/gmoh/mon/251229/state/20261017-livebt"
EQ = os.path.join(OUT, "eq")
LIVE = os.path.join(OUT, "live.json")
RES = os.path.join(OUT, "compare.json")

END = "2026-10-17"
ANCHORS = range(7)
W = dict(stock=0.60, spot=0.25, fut=0.15)
# 창 둘. 선물 계좌 송금이 끝난 다음날부터, 그리고 운영자가 말한 6월부터.
WINDOWS = [("2026-05-26", "선물 계좌 송금 직후"), ("2026-06-01", "6월 이후")]
TRANSFER_USD = 2000.0     # 계좌간 이동으로 볼 하루 변화 문턱


# ───────────────────────────────────────────────── 라이브
def live_frame():
    d = json.load(open(LIVE, encoding="utf-8"))
    rows = []
    for r in d:
        rows.append(dict(
            date=pd.Timestamp(r["date"]),
            stock=r["stock_krw"] / r["stock_fx"],
            spot=r["upbit_krw"] / r["stock_fx"],
            fut=r["fut_usdt"] if r["fut_usdt"] else r["fut_krw"] / r["fut_fx"],
            fx=r["stock_fx"],
        ))
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df["tot"] = df[["stock", "spot", "fut"]].sum(axis=1)
    return df


def live_returns(df):
    """계좌별 일간 수익률. 송금일은 그 계좌만 결측으로 둔다."""
    out = {}
    moved = []
    chg = df[["stock", "spot", "fut"]].diff()
    for i in range(1, len(df)):
        v = chg.iloc[i]
        if v.max() >= TRANSFER_USD and v.min() <= -TRANSFER_USD:
            moved.append(df.index[i])
    for k in ("stock", "spot", "fut"):
        r = df[k].pct_change()
        r.loc[moved] = np.nan
        out[k] = r
    out["tot"] = df["tot"].pct_change()      # 합은 송금에 보존된다
    return pd.DataFrame(out), moved


# ───────────────────────────────────────────────── 백테스트
def bt_curves():
    os.makedirs(EQ, exist_ok=True)
    need = [(k, a) for a in ANCHORS for k in ("stock", "spot", "fut")
            if not os.path.exists(os.path.join(EQ, f"{k}_a{a}.csv"))]
    if need:
        import alloc_grid_run as RUN
        import bt_coin_param_resweep as R
        RUN._init_resweep()
        R.END = END
        for kind, a in need:
            p = os.path.join(EQ, f"{kind}_a{a}.csv")
            print(f"  BT {kind} 앵커{a} …", flush=True)
            if kind == "stock":
                s = R.run_stock(a, False)
            elif kind == "spot":
                s = R.run_spot({}, R.PH_SPOT[a], False)
            else:
                s = R.run_fut({}, R.PH_FUT[a], False)
            s = s.dropna().rename("eq")
            tmp = f"{p}.tmp.{os.getpid()}"
            s.to_frame().to_csv(tmp)
            os.replace(tmp, p)
    out = {}
    for a in ANCHORS:
        for kind in ("stock", "spot", "fut"):
            out[(kind, a)] = pd.read_csv(os.path.join(EQ, f"{kind}_a{a}.csv"),
                                         index_col=0, parse_dates=True)["eq"]
    return out


def bt_composite(curves, a, idx):
    """앵커 a 의 합성 수익률을 라이브 날짜축에 맞춰 돌려준다."""
    parts = {}
    for kind in ("stock", "spot", "fut"):
        s = curves[(kind, a)].reindex(idx.union(curves[(kind, a)].index)).ffill().reindex(idx)
        parts[kind] = s
    df = pd.DataFrame(parts).dropna()
    r = df.pct_change()
    comp = (r * pd.Series(W)).sum(axis=1, skipna=False)
    return df, r, comp


# ───────────────────────────────────────────────── 지표
def stats(ret):
    r = ret.dropna()
    if len(r) < 2:
        return dict(n=len(r), cum=np.nan, cagr=np.nan, mdd=np.nan, vol=np.nan)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252.0
    mdd = float((eq / eq.cummax() - 1).min())
    return dict(n=int(len(r)), cum=float(eq.iloc[-1] - 1),
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan,
                mdd=mdd, vol=float(r.std() * np.sqrt(252)))


def fmt(s):
    return (f"누적 {s['cum']*100:+7.2f}%  연복리 {s['cagr']*100:+8.2f}%  "
            f"최대낙폭 {s['mdd']*100:7.2f}%  변동성 {s['vol']*100:6.2f}%  일수 {s['n']}")


def main():
    print("실거래 대 백테스트 — 2026")
    df = live_frame()
    lret, moved = live_returns(df)
    print(f"라이브 스냅샷 {len(df)}일 {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"계좌간 송금으로 본 날 {len(moved)}일: {[str(x.date()) for x in moved]}")
    print(f"환율 {df['fx'].iloc[0]:.1f} → {df['fx'].iloc[-1]:.1f} "
          f"({(df['fx'].iloc[-1]/df['fx'].iloc[0]-1)*100:+.2f}%)")
    print()

    curves = bt_curves()
    res = {"moved": [str(x.date()) for x in moved],
           "fx": [float(df['fx'].iloc[0]), float(df['fx'].iloc[-1])], "windows": {}}

    for start, label in WINDOWS:
        idx = df.index[df.index >= pd.Timestamp(start)]
        print("=" * 78)
        print(f"창: {start} ~ {END}  ({label})  영업일 {len(idx)}")
        print("=" * 78)
        w = {"label": label, "start": start, "n": int(len(idx)), "live": {}, "bt": {}}

        lw = lret.loc[idx]
        print("라이브")
        for k, nm in (("stock", "주식  "), ("spot", "업비트"), ("fut", "선물  "),
                      ("tot", "합계  ")):
            s = stats(lw[k])
            w["live"][k] = s
            print(f"  {nm} {fmt(s)}")

        # 라이브 합성은 계좌 비중이 60/25/15 에서 흘러간 실제 결과다.
        # 백테스트 합성은 창 시작일에 60/25/15 로 맞추고 그 뒤 재조정 없음 — 같은 규약.
        print("백테스트 (앵커 7개)")
        per = {k: [] for k in ("stock", "spot", "fut", "comp")}
        for a in ANCHORS:
            bdf, br, _ = bt_composite(curves, a, idx)
            sub = br.loc[idx].dropna(how="all")
            for k in ("stock", "spot", "fut"):
                per[k].append(stats(sub[k]))
            # 합성: 시작 비중 고정 후 흘러가게
            v = (bdf.loc[idx] / bdf.loc[idx].iloc[0]) * pd.Series(W)
            comp_eq = v.sum(axis=1)
            per["comp"].append(stats(comp_eq.pct_change()))
        w["bt"] = {k: per[k] for k in per}
        for k, nm in (("stock", "주식  "), ("spot", "업비트"), ("fut", "선물  "),
                      ("comp", "합계  ")):
            cums = [x["cum"] for x in per[k]]
            mdds = [x["mdd"] for x in per[k]]
            live_key = "tot" if k == "comp" else k
            lv = w["live"][live_key]["cum"]
            inside = min(cums) <= lv <= max(cums)
            gap = 0.0 if inside else (lv - max(cums) if lv > max(cums) else lv - min(cums))
            print(f"  {nm} 누적 {np.mean(cums)*100:+7.2f}% "
                  f"[{min(cums)*100:+7.2f} ~ {max(cums)*100:+7.2f}]  "
                  f"낙폭 평균 {np.mean(mdds)*100:7.2f}%  "
                  f"→ 라이브 {lv*100:+7.2f}% {'폭 안' if inside else f'폭 밖 {gap*100:+.2f}pp'}")
            w["bt"][k + "_verdict"] = dict(mean=float(np.mean(cums)), lo=float(min(cums)),
                                           hi=float(max(cums)), live=float(lv),
                                           inside=bool(inside), gap=float(gap))
        print()
        res["windows"][start] = w

    with open(RES, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, default=float)
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
