"""노출·비용 교란 점검과 선물 청산 진단 (2026-10-07, 사전등록 b0_robust_prereg.md).

본 측정(bt_b0_robust.py)은 자산곡선만 쌓는다. 사전등록이 따로 요구한 둘을 여기서 잰다.

  ① 노출 교란 — 어떤 팔이 "덜 담아서" 이긴 것은 아닌지. 비교군 사이 평균 투입 비중
     (1 - 현금 비중) 차이가 3퍼센트포인트를 넘으면 표시한다.
  ② 선물 청산 진단 — 청산 횟수와 파산 바닥 발동, 거래·리밸 횟수.

재는 법. 엔진을 B0_LEDGER=1 로 돌려 봉마다 자산별 평가액과 순자산을 받는다.
투입 비중 = 각 봉의 Σ|평가액| / 순자산 의 평균이다. 선물의 평가액은 증거금 기준이라
명목 노출은 증거금 × 그 코인 그날의 L(동적 per-coin, build_K2_signal 과 같은 값)로
되살린다 — 엔진이 쓰는 값 그대로라 재구성이지 추정이 아니다.

표본. 현물은 앵커 7곳 전부, 선물은 앵커 0·3·6 세 곳이다(1회 11초라 전수면 두 시간을
더 쓴다). 비용은 기본만 — 노출 교란은 팔 사이 비교라 비용 조건을 가르지 않는다.
이 축소는 판정 문턱이 아니라 표시용 진단이라 사전등록 범위 안이다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 b0_expo.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402
import bt_b0_robust as B  # noqa: E402

OUT = B.OUT
ANCH = {"sp": list(R.ANCHORS), "fu": [0, 3, 6]}
KEEP_ARMS = set(["b0"] + B.PLACEBO_5 + ["top1", "fullpool"])
FLAG_PP = 3.0


def run_one(sleeve: str, arm: str, ex, anchor: int):
    """장부를 켜고 한 벌 돌린다. 자산곡선은 이미 캐시에 있으니 여기선 장부만 쓴다."""
    os.environ["B0_LEDGER"] = "1"
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["B0_ARM"] = "" if arm == "b0" else arm
    os.environ["SELECT_NO_ABSORB"] = "1" if arm == "fullpool" else "0"
    if ex:
        os.environ["B0_EXCLUDE"] = ",".join(ex)
    else:
        os.environ.pop("B0_EXCLUDE", None)
    try:
        if sleeve == "sp":
            from unified_backtest import run as bt
            kw = dict(R.SPOT_BASE)
            kw.update(drift_threshold=B.DRIFT_SPOT)
            m = bt(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
                   tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
                   vol_mode="daily", phase_offset_bars=R.PH_SPOT[anchor],
                   post_flip_delay=5, selection="greedy", stop_kind="none", stop_pct=0.0,
                   dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
                   crash_threshold=-99.0, health_mode="mom2vol", **kw)
        else:
            from backtest_futures_v25 import run as bt
            kw = dict(R.FUT_BASE)
            kw.update(drift_threshold=B.DRIFT_FUT)
            m = bt(R._bars, R._funding, interval="D", leverage=R._k2, tx_cost=R.TX_FUT,
                   maint_rate=0.004, vol_days=90, vol_mode="daily", post_flip_delay=5,
                   phase_offset_bars=R.PH_FUT[anchor], health_mode="mom2vol",
                   start_date=R.START, end_date=R.END, **kw)
    finally:
        os.environ.pop("B0_LEDGER", None)
        os.environ["B0_ARM"] = ""
        os.environ["SELECT_NO_ABSORB"] = "0"
        os.environ.pop("B0_EXCLUDE", None)
    return m


def expo_of(sleeve: str, rows):
    """평균 투입 비중과(선물은) 평균 명목 노출. 둘 다 순자산 대비다."""
    inv, notion, n = 0.0, 0.0, 0
    for r in rows:
        pv = r["pv"]
        if pv <= 0:
            continue
        s = sum(abs(v) for v in r["pos"].values())
        inv += s / pv
        if sleeve == "fu":
            d = r["Date"]
            tot = 0.0
            for c, v in r["pos"].items():
                lv = R._k2.get(c)
                L = 2.0
                if lv is not None and len(lv):
                    i = lv.index.get_indexer([d], method="ffill")[0]
                    if i >= 0:
                        L = float(lv.iloc[i])
                tot += abs(v) * L
            notion += tot / pv
        n += 1
    if not n:
        return dict(invested=float("nan"), notional=float("nan"), bars=0)
    return dict(invested=inv / n, notional=(notion / n if sleeve == "fu" else inv / n),
                bars=n)


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"데이터 로드 {time.time() - t0:.0f}s", flush=True)

    jobs = [(s, a, e) for (s, a, e) in B.plan() if a in KEEP_ARMS]
    total = sum(len(ANCH[s]) for s, _a, _e in jobs)
    print(f"계획 {len(jobs)}조합 · {total}회", flush=True)

    recs = []
    done = 0
    for s, arm, ex in jobs:
        for a in ANCH[s]:
            m = run_one(s, arm, ex, a)
            e = expo_of(s, m["_ledger"])
            rec = dict(sleeve=s, arm=arm, excl=B.excl_key(ex), anchor=a, **e)
            if s == "fu":
                rec.update(liq=int(m.get("Liq", 0)), trades=int(m.get("Trades", 0)),
                           rebal=int(m.get("Rebal", 0)),
                           floor_hits=sum(1 for r in m["_ledger"] if "_FLOOR" in r["cash"]))
            recs.append(rec)
            done += 1
        if done % 50 < len(ANCH[s]):
            el = time.time() - t0
            print(f"  {done}/{total}회 · {el/60:.0f}분 · 남은 추정 "
                  f"{el/max(1,done)*(total-done)/60:.0f}분", flush=True)

    # 비교군별 집계 — (슬리브, 제외) 안에서 팔 사이 투입 비중 차이
    grp = defaultdict(dict)
    for r in recs:
        grp[(r["sleeve"], r["excl"])].setdefault(r["arm"], []).append(r)
    flags = []
    summary = []
    for (s, ex), arms in sorted(grp.items()):
        base = arms.get("b0")
        if not base:
            continue
        b_inv = float(np.mean([x["invested"] for x in base]))
        b_not = float(np.mean([x["notional"] for x in base]))
        for arm, xs in sorted(arms.items()):
            if arm == "b0":
                continue
            inv = float(np.mean([x["invested"] for x in xs]))
            not_ = float(np.mean([x["notional"] for x in xs]))
            d_inv, d_not = (b_inv - inv) * 100, (b_not - not_) * 100
            row = dict(sleeve=s, excl=ex, arm=arm, b0_invested=b_inv, arm_invested=inv,
                       d_invested_pp=d_inv, b0_notional=b_not, arm_notional=not_,
                       d_notional_pp=d_not)
            summary.append(row)
            if abs(d_inv) > FLAG_PP or abs(d_not) > FLAG_PP:
                flags.append(row)

    liq = [r for r in recs if r["sleeve"] == "fu"]
    liq_tot = sum(r.get("liq", 0) for r in liq)
    floor_tot = sum(r.get("floor_hits", 0) for r in liq)

    print(f"\n== 노출 교란 ==")
    print(f"  비교쌍 {len(summary)}개 중 3%포인트 초과 {len(flags)}개")
    for f in flags[:20]:
        print(f"  !! {f['sleeve']} {f['excl']:12s} {f['arm']:9s} "
              f"투입 {f['d_invested_pp']:+.1f}%p · 명목 {f['d_notional_pp']:+.1f}%p")
    print(f"\n== 선물 청산 진단 ==")
    print(f"  측정 {len(liq)}벌 · 청산 합계 {liq_tot} · 파산 바닥 발동 봉 합계 {floor_tot}")
    worst = sorted(liq, key=lambda r: -r.get("liq", 0))[:8]
    for r in worst:
        print(f"  {r['excl']:12s} {r['arm']:9s} 앵커{r['anchor']} 청산 {r.get('liq',0)} "
              f"· 거래 {r.get('trades',0)} · 리밸 {r.get('rebal',0)} "
              f"· 명목노출 {r['notional']:.2f}x")

    with open(os.path.join(OUT, "expo.json"), "w") as f:
        json.dump(dict(records=recs, pairs=summary, flags=flags,
                       liq_total=liq_tot, floor_total=floor_tot,
                       anchors=ANCH, flag_pp=FLAG_PP), f, ensure_ascii=False,
                  indent=1, default=str)
    print(f"\n저장 {OUT}/expo.json · 총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
