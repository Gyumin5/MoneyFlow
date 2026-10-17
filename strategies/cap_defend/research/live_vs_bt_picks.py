#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라이브 보유 종목과 백테스트 목표 종목을 날짜별로 맞대 본다 (2026-10-17).

묻는 것: 현물·선물이 백테스트가 고른 것과 같은 코인을 들고 있었나. 현금으로 빠져
있던 구간이 서로 맞나.

재는 방법
  ⑴ 라이브 — 서버 assets.db 일간 스냅샷의 보유 종목(업비트 KRW 평가액, 바이낸스
     명목 USDT). 비중은 그 계좌 안에서의 몫이고 현금을 포함한다.
  ⑵ 백테스트 — 채택 엔진의 목표 비중(_trace 의 combined). 스냅 위상을 라이브와
     맞출 방법이 없어 앵커 일곱을 전부 돌린다.
  ⑶ 같다·다르다는 종목 집합으로 센다. 비중의 미세한 차이는 보지 않는다 —
     라이브는 그날 체결까지의 결과이고 백테스트는 그날의 목표라 애초에 하루 어긋난다.

실행: cd .../research && python3 live_vs_bt_picks.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

OUT = "/home/gmoh/mon/251229/state/20261017-livebt"
LIVE2 = os.path.join(OUT, "live2.json")
RES = os.path.join(OUT, "picks.json")

START_VIEW = "2026-04-02"
END = "2026-10-17"
ANCHORS = range(7)
MIN_W = 0.01          # 이보다 작은 몫은 먼지로 보고 종목 집합에서 뺀다


def live_sets():
    d = json.load(open(LIVE2, encoding="utf-8"))
    sp, fu = {}, {}
    for r in d:
        up = r["up"]
        tot = up.get("tot_krw") or 0.0
        s = set()
        for k, v in (up.get("h") or {}).items():
            val = v[1] or 0.0
            if tot and val / tot >= MIN_W:
                s.add(k)
        sp[r["date"]] = s

        fo = r["fu"]
        ftot = fo.get("fut_tot_usd") or fo.get("tot_usd") or 0.0
        s2 = set()
        for k, v in (fo.get("h") or {}).items():
            not_ = v[1] or 0.0
            if ftot and not_ / ftot >= MIN_W:
                s2.add(k)
        fu[r["date"]] = s2
    return sp, fu


def bt_sets():
    import alloc_grid_run as RUN
    import bt_coin_param_resweep as R
    from unified_backtest import run as bt_run
    from backtest_futures_v25 import run as fbt_run

    RUN._init_resweep()
    R.END = END
    out = {"spot": {}, "fut": {}}

    for a in ANCHORS:
        os.environ["DRIFT_HEALTH_MODE"] = "refill"
        tr = []
        kw = dict(R.SPOT_BASE)
        bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=R.TX_SPOT, start_date=R.START, end_date=END,
               vol_mode="daily", phase_offset_bars=R.PH_SPOT[a],
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", _trace=tr, **kw)
        out["spot"][a] = _pick_map(tr)
        print(f"  BT spot 앵커{a} — {len(out['spot'][a])}일", flush=True)

        tr = []
        kw = dict(R.FUT_BASE)
        fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT, maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=R.PH_FUT[a], health_mode="mom2vol",
                start_date=R.START, end_date=END, _trace=tr, **kw)
        out["fut"][a] = _pick_map(tr)
        print(f"  BT fut  앵커{a} — {len(out['fut'][a])}일", flush=True)
    return out


def _pick_map(tr):
    m = {}
    for e in tr:
        d = pd.Timestamp(e["date"]).strftime("%Y-%m-%d")
        if d < START_VIEW:
            continue
        tgt = e.get("target") or {}
        s = set()
        for k, w in tgt.items():
            if k.upper() == "CASH":
                continue
            if (w or 0.0) >= MIN_W:
                s.add(k)
        m[d] = s
    return m


def cash_runs(series):
    """현금 100% 구간을 (시작, 끝, 일수) 로 묶는다."""
    runs, cur = [], None
    for d in sorted(series):
        if not series[d]:
            if cur is None:
                cur = [d, d, 1]
            else:
                cur[1] = d
                cur[2] += 1
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return [r for r in runs if r[2] >= 3]


def main():
    print("라이브 보유 대 백테스트 목표 — 종목")
    sp_live, fu_live = live_sets()
    bt = bt_sets()

    res = {"min_w": MIN_W, "live": {}, "bt": {}, "overlap": {}}
    for kind, live in (("spot", sp_live), ("fut", fu_live)):
        days = [d for d in sorted(live) if d >= START_VIEW]
        print()
        print("=" * 76)
        print(f"{'코인 현물' if kind == 'spot' else '코인 선물'} — 라이브 {len(days)}일")
        print("=" * 76)

        lr = cash_runs({d: live[d] for d in days})
        print("라이브 현금 100% 구간 (3일 이상): " +
              (", ".join(f"{a}~{b} {n}일" for a, b, n in lr) if lr else "없음"))
        for a in ANCHORS:
            br = cash_runs({d: bt[kind][a][d] for d in days if d in bt[kind][a]})
            print(f"  BT 앵커{a} 현금 구간: " +
                  (", ".join(f"{x}~{y} {n}일" for x, y, n in br) if br else "없음"))

        print()
        print("겹침 (라이브 보유 ∩ 백테스트 목표 / 합집합, 평균)")
        best = None
        for a in ANCHORS:
            j, both_cash, n = [], 0, 0
            for d in days:
                if d not in bt[kind][a]:
                    continue
                L, B = live[d], bt[kind][a][d]
                n += 1
                if not L and not B:
                    both_cash += 1
                    j.append(1.0)
                elif not (L | B):
                    j.append(1.0)
                else:
                    j.append(len(L & B) / len(L | B))
            m = sum(j) / len(j) if j else 0.0
            print(f"  앵커{a}: 자카드 평균 {m*100:5.1f}%  (둘 다 현금인 날 {both_cash}/{n})")
            if best is None or m > best[1]:
                best = (a, m)
        print(f"  → 가장 가까운 앵커 {best[0]}, 자카드 {best[1]*100:.1f}%")
        res["overlap"][kind] = dict(best_anchor=int(best[0]), best_jaccard=float(best[1]))
        res["live"][kind] = {d: sorted(live[d]) for d in days}
        res["bt"][kind] = {str(a): {d: sorted(bt[kind][a][d])
                                    for d in days if d in bt[kind][a]} for a in ANCHORS}

        # 가장 가까운 앵커로 날짜별 차이를 뽑는다
        a = best[0]
        print()
        print(f"앵커{a} 기준 종목이 갈린 날 (앞 25일)")
        cnt = 0
        for d in days:
            if d not in bt[kind][a]:
                continue
            L, B = live[d], bt[kind][a][d]
            if L == B:
                continue
            cnt += 1
            if cnt <= 25:
                print(f"  {d}  라이브 {sorted(L) or '현금'}  ≠  BT {sorted(B) or '현금'}")
        print(f"  — 갈린 날 합계 {cnt}일")
        res["overlap"][kind]["diff_days"] = int(cnt)

    with open(RES, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, default=str)
    print()
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
