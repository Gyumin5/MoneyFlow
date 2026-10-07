"""자산별 기여표 — 제외 대상 목록을 못박는 단계 (2026-10-07, 사전등록 b0_robust_prereg.md).

사전등록이 고정한 절차 그대로다. 슬리브마다 기본비용 B0 를 앵커 7곳 돌리고, 자산별로
둘을 잰다.

  순손익 기여 = 자산별 순손익 / 전일 슬리브 순자산 의 일별 합. 앵커 7곳 평균.
  보유 노출   = 절대 실효 노출 비율의 일별 합. 앵커 7곳 평균.

장부는 엔진이 B0_LEDGER 로 뱉는다. 자본 변동이 전부 코인 하나에 귀속되므로
자산별 기여의 통화 합 + 귀속 불가분 = 순자산 변화가 정확히 성립해야 한다.
그 대사를 같이 찍는다 — 안 맞으면 기여표를 쓰지 않는다.

제외 대상 = {비트코인, 이더리움} ∪ {기여 상위 5} ∪ {노출 상위 5}, 최대 12, 중복 제거.
동률은 보유 노출, 그 다음 심볼 순으로 푼다.
두 슬리브 공통 목록은 합성 비중(현물 0.25 · 선물 0.15)으로 가중한 기여로 다시 매긴다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 b0_contrib.py
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

OUT = "/home/gmoh/mon/251229/state/20261007-b0robust"
W_SPOT, W_FUT = 0.25, 0.15
TOP_N = 5
MAX_TARGETS = 12


def run_ledger(sleeve: str, anchor: int):
    """기본비용 B0 한 벌을 장부와 함께 돌린다."""
    os.environ["B0_LEDGER"] = "1"
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ.pop("B0_EXCLUDE", None)
    os.environ["B0_ARM"] = ""
    os.environ["SELECT_NO_ABSORB"] = "0"
    try:
        if sleeve == "sp":
            from unified_backtest import run as bt
            kw = dict(R.SPOT_BASE)
            kw.update(drift_threshold=0.10)
            m = bt(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
                   tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
                   vol_mode="daily", phase_offset_bars=R.PH_SPOT[anchor],
                   post_flip_delay=5, selection="greedy", stop_kind="none", stop_pct=0.0,
                   dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
                   crash_threshold=-99.0, health_mode="mom2vol", **kw)
        else:
            from backtest_futures_v25 import run as bt
            kw = dict(R.FUT_BASE)
            kw.update(drift_threshold=0.015)
            m = bt(R._bars, R._funding, interval="D", leverage=R._k2, tx_cost=R.TX_FUT,
                   maint_rate=0.004, vol_days=90, vol_mode="daily", post_flip_delay=5,
                   phase_offset_bars=R.PH_FUT[anchor], health_mode="mom2vol",
                   start_date=R.START, end_date=R.END, **kw)
    finally:
        os.environ.pop("B0_LEDGER", None)
    return m["_ledger"]


def tally(rows):
    """일별 자산 기여율과 노출율. 대사 잔차도 같이 낸다."""
    contrib = defaultdict(float)
    expo = defaultdict(float)
    prev_pos: dict = {}
    prev_pv = None
    resid = 0.0
    total_move = 0.0
    for r in rows:
        pv, pos, cash = r["pv"], r["pos"], r["cash"]
        if prev_pv is not None and prev_pv > 0:
            coins = set(pos) | set(prev_pos) | set(cash)
            day = {}
            for c in coins:
                day[c] = cash.get(c, 0.0) + pos.get(c, 0.0) - prev_pos.get(c, 0.0)
                contrib[c] += day[c] / prev_pv
            resid += (pv - prev_pv) - sum(day.values())
            total_move += abs(pv - prev_pv)
        if pv > 0:
            for c, v in pos.items():
                expo[c] += abs(v) / pv
        prev_pos, prev_pv = pos, pv
    return dict(contrib), dict(expo), resid, total_move


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

    out: dict = {}
    for sleeve, label in (("sp", "현물"), ("fu", "선물")):
        acc_c = defaultdict(list)
        acc_e = defaultdict(list)
        resids = []
        for a in R.ANCHORS:
            rows = run_ledger(sleeve, a)
            c, e, resid, move = tally(rows)
            for k, v in c.items():
                acc_c[k].append(v)
            for k, v in e.items():
                acc_e[k].append(v)
            resids.append(resid / move if move > 0 else 0.0)
            print(f"  {label} 앵커{a} 장부 {len(rows)}일 · 대사 잔차비 "
                  f"{resids[-1]:.2e} ({time.time() - t0:.0f}s)", flush=True)
        n = len(R.ANCHORS)
        contrib_all = {k: float(np.sum(v) / n) for k, v in acc_c.items()}
        # '_FLOOR' 는 코인이 아니라 선물의 파산 바닥(capital = max(capital, 0))이다.
        # 장부 대사에는 필요하고 제외 대상 순위에서는 빠져야 한다.
        floor = contrib_all.pop("_FLOOR", 0.0)
        contrib = contrib_all
        expo = {k: float(np.sum(v) / n) for k, v in acc_e.items()}
        worst = max(abs(x) for x in resids)
        print(f"  {label} 대사 최악 잔차비 {worst:.2e}", flush=True)
        if worst > 1e-9:
            print(f"  !! {label} 장부가 순자산 변화와 안 맞는다. 기여표를 쓰지 않는다.")
        top_c = sorted(contrib, key=lambda c: (-contrib[c], -expo.get(c, 0.0), c))[:TOP_N]
        top_e = sorted(expo, key=lambda c: (-expo[c], c))[:TOP_N]
        targets = []
        for c in ["BTC", "ETH"] + top_c + top_e:
            if c not in targets and c in contrib:
                targets.append(c)
        targets = targets[:MAX_TARGETS]
        grp2 = top_c[:2]
        grp3 = top_c[:3]
        out[sleeve] = dict(label=label, contrib=contrib, expo=expo,
                           top_contrib=top_c, top_expo=top_e, targets=targets, floor=floor,
                           group2=grp2, group3=grp3, resid_worst=worst)
        print(f"\n== {label} ==")
        print(f"  기여 상위: " + ", ".join(f"{c} {contrib[c]:+.2f}" for c in top_c))
        print(f"  노출 상위: " + ", ".join(f"{c} {expo[c]:.0f}" for c in top_e))
        print(f"  파산 바닥 기여 {floor:+.4f} (코인 아님)")
        print(f"  제외 대상 {len(targets)}종: {targets}\n")

    # 두 슬리브 공통 — 합성 비중으로 가중한 기여
    comm = defaultdict(float)
    for c, v in out["sp"]["contrib"].items():
        comm[c] += W_SPOT * v
    for c, v in out["fu"]["contrib"].items():
        comm[c] += W_FUT * v
    comm_rank = sorted(comm, key=lambda c: (-comm[c], c))
    out["common"] = dict(weighted=dict(comm), rank=comm_rank[:10],
                         group2=comm_rank[:2], group3=comm_rank[:3])
    print("== 두 슬리브 공통 (합성 가중 기여) ==")
    print("  " + ", ".join(f"{c} {comm[c]:+.2f}" for c in comm_rank[:6]))

    with open(os.path.join(OUT, "contrib.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n저장 {OUT}/contrib.json · 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
