"""최근 구간을 오늘까지 늘려서 다시 잰다.

flip_pfd_recent.py 는 저장된 곡선을 잘라 쓰므로 2026-09-18 에서 멈춘다. 여기서는
끝을 오늘로 두고 네 조합을 처음부터 다시 돌린다. 기본 비용만 본다 — 넉 달 구간에서
비용 5배는 따로 읽을 것이 없다.

곡선은 2020-10-01 부터 돌리고 2026-06-01 부터 잘라 잰다. 짧은 구간만 돌리면 SMA·
모멘텀 예열이 안 돼 첫 달이 왜곡된다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 flip_pfd_recent_extend.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

R.END = os.environ.get("EXTEND_END", "2026-10-17")

OUT = "/home/gmoh/mon/251229/state/20261018-fliplock"
EQDIR = os.path.join(OUT, "eq_ext")
SINCE = "2026-06-01"

ARMS = {
    "A 둘 다 넣음": (5, 5),
    "B 지금 라이브": (0, 0),
    "C 선물만 넣음": (0, 5),
    "D 현물만 넣음": (5, 0),
}


def _cached(name: str, fn):
    p = os.path.join(EQDIR, f"{name}.csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    s = fn()
    s.to_csv(p, header=["eq"])
    return s


def run_spot(pfd: int, phase: int) -> pd.Series:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    m = bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
               vol_mode="daily", phase_offset_bars=phase,
               post_flip_delay=pfd, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", **R.SPOT_BASE)
    return m["_equity"]


def run_fut(pfd: int, phase: int) -> pd.Series:
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    m = fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT, maint_rate=0.004, vol_days=90, vol_mode="daily",
                post_flip_delay=pfd, phase_offset_bars=phase, health_mode="mom2vol",
                start_date=R.START, end_date=R.END, **R.FUT_BASE)
    return m["_equity"]


def seg(eq: pd.Series) -> pd.Series:
    s = eq.loc[SINCE:].dropna()
    return s / s.iloc[0]


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print(f"끝 {R.END} · 코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    st, sp, fu = {}, {}, {}
    for a in R.ANCHORS:
        st[a] = _cached(f"st__a{a}", lambda a=a: R.run_stock(a, False))
        for pfd in (0, 5):
            sp[(pfd, a)] = _cached(f"sp_pfd{pfd}__a{a}",
                                   lambda p=pfd, a=a: run_spot(p, R.PH_SPOT[a]))
            fu[(pfd, a)] = _cached(f"fu_pfd{pfd}__a{a}",
                                   lambda p=pfd, a=a: run_fut(p, R.PH_FUT[a]))
        print(f"  앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)

    comp, sleeve = {k: [] for k in ARMS}, {k: [] for k in ARMS}
    end = None
    for a in R.ANCHORS:
        for name, (ps, pf) in ARMS.items():
            c = seg(R.build_alloc(st[a], sp[(ps, a)], fu[(pf, a)]))
            end = c.index[-1]
            comp[name].append(dict(ret=(c.iloc[-1] - 1) * 100,
                                   mdd=float((c / c.cummax() - 1).min()) * 100))
            s = seg(sp[(ps, a)])
            sleeve[name].append((s.iloc[-1] - 1) * 100)

    print(f"\n=== {SINCE} ~ {end.date()} · 합성 60/25/15 · 앵커 7곳 평균 · 기본 비용 ===")
    print(f"{'조합':<16}{'구간수익':>9}{'구간낙폭':>9}{'현물슬리브':>11}")
    for name in ARMS:
        print(f"{name:<16}{float(np.mean([x['ret'] for x in comp[name]])):>9.2f}"
              f"{float(np.mean([x['mdd'] for x in comp[name]])):>9.2f}"
              f"{float(np.mean(sleeve[name])):>11.2f}")
    b = float(np.mean([x["ret"] for x in comp["B 지금 라이브"]]))
    for name in ("A 둘 다 넣음", "C 선물만 넣음", "D 현물만 넣음"):
        d = float(np.mean([x["ret"] for x in comp[name]])) - b
        print(f"  {name} 은 지금 라이브보다 {d:+.2f}%포인트")

    with open(os.path.join(OUT, "recent_extend_result.json"), "w") as f:
        json.dump({"since": SINCE, "end": str(end.date()), "comp": comp,
                   "spot_sleeve": sleeve}, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/recent_extend_result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
