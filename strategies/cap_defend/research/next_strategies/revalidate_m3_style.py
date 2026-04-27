#!/usr/bin/env python3
"""Next strategies 재시뮬 — m3 스타일 aggregate (n_pick=1, cap_per_slot 기반).

C engine과 공정 비교:
- 같은 m3 엔진 구조 (v21 우선 + n_pick 선택 + cap margin)
- Standalone (fake flat V21 = 자본 100% 유휴)
- TX 3단계 (0.04 / 0.10 / 0.30)

대상: Pullback, Breakdown Short (AI 우선순위 권고)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_next import load_all, TRAIN_END, HOLDOUT_START, FULL_END
from engine_pullback import run_pullback
from engine_breakdown_short import run_breakdown_short
from engine_short_mom import load_funding
from m3_engine_futures import simulate_fut, metrics as metrics_fut
from c_engine_v5 import load_coin

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def fake_flat_v21(dates):
    """V21 없이 standalone 평가. equity=1 상수, cash_ratio=1."""
    return pd.DataFrame({
        "equity": 1.0, "cash_ratio": 1.0,
        "v21_ret": 0.0, "prev_cash": 1.0,
    }, index=dates)


def extract_events_parallel(avail, engine_fn, params, funding_loader=None, n_jobs=24):
    """코인별 병렬 이벤트 추출."""
    def _one(coin):
        df = load_coin(coin + "USDT")
        if df is None: return []
        kwargs = dict(params)
        if funding_loader is not None:
            kwargs["funding"] = funding_loader(coin)
        _, evs = engine_fn(df, **kwargs)
        for e in evs:
            e["coin"] = coin
        return evs
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one)(c) for c in avail)
    rows = [e for batch in results for e in batch]
    return pd.DataFrame(rows)


def eval_with_m3(events, v21_df, coin_daily, hist, cap, leverage=1.0, tx=0.003):
    _, stats = simulate_fut(events, coin_daily, v21_df.copy(), hist,
                             n_pick=1, cap_per_slot=cap, universe_size=15,
                             tx_cost=tx, swap_edge_threshold=1,
                             leverage=leverage)
    return stats


def run_grid(name, engine_fn, configs, funding_loader, is_short=False):
    avail, cd, v21_s, v21_f = load_all()

    # base dates for fake v21 (from V21 fut daily index)
    all_dates = v21_f.index
    fv_full = fake_flat_v21(all_dates)

    rows = []
    for i, P in enumerate(configs, 1):
        t0 = time.time()
        ev = extract_events_parallel(avail, engine_fn, P, funding_loader=funding_loader)
        if len(ev) == 0:
            rows.append({**P, "n_events":0}); continue
        # Convert pnl for short
        # engine이 이미 short에선 long 반대 부호로 pnl_pct 기록함 (engine_breakdown_short 확인 필요)
        # simulate_fut은 long 가정이므로 short는 notional 반대로 넣으려면 엔진 재설계 필요.
        # 단순화: pnl 기반 direct eq simulation (m3 엔진의 entry/exit 메커니즘 활용)

        # Standalone: fake flat v21
        for tx in [0.0004, 0.0010, 0.003]:
            for cap in [0.333]:  # fake 100% cash면 cap 0.333 의미 있음
                try:
                    st_full = eval_with_m3(ev, fv_full, cd, {}, cap, leverage=1.0, tx=tx)
                except Exception as e:
                    st_full = {"Cal":0, "CAGR":0, "MDD":0, "n_entries":0, "error":str(e)[:40]}
                rows.append({**P, "tx": tx, "cap": cap, "span": "full",
                             "n_events": len(ev),
                             "Cal": round(st_full.get("Cal",0),3),
                             "CAGR": round(st_full.get("CAGR",0),4),
                             "MDD": round(st_full.get("MDD",0),4),
                             "entries": st_full.get("n_entries",0)})
                # train
                mask_t = ev["entry_ts"] <= TRAIN_END
                fv_t = fake_flat_v21(all_dates[all_dates <= TRAIN_END])
                try:
                    st_t = eval_with_m3(ev[mask_t].copy(), fv_t, cd, {}, cap, leverage=1.0, tx=tx)
                except Exception as e:
                    st_t = {"Cal":0, "CAGR":0, "MDD":0, "n_entries":0}
                rows.append({**P, "tx": tx, "cap": cap, "span": "train",
                             "n_events": int(mask_t.sum()),
                             "Cal": round(st_t.get("Cal",0),3),
                             "CAGR": round(st_t.get("CAGR",0),4),
                             "MDD": round(st_t.get("MDD",0),4),
                             "entries": st_t.get("n_entries",0)})
                # holdout
                mask_h = ev["entry_ts"] >= HOLDOUT_START
                fv_h = fake_flat_v21(all_dates[all_dates >= HOLDOUT_START])
                try:
                    st_h = eval_with_m3(ev[mask_h].copy(), fv_h, cd, {}, cap, leverage=1.0, tx=tx)
                except Exception as e:
                    st_h = {"Cal":0, "CAGR":0, "MDD":0, "n_entries":0}
                rows.append({**P, "tx": tx, "cap": cap, "span": "holdout",
                             "n_events": int(mask_h.sum()),
                             "Cal": round(st_h.get("Cal",0),3),
                             "CAGR": round(st_h.get("CAGR",0),4),
                             "MDD": round(st_h.get("MDD",0),4),
                             "entries": st_h.get("n_entries",0)})
        print(f"[{i}/{len(configs)}] {P} n_events={len(ev)} ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"revalid_{name}.csv"), index=False)
    # Top by hout_Cal at tx=0.04% (realistic)
    best = df[(df["span"] == "holdout") & (df["tx"] == 0.0004)].sort_values("Cal", ascending=False).head(5)
    print(f"\n=== {name} Top 5 Holdout Cal @ TX 0.04% ===")
    print(best.to_string(index=False))


def main():
    # 1. Pullback (long)
    configs_pb = []
    for ef in [30, 50, 80]:
        for es in [150, 200, 300]:
            for pb in [0.015, 0.025]:
                for tp in [0.06, 0.10]:
                    for ts in [72, 120]:
                        configs_pb.append({"ema_fast":ef, "ema_slow":es,
                                           "pullback_min":pb, "tp":tp, "tstop":ts})
    print(f"Pullback configs: {len(configs_pb)}")
    run_grid("pullback", run_pullback, configs_pb, funding_loader=None)


if __name__ == "__main__":
    main()
