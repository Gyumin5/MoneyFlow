#!/usr/bin/env python3
"""추가 검증 C: Fut C에 intrabar stop-loss 추가.

Test 7 MAE: fut 전구간 lev_worst_mae -198% (wipeout 위험), n_wipeout_90=111건.
→ 진입가 대비 intrabar Low가 -10/-15/-20% 터치 시 청산.
stop_level 적용 시 Full/Train/Holdout Cal, MDD, n_stops 비교.

engine 수정 아닌 event-level post-filter 방식:
- 각 event의 bars_held 구간에서 Low 최소값 찾음 (MAE)
- MAE가 stop_level 이하면 stop으로 조기 청산 → new pnl_pct = stop_level
- 새 events df로 simulate_fut 재시뮬
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_FUT_OPTS, TRAIN_END, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, load_bars_1h, slice_v21,
                     run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def apply_intrabar_stop(ev: pd.DataFrame, stop_pct: float) -> pd.DataFrame:
    """진입 후 Low가 stop_pct 이하면 해당 시점 stop 체결.

    stop_pct 예: -0.10 means -10% intrabar.
    lev 미적용 원시 가격 기준. exit_px 업데이트.
    """
    if len(ev) == 0 or stop_pct is None:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"] = pd.to_datetime(ev["exit_ts"])
    bar_cache = {}
    stop_count = 0
    new_pnls = []
    new_exits = []
    new_reasons = []
    for _, r in ev.iterrows():
        c = r["coin"]
        if c not in bar_cache:
            bar_cache[c] = load_bars_1h(c)
        bars = bar_cache[c]
        if bars is None:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            continue
        idx = bars.index
        mask = (idx >= r["entry_ts"]) & (idx <= r["exit_ts"])
        sub = bars[mask]
        if len(sub) == 0:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            continue
        # entry 봉 이후 low 확인
        stop_price = float(r["entry_px"]) * (1 + stop_pct)
        # 첫 번째로 Low <= stop_price 인 봉
        hit = sub[sub["Low"] <= stop_price]
        if len(hit) > 0:
            hit_ts = hit.index[0]
            # entry 봉에서 즉시 맞으면 stop=entry_px*(1+stop_pct)로 체결 가정
            new_pnls.append(round(stop_pct * 100, 3))
            new_exits.append(hit_ts)
            new_reasons.append("stop")
            stop_count += 1
        else:
            new_pnls.append(r["pnl_pct"])
            new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["reason"] = new_reasons
    return ev


def run_split(name, v21, ev, cap):
    rows = []
    for span, s, e in [("full", v21.index[0], FULL_END),
                        ("train", v21.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21, s, e)
        evs = ev[(ev["entry_ts"] >= v21s.index[0]) & (ev["entry_ts"] <= v21s.index[-1])].copy()
        _, st = run_fut_combo(evs, kwargs["cd"], v21s.copy(), kwargs["hist"], cap)
        n_stops = int((ev["reason"] == "stop").sum())
        rows.append({"label": name, "span": span, "cap": cap,
                     "Cal": round(st["Cal"], 3),
                     "CAGR": round(st["CAGR"], 4),
                     "MDD": round(st["MDD"], 4),
                     "n_entries": st["n_entries"],
                     "n_stops_in_events": n_stops})
    return rows


def main():
    global kwargs
    v21_s, v21_f, hist, avail, cd = load_all()
    kwargs = {"cd": cd, "hist": hist}

    ev_f = load_cached_events("fut")

    print(f"Base fut events: {len(ev_f)}")

    stops = [None, -0.10, -0.15, -0.20, -0.25]
    rows = []
    for sp in stops:
        print(f"[stop={sp}] applying...")
        ev_mod = apply_intrabar_stop(ev_f, sp) if sp is not None else ev_f.copy()
        ev_mod["reason"] = ev_mod.get("reason", pd.Series([""] * len(ev_mod)))
        n_triggered = int((ev_mod["reason"] == "stop").sum())
        print(f"  stop triggered: {n_triggered}/{len(ev_mod)}")
        label_prefix = "none" if sp is None else f"stop{int(sp*100)}"
        for cap in CAP_FUT_OPTS:
            rows += run_split(f"fut_{label_prefix}_cap{cap}", v21_f, ev_mod, cap)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "addl_v3_stop.csv"), index=False)

    # Pivot
    print("\n=== FUT Stop 비교 (Cal) ===")
    print(df.pivot_table(index="label", columns="span", values="Cal").to_string())
    print("\n=== FUT Stop 비교 (MDD) ===")
    print(df.pivot_table(index="label", columns="span", values="MDD").to_string())
    print(f"\n저장: {OUT}/addl_v3_stop.csv")


if __name__ == "__main__":
    main()
