#!/usr/bin/env python3
"""Group C — 레짐 조건부.

C1. BTC SMA200 위/아래 진입 허용
C2. BTC 14d vol 한도 진입 필터
C3. SMA200 기준 cap 차등 (위=풀cap, 아래=cap/2)
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common3 import (
    load_all, load_cached_events, run_splits,
    filter_regime_sma200, filter_vol_regime,
    CAP_SPOT, CAP_FUT_OPTS, OUT,
)
import pandas as pd


def main():
    v21_spot, v21_fut, hist, _, cd = load_all()
    ev_spot = load_cached_events("spot")
    ev_fut  = load_cached_events("fut")

    rows = []
    rows += run_splits("baseline", "spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_splits("baseline", "fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    # C1 SMA200 요구/차단
    for require_above in [True, False]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = filter_regime_sma200(ev, require_above)
            lab = f"C1_sma200_{'above' if require_above else 'below'}"
            print(f"  {lab} {kind}: kept {len(ev2)}/{len(ev)}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # C2 vol regime
    for max_vol in [0.6, 0.8, 1.0, 1.2]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = filter_vol_regime(ev, max_vol)
            lab = f"C2_vol_le{max_vol:.1f}"
            print(f"  {lab} {kind}: kept {len(ev2)}/{len(ev)}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # C3 cap 차등 (아래에선 cap/2) — 구현: event를 두 그룹으로 나누어 별도 시뮬 후 비교 기준으로
    # 여기선 간소화: SMA 위만 풀 cap, 아래만 cap/2 — 각각 독립 시뮬 후 합계 근사는 복잡.
    # 대신 SMA 위 cap 풀 / SMA 위 cap 절반 / SMA 아래 cap 풀 세 조합으로 대체.
    for require_above in [True]:
        for cap_adj in [0.5, 1.0]:
            for kind, ev, v21, caps, cap_base in [
                ("spot", ev_spot, v21_spot, [CAP_SPOT], CAP_SPOT),
                ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS, None),
            ]:
                ev2 = filter_regime_sma200(ev, require_above)
                if kind == "spot":
                    caps_eff = [cap_base * cap_adj]
                else:
                    caps_eff = [c * cap_adj for c in caps]
                lab = f"C3_sma_above_cap_x{cap_adj:.1f}"
                rows += run_splits(lab, kind, ev2, v21, cd, hist, caps_eff)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_regime.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")
    piv = df[df["span"] == "holdout"].pivot_table(
        index="label", columns=["kind", "cap"], values="Cal")
    print("\n=== Holdout Cal ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
