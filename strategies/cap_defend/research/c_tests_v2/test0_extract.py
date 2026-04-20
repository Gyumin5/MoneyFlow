#!/usr/bin/env python3
"""Test 0: 이벤트 공통 캐시 추출.

1회 실행해 현물/선물 기본 파라미터로 이벤트 DataFrame을 pickle로 저장.
이후 test1~8에서 재사용해 추출 반복 제거 (Codex 권고).
출력: c_tests_v2/cache/events_spot.pkl, events_fut.pkl
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import P_SPOT, P_FUT, load_all, extract_events, CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    print(f"Coins: {len(avail)}")

    print("\n[1/2] Extracting spot events...")
    ev_s = extract_events(avail, P_SPOT)
    ev_s.to_pickle(os.path.join(CACHE_DIR, "events_spot.pkl"))
    ev_s.to_csv(os.path.join(CACHE_DIR, "events_spot.csv"), index=False)
    print(f"  spot events: {len(ev_s)}")
    print(f"  reasons: {ev_s['reason'].value_counts().to_dict()}")
    print(f"  period: {ev_s['entry_ts'].min()} → {ev_s['entry_ts'].max()}")

    print("\n[2/2] Extracting fut events...")
    ev_f = extract_events(avail, P_FUT)
    ev_f.to_pickle(os.path.join(CACHE_DIR, "events_fut.pkl"))
    ev_f.to_csv(os.path.join(CACHE_DIR, "events_fut.csv"), index=False)
    print(f"  fut events: {len(ev_f)}")
    print(f"  reasons: {ev_f['reason'].value_counts().to_dict()}")
    print(f"  period: {ev_f['entry_ts'].min()} → {ev_f['entry_ts'].max()}")

    # 데이터 완결성 체크
    print("\n=== Data integrity quick check ===")
    missing = [c for c in avail if c not in cd]
    if missing:
        print(f"⚠  coin_daily missing: {missing}")
    else:
        print(f"✓  coin_daily: {len(cd)} coins OK")
    # universe 기간
    hist_dates = sorted(pd.Timestamp(d) for d in hist.keys())
    print(f"✓  universe snapshots: {len(hist_dates)} ({hist_dates[0]} ~ {hist_dates[-1]})")

    print(f"\n저장: {CACHE_DIR}/")


if __name__ == "__main__":
    main()
