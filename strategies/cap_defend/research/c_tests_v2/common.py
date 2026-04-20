"""C tests v2 - 공통 헬퍼.

수정사항 (AI 2차 리뷰 반영):
- FULL_END 통일 (load_coin default 2026-03-30 일치)
- 이벤트 공통 캐시 지원 (test0_extract 생성)
- v21 slice 복제 방어 (in-place 오염 방지)
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, HERE)

from m3_engine_final import (load_v21, load_universe_hist, list_available_futures,
                             load_coin_daily, simulate, metrics as metrics_spot)
from m3_engine_futures import (load_v21_futures, simulate_fut, metrics as metrics_fut)
from c_engine_v5 import run_c_v5, load_coin

TRAIN_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
FULL_END = pd.Timestamp("2026-03-30")  # load_coin default와 일치

P_SPOT = {"dip_bars": 24, "dip_thr": -0.20, "tp": 0.04, "tstop": 24}
P_FUT  = {"dip_bars": 24, "dip_thr": -0.18, "tp": 0.08, "tstop": 48}
CAP_SPOT = 0.333
CAP_FUT_OPTS = [0.12, 0.25, 0.30]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def slice_v21(v21: pd.DataFrame, start, end) -> pd.DataFrame:
    """V21 슬라이스 + 정규화. 항상 독립 복사본 반환 (in-place 오염 방지)."""
    sub = v21[(v21.index >= start) & (v21.index <= end)].copy()
    if len(sub) < 30:
        return sub
    sub["equity"] = sub["equity"].astype(float) / float(sub["equity"].iloc[0])
    sub["v21_ret"] = sub["equity"].pct_change().fillna(0.0)
    sub["prev_cash"] = sub["cash_ratio"].shift(1).fillna(sub["cash_ratio"].iloc[0])
    return sub


def extract_events(avail, P, tx=0.003, fd=0) -> pd.DataFrame:
    """직접 추출 (캐시 미사용)."""
    rows = []
    for c in avail:
        df = load_coin(c + "USDT")
        if df is None: continue
        _, evs = run_c_v5(df, tx=tx, fill_delay=fd, **P)
        for e in evs:
            e["coin"] = c
            rows.append(e)
    return pd.DataFrame(rows)


def load_cached_events(kind: str, tx: float = 0.003, fd: int = 0) -> pd.DataFrame:
    """test0_extract가 저장한 캐시를 로드. 없으면 직접 추출 fallback."""
    if tx == 0.003 and fd == 0:
        path = os.path.join(CACHE_DIR, f"events_{kind}.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    # fallback: 직접 추출
    v21_s, v21_f, hist, avail, cd = load_all()
    P = P_SPOT if kind == "spot" else P_FUT
    return extract_events(avail, P, tx=tx, fd=fd)


_CACHE = {}

def load_all():
    """V21 + hist + avail + coin_daily. 프로세스 내 singleton 캐시."""
    if "data" in _CACHE:
        return _CACHE["data"]
    v21_s = load_v21()
    v21_f = load_v21_futures()
    hist = load_universe_hist()
    avail = sorted(list_available_futures())
    cd = load_coin_daily(avail)
    _CACHE["data"] = (v21_s, v21_f, hist, avail, cd)
    return _CACHE["data"]


def run_spot_combo(events, coin_daily, v21_slice, hist, cap):
    """in-place 오염 방지: v21_slice 복제 후 전달."""
    return simulate(events, coin_daily, v21_slice.copy(), hist,
                    n_pick=1, cap_per_slot=cap, universe_size=15,
                    tx_cost=0.003, swap_edge_threshold=1)


def run_fut_combo(events, coin_daily, v21_slice, hist, cap, lev=3.0):
    """in-place 오염 방지: v21_slice 복제 후 전달."""
    return simulate_fut(events, coin_daily, v21_slice.copy(), hist,
                        n_pick=1, cap_per_slot=cap, universe_size=15,
                        tx_cost=0.003, swap_edge_threshold=1, leverage=lev)


def load_bars_1h(coin: str) -> pd.DataFrame | None:
    """1h 원본 bar 로드 (MAE 계산 등)."""
    return load_coin(coin + "USDT") if not coin.endswith("USDT") else load_coin(coin)
