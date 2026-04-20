"""Next strategies common helpers — c_engine_v5 스타일 인터페이스."""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, HERE)

from c_engine_v5 import load_coin
from m3_engine_final import (load_v21, load_universe_hist, list_available_futures,
                             load_coin_daily, metrics as metrics_spot)
from m3_engine_futures import load_v21_futures, metrics as metrics_fut

TRAIN_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
FULL_END = pd.Timestamp("2026-03-30")


def extract_all(avail: list, engine_fn, extra_fn=None, **params):
    """전 코인 engine_fn 실행 → {coin: equity_series}, events DataFrame 반환.

    engine_fn(df, **params) → (eq_series, events_list).
    engine 내부 equity에는 tx/funding 모두 반영됨 (validator가 이걸 써야 공정).

    extra_fn: coin별 추가 인자 (예: funding data). fn(coin_symbol) → dict.
    """
    eq_map = {}
    rows = []
    for c in avail:
        df = load_coin(c + "USDT")
        if df is None: continue
        extra = extra_fn(c) if extra_fn else {}
        eq, evs = engine_fn(df, **extra, **params)
        eq_map[c] = eq
        for e in evs:
            e["coin"] = c
            rows.append(e)
    return eq_map, pd.DataFrame(rows)


def aggregate_ew_portfolio(eq_map: dict, initial: float = 10000.0) -> pd.Series:
    """Per-coin engine equity를 EW 포트폴리오로 집계 (tx + funding 보존).

    각 코인 1h equity를 daily last-resample → 일간 수익률 → 크로스섹션 평균 (EW across N coins).
    동시 포지션 보유 시 자동 축소 (각 코인 1/N 가중, 비포지션 코인은 수익률 0).
    tx/funding은 per-coin eq에 이미 반영되어 있어 자연히 집계 반영.
    """
    if not eq_map:
        return pd.Series(dtype=float, name="equity")
    daily_eq = {}
    for c, eq in eq_map.items():
        if eq is None or len(eq) == 0: continue
        s = eq.copy()
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx
        # 1h → daily last, ffill (데이터 없는 날 유지)
        d = s.resample("D").last().ffill()
        daily_eq[c] = d

    df = pd.DataFrame(daily_eq).dropna(how="all")
    # 각 코인 수익률 (초기 대비)
    rets = df.pct_change().fillna(0.0)
    # EW 평균
    port_ret = rets.mean(axis=1)
    port_eq = initial * (1 + port_ret).cumprod()
    port_eq.name = "equity"
    return port_eq


def extract_events_only(avail: list, engine_fn, **params) -> pd.DataFrame:
    """이벤트만 수집 (과거 호환). aggregate에는 쓰지 말 것."""
    rows = []
    for c in avail:
        df = load_coin(c + "USDT")
        if df is None: continue
        _, evs = engine_fn(df, **params)
        for e in evs:
            e["coin"] = c
            rows.append(e)
    return pd.DataFrame(rows)


def simple_metrics(eq: pd.Series, bpy: int = 365) -> dict:
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {"CAGR": 0.0, "MDD": 0.0, "Cal": 0.0, "Sharpe": 0.0}
    days = (eq.index[-1] - eq.index[0]).days
    yrs = days / 365.25 if days > 0 else 0.001
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0.0
    std = float(rets.std())
    sh = float(rets.mean() / std * np.sqrt(bpy)) if std > 0 else 0.0
    return {"CAGR": round(cagr, 4), "MDD": round(mdd, 4),
            "Cal": round(cal, 3), "Sharpe": round(sh, 3)}


def split_metrics(eq: pd.Series) -> dict:
    def slc(s, start, end):
        sub = s.copy()
        idx = pd.to_datetime(sub.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        sub.index = idx
        sub = sub[(sub.index >= start) & (sub.index <= end)]
        if len(sub) < 30:
            return {"CAGR": 0, "MDD": 0, "Cal": 0}
        return simple_metrics(sub / sub.iloc[0])
    return {
        "full": slc(eq, eq.index[0], FULL_END),
        "train": slc(eq, eq.index[0], TRAIN_END),
        "holdout": slc(eq, HOLDOUT_START, FULL_END),
    }


_CACHE = {}
def load_all():
    if "d" in _CACHE:
        return _CACHE["d"]
    avail = sorted(list_available_futures())
    cd = load_coin_daily(avail)
    v21_s = load_v21()
    v21_f = load_v21_futures()
    _CACHE["d"] = (avail, cd, v21_s, v21_f)
    return _CACHE["d"]
