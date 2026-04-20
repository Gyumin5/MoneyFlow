#!/usr/bin/env python3
"""Phase 파이프라인 공통 헬퍼.

- parse_tag: phase1 cfg_id 형식 파싱
- build_trace / run_single_target: 선물/현물 단일 타깃 실행
- emit_daily_equity: csv.gz 저장 (Date, Value, Ret)
- summarize_phase_raw: raw rows → summary (mMetric/sMetric/wMDD)
- write_manifest: atomic json 쓰기 (tmp+os.replace)
- atomic_append_csv: partial-write 내성 append
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
for p in (REPO, CAP, os.path.join(REPO, "trade")):
    if p not in sys.path:
        sys.path.insert(0, p)

FULL_END = os.environ.get("PHASE_END", "2026-04-13")

TAG_RE = re.compile(
    r"^(?P<asset>spot|fut)_(?P<iv>1D|4h|2h)_S(?P<sma>\d+)_M(?P<ms>\d+)_(?P<ml>\d+)"
    r"_(?P<vtag>[db]\d+\.\d+)_SN(?P<snap>\d+)_L(?P<lev>\d+)$"
)

RAW_COLUMNS = [
    "tag", "anchor", "asset", "lev",
    "Sh", "Cal", "CAGR", "MDD", "CVaR5", "Ulcer", "TUW",
    "rebal", "liq", "error",
]


def parse_tag(tag: str) -> dict:
    m = TAG_RE.match(str(tag))
    if not m:
        raise ValueError(f"unparsed tag: {tag}")
    d = m.groupdict()
    vol_mode = "daily" if d["vtag"].startswith("d") else "bar"
    vol_thr = float(d["vtag"][1:])
    return {
        "asset": d["asset"],
        "interval": "D" if d["iv"] == "1D" else d["iv"],
        "sma": int(d["sma"]),
        "ms": int(d["ms"]),
        "ml": int(d["ml"]),
        "vol_mode": vol_mode,
        "vol_thr": vol_thr,
        "snap": int(d["snap"]),
        "lev": int(d["lev"]),
    }


# ─── atomic io ───
def write_manifest(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {**data, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_manifest(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def atomic_append_csv(path: str, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if need_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ─── equity + metrics ───
def equity_metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    zero = {"Sh": 0.0, "Cal": 0.0, "CAGR": 0.0, "MDD": 0.0,
            "CVaR5": 0.0, "Ulcer": 0.0, "TUW": 0.0}
    if len(eq) < 2:
        return zero
    ed = eq.resample("D").last().dropna()
    if len(ed) < 2 or ed.iloc[0] <= 0:
        return zero
    yrs = (ed.index[-1] - ed.index[0]).days / 365.25
    cagr = (ed.iloc[-1] / ed.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dr = ed.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0.0
    dd = ed / ed.cummax() - 1
    mdd = float(dd.min())
    cal = cagr / abs(mdd) if mdd else 0.0
    cvar5 = float(np.percentile(dr, 5)) if len(dr) > 10 else 0.0
    ulcer = float(np.sqrt((dd ** 2).mean()))
    tuw = float((dd < -0.05).mean())
    return {"Sh": sh, "Cal": cal, "CAGR": cagr, "MDD": mdd,
            "CVaR5": cvar5, "Ulcer": ulcer, "TUW": tuw}


def emit_daily_equity(eq: pd.Series, out_path: str, horizon_days: int = 252) -> None:
    """anchor 시작부터 horizon_days 만큼만 저장. csv.gz."""
    eq = eq.dropna()
    if len(eq) < 2:
        return
    ed = eq.resample("D").last().dropna()
    if len(ed) == 0:
        return
    cutoff = ed.index[0] + pd.Timedelta(days=horizon_days)
    ed = ed[ed.index <= cutoff]
    ret = ed.pct_change().fillna(0.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame({"Date": ed.index.strftime("%Y-%m-%d"),
                       "Value": ed.values, "Ret": ret.values})
    tmp = f"{out_path}.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    os.replace(tmp, out_path)


def load_equity_gz(path: str) -> pd.Series | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, compression="gzip", parse_dates=["Date"])
        return df.set_index("Date")["Value"].astype(float)
    except Exception:
        return None


# ─── single-target runners ───
_FUT_DATA = None
_SPOT_READY = False


def preload_futures():
    global _FUT_DATA
    if _FUT_DATA is not None:
        return _FUT_DATA
    from backtest_futures_full import load_data
    _FUT_DATA = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    return _FUT_DATA


def preload_spot():
    global _SPOT_READY
    if _SPOT_READY:
        return
    import run_current_coin_v20_backtest as spot_bt
    univ = spot_bt.load_universe(top_n=40)
    bars = spot_bt.load_price_bars(univ)
    spot_bt.load_price_bars = lambda um: bars
    spot_bt.load_universe = lambda top_n=40: univ
    _SPOT_READY = True


def run_single_target(asset: str, cfg: dict, lev: float, anchor: str,
                      end: str = FULL_END, want_equity: bool = False) -> dict:
    """asset in {spot, fut}. cfg keys: interval, sma, ms, ml, vol_mode, vol_thr, snap."""
    if asset == "spot":
        return _run_spot(cfg, anchor, end, want_equity)
    return _run_futures(cfg, lev, anchor, end, want_equity)


def _cfg_to_member(cfg: dict) -> dict:
    # V21부터 4h 멤버 제거. D봉 템플릿만 사용.
    # 연구 파이프라인은 4h도 탐색 가능하지만 템플릿은 D 기반 공통 필드만 참조
    from coin_live_engine import MEMBER_D_SMA50
    template = MEMBER_D_SMA50
    member = dict(template)
    member.update({
        "interval": cfg["interval"],
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "snap_interval_bars": cfg["snap"],
        "vol_mode": cfg["vol_mode"],
        "vol_threshold": cfg["vol_thr"],
        "gap_threshold": -1.0,
        "exclusion_days": 0,
    })
    return member


def _run_spot(cfg: dict, anchor: str, end: str, want_equity: bool) -> dict:
    preload_spot()
    import run_current_coin_v20_backtest as spot_bt
    member = _cfg_to_member(cfg)
    orig_m = spot_bt.MEMBERS
    orig_w = spot_bt.ENSEMBLE_WEIGHTS
    spot_bt.MEMBERS = {"single": member}
    spot_bt.ENSEMBLE_WEIGHTS = {"single": 1.0}
    try:
        res = spot_bt.run_backtest(start=anchor, end=end)
    finally:
        spot_bt.MEMBERS = orig_m
        spot_bt.ENSEMBLE_WEIGHTS = orig_w
    eq = res["equity"]
    m = equity_metrics(eq)
    m["rebal"] = int(res.get("rebal_count", 0))
    m["liq"] = 0
    if want_equity:
        m["_equity"] = eq
    return m


def run_spot_ensemble(member_cfgs: dict, weights: dict,
                      anchor: str, end: str = FULL_END,
                      want_equity: bool = False) -> dict:
    """Spot 멀티 멤버 앙상블. member_cfgs/weights는 같은 key 사용.
    coin_live_engine MEMBERS/ENSEMBLE_WEIGHTS에 주입해 단일 계정 비중 합산 실행."""
    preload_spot()
    import run_current_coin_v20_backtest as spot_bt
    members = {k: _cfg_to_member(v) for k, v in member_cfgs.items()}
    total = sum(weights.values())
    norm_w = {k: w / total for k, w in weights.items()} if total > 0 else weights
    orig_m = spot_bt.MEMBERS
    orig_w = spot_bt.ENSEMBLE_WEIGHTS
    spot_bt.MEMBERS = members
    spot_bt.ENSEMBLE_WEIGHTS = norm_w
    try:
        res = spot_bt.run_backtest(start=anchor, end=end)
    finally:
        spot_bt.MEMBERS = orig_m
        spot_bt.ENSEMBLE_WEIGHTS = orig_w
    eq = res["equity"]
    m = equity_metrics(eq)
    m["rebal"] = int(res.get("rebal_count", 0))
    m["liq"] = 0
    if want_equity:
        m["_equity"] = eq
    return m


def _run_futures(cfg: dict, lev: float, anchor: str, end: str, want_equity: bool) -> dict:
    data = preload_futures()
    from backtest_futures_full import run as bt_run
    from futures_ensemble_engine import SingleAccountEngine, combine_targets
    from run_futures_fixedlev_search import FIXED_CFG

    params = dict(FIXED_CFG)
    params.update({
        "vol_mode": cfg["vol_mode"],
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "vol_threshold": cfg["vol_thr"],
        "snap_interval_bars": cfg["snap"],
    })
    iv = cfg["interval"]
    bars, funding = data[iv]
    bars_1h, funding_1h = data["1h"]
    trace: list = []
    bt_run(bars, funding, interval=iv, leverage=1.0,
           start_date=anchor, end_date=end, _trace=trace, **params)
    dates = bars_1h["BTC"].index
    ddates = dates[(dates >= anchor) & (dates <= end)]
    combined = combine_targets({"x": trace}, {"x": 1.0}, ddates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(lev), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always",
    )
    m = engine.run(combined)
    eq = m.get("_equity")
    if eq is not None and not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    ext = equity_metrics(eq) if eq is not None else {
        "CVaR5": 0.0, "Ulcer": 0.0, "TUW": 0.0}
    out = {
        "Sh": float(m.get("Sharpe", 0)),
        "Cal": float(m.get("Cal", 0)),
        "CAGR": float(m.get("CAGR", 0)),
        "MDD": float(m.get("MDD", 0)),
        "CVaR5": float(ext.get("CVaR5", 0)),
        "Ulcer": float(ext.get("Ulcer", 0)),
        "TUW": float(ext.get("TUW", 0)),
        "rebal": int(m.get("Rebal", 0)),
        "liq": int(m.get("Liq", 0)),
    }
    if want_equity and eq is not None:
        out["_equity"] = eq
    return out


def build_trace(asset: str, cfg: dict, lev: float, anchor: str, end: str = FULL_END) -> dict:
    """Phase-3 ensemble용: combine_targets 입력(trace) + 파라미터 반환."""
    if asset != "fut":
        raise ValueError("build_trace currently supports futures only")
    data = preload_futures()
    from backtest_futures_full import run as bt_run
    from run_futures_fixedlev_search import FIXED_CFG
    params = dict(FIXED_CFG)
    params.update({
        "vol_mode": cfg["vol_mode"],
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "vol_threshold": cfg["vol_thr"],
        "snap_interval_bars": cfg["snap"],
    })
    iv = cfg["interval"]
    bars, funding = data[iv]
    trace: list = []
    bt_run(bars, funding, interval=iv, leverage=1.0,
           start_date=anchor, end_date=end, _trace=trace, **params)
    return {"trace": trace, "interval": iv, "lev": lev}


# ─── summary ───
def summarize_phase_raw(df: pd.DataFrame) -> pd.DataFrame:
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""]
    rows = []
    metric_cols = ["Sh", "Cal", "CAGR", "MDD", "CVaR5", "Ulcer", "TUW"]
    for (tag, asset, lev), g in df.groupby(["tag", "asset", "lev"]):
        d = {"tag": tag, "asset": asset, "lev": lev, "n": len(g)}
        for c in metric_cols:
            if c in g:
                d[f"m{c}"] = float(g[c].mean())
                if c in ("Sh", "Cal", "CAGR"):
                    d[f"s{c}"] = float(g[c].std())
        if "MDD" in g:
            d["wMDD"] = float(g["MDD"].min())
        if "Cal" in g:
            d["win_rate"] = float((g["Cal"] > 0).mean())
        if "rebal" in g:
            d["rebal_mean"] = float(g["rebal"].mean())
        if "liq" in g:
            d["liq_sum"] = int(g["liq"].sum()) if g["liq"].notna().any() else 0
        rows.append(d)
    return pd.DataFrame(rows)
