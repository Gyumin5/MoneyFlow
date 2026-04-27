"""c_tests_v3 공통 헬퍼.

c_tests_v2/common.py 상위에 얹어 다음을 제공:
- v3 테스트들이 공통으로 쓰는 split runner
- event 전처리 함수(trailing/momentum/tstop/time-stop/partial-TP 등)
- BTC daily regime 로더
"""
from __future__ import annotations
import os, sys
from functools import lru_cache
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", "c_tests_v2"))
sys.path.insert(0, V2)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from common import (  # noqa: E402  (from c_tests_v2)
    TRAIN_END, HOLDOUT_START, FULL_END,
    P_SPOT, P_FUT, CAP_SPOT, CAP_FUT_OPTS,
    load_all, load_cached_events, load_bars_1h, slice_v21,
    run_spot_combo, run_fut_combo,
)

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────
# Split runner
# ─────────────────────────────────────────────

def run_splits(label: str, kind: str, ev: pd.DataFrame,
               v21: pd.DataFrame, cd: dict, hist: dict,
               caps: list[float]) -> list[dict]:
    """kind='spot' | 'fut'. ev는 이미 가공된 events."""
    rows = []
    runner = run_fut_combo if kind == "fut" else run_spot_combo
    for span, start, end in [
        ("full",    v21.index[0], FULL_END),
        ("train",   v21.index[0], TRAIN_END),
        ("holdout", HOLDOUT_START, FULL_END),
    ]:
        v21s = slice_v21(v21, start, end)
        if v21s is None or len(v21s) < 30:
            continue
        ev_s = ev.copy()
        if len(ev_s):
            ev_s["entry_ts"] = pd.to_datetime(ev_s["entry_ts"])
            ev_s = ev_s[(ev_s["entry_ts"] >= v21s.index[0]) &
                        (ev_s["entry_ts"] <= v21s.index[-1])]
        for cap in caps:
            if len(ev_s) == 0:
                rows.append({"label": label, "kind": kind, "span": span,
                             "cap": cap, "Cal": 0, "CAGR": 0, "MDD": 0,
                             "n_entries": 0})
                continue
            _, st = runner(ev_s, cd, v21s, hist, cap)
            rows.append({
                "label": label, "kind": kind, "span": span, "cap": cap,
                "Cal":       round(st.get("Cal", 0), 3),
                "CAGR":      round(st.get("CAGR", 0), 4),
                "MDD":       round(st.get("MDD", 0), 4),
                "Sharpe":    round(st.get("Sharpe", 0), 3),
                "n_entries": int(st.get("n_entries", 0)),
                "n_liq":     int(st.get("n_liquidations", 0) or 0),
            })
    return rows


# ─────────────────────────────────────────────
# Event transformations (post-filter)
# ─────────────────────────────────────────────

@lru_cache(maxsize=None)
def _bars_cached(coin: str) -> pd.DataFrame | None:
    return load_bars_1h(coin)


def _get_bars_span(coin: str, t0, t1):
    bars = _bars_cached(coin)
    if bars is None:
        return None
    mask = (bars.index >= t0) & (bars.index <= t1)
    return bars[mask]


def apply_trailing_stop(ev: pd.DataFrame, trail_pct: float) -> pd.DataFrame:
    """진입 후 (prev bar까지의) High 최고치 대비 trail_pct 이하로 Low 이탈 시 청산.
    1h봉 내 H/L 순서 불확실 → running_high는 shift(1)까지로 보수화.
    exit_px = trigger 시점 running_high * (1+trail_pct).
    """
    if len(ev) == 0 or trail_pct is None:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exits, new_reasons, new_exit_px = [], [], [], []
    for _, r in ev.iterrows():
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) == 0:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_exit_px.append(r["exit_px"])
            continue
        # 보수: prev bar까지의 running_high 사용
        running_high_prev = sub["High"].cummax().shift(1)
        trig_prices = running_high_prev * (1 + trail_pct)
        mask = (sub["Low"] <= trig_prices) & running_high_prev.notna()
        if mask.any():
            hit_ts = sub.index[mask][0]
            ref_high = float(running_high_prev.loc[hit_ts])
            exit_px = ref_high * (1 + trail_pct)
            pnl = (exit_px / float(r["entry_px"]) - 1.0) * 100.0
            new_pnls.append(round(pnl, 3))
            new_exits.append(hit_ts)
            new_reasons.append("trail")
            new_exit_px.append(float(exit_px))
        else:
            new_pnls.append(r["pnl_pct"])
            new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            new_exit_px.append(r["exit_px"])
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["exit_px"] = new_exit_px
    ev["reason"]  = new_reasons
    return ev


def apply_momentum_exit(ev: pd.DataFrame, window_h: int) -> pd.DataFrame:
    """진입 후 window_h 시간 내 양봉(Close>Open) 0건이면 확인 봉 Close로 청산.
    확인 시점 = first_h 마지막 봉의 Close.
    """
    if len(ev) == 0 or not window_h:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exits, new_reasons, new_exit_px = [], [], [], []
    for _, r in ev.iterrows():
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) < 2:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_exit_px.append(r["exit_px"])
            continue
        first_h = sub.iloc[1:1 + window_h]
        if len(first_h) == 0:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_exit_px.append(r["exit_px"])
            continue
        green = first_h["Close"] > first_h["Open"]
        if not green.any():
            exit_ts = first_h.index[-1]
            exit_px = float(first_h["Close"].iloc[-1])  # ← Close 확인 시점
            pnl = (exit_px / float(r["entry_px"]) - 1.0) * 100.0
            new_pnls.append(round(pnl, 3))
            new_exits.append(exit_ts)
            new_reasons.append("momentum_exit")
            new_exit_px.append(exit_px)
        else:
            new_pnls.append(r["pnl_pct"])
            new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            new_exit_px.append(r["exit_px"])
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["exit_px"] = new_exit_px
    ev["reason"]  = new_reasons
    return ev


def apply_tstop(ev: pd.DataFrame, tstop_h: int) -> pd.DataFrame:
    """bars_held를 tstop_h 이하로 제한. 초과분은 tstop_h 번째 봉 Close로 청산.
    exit_px도 업데이트 (엔진 settle에 사용)."""
    if len(ev) == 0 or not tstop_h:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exits, new_reasons, new_bars, new_exit_px = [], [], [], [], []
    for _, r in ev.iterrows():
        if int(r["bars_held"]) <= tstop_h:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_bars.append(r["bars_held"])
            new_exit_px.append(r["exit_px"]); continue
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) < tstop_h:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_bars.append(r["bars_held"])
            new_exit_px.append(r["exit_px"]); continue
        cut = sub.iloc[tstop_h]
        exit_ts = cut.name
        exit_px = float(cut["Close"])
        pnl = (exit_px / float(r["entry_px"]) - 1.0) * 100.0
        new_pnls.append(round(pnl, 3))
        new_exits.append(exit_ts)
        new_reasons.append("tstop_cut")
        new_bars.append(tstop_h)
        new_exit_px.append(exit_px)
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["exit_px"] = new_exit_px
    ev["reason"]  = new_reasons
    ev["bars_held"] = new_bars
    return ev


def apply_partial_tp(ev: pd.DataFrame, tp_full: float, split: float = 0.5) -> pd.DataFrame:
    """부분 익절 근사. half_tp 도달 시 split 비중 익절, 나머지는 원 exit 유지.
    ⚠ 근사: slot은 여전히 원 exit까지 점유 (자본 효율성 실제보다 낮게 평가).
    엔진 settle은 exit_px 사용하므로 exit_px도 가중평균으로 조정."""
    if len(ev) == 0 or not tp_full:
        return ev
    half_tp = tp_full * 0.5
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exit_px = [], []
    for _, r in ev.iterrows():
        entry_px = float(r["entry_px"])
        orig_pnl = float(r["pnl_pct"]) / 100.0
        orig_exit_px = float(r["exit_px"])
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) == 0:
            new_pnls.append(r["pnl_pct"]); new_exit_px.append(orig_exit_px); continue
        half_target = entry_px * (1 + half_tp)
        hit = sub[sub["High"] >= half_target]
        if len(hit) > 0:
            combined = split * half_tp + (1 - split) * orig_pnl
            # exit_px도 가중 평균
            half_exit_px = half_target
            eff_exit_px = split * half_exit_px + (1 - split) * orig_exit_px
            new_pnls.append(round(combined * 100.0, 3))
            new_exit_px.append(eff_exit_px)
        else:
            new_pnls.append(r["pnl_pct"])
            new_exit_px.append(orig_exit_px)
    ev["pnl_pct"] = new_pnls
    ev["exit_px"] = new_exit_px
    return ev


def apply_intrabar_stop(ev: pd.DataFrame, stop_pct: float) -> pd.DataFrame:
    """Fixed intrabar stop. exit_px = entry_px*(1+stop_pct) 업데이트."""
    if len(ev) == 0 or stop_pct is None:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exits, new_reasons, new_exit_px = [], [], [], []
    for _, r in ev.iterrows():
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) == 0:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_exit_px.append(r["exit_px"])
            continue
        stop_price = float(r["entry_px"]) * (1 + stop_pct)
        hit = sub[sub["Low"] <= stop_price]
        if len(hit) > 0:
            new_pnls.append(round(stop_pct * 100, 3))
            new_exits.append(hit.index[0])
            new_reasons.append("stop")
            new_exit_px.append(stop_price)
        else:
            new_pnls.append(r["pnl_pct"])
            new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            new_exit_px.append(r["exit_px"])
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["exit_px"] = new_exit_px
    ev["reason"]  = new_reasons
    return ev


def apply_time_conditioned_stop(ev: pd.DataFrame, stop_pct: float,
                                  min_hours: int) -> pd.DataFrame:
    """min_hours 경과 후에만 intrabar stop 활성. exit_px 업데이트."""
    if len(ev) == 0 or stop_pct is None:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exits, new_reasons, new_exit_px = [], [], [], []
    for _, r in ev.iterrows():
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) <= min_hours:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); new_exit_px.append(r["exit_px"])
            continue
        stop_price = float(r["entry_px"]) * (1 + stop_pct)
        active = sub.iloc[min_hours:]
        hit = active[active["Low"] <= stop_price]
        if len(hit) > 0:
            new_pnls.append(round(stop_pct * 100, 3))
            new_exits.append(hit.index[0])
            new_reasons.append("tstop_loss")
            new_exit_px.append(stop_price)
        else:
            new_pnls.append(r["pnl_pct"])
            new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
            new_exit_px.append(r["exit_px"])
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["exit_px"] = new_exit_px
    ev["reason"]  = new_reasons
    return ev


# ─────────────────────────────────────────────
# Entry filters
# ─────────────────────────────────────────────

def filter_next_bar_dip(ev: pd.DataFrame, next_drop: float) -> pd.DataFrame:
    """시그널 봉 다음 1h 봉이 next_drop 이하면 진입 스킵.
    통과 시 entry를 다음 봉 Open으로 지연 (look-ahead bias 방지).
    """
    if len(ev) == 0 or next_drop is None:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    keep, new_entry_ts, new_entry_px, new_pnl = [], [], [], []
    for _, r in ev.iterrows():
        bars = _bars_cached(r["coin"])
        if bars is None:
            keep.append(True); new_entry_ts.append(r["entry_ts"])
            new_entry_px.append(r["entry_px"]); new_pnl.append(r["pnl_pct"])
            continue
        after = bars[bars.index > r["entry_ts"]]
        if len(after) < 1:
            keep.append(True); new_entry_ts.append(r["entry_ts"])
            new_entry_px.append(r["entry_px"]); new_pnl.append(r["pnl_pct"])
            continue
        nxt = after.iloc[0]
        ret = float(nxt["Close"]) / float(r["entry_px"]) - 1.0
        ok = ret > next_drop
        keep.append(bool(ok))
        if ok:
            # 다음 봉 Open으로 진입 지연
            new_e_ts = nxt.name
            new_e_px = float(nxt["Open"])
            new_entry_ts.append(new_e_ts)
            new_entry_px.append(new_e_px)
            # pnl 재계산 (exit_px는 원래 유지)
            new_p = (float(r["exit_px"]) / new_e_px - 1.0) * 100.0
            new_pnl.append(round(new_p, 3))
        else:
            new_entry_ts.append(r["entry_ts"])
            new_entry_px.append(r["entry_px"])
            new_pnl.append(r["pnl_pct"])
    ev["entry_ts"] = new_entry_ts
    ev["entry_px"] = new_entry_px
    ev["pnl_pct"]  = new_pnl
    return ev[keep].reset_index(drop=True)


def filter_bounce_confirm(ev: pd.DataFrame, window_h: int) -> pd.DataFrame:
    """시그널 후 window_h 내 양봉 있으면 해당 양봉 Open에서 진입.
    없으면 진입 스킵. look-ahead 방지 위해 entry_ts/entry_px 재설정.
    """
    if len(ev) == 0 or not window_h:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    keep, new_entry_ts, new_entry_px, new_pnl = [], [], [], []
    for _, r in ev.iterrows():
        bars = _bars_cached(r["coin"])
        if bars is None:
            keep.append(True); new_entry_ts.append(r["entry_ts"])
            new_entry_px.append(r["entry_px"]); new_pnl.append(r["pnl_pct"])
            continue
        after = bars[bars.index > r["entry_ts"]].iloc[:window_h]
        if len(after) == 0:
            keep.append(True); new_entry_ts.append(r["entry_ts"])
            new_entry_px.append(r["entry_px"]); new_pnl.append(r["pnl_pct"])
            continue
        green_mask = after["Close"] > after["Open"]
        if green_mask.any():
            first_green_ts = after.index[green_mask][0]
            # 다음 봉 Open (양봉 "확인" 후 실제 진입은 그 다음 봉 시가)
            after_green = bars[bars.index > first_green_ts]
            if len(after_green) == 0:
                keep.append(False); new_entry_ts.append(r["entry_ts"])
                new_entry_px.append(r["entry_px"]); new_pnl.append(r["pnl_pct"])
                continue
            entry_row = after_green.iloc[0]
            new_e_ts = entry_row.name
            new_e_px = float(entry_row["Open"])
            # 지연된 entry가 exit_ts 이후면 무효
            if new_e_ts >= r["exit_ts"]:
                keep.append(False); new_entry_ts.append(r["entry_ts"])
                new_entry_px.append(r["entry_px"]); new_pnl.append(r["pnl_pct"])
                continue
            keep.append(True)
            new_entry_ts.append(new_e_ts)
            new_entry_px.append(new_e_px)
            new_p = (float(r["exit_px"]) / new_e_px - 1.0) * 100.0
            new_pnl.append(round(new_p, 3))
        else:
            keep.append(False)
            new_entry_ts.append(r["entry_ts"])
            new_entry_px.append(r["entry_px"])
            new_pnl.append(r["pnl_pct"])
    ev["entry_ts"] = new_entry_ts
    ev["entry_px"] = new_entry_px
    ev["pnl_pct"]  = new_pnl
    return ev[keep].reset_index(drop=True)


# ─────────────────────────────────────────────
# BTC regime
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _btc_daily():
    """BTC 일봉 종가 (shift 1 적용해 해당 날짜의 '전일까지' 확정 종가만 사용).
    resample("D").last()는 하루 종료 시점 종가(23:59 마감) → 당일 14:00 entry에서
    참조 시 look-ahead. shift(1)로 전일 종가 기준 교정.
    """
    bars = _bars_cached("BTC")
    if bars is None:
        return None
    daily = bars["Close"].resample("D").last().dropna().shift(1).dropna()
    return daily


def btc_regime_at(ts) -> dict:
    """해당 ts 시점 BTC regime. daily은 이미 전일까지 확정 종가."""
    daily = _btc_daily()
    if daily is None:
        return {"above_sma200": None, "vol14d": None}
    d = daily[daily.index <= ts]
    if len(d) < 200:
        return {"above_sma200": None, "vol14d": None}
    sma200 = d.iloc[-200:].mean()
    last = d.iloc[-1]
    rets = d.pct_change().dropna().iloc[-14:]
    import numpy as np
    vol14 = float(np.std(rets) * np.sqrt(365)) if len(rets) > 1 else None
    return {"above_sma200": last > sma200, "vol14d": vol14}


def filter_regime_sma200(ev: pd.DataFrame, require_above: bool = True) -> pd.DataFrame:
    """BTC가 SMA200 위(require_above=True)일 때만 진입 허용."""
    if len(ev) == 0:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    keep = []
    for _, r in ev.iterrows():
        reg = btc_regime_at(r["entry_ts"])
        if reg["above_sma200"] is None:
            keep.append(True); continue
        keep.append(bool(reg["above_sma200"]) == bool(require_above))
    return ev[keep].reset_index(drop=True)


def filter_vol_regime(ev: pd.DataFrame, max_vol14d: float) -> pd.DataFrame:
    """BTC 14d vol(연환산)이 max_vol14d 이하일 때만 진입."""
    if len(ev) == 0 or max_vol14d is None:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    keep = []
    for _, r in ev.iterrows():
        reg = btc_regime_at(r["entry_ts"])
        if reg["vol14d"] is None:
            keep.append(True); continue
        keep.append(float(reg["vol14d"]) <= max_vol14d)
    return ev[keep].reset_index(drop=True)
