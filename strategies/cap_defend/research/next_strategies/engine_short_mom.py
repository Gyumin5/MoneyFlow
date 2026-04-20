#!/usr/bin/env python3
"""Short Momentum engine — lookahead 제거 + funding 이벤트에 기록.

시그널:
  ema_fast = EMA(Close, 50)
  ema_slow = EMA(Close, 200)
  regime_down = Close < ema_slow AND ema_fast < ema_slow
  swing_low = rolling_min(Low, 72).shift(1)
  short_sig = regime_down AND (Close < swing_low) AND (Close/ema_fast - 1 < -0.01)

Short entry: t-1 close 시그널 → t open 매도.
Exit: t-1 close/ema_fast 기준 → t open cover.
Funding: 포지션 유지 시간 동안 누적. 숏은 funding 양수일 때 수익.
이벤트에 funding_pnl 기록.
"""
from __future__ import annotations
import os
import pandas as pd


def run_short_mom(df, funding=None,
                   ema_fast=50, ema_slow=200, swing_window=72,
                   tp=0.06, tstop=72, stop_above_ema=0.02,
                   tx=0.003, sell_at="open", cover_at="open", fill_delay=0):
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["swing_low"] = df["Low"].rolling(swing_window).min().shift(1)
    df["regime_dn"] = (df["Close"] < df["ema_slow"]) & (df["ema_fast"] < df["ema_slow"])
    df["short_sig"] = (
        df["regime_dn"]
        & (df["Close"] < df["swing_low"])
        & ((df["Close"] / df["ema_fast"] - 1) < -0.01)
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_ema_fast"] = df["ema_fast"].shift(1)

    if funding is not None:
        funding = funding.reindex(df.index).fillna(0.0)
    else:
        funding = pd.Series(0.0, index=df.index)

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    funding_accum_pct = 0.0  # 이벤트별 funding 누적 (log-return 근사로 %)
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        if position < 0:
            cover_px = row[cover_at.capitalize()]
            prev_close = row["prev_close"]
            prev_ema = row["prev_ema_fast"]
            running_pnl = (1.0 - prev_close / entry_price) if pd.notna(prev_close) else 0.0
            exit_now = False
            reason = None
            if running_pnl >= tp:
                exit_now = True; reason = "TP"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(prev_close) and pd.notna(prev_ema) and prev_close > prev_ema * (1 + stop_above_ema):
                exit_now = True; reason = "stop"
            if exit_now:
                exit_bar_ret = (1.0 - cover_px / prev_close) if pd.notna(prev_close) else 0.0
                eq *= (1 + exit_bar_ret - tx)
                total_pnl = 1.0 - cover_px / entry_price
                events.append({
                    "entry_ts": entry_ts, "exit_ts": ts,
                    "entry_px": float(entry_price), "exit_px": float(cover_px),
                    "pnl_pct": round(float(total_pnl) * 100, 3),
                    "funding_pnl_pct": round(float(funding_accum_pct) * 100, 3),
                    "pnl_net_pct": round((float(total_pnl) + float(funding_accum_pct) - 2 * tx) * 100, 3),
                    "bars_held": int(bars_held),
                    "reason": reason,
                })
                position = 0; bars_held = 0; entry_ts = None
                funding_accum_pct = 0.0

        if position == 0 and row["short_sig"]:
            sell_px = row[sell_at.capitalize()]
            entry_price = float(sell_px); entry_ts = ts; position = -1; bars_held = 0
            funding_accum_pct = 0.0
            bar_ret = 1.0 - row["Close"] / entry_price
            eq *= (1 + bar_ret - tx)
            f = float(funding.iloc[i])
            eq *= (1 + f)
            funding_accum_pct += f
            bars_held += 1
        elif position < 0:
            prev_close = row["prev_close"]
            if pd.notna(prev_close):
                bar_ret = 1.0 - row["Close"] / prev_close
                eq *= (1 + bar_ret)
            f = float(funding.iloc[i])
            eq *= (1 + f)
            funding_accum_pct += f
            bars_held += 1

        equity.append(eq)

    return pd.Series(equity, index=df.index, name="equity"), events


def load_funding(coin: str) -> pd.Series | None:
    """Binance funding rate CSV. 8h funding을 단일 시점에 적용."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "futures"))
    path = os.path.join(root, f"{coin}USDT_funding.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    tcol = "fundingTime" if "fundingTime" in df.columns else df.columns[0]
    rcol = "fundingRate" if "fundingRate" in df.columns else df.columns[1]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.set_index(tcol)
    s = df[rcol].astype(float)
    s.name = "funding"
    return s
