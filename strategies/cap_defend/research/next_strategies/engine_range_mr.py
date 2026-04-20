#!/usr/bin/env python3
"""Range Mean Reversion engine — 횡보장 반등 롱 (C와 중복 여부 검증용).

시그널:
  bb_mid = SMA(Close, bb_n)
  bb_std = std(Close, bb_n)
  bb_low = bb_mid - 2*bb_std
  bb_width = (bb_up - bb_low) / bb_mid
  rsi = RSI(Close, 14)
  regime_range = bb_width < bb_width_max
  no_strong_trend = |Close/SMA(Close,168) - 1| < 0.05
  dip_sig = (Close <= bb_low) AND (rsi < rsi_thr) AND regime_range AND no_strong_trend

Exit:
  - TP: +tp
  - bb_mid touch: Close >= bb_mid (mean reversion 완료)
  - stop: -stop_loss
  - time stop: tstop
"""
from __future__ import annotations
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def run_range_mr(df, bb_n=20, rsi_thr=30, bb_width_max=0.06,
                  tp=0.04, stop_loss=0.03, tstop=48,
                  tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["bb_mid"] = df["Close"].rolling(bb_n).mean()
    df["bb_std"] = df["Close"].rolling(bb_n).std()
    df["bb_low"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_up"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_width"] = (df["bb_up"] - df["bb_low"]) / df["bb_mid"]
    df["rsi"] = rsi(df["Close"], 14)
    df["sma168"] = df["Close"].rolling(168).mean()
    df["trend_abs"] = (df["Close"] / df["sma168"] - 1).abs()
    df["range_sig"] = (
        (df["Close"] <= df["bb_low"])
        & (df["rsi"] < rsi_thr)
        & (df["bb_width"] < bb_width_max)
        & (df["trend_abs"] < 0.05)
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_bb_mid"] = df["bb_mid"].shift(1)

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            sell_px = row[sell_at.capitalize()]
            prev_close = row["prev_close"]
            prev_mid = row["prev_bb_mid"]
            running_pnl = (prev_close / entry_price - 1.0) if pd.notna(prev_close) else 0.0
            exit_now = False
            reason = None
            if running_pnl >= tp:
                exit_now = True; reason = "TP"
            elif running_pnl <= -stop_loss:
                exit_now = True; reason = "stop"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(prev_close) and pd.notna(prev_mid) and prev_close >= prev_mid:
                exit_now = True; reason = "mean"
            if exit_now:
                exit_bar_ret = sell_px / prev_close - 1.0 if pd.notna(prev_close) else 0.0
                eq *= (1 + exit_bar_ret - tx)
                total_pnl = sell_px / entry_price - 1.0
                events.append({
                    "entry_ts": entry_ts, "exit_ts": ts,
                    "entry_px": float(entry_price), "exit_px": float(sell_px),
                    "pnl_pct": round(float(total_pnl) * 100, 3),
                    "pnl_net_pct": round((float(total_pnl) - 2 * tx) * 100, 3),
                    "bars_held": int(bars_held),
                    "reason": reason,
                })
                position = 0; bars_held = 0; entry_ts = None

        if position == 0 and row["range_sig"]:
            buy_px = row[buy_at.capitalize()]
            entry_price = float(buy_px); entry_ts = ts; position = 1; bars_held = 0
            bar_ret = row["Close"] / entry_price - 1.0
            eq *= (1 + bar_ret - tx)
            bars_held += 1
        elif position > 0:
            prev_close = row["prev_close"]
            if pd.notna(prev_close):
                bar_ret = row["Close"] / prev_close - 1.0
                eq *= (1 + bar_ret)
            bars_held += 1

        equity.append(eq)

    return pd.Series(equity, index=df.index, name="equity"), events
