#!/usr/bin/env python3
"""MACD Bullish Crossover — 모멘텀 전환 롱.

MACD(12,26,9) 시그널선 돌파 + regime filter.
Pullback/VBO와 다른 알파: oscillator 전환 기반.
"""
from __future__ import annotations
import pandas as pd


def run_macd_cross(df, fast=12, slow=26, signal=9, sma_regime=240,
                    tp=0.06, tstop=72, stop_loss=0.04,
                    tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    ema_f = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["sma_r"] = df["Close"].rolling(sma_regime).mean()
    df["cross_up"] = (
        (df["macd"] > df["signal"])
        & (df["macd"].shift(1) <= df["signal"].shift(1))
        & (df["Close"] > df["sma_r"])
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_macd"] = df["macd"].shift(1)
    df["prev_sig"] = df["signal"].shift(1)

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for _, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            sell_px = row[sell_at.capitalize()]
            prev_close = row["prev_close"]
            running = (prev_close / entry_price - 1.0) if pd.notna(prev_close) else 0.0
            exit_now = False; reason = None
            if running >= tp:
                exit_now = True; reason = "TP"
            elif running <= -stop_loss:
                exit_now = True; reason = "stop"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(row["prev_macd"]) and pd.notna(row["prev_sig"]) and row["prev_macd"] < row["prev_sig"]:
                exit_now = True; reason = "cross_dn"
            if exit_now:
                bar_ret = sell_px / prev_close - 1.0 if pd.notna(prev_close) else 0.0
                eq *= (1 + bar_ret - tx)
                total = sell_px / entry_price - 1.0
                events.append({
                    "entry_ts": entry_ts, "exit_ts": ts,
                    "entry_px": float(entry_price), "exit_px": float(sell_px),
                    "pnl_pct": round(float(total) * 100, 3),
                    "pnl_net_pct": round((float(total) - 2 * tx) * 100, 3),
                    "bars_held": int(bars_held), "reason": reason,
                })
                position = 0; bars_held = 0; entry_ts = None

        if position == 0 and row["cross_up"]:
            buy_px = row[buy_at.capitalize()]
            entry_price = float(buy_px); entry_ts = ts; position = 1; bars_held = 0
            eq *= (1 + row["Close"] / entry_price - 1.0 - tx)
            bars_held += 1
        elif position > 0:
            prev_close = row["prev_close"]
            if pd.notna(prev_close):
                eq *= (1 + row["Close"] / prev_close - 1.0)
            bars_held += 1

        equity.append(eq)

    return pd.Series(equity, index=df.index, name="equity"), events
