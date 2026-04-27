#!/usr/bin/env python3
"""RSI Reversal — 과매도 반등 롱 (C dip-buy와 다른 축).

RSI(close, period) < rsi_low → 진입
Exit: RSI > rsi_exit 또는 TP/tstop
C는 가격 dip 기준, 이건 oscillator 기반 → 다른 시그널
"""
from __future__ import annotations
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def run_rsi_reversal(df, rsi_period=14, rsi_low=25, rsi_exit=55,
                      tp=0.06, tstop=48, stop_loss=0.05,
                      tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["rsi"] = rsi(df["Close"], rsi_period)
    df["sma_long"] = df["Close"].rolling(240).mean()  # regime filter
    df["sig"] = (
        (df["rsi"] < rsi_low) & (df["Close"] > df["sma_long"])
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_rsi"] = df["rsi"].shift(1)

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
            prev_rsi_v = row["prev_rsi"]
            running = (prev_close / entry_price - 1.0) if pd.notna(prev_close) else 0.0
            exit_now = False; reason = None
            if running >= tp:
                exit_now = True; reason = "TP"
            elif running <= -stop_loss:
                exit_now = True; reason = "stop"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(prev_rsi_v) and prev_rsi_v > rsi_exit:
                exit_now = True; reason = "rsi_exit"
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

        if position == 0 and row["sig"]:
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
