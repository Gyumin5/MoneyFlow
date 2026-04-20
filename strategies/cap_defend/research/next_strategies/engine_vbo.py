#!/usr/bin/env python3
"""Volatility Breakout engine — lookahead 제거판.

Exit 조건은 전 bar 기준으로 판정 → 현 bar open 체결.
trailing stop도 전 bar까지의 High 최대값 사용.
"""
from __future__ import annotations
import pandas as pd


def wilder_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def run_vbo(df, donch_window=48, atr_window=14, sma_regime=168,
            trail_atr_mult=2.0, tp=0.08, tstop=96, vol_filter_max=0.035,
            tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["atr"] = wilder_atr(df, atr_window)
    df["donch_high"] = df["High"].rolling(donch_window).max().shift(1)
    df["sma_r"] = df["Close"].rolling(sma_regime).mean()
    df["vol_ok"] = (df["atr"] / df["Close"]) < vol_filter_max
    df["regime_up"] = df["Close"] > df["sma_r"]
    df["brk_sig"] = (
        (df["Close"] >= df["donch_high"])
        & df["vol_ok"] & df["regime_up"]
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_high"] = df["High"].shift(1)

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    entry_atr = 0.0
    highest_prev = 0.0  # 전 bar까지의 High 최대값
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            sell_px = row[sell_at.capitalize()]
            prev_close = row["prev_close"]
            prev_high = row["prev_high"]
            # trailing stop: 전 bar까지의 최대 High
            if pd.notna(prev_high):
                highest_prev = max(highest_prev, float(prev_high))
            trail_stop = highest_prev - trail_atr_mult * entry_atr
            running_pnl = (prev_close / entry_price - 1.0) if pd.notna(prev_close) else 0.0
            exit_now = False
            reason = None
            if running_pnl >= tp:
                exit_now = True; reason = "TP"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(prev_close) and prev_close < trail_stop:
                exit_now = True; reason = "trail"
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
                highest_prev = 0.0

        if position == 0 and row["brk_sig"] and pd.notna(row["atr"]):
            buy_px = row[buy_at.capitalize()]
            entry_price = float(buy_px); entry_ts = ts; position = 1; bars_held = 0
            entry_atr = float(row["atr"])
            highest_prev = float(row["High"])
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
