#!/usr/bin/env python3
"""Breakdown Short engine — Donchian breakdown 숏.

시그널:
  donch_low = rolling_min(Low, 48).shift(1)
  breakdown = Close < donch_low AND Close < SMA(Close, 168)
"""
from __future__ import annotations
import os
import pandas as pd

from engine_short_mom import load_funding  # re-use


def run_breakdown_short(df, funding=None,
                        donch_window=48, sma_regime=168,
                        tp=0.06, tstop=72, stop_above_ref=0.03,
                        tx=0.003, sell_at="open", cover_at="open", fill_delay=0):
    df = df.copy()
    df["donch_low"] = df["Low"].rolling(donch_window).min().shift(1)
    df["sma_r"] = df["Close"].rolling(sma_regime).mean()
    df["bd_sig"] = (
        (df["Close"] < df["donch_low"])
        & (df["Close"] < df["sma_r"])
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_donch_low"] = df["donch_low"].shift(1)

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
    funding_accum_pct = 0.0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        if position < 0:
            cover_px = row[cover_at.capitalize()]
            prev_close = row["prev_close"]
            prev_donch = row["prev_donch_low"]
            running_pnl = (1.0 - prev_close / entry_price) if pd.notna(prev_close) else 0.0
            exit_now = False
            reason = None
            if running_pnl >= tp:
                exit_now = True; reason = "TP"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(prev_close) and pd.notna(prev_donch) and prev_close > prev_donch * (1 + stop_above_ref):
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

        if position == 0 and row["bd_sig"]:
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
