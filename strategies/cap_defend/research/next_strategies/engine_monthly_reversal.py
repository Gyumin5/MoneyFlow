#!/usr/bin/env python3
"""Monthly Reversal — 30일 저점 근접 롱 (daily bar 기반).

30d low에 근접(±2%) + 장기 상승 추세 (SMA 90일 위) → 매수
hold: tp 또는 30d mean touch or tstop
"""
from __future__ import annotations
import pandas as pd


def run_monthly_reversal(df, low_days=30, near_low_pct=0.03,
                          sma_days=90, tp=0.08, tstop=15, stop_loss=0.06,
                          tx=0.003, buy_at="open", sell_at="open", fill_delay=0):
    df = df.copy()
    df_d = df["Close"].resample("D").last()
    df_d_hi = df["High"].resample("D").max()
    df_d_lo = df["Low"].resample("D").min()
    df_d_op = df["Open"].resample("D").first()

    daily = pd.DataFrame({
        "Close": df_d, "Open": df_d_op, "High": df_d_hi, "Low": df_d_lo,
    }).dropna()
    daily["low_N"] = daily["Low"].rolling(low_days).min().shift(1)
    daily["sma_L"] = daily["Close"].rolling(sma_days).mean()
    # dip_dist: 저점 대비 현재가 위치 [-0.01, near_low_pct]
    dist = daily["Close"] / daily["low_N"] - 1
    daily["near"] = (dist >= -0.01) & (dist <= near_low_pct)
    daily["regime"] = daily["Close"] > daily["sma_L"]
    daily["sig"] = (daily["near"] & daily["regime"]).shift(1 + fill_delay).fillna(False)
    daily["prev_close"] = daily["Close"].shift(1)

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for ts, row in daily.iterrows():
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

    return pd.Series(equity, index=daily.index, name="equity"), events
