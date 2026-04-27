#!/usr/bin/env python3
"""Weekly Momentum Rotation — 주간 수익률 상위 롱 (daily bar 기반).

1h bar 데이터를 daily로 resample 후 7일 rolling return 상위 코인 매매.
단일 코인 engine은 self-relative (절대 기준):
  7d_return > thr AND Close > SMA(30d) → long
  hold: rebal_days 후 자동 청산 or 7d_return < exit_thr

V21 (4h trend)과 다른 timeframe, C (1h dip)와 다른 방향.
"""
from __future__ import annotations
import pandas as pd


def run_weekly_rotation(df, mom_days=7, mom_thr=0.10, exit_thr=0.02,
                         sma_days=30, hold_days=5, stop_loss=0.06,
                         tx=0.003, buy_at="open", sell_at="open", fill_delay=0):
    """df: 1h bar. daily resample 후 signal."""
    df = df.copy()
    # daily close
    df_d = df["Close"].resample("D").last()
    df_d_hi = df["High"].resample("D").max()
    df_d_lo = df["Low"].resample("D").min()
    df_d_op = df["Open"].resample("D").first()

    daily = pd.DataFrame({
        "Close": df_d, "Open": df_d_op, "High": df_d_hi, "Low": df_d_lo,
    }).dropna()
    daily["mom"] = daily["Close"] / daily["Close"].shift(mom_days) - 1
    daily["sma"] = daily["Close"].rolling(sma_days).mean()
    daily["sig"] = (
        (daily["mom"] > mom_thr) & (daily["Close"] > daily["sma"])
    ).shift(1 + fill_delay).fillna(False)
    daily["prev_close"] = daily["Close"].shift(1)
    daily["prev_mom"] = daily["mom"].shift(1)

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
            if bars_held >= hold_days:
                exit_now = True; reason = "hold"
            elif running <= -stop_loss:
                exit_now = True; reason = "stop"
            elif pd.notna(row["prev_mom"]) and row["prev_mom"] < exit_thr:
                exit_now = True; reason = "mom_dn"
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
