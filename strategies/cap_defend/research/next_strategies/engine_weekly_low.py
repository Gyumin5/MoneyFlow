#!/usr/bin/env python3
"""Weekly Low Entry — 주간 저점 매수 롱.

최근 168h(1주일) 최저가 근접 → 롱 진입
단순, 저빈도, 극단 dip 전용
"""
from __future__ import annotations
import pandas as pd


def run_weekly_low(df, week_hours=168, near_low_pct=0.02,
                    sma_regime=720,  # 1개월 이평
                    tp=0.05, tstop=96, stop_loss=0.04,
                    tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["week_low"] = df["Low"].rolling(week_hours).min().shift(1)
    df["sma_r"] = df["Close"].rolling(sma_regime).mean()
    # 저점 근접 = [-0.01, 0.02] 범위 (떨어지는 칼날 방지)
    dist = df["Close"] / df["week_low"] - 1
    df["near_low"] = (dist >= -0.01) & (dist <= near_low_pct)
    df["regime_up"] = df["Close"] > df["sma_r"]  # 장기 상승 필터
    df["sig"] = (df["near_low"] & df["regime_up"]).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)

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
