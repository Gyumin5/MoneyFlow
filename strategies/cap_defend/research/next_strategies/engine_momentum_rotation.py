#!/usr/bin/env python3
"""Momentum Rotation — 최근 상승률 상위 코인 롱 보유 (single-coin perspective).

단일 코인 관점에서는 "최근 momentum 양수" 지속 시 hold.
각 코인별 event: momentum > thr 구간 진입, 꺾이면 exit.
cross-sectional rotation은 aggregate 레벨 (top N 선정).
"""
from __future__ import annotations
import pandas as pd


def run_momentum_rotation(df, mom_lookback=168, mom_thr=0.05,
                           exit_mom_thr=0.0,
                           tp=0.15, tstop=168, stop_loss=0.05,
                           tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["mom"] = df["Close"] / df["Close"].shift(mom_lookback) - 1.0
    df["sma_r"] = df["Close"].rolling(240).mean()
    df["sig"] = (
        (df["mom"] > mom_thr) & (df["Close"] > df["sma_r"])
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_mom"] = df["mom"].shift(1)

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
            elif pd.notna(row["prev_mom"]) and row["prev_mom"] < exit_mom_thr:
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

    return pd.Series(equity, index=df.index, name="equity"), events
