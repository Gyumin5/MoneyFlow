#!/usr/bin/env python3
"""Pullback Trend Entry engine — 수정판.

Codex 지적 반영:
- Exit lookahead 제거: 전 bar data로 조건 판정 → 현 bar open 체결
- TX 비용 eq 반영 유지, pnl_net_pct 이벤트에 추가 기록 (validator가 이걸 쓰면 tx 반영됨)

시그널 (1h bar):
  EMA_fast = EMA(Close, ema_fast)
  EMA_slow = EMA(Close, ema_slow)
  trend_up = Close > EMA_slow AND EMA_fast > EMA_slow
  pullback_depth = (EMA_fast - Low_6) / EMA_fast
  reclaim = Close > EMA_fast
  dip_sig = trend_up AND pb_depth in [pb_min, pb_max) AND reclaim

Entry: signal 확정 (t-1 close 기준) → t open 매수
Exit: t-1 bar의 close/trail 조건 → t open 청산
"""
from __future__ import annotations
import pandas as pd


def run_pullback(df, ema_fast=50, ema_slow=200,
                 pullback_min=0.015, pullback_max=0.08,
                 tp=0.06, tstop=72, trail_drop=0.02,
                 tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["low_6"] = df["Low"].rolling(6).min()
    df["trend_up"] = (df["Close"] > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"])
    df["pb_depth"] = (df["ema_fast"] - df["low_6"]) / df["ema_fast"]
    df["reclaim"] = df["Close"] > df["ema_fast"]
    # signal at t-1 close → act at t. fill_delay 추가 지연 가능
    df["pb_sig"] = (df["trend_up"]
                    & (df["pb_depth"] >= pullback_min)
                    & (df["pb_depth"] < pullback_max)
                    & df["reclaim"]).shift(1 + fill_delay).fillna(False)

    # prev bar fields for exit decision (lookahead 제거)
    df["prev_close"] = df["Close"].shift(1)
    df["prev_ema_fast"] = df["ema_fast"].shift(1)

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        # Exit: prev bar 기준 판정, 현 bar open 체결
        if position > 0:
            sell_px = row[sell_at.capitalize()]
            prev_close = row["prev_close"]
            prev_ema = row["prev_ema_fast"]
            # running total pnl (prev bar close 기준)
            running_pnl = (prev_close / entry_price - 1.0) if pd.notna(prev_close) else 0.0
            exit_now = False
            reason = None
            if running_pnl >= tp:
                exit_now = True; reason = "TP"
            elif bars_held >= tstop:
                exit_now = True; reason = "timeout"
            elif pd.notna(prev_close) and pd.notna(prev_ema) and prev_close < prev_ema * (1 - trail_drop):
                exit_now = True; reason = "trail"
            if exit_now:
                # execute at bar t open
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

        if position == 0 and row["pb_sig"]:
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
