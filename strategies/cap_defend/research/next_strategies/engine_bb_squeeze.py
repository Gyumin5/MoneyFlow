#!/usr/bin/env python3
"""Bollinger Band Squeeze Breakout — 저변동성 압축 후 돌파 롱.

BB width가 최근 N봉 최저 수준 → 변동성 압축
그 후 Close가 bb_upper 돌파 → 롱 진입
VBO(Donchian)와 구분: BB 정규화 기준 + squeeze 선행
"""
from __future__ import annotations
import pandas as pd


def run_bb_squeeze(df, bb_n=20, bb_std=2.0, squeeze_lookback=50,
                   tp=0.08, tstop=96, stop_loss=0.04,
                   tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    df["bb_mid"] = df["Close"].rolling(bb_n).mean()
    df["bb_std"] = df["Close"].rolling(bb_n).std()
    df["bb_up"] = df["bb_mid"] + bb_std * df["bb_std"]
    df["bb_lo"] = df["bb_mid"] - bb_std * df["bb_std"]
    df["bb_width"] = (df["bb_up"] - df["bb_lo"]) / df["bb_mid"]
    df["width_min"] = df["bb_width"].rolling(squeeze_lookback).min()
    df["is_squeezed"] = df["bb_width"] <= df["width_min"] * 1.2
    # squeeze 상태에서 상단 돌파 (prev was squeeze, now close > prev bb_up)
    df["breakout"] = (
        df["is_squeezed"].shift(1).fillna(False)
        & (df["Close"] > df["bb_up"].shift(1))
    ).shift(1 + fill_delay).fillna(False)
    df["prev_close"] = df["Close"].shift(1)

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

        if position == 0 and row["breakout"]:
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
