#!/usr/bin/env python3
"""Funding Rate Skew — 극단 funding 역매매 롱.

Funding rate이 극단적으로 음수 → 숏이 롱에게 지불 → 롱 편향 (반등 기대)
또는 음수 funding 지속 → 숏 과포지션 → 쏠림 해소 롱
"""
from __future__ import annotations
import os
import pandas as pd


def load_funding(coin: str) -> pd.Series | None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "futures"))
    path = os.path.join(root, f"{coin}USDT_funding.csv")
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    tcol = "fundingTime" if "fundingTime" in df.columns else df.columns[0]
    rcol = "fundingRate" if "fundingRate" in df.columns else df.columns[1]
    df[tcol] = pd.to_datetime(df[tcol])
    return df.set_index(tcol)[rcol].astype(float)


def run_funding_skew(df, funding=None,
                      fund_thr=-0.001, fund_lookback=3,  # 최근 3 funding 중 2 이상 극단 음수
                      tp=0.04, tstop=24, stop_loss=0.03,
                      tx=0.003, buy_at="high", sell_at="open", fill_delay=0):
    df = df.copy()
    if funding is not None:
        # 8h funding을 1h 인덱스로 ffill (3 bars 동안 동일 rate)
        funding_1h = funding.reindex(df.index, method="ffill").fillna(0.0)
        df["f"] = funding_1h
        # rolling 후 sum (Series < scalar → bool mask → rolling.sum)
        df["f_neg_count"] = (df["f"] < fund_thr).rolling(fund_lookback * 8).sum()
        # sig: 최근 fund_lookback(default 3) funding 중 2회 이상 극단 음수 + 현 bar도 음수
        df["sig"] = (
            (df["f"] < fund_thr) & (df["f_neg_count"] >= 2)
        ).shift(1 + fill_delay).fillna(False)
    else:
        return pd.Series([10000.0] * len(df), index=df.index, name="equity"), []
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
