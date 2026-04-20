from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Lot:
    coin: str
    cap_rank: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    pnl_pct: float
    qty: float = 0.0
    last_px: float = 0.0
    entered: bool = False

    @property
    def value(self) -> float:
        return self.qty * self.last_px if self.entered else 0.0


def _calc_metrics(eq: pd.Series) -> dict[str, float]:
    rets = eq.pct_change().dropna()
    if rets.empty or eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return {"sharpe": 0.0, "cagr": 0.0, "mdd": 0.0, "final": float(eq.iloc[-1])}
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = days / 365.25
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    mdd = (eq / eq.cummax() - 1.0).min()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "final": float(eq.iloc[-1]),
    }


def simulate(
    events: pd.DataFrame,
    coin_daily_close: dict[str, pd.Series],
    v21_daily: pd.DataFrame,
    hist_universe: dict[pd.Timestamp, list[str]],
    n_pick: int = 1,
    cap_per_slot: float = 0.333,
    universe_size: int = 15,
    tx_cost: float = 0.003,
    swap_edge_threshold: int = 1,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Exact M3 lot engine.

    Unit convention:
    - `v21_daily["equity"]` is the standalone V21 portfolio equity, normalized to 1.0.
    - `c_delta` is only the incremental PnL of the C sleeve versus leaving that sleeve in V21 cash.
    - Final portfolio equity is always `v21_alone_equity + c_delta`.
    - Lot notionals / tx / MTM are handled in the same normalized currency unit as `equity`.
    """
    if n_pick < 1 or n_pick > 5:
        raise ValueError("n_pick must be in [1, 5]")
    need_cols = {
        "entry_ts",
        "exit_ts",
        "entry_px",
        "exit_px",
        "pnl_pct",
        "bars_held",
        "coin",
    }
    if not need_cols.issubset(events.columns):
        raise ValueError(f"events missing columns: {sorted(need_cols - set(events.columns))}")
    if not {"equity", "cash_ratio", "v21_ret", "prev_cash"}.issubset(v21_daily.columns):
        raise ValueError("v21_daily must contain equity, cash_ratio, v21_ret, prev_cash")

    idx = pd.DatetimeIndex(v21_daily.index).sort_values().normalize()
    v21 = v21_daily.copy()
    v21.index = idx
    v21 = v21.loc[~v21.index.duplicated(keep="last")].sort_index()

    hist_dates = sorted(pd.Timestamp(d).normalize() for d in hist_universe)
    hist_map = {pd.Timestamp(d).normalize(): coins for d, coins in hist_universe.items()}
    px_map = {c: s.sort_index() for c, s in coin_daily_close.items()}

    def cap_rank(date: pd.Timestamp, coin: str) -> int:
        pos = bisect_right(hist_dates, date.normalize()) - 1
        if pos < 0:
            return 10**9
        coins = hist_map[hist_dates[pos]]
        try:
            return coins.index(coin)
        except ValueError:
            return 10**9

    def close_px(coin: str, date: pd.Timestamp, fallback: float) -> float:
        ser = px_map.get(coin)
        if ser is None or ser.empty:
            return float(fallback)
        px = ser.asof(date)
        return float(px) if pd.notna(px) and px > 0 else float(fallback)

    ev = events.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_ts"]).dt.normalize()
    ev["exit_date"] = pd.to_datetime(ev["exit_ts"]).dt.normalize()
    ev["cap_rank"] = [cap_rank(d, c) for d, c in zip(ev["entry_date"], ev["coin"])]
    ev = ev[ev["cap_rank"] < universe_size].sort_values(["entry_ts", "cap_rank"]).reset_index(drop=True)
    by_day = {d: g.to_dict("records") for d, g in ev.groupby("entry_date", sort=False)}

    lots: list[Lot] = []
    c_delta = 0.0
    port_vals = []
    stats: dict[str, Any] = {
        "n_entries": 0,
        "n_natural_exits": 0,
        "n_swaps": 0,
        "n_shrinks": 0,
        "n_expands": 0,
        "n_forced_zero": 0,
        "total_tx_cost": 0.0,
        "max_cash_overshoot": 0.0,
        "max_slots": 0,
    }

    for date in v21.index:
        v21_eq = float(v21.at[date, "equity"])
        base_cash_ratio = float(v21.at[date, "prev_cash"])

        # 1) Open lots: MTM to today close, except natural exits which settle at event exit_px.
        still_open: list[Lot] = []
        for lot in lots:
            if lot.exit_date <= date:
                settle_px = float(lot.exit_px)
                c_delta += lot.qty * (settle_px - lot.last_px)
                tx = lot.qty * settle_px * tx_cost
                c_delta -= tx
                stats["total_tx_cost"] += tx
                stats["n_natural_exits"] += 1
                continue
            px = close_px(lot.coin, date, lot.last_px or lot.entry_px)
            c_delta += lot.qty * (px - lot.last_px)
            lot.last_px = px
            lot.cap_rank = cap_rank(date, lot.coin)
            still_open.append(lot)
        lots = still_open

        # 2) Candidate lineup: fill vacancies first, then swap only for sufficiently better rank.
        open_coins = {lot.coin for lot in lots}
        for row in by_day.get(date, []):
            coin = row["coin"]
            if coin in open_coins:
                continue
            new_lot = Lot(
                coin=coin,
                cap_rank=int(row["cap_rank"]),
                entry_date=row["entry_date"],
                exit_date=row["exit_date"],
                entry_px=float(row["entry_px"]),
                exit_px=float(row["exit_px"]),
                pnl_pct=float(row["pnl_pct"]),
            )
            if len(lots) < n_pick:
                lots.append(new_lot)
                open_coins.add(coin)
                continue
            worst = max(lots, key=lambda x: x.cap_rank)
            if worst.cap_rank - new_lot.cap_rank >= swap_edge_threshold:
                if worst.entered and worst.qty > 0:
                    tx = worst.value * tx_cost
                    c_delta -= tx
                    stats["total_tx_cost"] += tx
                lots.remove(worst)
                lots.append(new_lot)
                open_coins.discard(worst.coin)
                open_coins.add(coin)
                stats["n_swaps"] += 1

        # 3) Daily cap allocation. Budget uses previous-day V21 cash ratio, avoiding look-ahead.
        alloc_base_eq = v21_eq + c_delta
        slot_cap = max(cap_per_slot * alloc_base_eq, 0.0)
        cash_budget = max(base_cash_ratio * alloc_base_eq, 0.0)
        lots.sort(key=lambda x: (x.cap_rank, x.entry_date, x.coin))
        remaining = cash_budget
        rebalanced: list[Lot] = []

        for lot in lots:
            cur_px = lot.last_px if lot.entered else close_px(lot.coin, date, lot.entry_px)
            cur_val = lot.qty * cur_px if lot.entered else 0.0
            target = min(slot_cap, remaining)

            if lot.entered:
                if target < cur_val - 1e-12:
                    sell_val = cur_val - target
                    tx = sell_val * tx_cost
                    c_delta -= tx
                    stats["total_tx_cost"] += tx
                    stats["n_shrinks"] += 1
                    lot.qty = target / cur_px if target > 0 else 0.0
                    lot.last_px = cur_px
                elif target > cur_val + 1e-12:
                    buy_val = target - cur_val
                    tx = buy_val * tx_cost
                    c_delta -= tx
                    stats["total_tx_cost"] += tx
                    stats["n_expands"] += 1
                    lot.qty += buy_val / cur_px if cur_px > 0 else 0.0
                    lot.last_px = cur_px
            elif target > 0:
                qty = target / lot.entry_px
                tx = target * tx_cost
                mark_px = close_px(lot.coin, date, lot.entry_px)
                c_delta -= tx
                c_delta += qty * (mark_px - lot.entry_px)
                stats["total_tx_cost"] += tx
                stats["n_entries"] += 1
                lot.qty = qty
                lot.last_px = mark_px
                lot.entered = True

            if lot.entered and lot.qty > 0:
                rebalanced.append(lot)
            elif lot.entered and lot.qty <= 0:
                stats["n_forced_zero"] += 1

            remaining -= target
            if remaining < 0 and abs(remaining) > stats["max_cash_overshoot"]:
                stats["max_cash_overshoot"] = abs(remaining)

        lots = rebalanced
        stats["max_slots"] = max(stats["max_slots"], len(lots))
        port_vals.append(v21_eq + c_delta)

    port_equity = pd.Series(port_vals, index=v21.index, name="port_equity")
    port_equity = port_equity / float(port_equity.iloc[0])

    out = {
        **stats,
        **_calc_metrics(port_equity),
        "final_c_delta": float(c_delta),
        "final_active_lots": len(lots),
    }
    return port_equity, out
