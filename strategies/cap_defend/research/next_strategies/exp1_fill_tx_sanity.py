#!/usr/bin/env python3
"""실험 1: 체결 하네스 sanity check.

모든 신규 엔진에 대해 BTC 단독 기준:
- buy_at: open / high / midpoint((open+high)/2)
- tx: 0.0004 / 0.0010 / 0.003

목적: gross edge 존재 여부 1차 확인. buy_at 차이만으로 부호 뒤집히면 즉시 폐기.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_engine_v5 import load_coin
from engine_pullback import run_pullback
from engine_vbo import run_vbo
from engine_range_mr import run_range_mr
from engine_rsi_reversal import run_rsi_reversal
from engine_bb_squeeze import run_bb_squeeze
from engine_macd_cross import run_macd_cross
from engine_weekly_low import run_weekly_low
from engine_momentum_rotation import run_momentum_rotation

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def summarize(eq, evs):
    if len(evs) == 0:
        return {"n": 0, "final": eq.iloc[-1] if len(eq) else 10000,
                "win_rate": 0, "gross_exp": 0}
    pnls = [e["pnl_pct"] for e in evs]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": len(evs),
        "final": round(float(eq.iloc[-1]), 1),
        "win_rate": round(wins / len(evs), 2),
        "gross_exp": round(sum(pnls) / len(evs), 2),  # gross per trade (%)
    }


# midpoint entry helper
def run_with_midpoint(engine_fn, df, **kwargs):
    """engine이 buy_at=midpoint 지원 안 하면 df 수정해서 근사."""
    df2 = df.copy()
    # midpoint를 새 High로 대체 (buy_at='high' 유지 but 값은 midpoint)
    df2["High_orig"] = df2["High"]
    df2["High"] = (df2["Open"] + df2["High"]) / 2
    kwargs.pop("buy_at", None)
    return engine_fn(df2, buy_at="high", **kwargs)


def main():
    df = load_coin("BTCUSDT")
    print(f"BTC data: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")

    engines = [
        ("pullback",  run_pullback,           {}),
        ("vbo",       run_vbo,                {}),
        ("range_mr",  run_range_mr,           {}),
        ("rsi_rev",   run_rsi_reversal,       {}),
        ("bb_squeeze",run_bb_squeeze,         {}),
        ("macd",      run_macd_cross,         {}),
        ("weekly_low",run_weekly_low,         {}),
        ("mom_rot",   run_momentum_rotation,  {}),
    ]

    rows = []
    for name, fn, extra in engines:
        for buy_at in ["open", "high", "midpoint"]:
            for tx in [0.0004, 0.001, 0.003]:
                kwargs = dict(extra, tx=tx)
                try:
                    if buy_at == "midpoint":
                        eq, evs = run_with_midpoint(fn, df, **kwargs)
                    else:
                        eq, evs = fn(df, buy_at=buy_at, **kwargs)
                except Exception as e:
                    rows.append({"engine": name, "buy_at": buy_at, "tx": tx, "err": str(e)[:50]})
                    continue
                s = summarize(eq, evs)
                rows.append({"engine": name, "buy_at": buy_at, "tx": tx, **s})
                print(f"{name:12s} buy={buy_at:10s} tx={tx*100:.2f}% "
                      f"n={s['n']:4d} final={s['final']:8.1f} "
                      f"win={s['win_rate']:.0%} gross/tr={s['gross_exp']:+.2f}%")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "exp1_fill_tx_sanity.csv"), index=False)

    # Summary table
    print("\n=== Final equity summary (BTC from 10000) ===")
    pv = df_out.pivot_table(index="engine", columns=["buy_at", "tx"],
                             values="final", aggfunc="first")
    print(pv.to_string())
    print(f"\n저장: {OUT}/exp1_fill_tx_sanity.csv")


if __name__ == "__main__":
    main()
