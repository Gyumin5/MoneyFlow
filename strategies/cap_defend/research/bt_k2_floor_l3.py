"""K2 하방 floor 를 L3 로 올리는 변형 검증.

현행 per-coin: >1.075 → L4, >1.025 → L3, else L2 (floor 2)
제안: floor 를 L3 로 → 눌림목에서도 2 로 안 떨어지고 3 유지

변형:
- baseline (per-coin floor 2)
- per-coin floor 3, BTC_cap floor 2 (BTC bear 면 여전히 cap=2 로 눌림)
- per-coin floor 3, BTC_cap floor 3 (전역 최소 L3)

가설: 모멘텀 코인 눌림목은 반등 → L3 유지가 상방 캡처. trade-off: MDD/청산↑
BNB+SOL 제외, baseline spec. look-ahead: shift(1) 유지.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from unified_backtest import load_data
from bt_k2_l1_downside import run_fut, run_spot, run_stock, build_alloc, metrics

K2_HYST = 0.025


def build_K2(bars, pc_floor=2.0, cap_floor=2.0,
             btc_cap_sma_period=42, btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
             k2_sma_period=7, l_mid=3.0, l_max=4.0):
    btc_df = bars.get('BTC')
    if btc_df is None: return {}
    btc_close = pd.Series(btc_df['Close'].values, index=btc_df.index)
    btc_sma = btc_close.rolling(btc_cap_sma_period).mean()
    btc_ratio = btc_close / btc_sma
    btc_cap = pd.Series(np.where(btc_ratio > btc_cap_thr_max, l_max,
                        np.where(btc_ratio > btc_cap_thr_mid, l_mid, cap_floor)),
                        index=btc_ratio.index).shift(1).ffill().fillna(cap_floor)
    thr_max = 1.0 + K2_HYST * 3
    thr_mid = 1.0 + K2_HYST
    out = {}
    for coin in bars:
        close = bars[coin]['Close']
        sma = close.rolling(k2_sma_period).mean()
        ratio = close / sma
        base = np.where(ratio > thr_max, l_max, np.where(ratio > thr_mid, l_mid, pc_floor))
        pc = pd.Series(base, index=close.index).shift(1).ffill().fillna(pc_floor)
        idx = pc.index.intersection(btc_cap.index)
        out[coin] = pd.Series(np.minimum(pc.loc[idx].values, btc_cap.loc[idx].values), index=idx)
    return out


def main():
    t0 = time.time()
    bars_full, _ = load_data('D')
    eq_st = run_stock()
    eq_sp = run_spot()
    print(f"[stock] CAGR {metrics(eq_st)[0]:.1f}% | [spot] CAGR {metrics(eq_sp)[0]:.1f}%\n")

    variants = [
        ('baseline (floor L2)',        2.0, 2.0),
        ('per-coin floor L3, cap L2',  3.0, 2.0),
        ('전역 floor L3 (cap도 L3)',    3.0, 3.0),
    ]
    print(f"{'variant':<26} {'fut_CAGR':>9} {'fut_MDD':>8} {'fut_Cal':>8} | {'al_CAGR':>8} {'al_MDD':>8} {'al_Sh':>6} {'al_Cal':>7}")
    for tag, pcf, capf in variants:
        k2 = build_K2(bars_full, pc_floor=pcf, cap_floor=capf)
        eq_fu = run_fut(k2)
        if eq_fu is None:
            print(f"{tag:<26} FAILED"); continue
        m_fu = metrics(eq_fu)
        alloc = build_alloc(eq_st, eq_sp, eq_fu)
        m_al = metrics(alloc)
        print(f"{tag:<26} {m_fu[0]:>8.1f}% {m_fu[1]:>+7.1f}% {m_fu[3]:>8.2f} | "
              f"{m_al[0]:>7.1f}% {m_al[1]:>+7.1f}% {m_al[2]:>6.2f} {m_al[3]:>7.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
