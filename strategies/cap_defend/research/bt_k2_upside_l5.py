"""K2 상방 L5 티어 추가 검증.

현행: >1.075 → L4 (상한 4). per-coin & BTC_cap 모두 ceiling L4.
제안: 강한 모멘텀에 L5 추가. 최종 L=5 되려면 per-coin AND BTC_cap 둘 다 5 허용 필요.

변형 (thr 스페이싱 = hyst 0.025 연장: L4=1.075=1+.025*3, L5=1+.025*5=1.125):
- baseline (ceiling L4)
- L5 thr 1.100 (per-coin) / BTC_cap 1.08
- L5 thr 1.125 / BTC_cap 1.10
- L5 thr 1.150 / BTC_cap 1.12

가설: 포물선 구간 5배로 상방 극대화. trade-off: 청산거리↓ → MDD/청산↑ (CROSS 모델 반영).
BNB+SOL 제외, baseline spec. look-ahead: shift(1).
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


def build_K2_L5(bars, pc_thr_l5=None, btc_thr_l5=None, l_top=5.0,
                btc_cap_sma_period=42, btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
                k2_sma_period=7, l_min=2.0, l_mid=3.0, l_max=4.0):
    btc_df = bars.get('BTC')
    if btc_df is None: return {}
    btc_close = pd.Series(btc_df['Close'].values, index=btc_df.index)
    btc_sma = btc_close.rolling(btc_cap_sma_period).mean()
    br = btc_close / btc_sma
    cap = np.where(br > btc_cap_thr_max, l_max, np.where(br > btc_cap_thr_mid, l_mid, l_min))
    if btc_thr_l5 is not None:
        cap = np.where(br > btc_thr_l5, l_top, cap)
    btc_cap = pd.Series(cap, index=br.index).shift(1).ffill().fillna(l_min)
    thr_max = 1.0 + K2_HYST * 3
    thr_mid = 1.0 + K2_HYST
    out = {}
    for coin in bars:
        close = bars[coin]['Close']
        sma = close.rolling(k2_sma_period).mean()
        ratio = close / sma
        base = np.where(ratio > thr_max, l_max, np.where(ratio > thr_mid, l_mid, l_min))
        if pc_thr_l5 is not None:
            base = np.where(ratio > pc_thr_l5, l_top, base)
        pc = pd.Series(base, index=close.index).shift(1).ffill().fillna(l_min)
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
        ('baseline (ceiling L4)',  None,  None),
        ('L5 pc1.100 / btc1.08',  1.100, 1.08),
        ('L5 pc1.125 / btc1.10',  1.125, 1.10),
        ('L5 pc1.150 / btc1.12',  1.150, 1.12),
    ]
    print(f"{'variant':<24} {'fut_CAGR':>9} {'fut_MDD':>8} {'fut_Cal':>8} | {'al_CAGR':>8} {'al_MDD':>8} {'al_Sh':>6} {'al_Cal':>7}")
    for tag, pc5, btc5 in variants:
        k2 = build_K2_L5(bars_full, pc_thr_l5=pc5, btc_thr_l5=btc5)
        eq_fu = run_fut(k2)
        if eq_fu is None:
            print(f"{tag:<24} FAILED"); continue
        m_fu = metrics(eq_fu)
        alloc = build_alloc(eq_st, eq_sp, eq_fu)
        m_al = metrics(alloc)
        print(f"{tag:<24} {m_fu[0]:>8.1f}% {m_fu[1]:>+7.1f}% {m_fu[3]:>8.2f} | "
              f"{m_al[0]:>7.1f}% {m_al[1]:>+7.1f}% {m_al[2]:>6.2f} {m_al[3]:>7.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
