"""K2 per-coin L3 희소성 진단.

질문: 왜 per-coin L 이 2↔4 점프하고 3 은 거의 안 찍히나?
가설: L3 밴드(close/SMA7 비율 1.025~1.075, 폭 5%p)가 SMA7 비율의
      하루 변동폭보다 좁아서 하루 만에 밴드를 건너뜀.

검증:
1) 전체 기간 per-coin L (final, BTC_cap min 적용) 분포 — L2/L3/L4 비율
2) per-coin K2 단독 (min 전) L 분포
3) close/SMA{7,14,21} 비율의 하루 변동폭(|Δratio|) 분포 vs L3 밴드 폭(0.05)
4) 2021-02 보유 코인의 일별 ratio + L 궤적
5) SMA 기간을 7→14→21 로 늘리면 L3 비율이 얼마나 늘어나나
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from unified_backtest import load_data

K2_HYST = 0.025
THR_MID = 1.0 + K2_HYST       # 1.025
THR_MAX = 1.0 + K2_HYST * 3   # 1.075
L_MIN, L_MID, L_MAX = 2.0, 3.0, 4.0


def classify(ratio):
    return np.where(ratio > THR_MAX, L_MAX, np.where(ratio > THR_MID, L_MID, L_MIN))


def per_coin_L(close, sma_period):
    sma = close.rolling(sma_period).mean()
    ratio = close / sma
    L = pd.Series(classify(ratio), index=close.index).shift(1)
    return ratio, L


def btc_cap_series(btc_close, sma_period=42, thr_mid=1.015, thr_max=1.05):
    sma = btc_close.rolling(sma_period).mean()
    r = btc_close / sma
    cap = pd.Series(np.where(r > thr_max, L_MAX, np.where(r > thr_mid, L_MID, L_MIN)),
                    index=r.index).shift(1).ffill().fillna(L_MIN)
    return cap


def dist(L):
    L = L.dropna()
    n = len(L)
    if n == 0: return (0, 0, 0)
    return (np.mean(L == L_MIN)*100, np.mean(L == L_MID)*100, np.mean(L == L_MAX)*100)


def main():
    bars, _ = load_data('D')
    EXCLUDE = {'BNB', 'SOL'}
    coins = [c for c in bars if c not in ('CASH',) and c not in EXCLUDE]
    btc_close = pd.Series(bars['BTC']['Close'].values, index=bars['BTC'].index)
    cap = btc_cap_series(btc_close)

    print("=== 1) per-coin K2 단독 L 분포 (전체기간, BNB/SOL 제외) ===")
    print(f"  L3 밴드: ratio ∈ ({THR_MID:.3f}, {THR_MAX:.3f}], 폭 {THR_MAX-THR_MID:.3f} (5.0%p)\n")
    print(f"  {'SMA':<5} {'L2%':>7} {'L3%':>7} {'L4%':>7}   {'med|Δratio|':>11} {'p75|Δ|':>8} {'p90|Δ|':>8}")
    for smap in [7, 10, 14, 21, 28]:
        l2s, l3s, l4s, draws = [], [], [], []
        for c in coins:
            close = bars[c]['Close']
            if len(close.dropna()) < smap + 30: continue
            ratio, L = per_coin_L(close, smap)
            d2, d3, d4 = dist(L)
            l2s.append(d2); l3s.append(d3); l4s.append(d4)
            draws.append(ratio.diff().abs().dropna())
        alld = pd.concat(draws)
        print(f"  {smap:<5} {np.mean(l2s):>6.1f}% {np.mean(l3s):>6.1f}% {np.mean(l4s):>6.1f}%   "
              f"{alld.median():>10.3f} {alld.quantile(.75):>8.3f} {alld.quantile(.90):>8.3f}")

    print("\n  → med|Δratio| 가 L3 밴드 폭(0.050)보다 크면 하루 만에 밴드 건너뜀 = L3 skip")

    print("\n=== 2) final L (min(BTC_cap, K2)) 분포, SMA7 (현행) ===")
    l2s, l3s, l4s = [], [], []
    for c in coins:
        close = bars[c]['Close']
        if len(close.dropna()) < 40: continue
        ratio, L = per_coin_L(close, 7)
        idx = L.index.intersection(cap.index)
        finalL = pd.Series(np.minimum(L.loc[idx].values, cap.loc[idx].values), index=idx)
        d2, d3, d4 = dist(finalL)
        l2s.append(d2); l3s.append(d3); l4s.append(d4)
    print(f"  L2 {np.mean(l2s):.1f}%  L3 {np.mean(l3s):.1f}%  L4 {np.mean(l4s):.1f}%")

    print("\n=== 3) BTC_cap 단독 분포 (SMA42, thr 1.015/1.05) ===")
    print(f"  L2 {dist(cap)[0]:.1f}%  L3 {dist(cap)[1]:.1f}%  L4 {dist(cap)[2]:.1f}%")
    print("  → BTC_cap L3 밴드: ratio 1.015~1.05 (폭 3.5%p)")

    print("\n=== 4) 2021-02 보유 코인 일별 ratio/L 궤적 (SMA7) ===")
    feb = pd.Timestamp('2021-02-01'); mar = pd.Timestamp('2021-03-01')
    for c in ['BTC', 'ETH', 'ADA', 'LINK', 'DOT']:
        if c not in bars: continue
        close = bars[c]['Close']
        ratio, L = per_coin_L(close, 7)
        idx = L.index.intersection(cap.index)
        finalL = pd.Series(np.minimum(L.loc[idx].values, cap.loc[idx].values), index=idx)
        seg_r = ratio[(ratio.index >= feb) & (ratio.index < mar)]
        seg_L = finalL[(finalL.index >= feb) & (finalL.index < mar)]
        if len(seg_r) == 0: continue
        rstr = " ".join(f"{v:.2f}" for v in seg_r.values[:14])
        lstr = " ".join(f"{int(v)}" for v in seg_L.values[:14])
        print(f"  {c:<5} ratio: {rstr}")
        print(f"  {'':5} L    : {lstr}")

    print("\n=== 5) SMA 늘릴 때 final L 분포 변화 (min BTC_cap 적용) ===")
    print(f"  {'SMA':<5} {'L2%':>7} {'L3%':>7} {'L4%':>7}")
    for smap in [7, 14, 21]:
        l2s, l3s, l4s = [], [], []
        for c in coins:
            close = bars[c]['Close']
            if len(close.dropna()) < smap + 30: continue
            ratio, L = per_coin_L(close, smap)
            idx = L.index.intersection(cap.index)
            finalL = pd.Series(np.minimum(L.loc[idx].values, cap.loc[idx].values), index=idx)
            d2, d3, d4 = dist(finalL)
            l2s.append(d2); l3s.append(d3); l4s.append(d4)
        print(f"  {smap:<5} {np.mean(l2s):>6.1f}% {np.mean(l3s):>6.1f}% {np.mean(l4s):>6.1f}%")


if __name__ == "__main__":
    main()
