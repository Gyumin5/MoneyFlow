"""V25 라이브 trigger 검증 BT — cycle 7 ai-debate action_steps 적용.

목적:
1. V24 ISO vs V24 CROSS L3 분리 (라이브 V24 = ISO 였음)
2. T1/T3U grid 확장 — T1=10~25 × T3U=15~30 plateau 확인
3. stock canary 변형 (always-ON vs hyst SMA300)
4. cost stress (tx 1.5x)
"""
from __future__ import annotations
import sys, os, importlib.util
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
spec = importlib.util.spec_from_file_location("bt_cross", "/tmp/bt_fut_cross.py")
assert spec and spec.loader
bt_cross = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt_cross)
os.environ['DRIFT_HEALTH_MODE'] = 'refill'


def metrics(s):
    s = s.dropna()
    if len(s) < 30: return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    peak = s.cummax(); mdd = float((s / peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    sh = (s.pct_change().mean() / s.pct_change().std() * np.sqrt(365)) if s.pct_change().std() > 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, Sharpe=sh)


# ─── fut sleeve 시그널 ───
bars0, funding0 = bt_cross.load_data('D')
btc_close = pd.Series(bars0['BTC']['Close'].values, index=bars0['BTC'].index)
btc_sma42 = btc_close.rolling(42).mean()
btc_ratio = btc_close / btc_sma42
btc_cap_signal = pd.Series(np.where(btc_ratio > 1.05, 4.0,
                          np.where(btc_ratio > 1.015, 3.0, 2.0)),
                          index=btc_ratio.index).shift(1).ffill().fillna(2.0)


def signal_K2(period=7, h=0.025):
    out = {}
    for c in bars0:
        close = bars0[c]['Close']
        sma = close.rolling(period).mean()
        ratio = close / sma
        sig = pd.Series(np.where(ratio > (1 + h * 3), 4.0,
                        np.where(ratio > (1 + h), 3.0, 2.0)), index=close.index)
        sig = sig.shift(1).ffill().fillna(2.0)
        idx = sig.index.intersection(btc_cap_signal.index)
        out[c] = pd.Series(np.minimum(sig.loc[idx].values, btc_cap_signal.loc[idx].values), index=idx)
    return out


def run_fut_bt(leverage, tx=0.0006, maint=0.004):
    return bt_cross.run(bars0, funding0, interval='D', leverage=leverage,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=tx, maint_rate=maint, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date='2020-10-01', end_date='2026-05-13')['_equity']


RES_DIR = '/home/gmoh/mon/251229/strategies/cap_defend/research/alloc_reopt_2026_05'
print("=== sleeve equity 준비 ===")
stock_eq = pd.read_csv(f'{RES_DIR}/stock_equity.csv', index_col='Date', parse_dates=True)['Value']
spot_eq = pd.read_csv(f'{RES_DIR}/spot_equity.csv', index_col='Date', parse_dates=True)['Value']

print("fut V24 L3 (CROSS L=3 scalar) BT ...")
fut_v24_cross = run_fut_bt(3.0)
print(f"  V24 L3 CROSS: Cal={metrics(fut_v24_cross)['Cal']:.2f}")

print("fut V25 K2 (CROSS, K2 sig) BT ...")
fut_v25 = run_fut_bt(signal_K2(7, 0.025))
print(f"  V25 K2 CROSS: Cal={metrics(fut_v25)['Cal']:.2f}")

start = max(stock_eq.index[0], spot_eq.index[0], fut_v24_cross.index[0])
end = min(stock_eq.index[-1], spot_eq.index[-1], fut_v24_cross.index[-1])

def align(s):
    return s[(s.index >= start) & (s.index <= end)].reindex(pd.date_range(start, end, freq='D')).ffill().dropna()

stock_a = align(stock_eq)
spot_a = align(spot_eq)
fut_v24_a = align(fut_v24_cross)
fut_v25_a = align(fut_v25)


# ─── alloc BT (cycle 7 보강) ───
BUF_STOCK, BUF_SPOT, BUF_FUT = 0.07, 0.01, 0.01


def alloc_bt(stock, spot, fut, w_st=0.60, w_sp=0.25, w_ft=0.15,
             t1=None, t3u=None, stock_canary_mode='always_on'):
    """stock_canary_mode: 'always_on' (단순화) / 'sma300' (V24 spec hysteresis 1.5%)."""
    r_st = stock.pct_change().fillna(0) * (1 - BUF_STOCK)
    r_sp = spot.pct_change().fillna(0) * (1 - BUF_SPOT)
    r_ft = fut.pct_change().fillna(0) * (1 - BUF_FUT)

    target = np.array([w_st, w_sp, w_ft])
    val = target.copy()

    btc_sub = btc_close.reindex(stock.index).ffill()
    btc_sma_sub = btc_sub.rolling(42).mean()
    btc_canary = (btc_sub > btc_sma_sub * 1.015).shift(1).fillna(False)

    if stock_canary_mode == 'sma300':
        # V24 stock canary (EEM Risk-On) 근사 — BTC SMA300 hyst 2% 사용 (proxy: stock_eq 자체 회복 신호)
        # stock_eq drawdown < 5% 면 OFF (방어) 가정
        stock_dd = stock / stock.cummax() - 1
        stock_canary_series = (stock_dd >= -0.05).shift(1).fillna(True)
    else:
        stock_canary_series = pd.Series(True, index=stock.index)

    eq_list = [val.sum()]
    fire_log = []
    for i in range(1, len(stock.index)):
        date = stock.index[i]
        val[0] *= (1 + r_st.iloc[i])
        val[1] *= (1 + r_sp.iloc[i])
        val[2] *= (1 + r_ft.iloc[i])
        total = val.sum()
        cur_w = val / total if total > 0 else target.copy()

        if t1 is not None and t3u is not None:
            ht = abs(cur_w - target).sum() / 2
            canary = [bool(stock_canary_series.iloc[i]),
                      bool(btc_canary.iloc[i]), bool(btc_canary.iloc[i])]
            rel_under = max(((target[j] - cur_w[j]) / target[j]) if (canary[j] and target[j] > 0) else 0
                            for j in range(3))
            fire = (ht >= t1) or (rel_under >= t3u)
            if fire:
                val = total * target
                fire_log.append((date, 'T1' if ht >= t1 else 'T3U'))
        else:
            val = total * target
        eq_list.append(val.sum())
    eq_s = pd.Series(eq_list, index=stock.index)
    eq_s.fire_log = fire_log
    return eq_s


print("\n" + "=" * 90)
print("[A] V24 ISO vs CROSS 분리 (단독 sleeve)")
print("=" * 90)
print("주의: bt_cross 엔진은 CROSS 청산 모델만 지원. ISO 정확 비교는 backtest_futures_full.py 별도.")
print(f"V24 L3 CROSS 단독: CAGR {metrics(fut_v24_cross)['CAGR']*100:.1f}%, MDD {metrics(fut_v24_cross)['MDD']*100:.1f}%, Cal {metrics(fut_v24_cross)['Cal']:.2f}")
# ISO L3 실측 — backtest_futures_full.py 결과 (CLAUDE.md 메모리): CAGR 256%, MDD -63%, Cal 4.05
print("V24 L3 ISO  단독 (참고, backtest_futures_full): CAGR 256%, MDD -63%, Cal 4.05 (CLAUDE.md)")


print("\n" + "=" * 90)
print("[B] T1/T3U grid 확장 — V25 K2 + alloc 60/25/15 + buffer 7/1/1")
print("=" * 90)
label = 'T1/T3U'
print(f"{label:<8}", end='')
t3u_grid = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
t1_grid = [0.10, 0.13, 0.15, 0.18, 0.20, 0.23, 0.25]
for t3u in t3u_grid:
    print(f" {int(t3u*100):>5}%", end='')
print()
grid_results = {}
for t1 in t1_grid:
    print(f"{int(t1*100):>3}%    ", end='')
    for t3u in t3u_grid:
        eq = alloc_bt(stock_a, spot_a, fut_v25_a, t1=t1, t3u=t3u)
        m = metrics(eq); grid_results[(t1, t3u)] = m
        print(f" {m['Cal']:>5.2f}", end='')
    print()

# 최고 Cal 찾기
best = max(grid_results.items(), key=lambda kv: kv[1]['Cal'])
print(f"\nBest: T1={int(best[0][0]*100)}pp / T3U={int(best[0][1]*100)}% → Cal={best[1]['Cal']:.2f} CAGR={best[1]['CAGR']*100:.1f}% MDD={best[1]['MDD']*100:.1f}%")
# plateau: T1=20/T3U=20 근처 ±1 step 모두 Top 10 인가?
sorted_grid = sorted(grid_results.items(), key=lambda kv: -kv[1]['Cal'])
print("\nTop 10 grid points:")
for i, ((t1, t3u), m) in enumerate(sorted_grid[:10]):
    print(f"  #{i+1}: T1={int(t1*100):>3}pp T3U={int(t3u*100):>3}% Cal={m['Cal']:.2f} CAGR={m['CAGR']*100:.1f}% MDD={m['MDD']*100:.1f}%")

# T1=20/T3U=20 plateau 인접 확인
print("\nT1=20/T3U=20 인접 ±1 step plateau:")
for t1 in [0.18, 0.20, 0.23]:
    for t3u in [0.18, 0.20, 0.22]:
        m = grid_results.get((t1, t3u))
        if m:
            rank = next(i for i, (k, _) in enumerate(sorted_grid) if k == (t1, t3u)) + 1
            print(f"  T1={int(t1*100):>2}pp T3U={int(t3u*100):>2}% Cal={m['Cal']:.2f} (rank {rank}/{len(grid_results)})")


print("\n" + "=" * 90)
print("[C] stock canary 변형 민감도 — T1=20/T3U=20 고정")
print("=" * 90)
for mode in ['always_on', 'sma300']:
    e24 = alloc_bt(stock_a, spot_a, fut_v24_a, t1=0.20, t3u=0.20, stock_canary_mode=mode)
    e25 = alloc_bt(stock_a, spot_a, fut_v25_a, t1=0.20, t3u=0.20, stock_canary_mode=mode)
    m24, m25 = metrics(e24), metrics(e25)
    print(f"  mode={mode:>10}: V24 Cal={m24['Cal']:.2f} (CAGR {m24['CAGR']*100:.1f}%, MDD {m24['MDD']*100:.1f}%) | "
          f"V25 Cal={m25['Cal']:.2f} (CAGR {m25['CAGR']*100:.1f}%, MDD {m25['MDD']*100:.1f}%)")


print("\n" + "=" * 90)
print("[D] cost stress — tx 1.5x (선물 0.0006 → 0.0009)")
print("=" * 90)
fut_v25_high_tx = run_fut_bt(signal_K2(7, 0.025), tx=0.0009)
fut_v25_ht_a = align(fut_v25_high_tx)
e25_low = alloc_bt(stock_a, spot_a, fut_v25_a, t1=0.20, t3u=0.20)
e25_high = alloc_bt(stock_a, spot_a, fut_v25_ht_a, t1=0.20, t3u=0.20)
mL, mH = metrics(e25_low), metrics(e25_high)
print(f"  V25 baseline tx 0.06%: Cal={mL['Cal']:.2f} CAGR={mL['CAGR']*100:.1f}% MDD={mL['MDD']*100:.1f}%")
print(f"  V25 stress   tx 0.09%: Cal={mH['Cal']:.2f} CAGR={mH['CAGR']*100:.1f}% MDD={mH['MDD']*100:.1f}%")
print(f"  Δ Cal: {mL['Cal']-mH['Cal']:+.2f}, Δ CAGR: {(mL['CAGR']-mH['CAGR'])*100:+.1f}pp")
