"""V24 vs V25 자산배분 BT — 60/25/15 + sleeve cash buffer 7/1/1, 4 trigger 조합 교차.

사용자 요청 (2026-05-28):
- alloc 60/25/15 고정
- sleeve cash buffer: stock 7%, spot 1%, fut 1% (총 ≈ 4.6% cash, alpha 비획득)
- 전략 비교: V24 (fut L3 ISO 고정) vs V25 (fut K2 동적 L 2/3/4 + CROSS)
- 리밸 trigger 교차: V24 라이브 (T1=13/T3U=20) + V25 라이브 (T1=20/T3U=25) — 4 조합
- look-ahead 차단 (shift(1)), daily 평가, fire 분포 포함

bt_cross 엔진 (선물 CROSS 청산 모델, 동적 leverage 지원) + 사전 저장된 stock/spot equity 사용.
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

# ─── helpers ───
def metrics(s):
    s = s.dropna()
    if len(s) < 30: return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    if yrs <= 0 or s.iloc[0] <= 0: return None
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    peak = s.cummax(); mdd = float((s / peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    sh = (s.pct_change().mean() / s.pct_change().std() * np.sqrt(365)) if s.pct_change().std() > 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, Sharpe=sh)


def yearly_cal(eq):
    out = {}
    for y in [2021, 2022, 2023, 2024, 2025]:
        sub = eq[eq.index.year == y]
        m = metrics(sub) if len(sub) > 50 else None
        out[y] = m['Cal'] if m else 0
    return out


# ─── fut sleeve BT ───
bars0, funding0 = bt_cross.load_data('D')
btc_close = pd.Series(bars0['BTC']['Close'].values, index=bars0['BTC'].index)
btc_sma42 = btc_close.rolling(42).mean()
btc_ratio = btc_close / btc_sma42
btc_cap_signal = pd.Series(np.where(btc_ratio > 1.05, 4.0,
                          np.where(btc_ratio > 1.015, 3.0, 2.0)),
                          index=btc_ratio.index).shift(1).ffill().fillna(2.0)


def signal_K2(period=7, h=0.025):
    """V25 K2: per-coin SMA ratio → min(BTC_cap, K2)."""
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


def run_fut_bt(leverage):
    return bt_cross.run(bars0, funding0, interval='D', leverage=leverage,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=0.0006, maint_rate=0.004, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date='2020-10-01', end_date='2026-05-13')['_equity']


RES_DIR = '/home/gmoh/mon/251229/strategies/cap_defend/research/alloc_reopt_2026_05'
print("=== sleeve equity 준비 ===")
stock_eq = pd.read_csv(f'{RES_DIR}/stock_equity.csv', index_col='Date', parse_dates=True)['Value']
spot_eq = pd.read_csv(f'{RES_DIR}/spot_equity.csv', index_col='Date', parse_dates=True)['Value']
print(f"  stock equity: {stock_eq.index[0].date()} ~ {stock_eq.index[-1].date()} n={len(stock_eq)}")
print(f"  spot equity:  {spot_eq.index[0].date()} ~ {spot_eq.index[-1].date()} n={len(spot_eq)}")

print("\nfut sleeve BT 실행 중 ...")
fut_v24 = run_fut_bt(3.0)
fut_v25 = run_fut_bt(signal_K2(7, 0.025))
print(f"  fut V24 L3 단독:  Cal={metrics(fut_v24)['Cal']:.2f} CAGR={metrics(fut_v24)['CAGR']*100:.1f}%")
print(f"  fut V25 K2 단독:  Cal={metrics(fut_v25)['Cal']:.2f} CAGR={metrics(fut_v25)['CAGR']*100:.1f}%")

# ─── align ───
start = max(stock_eq.index[0], spot_eq.index[0], fut_v24.index[0])
end = min(stock_eq.index[-1], spot_eq.index[-1], fut_v24.index[-1])
print(f"\n공통 기간: {start.date()} ~ {end.date()}")

def align(s):
    return s[(s.index >= start) & (s.index <= end)].reindex(pd.date_range(start, end, freq='D')).ffill().dropna()

stock_a = align(stock_eq)
spot_a = align(spot_eq)
fut_v24_a = align(fut_v24)
fut_v25_a = align(fut_v25)


# ─── alloc BT with per-sleeve cash buffer ───
# 각 sleeve 의 effective return = sleeve_return × (1 - buf). 버퍼 cash 는 0% (KRW/USDT).
# total target weight 는 자산배분 비중 (60/25/15) 그대로 유지 — 버퍼는 sleeve 내부.
BUF_STOCK = 0.07
BUF_SPOT = 0.01
BUF_FUT = 0.01


def alloc_bt(stock, spot, fut, w_st=0.60, w_sp=0.25, w_ft=0.15,
             buf=(BUF_STOCK, BUF_SPOT, BUF_FUT),
             t1=None, t3u=None):
    """T1+T3U_can drift trigger, 자산간 리밸. sleeve 내부 cash buffer 반영.
    t1=None → 일별 강제 rebal (이론 베이스라인).
    """
    bs, bp, bf = buf
    # sleeve 내부 invested return × (1-buf). buffer 는 idle cash.
    r_st = stock.pct_change().fillna(0) * (1 - bs)
    r_sp = spot.pct_change().fillna(0) * (1 - bp)
    r_ft = fut.pct_change().fillna(0) * (1 - bf)

    target = np.array([w_st, w_sp, w_ft])
    val = target.copy()

    btc_sub = btc_close.reindex(stock.index).ffill()
    btc_sma_sub = btc_sub.rolling(42).mean()
    btc_canary = (btc_sub > btc_sma_sub * 1.015).shift(1).fillna(False)

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
            canary = [True, bool(btc_canary.iloc[i]), bool(btc_canary.iloc[i])]
            rel_under = max(((target[j] - cur_w[j]) / target[j]) if (canary[j] and target[j] > 0) else 0
                            for j in range(3))
            fire = (ht >= t1) or (rel_under >= t3u)
            if fire:
                val = total * target
                fire_log.append((date, 'T1' if ht >= t1 else 'T3U', ht, rel_under))
        else:
            val = total * target  # daily rebal

        eq_list.append(val.sum())
    eq_s = pd.Series(eq_list, index=stock.index)
    eq_s.fire_log = fire_log
    return eq_s


# ─── BT matrix ───
print("\n" + "=" * 90)
print("V24 vs V25 자산배분 BT — alloc 60/25/15, buffer 7/1/1, 4 trigger 조합")
print("=" * 90)
print(f"{'전략':<8} {'trigger':<22} | {'CAGR':>7} {'MDD':>8} {'Cal':>5} {'Shrp':>5} | y21 y22 y23 y24 y25 | fires")
print("-" * 110)

def row(label_strat, label_trig, eq):
    m = metrics(eq); y = yearly_cal(eq)
    if m is None:
        print(f"{label_strat:<8} {label_trig:<22} | BT 실패"); return
    fires = len(getattr(eq, 'fire_log', []))
    print(f"{label_strat:<8} {label_trig:<22} | {m['CAGR']*100:6.1f}% {m['MDD']*100:7.1f}% {m['Cal']:5.2f} {m['Sharpe']:4.2f} | "
          f"{y[2021]:5.1f} {y[2022]:5.1f} {y[2023]:5.1f} {y[2024]:5.1f} {y[2025]:5.1f} | {fires}")


# 4 trigger 조합
triggers = [
    ('일별 강제 rebal',  None, None),
    ('T1=13 / T3U=20 (V24 라이브)',  0.13, 0.20),
    ('T1=20 / T3U=25 (V25 라이브)',  0.20, 0.25),
    ('T1=20 / T3U=20',  0.20, 0.20),
]

results = {}
for trig_label, t1, t3u in triggers:
    print(f"\n--- trigger: {trig_label} ---")
    e_v24 = alloc_bt(stock_a, spot_a, fut_v24_a, t1=t1, t3u=t3u)
    e_v25 = alloc_bt(stock_a, spot_a, fut_v25_a, t1=t1, t3u=t3u)
    row('V24', trig_label, e_v24)
    row('V25', trig_label, e_v25)
    results[trig_label] = {'V24': e_v24, 'V25': e_v25}

# 단독 sleeve reference
print('\n--- 단독 sleeve reference (5.6yr, no alloc) ---')
print(f"{'sleeve':<22} | {'CAGR':>7} {'MDD':>8} {'Cal':>5} {'Shrp':>5}")
for label, eq in [
    ('stock 단독 (KIS)', stock_a),
    ('spot 단독 (Upbit)', spot_a),
    ('fut V24 L3 단독',  fut_v24_a),
    ('fut V25 K2 단독',  fut_v25_a),
]:
    m = metrics(eq)
    print(f"{label:<22} | {m['CAGR']*100:6.1f}% {m['MDD']*100:7.1f}% {m['Cal']:5.2f} {m['Sharpe']:4.2f}")

print("\n=== Summary — alloc 60/25/15 + buffer 7/1/1 ===")
print(f"{'trigger':<28} | V24 Cal | V25 Cal | V24 CAGR | V25 CAGR | V24 MDD | V25 MDD")
print("-" * 110)
for trig_label, _, _ in triggers:
    eV = results[trig_label]
    m24 = metrics(eV['V24']); m25 = metrics(eV['V25'])
    print(f"{trig_label:<28} | {m24['Cal']:7.2f} | {m25['Cal']:7.2f} | "
          f"{m24['CAGR']*100:7.1f}% | {m25['CAGR']*100:7.1f}% | "
          f"{m24['MDD']*100:6.1f}% | {m25['MDD']*100:6.1f}%")

print("\n=== buffer 정합성 확인 ===")
buf_total = 0.60 * BUF_STOCK + 0.25 * BUF_SPOT + 0.15 * BUF_FUT
print(f"buffer total cash share: {buf_total*100:.2f}% of total assets")
print(f"  stock 60% × {BUF_STOCK*100:.0f}% = {0.60*BUF_STOCK*100:.1f}% cash inside stock account")
print(f"  spot  25% × {BUF_SPOT*100:.0f}%  = {0.25*BUF_SPOT*100:.2f}% cash inside spot account")
print(f"  fut   15% × {BUF_FUT*100:.0f}%   = {0.15*BUF_FUT*100:.2f}% cash inside fut account")
