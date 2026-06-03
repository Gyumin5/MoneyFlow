"""L5/L6 상방 티어 전체 검증 — 게이트 5종.

variants:
  baseline : ceiling L4
  L5       : pc>1.10→5, btc>1.08→5
  L6       : L5 + pc>1.25→6, btc>1.12→6  (hyst 연장: L5=1.10, L6=1.25 plateau 후보)

게이트:
  1. 제외조합 robustness (none/BNB/SOL/BNB+SOL)
  2. window rank-sum (504/756/1008 × 63/126/252) — BNB+SOL & full
  3. cost/maint stress (tx 1/3/5x × maint 0.4%/1.0%)
  4. bootstrap outlier (랜덤 1/2/3 제거 × 12)
  5. thr plateau grid (L5 pc 1.075~1.15, L6 pc 1.20~1.30)

출력은 /tmp/k2_l5l6_full.log. look-ahead: shift(1).
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from unified_backtest import load_data
from bt_k2_l1_downside import run_stock, build_alloc, metrics

START = "2020-10-01"; END = "2026-05-29"
K2_HYST = 0.025
_STOCK = None; _SPOT_CACHE = {}


def build_K2_tiered(bars, l5_pc=None, l5_btc=None, l6_pc=None, l6_btc=None,
                    btc_sma=42, btc_mid=1.015, btc_max=1.05, k2_sma=7,
                    l_min=2.0, l_mid=3.0, l_max=4.0, l5=5.0, l6=6.0):
    btc_df = bars.get('BTC')
    if btc_df is None: return {}
    bc = pd.Series(btc_df['Close'].values, index=btc_df.index)
    br = bc / bc.rolling(btc_sma).mean()
    cap = np.where(br > btc_max, l_max, np.where(br > btc_mid, l_mid, l_min))
    if l5_btc is not None: cap = np.where(br > l5_btc, l5, cap)
    if l6_btc is not None: cap = np.where(br > l6_btc, l6, cap)
    btc_cap = pd.Series(cap, index=br.index).shift(1).ffill().fillna(l_min)
    tmax = 1.0 + K2_HYST * 3; tmid = 1.0 + K2_HYST
    out = {}
    for coin in bars:
        close = bars[coin]['Close']
        ratio = close / close.rolling(k2_sma).mean()
        base = np.where(ratio > tmax, l_max, np.where(ratio > tmid, l_mid, l_min))
        if l5_pc is not None: base = np.where(ratio > l5_pc, l5, base)
        if l6_pc is not None: base = np.where(ratio > l6_pc, l6, base)
        pc = pd.Series(base, index=close.index).shift(1).ffill().fillna(l_min)
        idx = pc.index.intersection(btc_cap.index)
        out[coin] = pd.Series(np.minimum(pc.loc[idx].values, btc_cap.loc[idx].values), index=idx)
    return out


def run_fut(k2, exclude, tx=0.0006, maint=0.004):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'; os.environ['ANCHOR_TRADE_MODE'] = 'on'
    from backtest_futures_v25 import run as fbt_run
    bars_full, funding = load_data('D')
    bars = {c: df for c, df in bars_full.items() if c not in exclude}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=tx, maint_rate=maint,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95, start_date=START, end_date=END)
    return m.get('_equity') if m else None


def run_spot(exclude, tx=0.004):
    key = (tuple(sorted(exclude)), tx)
    if key in _SPOT_CACHE: return _SPOT_CACHE[key]
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from unified_backtest import run as bt_run
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=tx,
        start_date=START, end_date=END, sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily', n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy', stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol', exclude_assets=frozenset(exclude) if exclude else None)
    eq = m.get('_equity') if m else None
    _SPOT_CACHE[key] = eq
    return eq


def stock():
    global _STOCK
    if _STOCK is None: _STOCK = run_stock()
    return _STOCK


VARIANTS = [
    ('baseline L4', dict()),
    ('L5',          dict(l5_pc=1.10, l5_btc=1.08)),
    ('L6',          dict(l5_pc=1.10, l5_btc=1.08, l6_pc=1.25, l6_btc=1.12)),
]
EXCL = {'none': [], 'BNB': ['BNB'], 'SOL': ['SOL'], 'BNB+SOL': ['BNB', 'SOL']}


def window_rs(eq_dict, wins=(504, 756, 1008), strides=(63, 126, 252)):
    tags = list(eq_dict)
    common = None
    for t in tags:
        idx = eq_dict[t].dropna().index
        common = idx if common is None else common.intersection(idx)
    common = sorted(common)
    sums = defaultdict(float); wins_cnt = defaultdict(int); n_w = 0
    def cal(s, d0, d1):
        seg = s.loc[d0:d1].dropna()
        if len(seg) < 30: return None
        yrs = (seg.index[-1] - seg.index[0]).days / 365.25
        if yrs <= 0: return None
        cagr = (seg.iloc[-1]/seg.iloc[0]) ** (1/yrs) - 1
        peak = seg.cummax(); mdd = float((seg/peak - 1).min())
        return cagr/abs(mdd) if mdd < 0 else 0
    for size in wins:
        for stride in strides:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                vals = {t: cal(eq_dict[t], d0, d1) for t in tags}
                if any(v is None for v in vals.values()): continue
                n_w += 1
                ranked = sorted(tags, key=lambda t: -vals[t])
                for r, t in enumerate(ranked):
                    sums[t] += (r + 1)
                wins_cnt[ranked[0]] += 1
    return sums, wins_cnt, n_w


def gate1_exclusion():
    print("\n========== GATE 1: 제외조합 robustness ==========", flush=True)
    bars_full, _ = load_data('D')
    eq_st = stock()
    store = {}  # (vtag, etag) -> alloc eq
    print(f"{'variant':<12} {'excl':<8} {'fut_Cal':>8} | {'al_CAGR':>8} {'al_MDD':>8} {'al_Sh':>6} {'al_Cal':>7}", flush=True)
    for vtag, kw in VARIANTS:
        for etag, ex in EXCL.items():
            k2 = build_K2_tiered(bars_full, **kw)
            eq_fu = run_fut(k2, set(ex)); eq_sp = run_spot(ex)
            if eq_fu is None or eq_sp is None:
                print(f"{vtag:<12} {etag:<8} FAIL", flush=True); continue
            al = build_alloc(eq_st, eq_sp, eq_fu)
            store[(vtag, etag)] = al
            mf = metrics(eq_fu); ma = metrics(al)
            print(f"{vtag:<12} {etag:<8} {mf[3]:>8.2f} | {ma[0]:>7.1f}% {ma[1]:>+7.1f}% {ma[2]:>6.2f} {ma[3]:>7.2f}", flush=True)
        print(flush=True)
    print("[delta vs baseline]", flush=True)
    for etag in EXCL:
        b = metrics(store[('baseline L4', etag)])[3]
        l5 = metrics(store[('L5', etag)])[3]
        l6 = metrics(store[('L6', etag)])[3]
        print(f"  {etag:<8} base {b:.2f} | L5 {l5:.2f} ({l5-b:+.2f}) | L6 {l6:.2f} ({l6-b:+.2f})", flush=True)
    return store


def gate2_window(store):
    print("\n========== GATE 2: window rank-sum ==========", flush=True)
    for etag in ['none', 'BNB+SOL']:
        eqs = {v[0]: store[(v[0], etag)] for v in VARIANTS}
        sums, wcnt, n_w = window_rs(eqs)
        print(f"\n  [exclude={etag}] n_windows={n_w}", flush=True)
        for t, s in sorted(sums.items(), key=lambda x: x[1]):
            print(f"    {t:<12} avg_rank={s/n_w:.3f}  win={wcnt[t]/n_w*100:.1f}%", flush=True)


def gate3_cost(store):
    print("\n========== GATE 3: cost/maint stress (BNB+SOL 제외) ==========", flush=True)
    bars_full, _ = load_data('D')
    eq_st = stock()
    ex = ['BNB', 'SOL']
    print(f"{'variant':<12} {'tx':>4} {'maint':>6} {'fut_CAGR':>9} {'fut_MDD':>8} {'fut_Cal':>8} {'al_Cal':>7}", flush=True)
    for vtag, kw in VARIANTS:
        for txm in [1, 3, 5]:
            for maint in [0.004, 0.010]:
                k2 = build_K2_tiered(bars_full, **kw)
                eq_fu = run_fut(k2, set(ex), tx=0.0006*txm, maint=maint)
                eq_sp = run_spot(ex, tx=0.004*txm)
                if eq_fu is None or eq_sp is None:
                    print(f"{vtag:<12} {txm}x  {maint} FAIL", flush=True); continue
                al = build_alloc(eq_st, eq_sp, eq_fu)
                mf = metrics(eq_fu); ma = metrics(al)
                print(f"{vtag:<12} {txm}x  {maint:>6.3f} {mf[0]:>8.1f}% {mf[1]:>+7.1f}% {mf[3]:>8.2f} {ma[3]:>7.2f}", flush=True)
        print(flush=True)


def gate4_bootstrap():
    print("\n========== GATE 4: bootstrap outlier ==========", flush=True)
    bars_full, _ = load_data('D')
    eq_st = stock()
    cands = sorted([c for c in bars_full if c not in ('BTC', 'CASH')])
    rng = np.random.RandomState(7)
    samples = []
    for nd in [1, 2, 3]:
        for _ in range(4):
            samples.append((nd, sorted(rng.choice(cands, size=nd, replace=False))))
    print(f"  {'drop':<22} {'base':>6} {'L5':>6} {'L6':>6} {'L5-b':>6} {'L6-b':>6}", flush=True)
    wins5 = 0; wins6 = 0
    for nd, picks in samples:
        ex = set(picks)
        eq_sp = run_spot(list(ex))
        if eq_sp is None: continue
        cals = {}
        for vtag, kw in VARIANTS:
            k2 = build_K2_tiered(bars_full, **kw)
            eq_fu = run_fut(k2, ex)
            if eq_fu is None: cals[vtag] = None; continue
            cals[vtag] = metrics(build_alloc(eq_st, eq_sp, eq_fu))[3]
        if any(v is None for v in cals.values()): continue
        b, l5, l6 = cals['baseline L4'], cals['L5'], cals['L6']
        if l5 > b: wins5 += 1
        if l6 > b: wins6 += 1
        print(f"  {','.join(picks):<22} {b:>6.2f} {l5:>6.2f} {l6:>6.2f} {l5-b:>+6.2f} {l6-b:>+6.2f}", flush=True)
    print(f"\n  L5 > base: {wins5}/{len(samples)}   L6 > base: {wins6}/{len(samples)}", flush=True)


def gate5_plateau():
    print("\n========== GATE 5: thr plateau grid (BNB+SOL 제외) ==========", flush=True)
    bars_full, _ = load_data('D')
    eq_st = stock(); ex = ['BNB', 'SOL']; eq_sp = run_spot(ex)
    print("\n  [L5 only, pc thr grid, btc=1.08 fixed]", flush=True)
    print(f"  {'pc_thr':>7} {'fut_Cal':>8} {'al_Cal':>7}", flush=True)
    for pc in [1.075, 1.10, 1.125, 1.15, 1.20]:
        k2 = build_K2_tiered(bars_full, l5_pc=pc, l5_btc=1.08)
        eq_fu = run_fut(k2, set(ex))
        if eq_fu is None: continue
        al = build_alloc(eq_st, eq_sp, eq_fu)
        print(f"  {pc:>7.3f} {metrics(eq_fu)[3]:>8.2f} {metrics(al)[3]:>7.2f}", flush=True)
    print("\n  [L6 add, l6_pc grid, L5=1.10/1.08, l6_btc=1.12 fixed]", flush=True)
    print(f"  {'l6_pc':>7} {'fut_Cal':>8} {'al_Cal':>7}", flush=True)
    for pc6 in [1.20, 1.25, 1.30, 1.35]:
        k2 = build_K2_tiered(bars_full, l5_pc=1.10, l5_btc=1.08, l6_pc=pc6, l6_btc=1.12)
        eq_fu = run_fut(k2, set(ex))
        if eq_fu is None: continue
        al = build_alloc(eq_st, eq_sp, eq_fu)
        print(f"  {pc6:>7.3f} {metrics(eq_fu)[3]:>8.2f} {metrics(al)[3]:>7.2f}", flush=True)


def main():
    t0 = time.time()
    print(f"[stock] CAGR {metrics(stock())[0]:.1f}%", flush=True)
    store = gate1_exclusion()
    gate2_window(store)
    gate3_cost(store)
    gate4_bootstrap()
    gate5_plateau()
    print(f"\n총 소요: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
