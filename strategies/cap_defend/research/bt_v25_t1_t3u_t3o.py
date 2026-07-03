"""V25 — alloc 트리거 T1 × T3U_can × T3O_can, 실제 라이브 카나리로 정확 구현.

이전 bt_v25_t1_t3u.py 는 카나리를 "20일평균>0" 프록시로 단순화 → 본 스크립트는
라이브 recommend_personal.py 와 동일한 sleeve 카나리를 그대로 구현한다.

라이브 카나리 (코드 확인, source of truth):
- stock: EEM SMA200, ±0.5% dead-zone hysteresis (stock_strategy_v25.eem_canary)
- spot : BTC SMA42, 진입 close>SMA*1.015 / 이탈 close<SMA*0.985 dead-zone (coin_live_engine)
- fut  : BTC SMA42 동일 (완료봉 기준 — 일봉 BT 에선 spot 과 동일 시계열)

트리거 (라이브 recommend_personal.py):
- T1     : half_turnover = (|Δstock|+|Δspot|+|Δfut|)/2 >= T1pp → 전체 60/25/15 reset
- T3U_can: sleeve 상대미달 (tgt-cur)/tgt >= T3U AND 그 sleeve 카나리 ON → 그 sleeve target 까지 끌어올림
- T3O_can: sleeve 상대과대 (cur-tgt)/tgt >= T3O AND 그 sleeve 카나리 (방향 옵션) → target 까지 트림
  · T3O 카나리 방향 모드: 'off'(대칭, 약세전환 시 트림) / 'on' / 'none'(무카나리)
- 발화 시 100% 가상 송금 (수수료 무시 — 수동 송금 가정)
- 우선순위: T1 > (단일 sleeve T3 부분조정) ; 동시 2+ sleeve 발화는 full reset

BNB·SOL 은 아웃라이어로 코인 현물·선물 유니버스에서 제외.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_excl_matrix import run_spot, run_fut, run_stock, metrics

START = "2020-10-01"; END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']
WEIGHTS = {'stock': 0.60, 'spot': 0.25, 'fut': 0.15}


def dead_zone_canary(close, sma_period, band):
    """라이브 dead-zone 카나리: 진입 close>SMA*(1+band), 이탈 close<SMA*(1-band), 그 외 prev 유지.
    첫 유효봉 seed = close>SMA. stateful (라이브와 동일). 반환 bool Series."""
    close = close.dropna().astype(float)
    sma = close.rolling(sma_period).mean()
    out = pd.Series(index=close.index, dtype=object)
    prev = None
    for i in range(len(close)):
        s = sma.iloc[i]; c = close.iloc[i]
        if pd.isna(s):
            out.iloc[i] = None; continue
        if prev is None:
            st = bool(c > s)
        elif c > s * (1 + band):
            st = True
        elif c < s * (1 - band):
            st = False
        else:
            st = prev
        out.iloc[i] = st; prev = st
    return out.ffill().fillna(False).astype(bool)


def load_canaries(common):
    """라이브 정의대로 stock/spot/fut 카나리 일별 bool 시계열 → common 인덱스에 ffill 정렬."""
    # --- BTC SMA42 ±1.5% (spot/fut 공통) ---
    from unified_backtest import load_data
    bars, _ = load_data('D')
    btc_close = bars['BTC']['Close']
    btc_can = dead_zone_canary(btc_close, 42, 0.015)
    spot_can = btc_can.reindex(common, method='ffill').fillna(False).astype(bool)
    fut_can = spot_can.copy()  # 일봉 BT 에선 fut(완료봉)=spot 동일 시계열
    # --- EEM SMA200 ±0.5% (stock) ---
    from stock_engine import load_prices, ALL_TICKERS
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    eem_can = dead_zone_canary(pdf['EEM'], 200, 0.005)
    stock_can = eem_can.reindex(common, method='ffill').fillna(False).astype(bool)
    return {'stock': stock_can, 'spot': spot_can, 'fut': fut_can}


def simulate_alloc(eq_st, eq_sp, eq_fu, can, T1=0.20, T3U=0.20, T3O=None, t3o_canary='off',
                   transfer_cost=0.0):
    """일별 alloc 시뮬. T3O=None 이면 과대-트림 트리거 끔(=현행 전략).
    t3o_canary: 'off'(과대 sleeve 카나리 OFF 일 때 트림) / 'on' / 'none'(무카나리).
    transfer_cost: 계좌간 송금 1회당 이동금액 비율 비용 (T1=ht, 단일 sleeve=|delta|/total 에 부과)."""
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    common = sorted(common)
    if len(common) < 30:
        return None
    r_st = eq_st.loc[common].pct_change().fillna(0)
    r_sp = eq_sp.loc[common].pct_change().fillna(0)
    r_fu = eq_fu.loc[common].pct_change().fillna(0)
    cser = {k: can[k].reindex(common).ffill().fillna(False).astype(bool) for k in can}
    rser = {'stock': r_st, 'spot': r_sp, 'fut': r_fu}

    pvs = {'stock': WEIGHTS['stock'], 'spot': WEIGHTS['spot'], 'fut': WEIGHTS['fut']}
    eq_alloc = []
    n_t1 = 0; n_t3u = 0; n_t3o = 0
    for i, d in enumerate(common):
        for k in pvs:
            pvs[k] *= (1 + rser[k].iloc[i])
        total = sum(pvs.values())
        if total <= 0:
            eq_alloc.append(0); continue
        cur_w = {k: pvs[k] / total for k in pvs}
        ht = sum(abs(cur_w[k] - WEIGHTS[k]) for k in WEIGHTS) / 2
        fire_t1 = ht >= T1

        fire = {}  # sleeve -> 'u'/'o'
        for k in WEIGHTS:
            tgt = WEIGHTS[k]
            can_on = bool(cser[k].iloc[i])
            rel_under = max(0.0, (tgt - cur_w[k]) / tgt)
            rel_over = max(0.0, (cur_w[k] - tgt) / tgt)
            if rel_under >= T3U and can_on:
                fire[k] = 'u'
            elif T3O is not None and rel_over >= T3O:
                gate = (not can_on) if t3o_canary == 'off' else (can_on if t3o_canary == 'on' else True)
                if gate:
                    fire[k] = 'o'

        moved_frac = 0.0  # 이동(송금) 금액 / total
        if fire_t1 or len(fire) >= 2:
            moved_frac = ht  # full reset: half-turnover 만큼 이동
            for k in WEIGHTS:
                pvs[k] = total * WEIGHTS[k]
            if fire_t1:
                n_t1 += 1
            else:  # 2+ sleeve 동시 발화 → full reset 로 처리, 카운트는 종류별
                n_t3u += sum(1 for v in fire.values() if v == 'u')
                n_t3o += sum(1 for v in fire.values() if v == 'o')
        elif len(fire) == 1:
            k_fire = next(iter(fire))
            others = [k for k in WEIGHTS if k != k_fire]
            w_oth = sum(WEIGHTS[o] for o in others)
            target_pv = total * WEIGHTS[k_fire]
            delta = target_pv - pvs[k_fire]  # under: +, over: -
            pvs[k_fire] = target_pv
            for o in others:
                pvs[o] -= delta * (WEIGHTS[o] / w_oth)
            moved_frac = abs(delta) / total
            if fire[k_fire] == 'u':
                n_t3u += 1
            else:
                n_t3o += 1
        if transfer_cost > 0 and moved_frac > 0:
            factor = 1.0 - moved_frac * transfer_cost
            for k in pvs:
                pvs[k] *= factor
        eq_alloc.append(sum(pvs.values()))
    return pd.Series(eq_alloc, index=common), n_t1, n_t3u, n_t3o


def main():
    t0 = time.time()
    print("[load sleeves: stock / spot(excl BNB,SOL) / fut(excl BNB,SOL)]")
    eq_st = run_stock()
    eq_sp = run_spot(20, 127, 217, 7, EXCLUDE)
    eq_fu = run_fut(18, 127, 95, 5, EXCLUDE)
    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    print(f"[load real canaries] common days={len(common)} {common[0].date()}~{common[-1].date()}")
    can = load_canaries(common)
    for k in can:
        s = can[k].reindex(common).ffill().fillna(False)
        print(f"  canary {k}: ON {s.mean()*100:.0f}% of days")

    # === TABLE 1: T1 × T3U grid (T3O off) = 현행 전략, 실 카나리 ===
    print("\n=== TABLE 1: T1 × T3U grid (현행, T3O off, 실 카나리) ===")
    print(f"  {'T1':<5} {'T3U':<5} {'CAGR':>7} {'MDD':>7} {'Sh':>6} {'Cal':>5} {'nT1':>4} {'nT3U':>5}")
    T1g = [0.15, 0.17, 0.20, 0.23, 0.25, 0.30]
    T3Ug = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    grid = []
    for T1 in T1g:
        for T3U in T3Ug:
            res = simulate_alloc(eq_st, eq_sp, eq_fu, can, T1=T1, T3U=T3U, T3O=None)
            if res is None: continue
            eq, n1, n3u, _ = res
            m = metrics(eq)
            grid.append((T1, T3U, m, n1, n3u))
            tag = "← base" if T1 == 0.20 and T3U == 0.20 else ""
            print(f"  {T1*100:>4.0f}% {T3U*100:>4.0f}% {m[0]:>6.1f}% {m[1]:>+6.1f}% {m[2]:>6.2f} {m[3]:>5.2f} {n1:>4} {n3u:>5} {tag}")
    base_m = next(m for T1, T3U, m, n1, n3u in grid if T1 == 0.20 and T3U == 0.20)
    print("  Top 5 by Cal:")
    for T1, T3U, m, n1, n3u in sorted(grid, key=lambda r: -r[2][3])[:5]:
        print(f"    T1={T1*100:.0f}% T3U={T3U*100:.0f}%: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Sh {m[2]:.2f} Cal {m[3]:.2f}")

    # === TABLE 2: T3O ablation @ base T1=20 T3U=20, 카나리 방향 3모드 ===
    print("\n=== TABLE 2: T3O ablation (T1=20 T3U=20 고정) ===")
    print(f"  base (T3O off): CAGR {base_m[0]:.1f}% MDD {base_m[1]:+.1f}% Sh {base_m[2]:.2f} Cal {base_m[3]:.2f}")
    print(f"  {'T3O':<5} {'canary':<7} {'CAGR':>7} {'MDD':>7} {'Sh':>6} {'Cal':>5} {'nT3O':>5} {'ΔCal':>6}")
    for mode in ['off', 'on', 'none']:
        for T3O in [0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
            res = simulate_alloc(eq_st, eq_sp, eq_fu, can, T1=0.20, T3U=0.20, T3O=T3O, t3o_canary=mode)
            if res is None: continue
            eq, n1, n3u, n3o = res
            m = metrics(eq)
            print(f"  {T3O*100:>4.0f}% {mode:<7} {m[0]:>6.1f}% {m[1]:>+6.1f}% {m[2]:>6.2f} {m[3]:>5.2f} {n3o:>5} {m[3]-base_m[3]:>+6.2f}")
        print()

    # === TABLE 3: best T1/T3U + T3O(off) 재확인 ===
    bestT1, bestT3U, bm, _, _ = sorted(grid, key=lambda r: -r[2][3])[0]
    print(f"=== TABLE 3: grid-best (T1={bestT1*100:.0f}% T3U={bestT3U*100:.0f}%) + T3O(off) sweep ===")
    print(f"  no-T3O: Cal {bm[3]:.2f}")
    for T3O in [0.15, 0.18, 0.20, 0.25]:
        res = simulate_alloc(eq_st, eq_sp, eq_fu, can, T1=bestT1, T3U=bestT3U, T3O=T3O, t3o_canary='off')
        eq, n1, n3u, n3o = res
        m = metrics(eq)
        print(f"  +T3O {T3O*100:.0f}% off: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Cal {m[3]:.2f} (nT3O {n3o})")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
