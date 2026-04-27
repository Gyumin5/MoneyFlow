#!/usr/bin/env python3
"""Upbit 1h 데이터로 V22 현물 C 슬리브 재백테.

Binance champion (s_dthr12_tp3 + A2_bounce_w1) 을 Upbit 실거래가로 재검증.
목적:
- Upbit 과민반응 (altcoin 프리미엄/디스카운트) 영향 측정
- 실체결 가정과 백테 수치 괴리 확인
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "c_tests_v3"))
sys.path.insert(0, os.path.join(HERE, "next_strategies"))

from c_engine_v5 import run_c_v5
from m3_engine_final import (simulate, load_universe_hist, load_coin_daily,
                              list_available_futures, load_v21)
from _common3 import filter_bounce_confirm, TRAIN_END, HOLDOUT_START, FULL_END, slice_v21, CAP_SPOT

try:
    import pyupbit
except ImportError:
    print("pyupbit 미설치. pip install pyupbit")
    sys.exit(1)

OUT = os.path.join(HERE, "upbit_c_out")
os.makedirs(OUT, exist_ok=True)
CACHE_DIR = os.path.join(OUT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

START = pd.Timestamp("2020-10-01", tz='UTC')
END = pd.Timestamp("2026-03-28", tz='UTC')


def fetch_upbit_1h(ticker: str) -> pd.DataFrame | None:
    """Upbit 1h OHLCV 전체 history 수집. cache file 활용."""
    cache = os.path.join(CACHE_DIR, f"{ticker.replace('KRW-', '')}_1h.pkl")
    if os.path.exists(cache):
        try:
            df = pd.read_pickle(cache)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass
    try:
        df = pyupbit.get_ohlcv_from(ticker, interval="minute60",
                                      fromDatetime="2020-10-01 00:00:00")
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={'open':'Open','high':'High','low':'Low',
                                  'close':'Close','volume':'Volume'})
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.to_pickle(cache)
        return df
    except Exception as e:
        print(f"    {ticker} fetch 실패: {e}")
        return None


def fetch_sequential(coins):
    """순차 fetch (pyupbit 세션 충돌 회피, rate limit 준수)."""
    out = {}
    print(f"  Upbit 1h fetch (sequential): {len(coins)} coins")
    for i, c in enumerate(coins):
        ticker = f"KRW-{c}"
        df = fetch_upbit_1h(ticker)
        if df is not None and len(df) > 100:
            out[c] = df
            print(f"    [{i+1}/{len(coins)}] {c}: {len(df)} bars")
        else:
            print(f"    [{i+1}/{len(coins)}] {c}: SKIP")
        time.sleep(0.15)  # rate limit
    return out


def extract_c_events_upbit(coins_data: dict, dip_bars=24, dip_thr=-0.12,
                             tp=0.03, tstop=24, tx=0.0005):
    """각 코인에서 C 이벤트 추출. TX cost 는 Upbit 0.05%."""
    all_events = []
    for coin, df in coins_data.items():
        if df is None or len(df) < dip_bars + 50:
            continue
        # Upbit 데이터 (index: timezone-aware). run_c_v5 는 tz-naive 가정일 수 있음.
        df2 = df.copy()
        if df2.index.tz is not None:
            df2.index = df2.index.tz_localize(None)
        try:
            _, evs = run_c_v5(df2, dip_bars=dip_bars, dip_thr=dip_thr,
                               tp=tp, tstop=tstop, tx=tx)
            for e in evs:
                e['coin'] = coin
            all_events.extend(evs)
        except Exception as e:
            print(f"  {coin} run_c_v5 실패: {e}")
    return pd.DataFrame(all_events)


def main():
    # 유니버스: Binance 에 있는 Top 15 중 Upbit 상장된 것만
    avail_fut = sorted(list_available_futures())
    print(f"Binance futures universe: {len(avail_fut)} coins")

    # Upbit KRW 마켓 전체
    print("Upbit KRW 마켓 조회...")
    krw_markets = pyupbit.get_tickers(fiat="KRW")
    krw_coins = {t.replace("KRW-", "") for t in krw_markets}
    print(f"  Upbit KRW: {len(krw_coins)} coins")

    # 교집합 (Binance futures universe ∩ Upbit KRW)
    target_coins = sorted(set(avail_fut) & krw_coins)
    print(f"  교집합: {len(target_coins)} coins")
    # 상위 30개만 (Top 15 + 여유, 백테 universe_size=15)
    target_coins = target_coins[:30]

    # Upbit 1h 데이터 수집
    print("\n1. Upbit 1h 데이터 fetch...")
    t0 = time.time()
    coins_data = fetch_sequential(target_coins)
    print(f"   획득: {len(coins_data)} coins ({time.time()-t0:.0f}s)")
    for c in sorted(coins_data.keys())[:10]:
        df = coins_data[c]
        print(f"   {c}: {len(df)} bars, {df.index[0]} ~ {df.index[-1]}")

    # 2. C 이벤트 재추출 (champion params, Upbit 1h 기반)
    print("\n2. C 이벤트 재추출 (Upbit 1h)...")
    t0 = time.time()
    ev = extract_c_events_upbit(coins_data,
                                  dip_bars=24, dip_thr=-0.12, tp=0.03, tstop=24,
                                  tx=0.0005)  # Upbit 0.05%
    print(f"   events: {len(ev)} ({time.time()-t0:.0f}s)")

    # A2 bounce 필터 적용 (Binance champion 과 동일)
    ev_filtered = filter_bounce_confirm(ev, 1)
    print(f"   A2 filter 후: {len(ev_filtered)}")

    # 3. V21 spot + C 시뮬
    hist = load_universe_hist()
    cd = load_coin_daily(avail_fut)
    v21_spot = load_v21()

    # START 이후만
    mask = (v21_spot.index >= START.tz_localize(None)) & (v21_spot.index <= END.tz_localize(None))
    v21s = v21_spot[mask].copy()
    v21s['equity'] = v21s['equity'].astype(float) / float(v21s['equity'].iloc[0])
    v21s['v21_ret'] = v21s['equity'].pct_change().fillna(0)
    v21s['prev_cash'] = v21s['cash_ratio'].shift(1).fillna(v21s['cash_ratio'].iloc[0])

    ev_s = ev_filtered.copy()
    if len(ev_s) > 0:
        ev_s['entry_ts'] = pd.to_datetime(ev_s['entry_ts'])
        ev_s = ev_s[(ev_s['entry_ts'] >= v21s.index[0]) & (ev_s['entry_ts'] <= v21s.index[-1])]
    print(f"\n3. 시뮬 (V21+C champion, Upbit events): evts after slice={len(ev_s)}")

    rows = []
    for span, start, end in [
        ("full", v21s.index[0], FULL_END.tz_localize('UTC').tz_localize(None) if FULL_END.tz is not None else FULL_END),
        ("train", v21s.index[0], TRAIN_END),
        ("holdout", HOLDOUT_START, FULL_END.tz_localize('UTC').tz_localize(None) if FULL_END.tz is not None else FULL_END),
    ]:
        try:
            vs = slice_v21(v21s, start, end)
            if vs is None: continue
            e_s = ev_s[(ev_s['entry_ts'] >= vs.index[0]) & (ev_s['entry_ts'] <= vs.index[-1])]
            _, st = simulate(e_s, cd, vs.copy(), hist,
                              n_pick=1, cap_per_slot=CAP_SPOT, universe_size=15,
                              tx_cost=0.0005, swap_edge_threshold=1)
            rows.append({
                'span': span,
                'events': len(e_s),
                'Cal': round(st.get('Cal', 0), 3),
                'CAGR': round(st.get('CAGR', 0)*100, 2),
                'MDD': round(st.get('MDD', 0)*100, 2),
                'Sharpe': round(st.get('Sharpe', 0), 2),
            })
        except Exception as e:
            print(f"  {span} simulate 실패: {e}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "upbit_c_result.csv"), index=False)
    print(f"\n=== Upbit 재백테 결과 (V22 champion) ===")
    print(df_out.to_string(index=False))
    print(f"\n저장: {OUT}/upbit_c_result.csv")


if __name__ == "__main__":
    main()
