"""
Enhanced Strategy Backtest — 기존 전략에 보조지표를 추가(교체X)
================================================================
ETF: Cap Defend Stock + 보조지표 필터/스코어링 보강
Crypto: Baseline + 보조지표 필터/스코어링 보강

Usage:
    python3 strategies/cap_defend/backtest_enhanced.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = './data'
UNIVERSE_FILE = './data/historical_universe.json'

STABLECOINS = {'USDT','USDC','BUSD','DAI','UST','TUSD','PAX','GUSD',
               'FRAX','LUSD','MIM','USDN','FDUSD','USDS','PYUSD','USDE'}

ETF_TICKERS = [
    'SPY','QQQ','EFA','EEM','VT','VEA','VNQ','VGK',
    'QUAL','MTUM','IQLT','IMTM','IWD','SCZ',
    'IEF','TLT','BND','AGG','BNDX','BIL','SHY','LQD','TIP',
    'GLD','DBC','PDBC','DBMF','KMLM','VWO','HYG','RWX',
]

ETF_STARTS = ['2012-01-01','2014-01-01','2016-01-01','2019-01-01','2021-01-01','2023-01-01']
CRYPTO_STARTS = ['2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01']
END_DATE = '2025-12-31'


# ===========================================================================
# Data loading
# ===========================================================================

def load_universe():
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE) as f:
            return json.load(f)
    return {}

def collect_crypto_tickers(hu):
    tickers = {'BTC-USD'}
    for symbols in hu.values():
        for s in symbols:
            t = s if s.endswith('-USD') else f"{s}-USD"
            if t.replace('-USD','') not in STABLECOINS:
                tickers.add(t)
    return tickers

def load_prices(tickers, buffer_start='2009-01-01'):
    data_dict = {}
    for ticker in tickers:
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp, parse_dates=['Date'])
            df = df.drop_duplicates(subset=['Date'], keep='first').set_index('Date')
            col = 'Adj Close' if 'Adj Close' in df else ('Adj_Close' if 'Adj_Close' in df else 'Close')
            if col in df:
                data_dict[ticker] = df[col]
        except:
            pass
    idx = pd.date_range(start=buffer_start, end=END_DATE, freq='D')
    return pd.DataFrame(data_dict).reindex(idx).ffill()


# ===========================================================================
# Pre-compute ALL indicators
# ===========================================================================

def precompute(data):
    ind = {}
    for col in data.columns:
        p = data[col]
        dr = p.pct_change()

        d = pd.DataFrame({'price': p})

        # --- SMA ---
        d['sma30']  = p.rolling(30).mean()
        d['sma50']  = p.rolling(50).mean()
        d['sma200'] = p.rolling(200).mean()

        # --- Momentum ---
        d['mom21']  = p / p.shift(21) - 1
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1
        d['mom_w']  = 0.5*d['mom63'] + 0.3*d['mom126'] + 0.2*d['mom252']

        # --- Volatility ---
        d['vol90'] = dr.rolling(90).std()

        # --- Sharpe ---
        for w in [126, 252]:
            rm = dr.rolling(w).mean()
            rs = dr.rolling(w).std()
            d[f'sharpe{w}'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)

        # --- RSI (14) ---
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs_rsi = gain / loss.replace(0, np.nan)
        d['rsi14'] = 100 - (100 / (1 + rs_rsi))

        # --- MACD (12,26,9) ---
        ema12 = p.ewm(span=12).mean()
        ema26 = p.ewm(span=26).mean()
        d['macd_line'] = ema12 - ema26
        d['macd_signal'] = d['macd_line'].ewm(span=9).mean()
        d['macd_hist'] = d['macd_line'] - d['macd_signal']

        # --- Bollinger Bands (20,2) ---
        bb_sma = p.rolling(20).mean()
        bb_std = p.rolling(20).std()
        d['bb_upper'] = bb_sma + 2*bb_std
        d['bb_lower'] = bb_sma - 2*bb_std
        d['bb_pctb'] = (p - d['bb_lower']) / (d['bb_upper'] - d['bb_lower']).replace(0, np.nan)

        # --- Stochastic (14,3) ---
        low_14  = p.rolling(14).min()
        high_14 = p.rolling(14).max()
        d['stoch_k'] = 100 * (p - low_14) / (high_14 - low_14).replace(0, np.nan)

        # --- CCI (20) ---
        tp_sma = p.rolling(20).mean()
        tp_mad = p.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        d['cci20'] = (p - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # --- ADX (14) approximation ---
        high = p * 1.005
        low  = p * 0.995
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = (high - low).rolling(14).mean().replace(0, np.nan)
        plus_di  = 100 * plus_dm.rolling(14).mean() / tr
        minus_di = 100 * minus_dm.rolling(14).mean() / tr
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        d['adx14'] = dx.rolling(14).mean()
        d['plus_di'] = plus_di
        d['minus_di'] = minus_di

        # --- Donchian (20) ---
        d['don_high'] = p.rolling(20).max()

        # --- Williams %R (14) ---
        d['willr14'] = -100 * (high_14 - p) / (high_14 - low_14).replace(0, np.nan)

        # --- ATR % (14) ---
        d['atr_pct'] = (high - low).rolling(14).mean() / p

        ind[col] = d
    return ind


def v(ind, t, col, d):
    if t not in ind: return np.nan
    try: return ind[t][col].loc[d]
    except: return np.nan

def has(ind, t, d, col='mom252'):
    return pd.notna(v(ind, t, col, d))


# ===========================================================================
# CRYPTO — Baseline + 보조지표 추가 조합
# ===========================================================================

def get_universe(date, hu, top_n=50):
    key = date.strftime("%Y-%m") + "-01"
    symbols = hu.get(key, [])
    if not symbols:
        avail = sorted([k for k in hu if k <= key], reverse=True)
        if avail: symbols = hu[avail[0]]
    final = []
    for s in symbols:
        t = s if s.endswith('-USD') else f"{s}-USD"
        if t.replace('-USD','') not in STABLECOINS:
            final.append(t)
            if len(final) >= top_n: break
    return final

def ivw(ind, picks, d):
    vols = {}
    for t in picks:
        vol = v(ind,t,'vol90',d)
        vols[t] = vol if pd.notna(vol) and vol > 0 else 0.001
    inv = {t: 1.0/vol for t, vol in vols.items()}
    tot = sum(inv.values())
    return {t: val/tot for t, val in inv.items()} if tot > 0 else {t: 1.0/len(picks) for t in picks}


# --- Baseline crypto ---
def crypto_base(d, ind, hu):
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'sma30',d)) and v(ind,c,'price',d) > v(ind,c,'sma30',d)]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# --- Health 추가 조합 ---
def crypto_h_macd(d, ind, hu):
    """Baseline health + MACD hist > 0."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and v(ind,c,'mom21',d) is not np.nan and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'macd_hist',d)) and v(ind,c,'macd_hist',d) > 0]  # 추가
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_h_rsi(d, ind, hu):
    """Baseline health + RSI 35-75."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'rsi14',d)) and 35 <= v(ind,c,'rsi14',d) <= 75]  # 추가
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_h_bb(d, ind, hu):
    """Baseline health + BB %B > 0.3 (above lower band area)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'bb_pctb',d)) and v(ind,c,'bb_pctb',d) > 0.3]  # 추가
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_h_macd_rsi(d, ind, hu):
    """Baseline health + MACD > 0 + RSI 35-75."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'macd_hist',d)) and v(ind,c,'macd_hist',d) > 0
               and pd.notna(v(ind,c,'rsi14',d)) and 35 <= v(ind,c,'rsi14',d) <= 75]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_h_donchian(d, ind, hu):
    """Baseline health + near 20d high (>95%)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'don_high',d)) and v(ind,c,'don_high',d) > 0
               and v(ind,c,'price',d) >= v(ind,c,'don_high',d) * 0.95]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# --- Scoring 보강 (health는 baseline, 점수에 보조지표 가산) ---
def crypto_s_rsi_bonus(d, ind, hu):
    """Sharpe score + RSI bonus (RSI 50-70 → +0.5 bonus)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        rsi = v(ind,c,'rsi14',d)
        if pd.notna(s1) and pd.notna(s2):
            base = s1 + s2
            if pd.notna(rsi) and 50 <= rsi <= 70:
                base += 0.5
            scores[c] = base
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_s_macd_bonus(d, ind, hu):
    """Sharpe score + MACD bonus (hist > 0 → +0.5)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        mh = v(ind,c,'macd_hist',d)
        if pd.notna(s1) and pd.notna(s2):
            base = s1 + s2
            if pd.notna(mh) and mh > 0:
                base += 0.5
            scores[c] = base
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_s_bb_bonus(d, ind, hu):
    """Sharpe score + BB %B bonus (%B > 0.7 → +0.5)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        pctb = v(ind,c,'bb_pctb',d)
        if pd.notna(s1) and pd.notna(s2):
            base = s1 + s2
            if pd.notna(pctb) and pctb > 0.7:
                base += 0.5
            scores[c] = base
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_s_cci_bonus(d, ind, hu):
    """Sharpe score + CCI bonus (CCI > 100 → +0.5)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        cci = v(ind,c,'cci20',d)
        if pd.notna(s1) and pd.notna(s2):
            base = s1 + s2
            if pd.notna(cci) and cci > 100:
                base += 0.5
            scores[c] = base
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_s_multi_bonus(d, ind, hu):
    """Sharpe score + multi-indicator bonus (RSI ok +0.2, MACD ok +0.2, BB ok +0.2)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2):
            base = s1 + s2
            rsi = v(ind,c,'rsi14',d)
            mh  = v(ind,c,'macd_hist',d)
            pctb = v(ind,c,'bb_pctb',d)
            if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
            if pd.notna(mh) and mh > 0: base += 0.2
            if pd.notna(pctb) and pctb > 0.5: base += 0.2
            scores[c] = base
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# --- Canary 보강 ---
def crypto_canary_rsi(d, ind, hu):
    """BTC SMA50 canary + BTC RSI > 45 (추가 카나리아)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    btc_rsi = v(ind,'BTC-USD','rsi14',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs: return {}, 'Off'
    if pd.isna(btc_rsi) or btc_rsi <= 45: return {}, 'RSIOff'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'

def crypto_canary_macd(d, ind, hu):
    """BTC SMA50 canary + BTC MACD hist > 0 (추가 카나리아)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    btc_mh = v(ind,'BTC-USD','macd_hist',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs: return {}, 'Off'
    if pd.isna(btc_mh) or btc_mh <= 0: return {}, 'MACDOff'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# --- 종합 강화 (health + scoring + canary 모두 보강) ---
def crypto_full_enhanced(d, ind, hu):
    """BTC SMA50+RSI>45 canary, health+MACD, scoring+multi bonus."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    btc_rsi = v(ind,'BTC-USD','rsi14',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs: return {}, 'Off'
    if pd.notna(btc_rsi) and btc_rsi <= 40: return {}, 'RSIOff'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and v(ind,c,'price',d) > v(ind,c,'sma30',d)
               and pd.notna(v(ind,c,'mom21',d)) and v(ind,c,'mom21',d) > 0
               and pd.notna(v(ind,c,'vol90',d)) and v(ind,c,'vol90',d) <= 0.10
               and pd.notna(v(ind,c,'macd_hist',d)) and v(ind,c,'macd_hist',d) > 0]
    if not healthy: return {}, 'NoH'
    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2):
            base = s1 + s2
            rsi = v(ind,c,'rsi14',d)
            pctb = v(ind,c,'bb_pctb',d)
            if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
            if pd.notna(pctb) and pctb > 0.5: base += 0.2
            scores[c] = base
    if not scores: return {}, 'NoS'
    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# ===========================================================================
# ETF — Cap Defend Stock + 보조지표 추가 조합
# ===========================================================================

def etf_base(d, ind):
    """Baseline Cap Defend Stock."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
                for t in off if has(ind,t,d)]
        rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r:
                best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def _etf_offensive(ind, off, d, extra_filter=None, score_bonus=None):
    """Helper: offensive selection with optional extra filter and scoring bonus."""
    rows = []
    for t in off:
        if not has(ind,t,d): continue
        mom = v(ind,t,'mom_w',d)
        sh  = v(ind,t,'sharpe126',d)
        if pd.isna(mom) or pd.isna(sh): continue
        if extra_filter and not extra_filter(ind, t, d): continue
        bonus = score_bonus(ind, t, d) if score_bonus else 0
        rows.append((t, mom, sh, bonus))
    return rows


def etf_filter_rsi(d, ind):
    """Cap Defend + RSI filter (skip overbought > 75)."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        def filt(ind, t, d):
            rsi = v(ind,t,'rsi14',d)
            return pd.isna(rsi) or rsi <= 75  # skip overbought
        rows = _etf_offensive(ind, off, d, extra_filter=filt)
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q','B']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def etf_filter_macd(d, ind):
    """Cap Defend + MACD filter (skip if MACD hist < 0)."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        def filt(ind, t, d):
            mh = v(ind,t,'macd_hist',d)
            return pd.isna(mh) or mh > 0
        rows = _etf_offensive(ind, off, d, extra_filter=filt)
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q','B']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def etf_filter_adx(d, ind):
    """Cap Defend + ADX filter (only strong trends ADX > 20)."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        def filt(ind, t, d):
            adx = v(ind,t,'adx14',d)
            pdi = v(ind,t,'plus_di',d)
            mdi = v(ind,t,'minus_di',d)
            if pd.isna(adx) or pd.isna(pdi) or pd.isna(mdi): return True
            return adx > 20 and pdi > mdi
        rows = _etf_offensive(ind, off, d, extra_filter=filt)
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q','B']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def etf_score_rsi(d, ind):
    """Cap Defend + RSI scoring bonus (RSI 40-65 → Mom+0.02)."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        rows = []
        for t in off:
            if not has(ind,t,d): continue
            mom = v(ind,t,'mom_w',d)
            sh  = v(ind,t,'sharpe126',d)
            rsi = v(ind,t,'rsi14',d)
            if pd.isna(mom) or pd.isna(sh): continue
            bonus = 0.02 if pd.notna(rsi) and 40 <= rsi <= 65 else 0
            rows.append((t, mom+bonus, sh+bonus))
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def etf_score_macd(d, ind):
    """Cap Defend + MACD scoring bonus (hist > 0 → Mom+0.02)."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        rows = []
        for t in off:
            if not has(ind,t,d): continue
            mom = v(ind,t,'mom_w',d)
            sh  = v(ind,t,'sharpe126',d)
            mh  = v(ind,t,'macd_hist',d)
            if pd.isna(mom) or pd.isna(sh): continue
            bonus = 0.02 if pd.notna(mh) and mh > 0 else 0
            rows.append((t, mom+bonus, sh+bonus))
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def etf_full_enhanced(d, ind):
    """Cap Defend + RSI filter + MACD filter + ADX filter + RSI score bonus."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']
    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)
    if risk_on:
        rows = []
        for t in off:
            if not has(ind,t,d): continue
            mom = v(ind,t,'mom_w',d)
            sh  = v(ind,t,'sharpe126',d)
            if pd.isna(mom) or pd.isna(sh): continue
            # Filters
            rsi = v(ind,t,'rsi14',d)
            mh  = v(ind,t,'macd_hist',d)
            if pd.notna(rsi) and rsi > 75: continue
            if pd.notna(mh) and mh < 0: continue
            # Bonus
            bonus = 0
            if pd.notna(rsi) and 40 <= rsi <= 65: bonus += 0.01
            if pd.notna(mh) and mh > 0: bonus += 0.01
            rows.append((t, mom+bonus, sh+bonus))
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


# ===========================================================================
# Backtest engines
# ===========================================================================

def run_etf_bt(data, ind, strat_func, start, end, capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))
    cash, hold, hist, rebals, prev_st = capital, {}, [], 0, None
    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date':today,'Value':pv})
        tgt, st = strat_func(today, ind)
        flip = prev_st is not None and st != prev_st
        prev_st = st
        if today in month_ends or flip:
            rebals += 1
            amt = pv*(1-tx); cash = amt; hold = {}
            for t,w in tgt.items():
                p = row.get(t,0) if pd.notna(row.get(t,0)) else 0
                if p > 0: a = amt*w; hold[t] = a/p; cash -= a
    df = pd.DataFrame(hist).set_index('Date'); df.attrs['rebals'] = rebals; return df


def run_crypto_bt(data, ind, strat_func, hu, start, end, capital=10000, tx=0.001, turnover_th=0.30):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))
    cash, hold, hist, rebals = capital, {}, [], 0
    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date':today,'Value':pv})
        tgt, st = strat_func(today, ind, hu)
        is_monthly = today in month_ends
        is_turn = False
        if hold and pv > 0:
            cur_w = {}
            for t,u in hold.items():
                p = row.get(t,0) if pd.notna(row.get(t,0)) else 0
                if p > 0: cur_w[t] = (u*p)/pv
            turn = sum(abs(tgt.get(t,0)-cur_w.get(t,0)) for t in set(cur_w)|set(tgt))/2
            is_turn = turn > turnover_th
        is_eject = False
        if hold:
            for held in hold:
                if held in ind:
                    # Use baseline health for ejection
                    p_ = v(ind,held,'price',today)
                    s30 = v(ind,held,'sma30',today)
                    m21 = v(ind,held,'mom21',today)
                    vol = v(ind,held,'vol90',today)
                    if all(pd.notna(x) for x in [p_,s30,m21,vol]):
                        if p_ <= s30 or m21 <= 0 or vol > 0.10:
                            is_eject = True; break
        if is_monthly or is_turn or is_eject:
            rebals += 1
            amt = pv*(1-tx); cash = amt; hold = {}
            for t,w in tgt.items():
                p = row.get(t,0) if pd.notna(row.get(t,0)) else 0
                if p > 0: a = amt*w; hold[t] = a/p; cash -= a
    df = pd.DataFrame(hist).set_index('Date'); df.attrs['rebals'] = rebals; return df


def metrics(vals):
    if len(vals) < 2: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'wr':0,'final':0}
    days = (vals.index[-1] - vals.index[0]).days
    if days <= 0: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'wr':0,'final':0}
    cagr = (vals.iloc[-1]/vals.iloc[0])**(365.25/days) - 1
    mdd = (vals/vals.cummax()-1).min()
    dr = vals.pct_change().dropna()
    sharpe = (dr.mean()/dr.std())*np.sqrt(252) if dr.std()>0 else 0
    ds = dr[dr<0]
    sortino = (dr.mean()/ds.std())*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
    wr = (dr>0).sum()/len(dr) if len(dr)>0 else 0
    return {'cagr':cagr,'mdd':mdd,'sharpe':sharpe,'sortino':sortino,'wr':wr,'final':vals.iloc[-1]}


def print_table(title, rows_data, start_dates, stab):
    for start in start_dates:
        period = rows_data[rows_data['Start']==start].sort_values('Sharpe', ascending=False)
        print(f"\n{'='*110}")
        print(f"  {title} — START: {start} ~ {END_DATE}")
        print(f"{'='*110}")
        print(f"{'#':>3} {'Strategy':<30} {'Final($)':>11} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'Reb':>5}")
        print(f"{'-'*110}")
        for i, (_,r) in enumerate(period.iterrows(),1):
            mark = ' ***' if i <= 3 else ''
            print(f"{i:>3} {r['Strategy']:<30} {r['Final']:>11,.0f} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['WinRate']:>6.1%} {r['Rebals']:>5}{mark}")

    print(f"\n{'='*110}")
    print(f"  {title} — STABILITY RANKING")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Strategy':<30} {'AvgCAGR':>9} {'AvgMDD':>9} {'AvgSharpe':>10} {'AvgSortino':>11} {'Score':>7}")
    print(f"{'-'*110}")
    for i, (_,r) in enumerate(stab.iterrows(),1):
        mark = ' <-- TOP' if i <= 5 else ''
        print(f"{i:>3} {r['Strategy']:<30} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} {r['Sharpe']:>10.2f} "
              f"{r['Sortino']:>11.2f} {r['Score']:>7.2f}{mark}")
    print(f"{'='*110}")


# ===========================================================================
# Main
# ===========================================================================

ETF_STRATEGIES = {
    'Baseline':             etf_base,
    '+RSI Filter(<75)':     etf_filter_rsi,
    '+MACD Filter(>0)':     etf_filter_macd,
    '+ADX Filter(>20)':     etf_filter_adx,
    '+RSI Score Bonus':     etf_score_rsi,
    '+MACD Score Bonus':    etf_score_macd,
    '+Full Enhanced':       etf_full_enhanced,
}

CRYPTO_STRATEGIES = {
    'Baseline':              crypto_base,
    '+H:MACD>0':             crypto_h_macd,
    '+H:RSI 35-75':          crypto_h_rsi,
    '+H:BB %B>0.3':          crypto_h_bb,
    '+H:MACD+RSI':           crypto_h_macd_rsi,
    '+H:Donchian95%':        crypto_h_donchian,
    '+S:RSI Bonus':          crypto_s_rsi_bonus,
    '+S:MACD Bonus':         crypto_s_macd_bonus,
    '+S:BB Bonus':           crypto_s_bb_bonus,
    '+S:CCI Bonus':          crypto_s_cci_bonus,
    '+S:Multi Bonus':        crypto_s_multi_bonus,
    '+C:BTC RSI>45':         crypto_canary_rsi,
    '+C:BTC MACD>0':         crypto_canary_macd,
    '+Full Enhanced':        crypto_full_enhanced,
}


def main():
    hu = load_universe()

    # --- Load ETF data ---
    print("Loading ETF data...")
    etf_data = load_prices(ETF_TICKERS, '2009-01-01')
    print(f"  {len([c for c in etf_data.columns])} ETFs loaded")

    # --- Load Crypto data ---
    crypto_tickers = collect_crypto_tickers(hu)
    print(f"Loading crypto data ({len(crypto_tickers)} tickers)...")
    crypto_data = load_prices(crypto_tickers, '2017-01-01')
    print(f"  {len([c for c in crypto_data.columns])} tickers loaded")

    # --- Pre-compute ---
    print("Pre-computing ETF indicators...")
    etf_ind = precompute(etf_data)
    print("Pre-computing crypto indicators...")
    crypto_ind = precompute(crypto_data)

    # === ETF Backtests ===
    total_etf = len(ETF_STRATEGIES) * len(ETF_STARTS)
    print(f"\nRunning {len(ETF_STRATEGIES)} ETF strategies x {len(ETF_STARTS)} periods = {total_etf}...")
    etf_rows = []
    n = 0
    for name, func in ETF_STRATEGIES.items():
        for start in ETF_STARTS:
            n += 1
            try:
                res = run_etf_bt(etf_data, etf_ind, func, start, END_DATE)
                m = metrics(res['Value'])
                etf_rows.append({
                    'Strategy': name, 'Start': start, 'Final': m['final'],
                    'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                    'Sortino': m['sortino'], 'WinRate': m['wr'],
                    'Rebals': res.attrs.get('rebals',0),
                })
            except:
                etf_rows.append({'Strategy':name,'Start':start,'Final':10000,
                    'CAGR':0,'MDD':0,'Sharpe':0,'Sortino':0,'WinRate':0,'Rebals':0})
            if n % 10 == 0: print(f"  ETF progress: {n}/{total_etf}")

    etf_df = pd.DataFrame(etf_rows)
    etf_stab = etf_df.groupby('Strategy').agg({
        'CAGR':'mean','MDD':'mean','Sharpe':'mean','Sortino':'mean','Rebals':'mean'
    }).reset_index()
    etf_stab['Score'] = etf_stab['Sharpe']*0.5 + (1+etf_stab['MDD'])*2 + etf_stab['CAGR']
    etf_stab = etf_stab.sort_values('Score', ascending=False)
    print_table("ETF ENHANCED", etf_df, ETF_STARTS, etf_stab)

    # === Crypto Backtests ===
    total_crypto = len(CRYPTO_STRATEGIES) * len(CRYPTO_STARTS)
    print(f"\nRunning {len(CRYPTO_STRATEGIES)} crypto strategies x {len(CRYPTO_STARTS)} periods = {total_crypto}...")
    crypto_rows = []
    n = 0
    for name, func in CRYPTO_STRATEGIES.items():
        for start in CRYPTO_STARTS:
            n += 1
            try:
                res = run_crypto_bt(crypto_data, crypto_ind, func, hu, start, END_DATE)
                m = metrics(res['Value'])
                crypto_rows.append({
                    'Strategy': name, 'Start': start, 'Final': m['final'],
                    'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                    'Sortino': m['sortino'], 'WinRate': m['wr'],
                    'Rebals': res.attrs.get('rebals',0),
                })
            except:
                crypto_rows.append({'Strategy':name,'Start':start,'Final':10000,
                    'CAGR':0,'MDD':0,'Sharpe':0,'Sortino':0,'WinRate':0,'Rebals':0})
            if n % 10 == 0: print(f"  Crypto progress: {n}/{total_crypto}")

    crypto_df = pd.DataFrame(crypto_rows)
    crypto_stab = crypto_df.groupby('Strategy').agg({
        'CAGR':'mean','MDD':'mean','Sharpe':'mean','Sortino':'mean','Rebals':'mean'
    }).reset_index()
    crypto_stab['Score'] = crypto_stab['Sharpe']*0.5 + (1+crypto_stab['MDD'])*2 + crypto_stab['CAGR']
    crypto_stab = crypto_stab.sort_values('Score', ascending=False)
    print_table("CRYPTO ENHANCED", crypto_df, CRYPTO_STARTS, crypto_stab)


if __name__ == '__main__':
    main()
