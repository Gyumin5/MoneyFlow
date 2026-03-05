"""
Crypto Strategy Backtest V2 — New indicator-based strategies
=============================================================
RSI, MACD, Bollinger Bands, ADX, Stochastic, CCI, etc.

Usage:
    python3 strategies/cap_defend/backtest_crypto_v2.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = './data'
UNIVERSE_FILE = './data/historical_universe.json'

STABLECOINS = {'USDT','USDC','BUSD','DAI','UST','TUSD','PAX','GUSD',
               'FRAX','LUSD','MIM','USDN','FDUSD','USDS','PYUSD','USDE'}

START_DATES = ['2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01']
END_DATE = '2025-12-31'


def load_universe():
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE) as f:
            return json.load(f)
    return {}


def collect_tickers(hist_universe):
    tickers = {'BTC-USD'}
    for symbols in hist_universe.values():
        for s in symbols:
            t = s if s.endswith('-USD') else f"{s}-USD"
            sym = t.replace('-USD', '')
            if sym not in STABLECOINS:
                tickers.add(t)
    return tickers


def load_data(tickers):
    buffer_start = '2017-01-01'
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
    data = pd.DataFrame(data_dict).reindex(idx).ffill()
    print(f"Loaded {len(data_dict)} tickers, range: {data.index[0].date()} ~ {data.index[-1].date()}")
    return data


# ---------------------------------------------------------------------------
# Pre-compute ALL indicators
# ---------------------------------------------------------------------------

def precompute(data):
    ind = {}
    for col in data.columns:
        p = data[col]
        dr = p.pct_change()

        d = pd.DataFrame({'price': p})

        # === Classic ===
        d['sma30']  = p.rolling(30).mean()
        d['sma50']  = p.rolling(50).mean()

        d['mom7']   = p / p.shift(7) - 1
        d['mom21']  = p / p.shift(21) - 1
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1

        d['vol90']  = dr.rolling(90).std()

        for w in [126, 252]:
            rm = dr.rolling(w).mean()
            rs = dr.rolling(w).std()
            d[f'sharpe{w}'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)

        d['m13612w'] = (12*(p/p.shift(21)-1) + 4*(p/p.shift(63)-1)
                        + 2*(p/p.shift(126)-1) + (p/p.shift(252)-1))

        # Health check (baseline)
        d['h_base'] = (p > d['sma30']) & (d['mom21'] > 0) & (d['vol90'] <= 0.10)

        # === RSI (14-day) ===
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs_rsi = gain / loss.replace(0, np.nan)
        d['rsi14'] = 100 - (100 / (1 + rs_rsi))

        # === MACD (12,26,9) ===
        ema12 = p.ewm(span=12).mean()
        ema26 = p.ewm(span=26).mean()
        d['macd_line'] = ema12 - ema26
        d['macd_signal'] = d['macd_line'].ewm(span=9).mean()
        d['macd_hist'] = d['macd_line'] - d['macd_signal']

        # === Bollinger Bands (20, 2) ===
        bb_sma = p.rolling(20).mean()
        bb_std = p.rolling(20).std()
        d['bb_upper'] = bb_sma + 2*bb_std
        d['bb_lower'] = bb_sma - 2*bb_std
        d['bb_pctb'] = (p - d['bb_lower']) / (d['bb_upper'] - d['bb_lower']).replace(0, np.nan)

        # === Stochastic Oscillator (14,3) ===
        low_14  = p.rolling(14).min()
        high_14 = p.rolling(14).max()
        d['stoch_k'] = 100 * (p - low_14) / (high_14 - low_14).replace(0, np.nan)

        # === CCI (20-day) ===
        tp_sma = p.rolling(20).mean()
        tp_mad = p.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        d['cci20'] = (p - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # === Donchian (20) ===
        d['don_high'] = p.rolling(20).max()

        # === Williams %R (14) ===
        d['willr14'] = -100 * (high_14 - p) / (high_14 - low_14).replace(0, np.nan)

        # --- New health checks using new indicators ---
        # RSI health: RSI > 40 AND < 80 AND above SMA30
        d['h_rsi'] = (p > d['sma30']) & (d['rsi14'] > 40) & (d['rsi14'] < 80) & (d['vol90'] <= 0.10)
        # MACD health: MACD hist > 0 AND above SMA30
        d['h_macd'] = (p > d['sma30']) & (d['macd_hist'] > 0) & (d['vol90'] <= 0.10)
        # Composite health: RSI(40-75) AND MACD>0 AND above SMA30
        d['h_combo'] = (p > d['sma30']) & (d['rsi14'] > 40) & (d['rsi14'] < 75) & (d['macd_hist'] > 0) & (d['vol90'] <= 0.10)

        ind[col] = d
    return ind


def get_universe(date, hist_universe, top_n=50):
    key = date.strftime("%Y-%m") + "-01"
    symbols = hist_universe.get(key, [])
    if not symbols:
        avail = sorted([k for k in hist_universe if k <= key], reverse=True)
        if avail:
            symbols = hist_universe[avail[0]]
    final = []
    for s in symbols:
        t = s if s.endswith('-USD') else f"{s}-USD"
        if t.replace('-USD','') not in STABLECOINS:
            final.append(t)
            if len(final) >= top_n:
                break
    return final


def v(ind, ticker, col, date):
    if ticker not in ind: return np.nan
    try: return ind[ticker][col].loc[date]
    except: return np.nan


def ivw(ind, picks, date, vol_col='vol90'):
    vols = {}
    for t in picks:
        vol = v(ind, t, vol_col, date)
        vols[t] = vol if pd.notna(vol) and vol > 0 else 0.001
    inv = {t: 1.0/vol for t, vol in vols.items()}
    tot = sum(inv.values())
    return {t: val/tot for t, val in inv.items()} if tot > 0 else {t: 1.0/len(picks) for t in picks}


def ew(picks):
    return {t: 1.0/len(picks) for t in picks} if picks else {}


# ---------------------------------------------------------------------------
# Baseline (for comparison)
# ---------------------------------------------------------------------------

def c00_baseline(d, ind, hu):
    """Current baseline: BTC MA50, h_base, Sharpe, IVW Top5."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# ---------------------------------------------------------------------------
# NEW STRATEGIES (보조지표 기반)
# ---------------------------------------------------------------------------

def c01_rsi_scoring(d, ind, hu):
    """RSI Scoring: BTC canary → health_base → rank by RSI (50-70 sweet spot).
    RSI가 50~70인 코인 = 상승 추세 진행 중, RSI 값으로 순위."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        rsi = v(ind,c,'rsi14',d)
        if pd.notna(rsi) and 40 <= rsi <= 75:
            # Distance from 60 (optimal mid-trend) → prefer mid-range RSI
            scores[c] = -abs(rsi - 60)  # closer to 60 = higher score
    if not scores: return {}, 'NoRSI'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c02_macd_health(d, ind, hu):
    """MACD Health: h_macd (MACD hist > 0) instead of mom21 for health.
    MACD 히스토그램으로 건강 체크, Sharpe로 순위."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_macd',d)) and v(ind,c,'h_macd',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c03_rsi_health(d, ind, hu):
    """RSI Health: h_rsi (RSI 40-80) instead of mom21 for health.
    RSI로 건강 체크, Sharpe로 순위."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_rsi',d)) and v(ind,c,'h_rsi',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c04_combo_health(d, ind, hu):
    """Combo Health: RSI(40-75) AND MACD>0 AND SMA30 AND vol<=10%.
    가장 엄격한 다중 지표 건강 체크."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_combo',d)) and v(ind,c,'h_combo',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c05_macd_scoring(d, ind, hu):
    """MACD Scoring: health_base → rank by MACD histogram.
    MACD 히스토그램 크기로 종목 순위 (추세 강도)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        mh = v(ind,c,'macd_hist',d)
        if pd.notna(mh) and mh > 0:
            scores[c] = mh
    if not scores: return {}, 'NoMACD'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c06_bb_scoring(d, ind, hu):
    """BB %B Scoring: health_base → rank by Bollinger %B.
    BB 상단에 가까울수록 높은 점수 (추세 돌파)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        pctb = v(ind,c,'bb_pctb',d)
        if pd.notna(pctb) and pctb > 0.5:  # above middle band
            scores[c] = pctb
    if not scores: return {}, 'NoBB'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c07_stoch_scoring(d, ind, hu):
    """Stochastic Scoring: health_base → rank by Stochastic %K.
    %K가 적당히 높은 (50-80) 코인 선택."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        stk = v(ind,c,'stoch_k',d)
        if pd.notna(stk) and 40 <= stk <= 85:
            scores[c] = stk
    if not scores: return {}, 'NoStoch'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c08_cci_scoring(d, ind, hu):
    """CCI Scoring: health_base → rank by CCI (positive = uptrend).
    CCI 양수 중 높은 순 선택."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        cci = v(ind,c,'cci20',d)
        if pd.notna(cci) and cci > 0:
            scores[c] = cci
    if not scores: return {}, 'NoCCI'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c09_donchian_breakout(d, ind, hu):
    """Donchian Breakout: health_base → price near 20d high.
    20일 최고가 근접 코인 = 돌파 추세."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        price = v(ind,c,'price',d)
        don_h = v(ind,c,'don_high',d)
        if pd.notna(price) and pd.notna(don_h) and don_h > 0:
            ratio = price / don_h  # closer to 1.0 = near breakout
            if ratio > 0.95:
                scores[c] = ratio
    if not scores: return {}, 'NoDon'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c10_multi_indicator(d, ind, hu):
    """Multi-Indicator: health_base → composite score from RSI+MACD+BB+Sharpe.
    4개 지표 종합 점수."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        rsi  = v(ind,c,'rsi14',d)
        mh   = v(ind,c,'macd_hist',d)
        pctb = v(ind,c,'bb_pctb',d)
        s126 = v(ind,c,'sharpe126',d)
        if any(pd.isna(x) for x in [rsi,mh,pctb,s126]): continue

        # Normalize and combine
        rsi_n  = rsi / 100.0
        macd_n = 1.0 if mh > 0 else 0.0
        bb_n   = max(0, min(1, pctb))
        sh_n   = max(-1, min(3, s126)) / 3.0  # normalize sharpe to ~0-1

        composite = 0.25*rsi_n + 0.25*macd_n + 0.2*bb_n + 0.3*sh_n
        scores[c] = composite

    if not scores: return {}, 'NoMulti'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c11_rsi_canary(d, ind, hu):
    """RSI Canary: BTC RSI > 45 (instead of SMA50), baseline scoring.
    BTC의 RSI로 시장 상태 판단."""
    btc_rsi = v(ind,'BTC-USD','rsi14',d)
    if pd.isna(btc_rsi) or btc_rsi <= 45:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c12_macd_canary(d, ind, hu):
    """MACD Canary: BTC MACD histogram > 0 (instead of SMA50).
    BTC의 MACD로 시장 상태 판단."""
    btc_mh = v(ind,'BTC-USD','macd_hist',d)
    if pd.isna(btc_mh) or btc_mh <= 0:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# ---------------------------------------------------------------------------
# Backtest engine (same as V1: health ejection + turnover trigger)
# ---------------------------------------------------------------------------

def run_bt(data, ind, strat_func, hu, start, end, capital=10000, tx=0.001,
           health_col='h_base', turnover_th=0.30):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))

    cash = capital
    hold = {}
    hist = []
    rebals = 0

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u * (row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        tgt, st = strat_func(today, ind, hu)

        is_monthly = today in month_ends

        is_turn = False
        if hold and pv > 0:
            cur_w = {}
            for t, u in hold.items():
                p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                if p > 0: cur_w[t] = (u*p)/pv
            all_t = set(cur_w) | set(tgt)
            turn = sum(abs(tgt.get(t,0)-cur_w.get(t,0)) for t in all_t)/2
            is_turn = turn > turnover_th

        is_eject = False
        if hold and health_col:
            for held in hold:
                if held in ind:
                    hv = v(ind, held, health_col, today)
                    if pd.notna(hv) and not hv:
                        is_eject = True
                        break

        if is_monthly or is_turn or is_eject:
            rebals += 1
            amt = pv * (1 - tx)
            cash = amt
            hold = {}
            for t, w in tgt.items():
                p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                if p > 0:
                    a = amt * w
                    hold[t] = a / p
                    cash -= a

    df = pd.DataFrame(hist).set_index('Date')
    df.attrs['rebals'] = rebals
    return df


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = {
    'Baseline (현재)':       (c00_baseline,       'h_base',  0.30),
    'RSI Scoring':          (c01_rsi_scoring,    'h_base',  0.30),
    'MACD Health':          (c02_macd_health,    'h_macd',  0.30),
    'RSI Health':           (c03_rsi_health,     'h_rsi',   0.30),
    'Combo Health':         (c04_combo_health,   'h_combo', 0.30),
    'MACD Scoring':         (c05_macd_scoring,   'h_base',  0.30),
    'BB %B Scoring':        (c06_bb_scoring,     'h_base',  0.30),
    'Stoch Scoring':        (c07_stoch_scoring,  'h_base',  0.30),
    'CCI Scoring':          (c08_cci_scoring,    'h_base',  0.30),
    'Donchian Break':       (c09_donchian_breakout,'h_base',0.30),
    'Multi-Indicator':      (c10_multi_indicator,'h_base',  0.30),
    'RSI Canary':           (c11_rsi_canary,     'h_base',  0.30),
    'MACD Canary':          (c12_macd_canary,    'h_base',  0.30),
}


def main():
    hu = load_universe()
    tickers = collect_tickers(hu)
    print(f"Total coin tickers: {len(tickers)}")
    data = load_data(tickers)
    print("Pre-computing indicators (classic + RSI/MACD/BB/Stoch/CCI/Donchian/WillR)...")
    ind = precompute(data)
    print(f"Pre-computed indicators for {len(ind)} tickers.")

    total = len(STRATEGIES) * len(START_DATES)
    print(f"\nRunning {len(STRATEGIES)} strategies x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for name, (func, hcol, turnover) in STRATEGIES.items():
        for start in START_DATES:
            n += 1
            try:
                res = run_bt(data, ind, func, hu, start, END_DATE,
                             health_col=hcol, turnover_th=turnover)
                m = metrics(res['Value'])
                rows.append({
                    'Strategy': name, 'Start': start, 'Final': m['final'],
                    'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                    'Sortino': m['sortino'], 'WinRate': m['wr'],
                    'Rebals': res.attrs.get('rebals', 0),
                })
            except Exception as e:
                rows.append({
                    'Strategy': name, 'Start': start, 'Final': 10000,
                    'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'WinRate': 0, 'Rebals': 0,
                })
            if n % 10 == 0:
                print(f"  Progress: {n}/{total}")

    df = pd.DataFrame(rows)

    for start in START_DATES:
        period = df[df['Start'] == start].sort_values('Sharpe', ascending=False)

        print(f"\n{'='*105}")
        print(f"  CRYPTO V2 STRATEGIES — START: {start} ~ {END_DATE}")
        print(f"{'='*105}")
        print(f"{'#':>3} {'Strategy':<22} {'Final($)':>11} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'Reb':>5}")
        print(f"{'-'*105}")

        for i, (_, r) in enumerate(period.iterrows(), 1):
            mark = ' ***' if i <= 3 else ''
            print(f"{i:>3} {r['Strategy']:<22} {r['Final']:>11,.0f} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['WinRate']:>6.1%} {r['Rebals']:>5}{mark}")

    # Stability ranking
    stab = df.groupby('Strategy').agg({
        'CAGR':'mean','MDD':'mean','Sharpe':'mean','Sortino':'mean','Rebals':'mean'
    }).reset_index()
    stab['Score'] = stab['Sharpe']*0.5 + (1+stab['MDD'])*2 + stab['CAGR']
    stab = stab.sort_values('Score', ascending=False)

    print(f"\n{'='*105}")
    print(f"  CRYPTO V2 STABILITY RANKING — Average across all start dates")
    print(f"{'='*105}")
    print(f"{'#':>3} {'Strategy':<22} {'AvgCAGR':>9} {'AvgMDD':>9} {'AvgSharpe':>10} {'AvgSortino':>11} {'Score':>7}")
    print(f"{'-'*105}")
    for i, (_, r) in enumerate(stab.iterrows(), 1):
        mark = ' <-- TOP' if i <= 5 else ''
        print(f"{i:>3} {r['Strategy']:<22} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} {r['Sharpe']:>10.2f} "
              f"{r['Sortino']:>11.2f} {r['Score']:>7.2f}{mark}")
    print(f"{'='*105}")


if __name__ == '__main__':
    main()
