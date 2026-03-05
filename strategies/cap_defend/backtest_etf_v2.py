"""
ETF Strategy Backtest V2 — New indicator-based strategies
==========================================================
RSI, MACD, Bollinger Bands, ADX, Stochastic, CCI, Donchian, etc.

Usage:
    python3 strategies/cap_defend/backtest_etf_v2.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = './data'

ALL_ETF_TICKERS = [
    'SPY','QQQ','EFA','EEM','VT','VEA','VNQ','VGK',
    'QUAL','MTUM','IQLT','IMTM','IWD','SCZ',
    'IEF','TLT','BND','AGG','BNDX','BIL','SHY','LQD','TIP',
    'GLD','DBC','PDBC','DBMF','KMLM',
    'VWO','HYG','RWX',
]

START_DATES = ['2012-01-01','2014-01-01','2016-01-01','2019-01-01','2021-01-01','2023-01-01']
END_DATE = '2025-12-31'


def load_etf_data():
    buffer_start = '2009-01-01'
    data_dict = {}
    for ticker in ALL_ETF_TICKERS:
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
    print(f"Loaded {len(data_dict)} ETFs, range: {data.index[0].date()} ~ {data.index[-1].date()}")
    return data


# ---------------------------------------------------------------------------
# Pre-compute indicators — classic + new technical indicators
# ---------------------------------------------------------------------------

def precompute(data):
    ind = {}
    for col in data.columns:
        p = data[col]
        dr = p.pct_change()

        d = pd.DataFrame({'price': p})

        # === Classic ===
        d['sma50']  = p.rolling(50).mean()
        d['sma200'] = p.rolling(200).mean()
        d['sma210'] = p.rolling(210).mean()

        d['mom21']  = p / p.shift(21) - 1
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1

        d['m13612w'] = (12*(p/p.shift(21)-1) + 4*(p/p.shift(63)-1)
                        + 2*(p/p.shift(126)-1) + (p/p.shift(252)-1))
        d['mom_avg'] = (d['mom21'] + d['mom63'] + d['mom126'] + d['mom252']) / 4
        d['mom_w']   = 0.5*d['mom63'] + 0.3*d['mom126'] + 0.2*d['mom252']

        d['vol90'] = dr.rolling(90).std()
        rm = dr.rolling(126).mean()
        rs = dr.rolling(126).std()
        d['sharpe126'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)

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

        # === ADX (14-day) ===
        high = p * 1.005  # approximate (no OHLC, use close +/- 0.5%)
        low  = p * 0.995
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = (high - low).rolling(14).mean()
        tr = tr.replace(0, np.nan)
        plus_di  = 100 * plus_dm.rolling(14).mean() / tr
        minus_di = 100 * minus_dm.rolling(14).mean() / tr
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        d['adx14'] = dx.rolling(14).mean()
        d['plus_di'] = plus_di
        d['minus_di'] = minus_di

        # === Stochastic Oscillator (14,3) ===
        low_14  = p.rolling(14).min()
        high_14 = p.rolling(14).max()
        d['stoch_k'] = 100 * (p - low_14) / (high_14 - low_14).replace(0, np.nan)
        d['stoch_d'] = d['stoch_k'].rolling(3).mean()

        # === CCI (20-day) ===
        tp = p  # typical price ≈ close (no OHLC)
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        d['cci20'] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # === Donchian Channel (20-day) ===
        d['don_high'] = p.rolling(20).max()
        d['don_low']  = p.rolling(20).min()

        # === ATR approximation (14-day) ===
        d['atr14'] = (high - low).rolling(14).mean()
        d['atr_pct'] = d['atr14'] / p

        # === Rate of Change (21, 63) ===
        d['roc21'] = d['mom21'] * 100
        d['roc63'] = d['mom63'] * 100

        # === Williams %R (14) ===
        d['willr14'] = -100 * (high_14 - p) / (high_14 - low_14).replace(0, np.nan)

        ind[col] = d
    return ind


def v(ind, ticker, col, date):
    if ticker not in ind: return np.nan
    try: return ind[ticker][col].loc[date]
    except: return np.nan

def has(ind, ticker, date, min_col='mom252'):
    return pd.notna(v(ind, ticker, min_col, date))


# ---------------------------------------------------------------------------
# Cap Defend Stock (baseline for comparison)
# ---------------------------------------------------------------------------

def s00_cap_defend(d, ind):
    """Baseline: Current Cap Defend stock sleeve."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']

    risk_on = all(
        has(ind, c, d) and v(ind, c, 'price', d) > v(ind, c, 'sma200', d)
        for c in canary
    )
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


# ---------------------------------------------------------------------------
# NEW STRATEGIES (보조지표 기반)
# ---------------------------------------------------------------------------

def s01_rsi_rotation(d, ind):
    """RSI Rotation: Canary(VT>SMA200) → Top ETFs by RSI sweet spot (40-70).
    RSI 40-70인 종목만 선택 (추세 중반), 모멘텀으로 순위."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','VNQ','VGK']
    defe = ['IEF','TLT','BIL']

    risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in canary)

    if risk_on:
        cands = []
        for t in off:
            rsi = v(ind,t,'rsi14',d)
            mom = v(ind,t,'mom63',d)
            if pd.isna(rsi) or pd.isna(mom): continue
            if 40 <= rsi <= 70:  # sweet spot: not overbought, not oversold
                cands.append((t, mom))
        cands.sort(key=lambda x: x[1], reverse=True)
        picks = [t for t,_ in cands[:5]]
        if not picks: return {'BIL': 1.0}, 'NoRSI'
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        return {'IEF': 1.0}, 'Off'


def s02_macd_momentum(d, ind):
    """MACD Momentum: Canary(MACD histogram > 0 for SPY) →
    Rank by MACD histogram strength, top 5."""
    spy_hist = v(ind,'SPY','macd_hist',d)
    if pd.isna(spy_hist) or spy_hist <= 0:
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','VGK','QUAL','MTUM']
    scores = []
    for t in off:
        mh = v(ind,t,'macd_hist',d)
        if pd.notna(mh) and mh > 0:
            scores.append((t, mh))
    scores.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in scores[:5]]
    if not picks: return {'BIL': 1.0}, 'NoMACD'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s03_bb_breakout(d, ind):
    """Bollinger Band Breakout: VT canary → Buy ETFs above upper BB (%B > 0.8).
    강한 추세 돌파 전략."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        pctb = v(ind,t,'bb_pctb',d)
        mom = v(ind,t,'mom63',d)
        if pd.notna(pctb) and pd.notna(mom) and pctb > 0.8:
            cands.append((t, mom))
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks:
        # Fallback: top momentum
        rows = [(t, v(ind,t,'mom63',d)) for t in off if pd.notna(v(ind,t,'mom63',d))]
        rows.sort(key=lambda x: x[1], reverse=True)
        picks = [t for t,_ in rows[:3]]
    if not picks: return {'BIL': 1.0}, 'NoBB'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s04_adx_trend(d, ind):
    """ADX Trend Filter: Only invest when ADX > 25 (strong trend).
    ADX+DI방향으로 추세 강도 확인, 모멘텀으로 종목 선택."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        adx = v(ind,t,'adx14',d)
        pdi = v(ind,t,'plus_di',d)
        mdi = v(ind,t,'minus_di',d)
        mom = v(ind,t,'mom63',d)
        if pd.isna(adx) or pd.isna(pdi) or pd.isna(mdi) or pd.isna(mom): continue
        if adx > 25 and pdi > mdi:  # strong uptrend
            cands.append((t, mom))
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks: return {'BIL': 1.0}, 'NoTrend'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s05_stoch_mean_rev(d, ind):
    """Stochastic Mean Reversion: Buy oversold (%K < 20) in uptrend.
    장기 상승추세에서 단기 과매도 매수."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        stk = v(ind,t,'stoch_k',d)
        sma = v(ind,t,'sma200',d)
        price = v(ind,t,'price',d)
        if pd.isna(stk) or pd.isna(sma) or pd.isna(price): continue
        if price > sma and stk < 30:  # oversold in uptrend
            cands.append((t, -stk))  # lower stoch = more oversold = priority
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks:
        # Fallback: top momentum when no oversold
        rows = [(t, v(ind,t,'mom_w',d)) for t in off
                if pd.notna(v(ind,t,'mom_w',d)) and v(ind,t,'mom_w',d) > 0]
        rows.sort(key=lambda x: x[1], reverse=True)
        picks = [t for t,_ in rows[:3]]
    if not picks: return {'BIL': 1.0}, 'NoOversold'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s06_cci_momentum(d, ind):
    """CCI Momentum: CCI > +100 = strong uptrend, rank by CCI.
    CCI 기반 추세 추종."""
    if not (has(ind,'SPY',d) and v(ind,'SPY','cci20',d) and v(ind,'SPY','cci20',d) > 0):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        cci = v(ind,t,'cci20',d)
        if pd.notna(cci) and cci > 0:
            cands.append((t, cci))
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks: return {'BIL': 1.0}, 'NoCCI'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s07_donchian_breakout(d, ind):
    """Donchian Channel Breakout: price at 20d high → trend following.
    돈치안 채널 상단 돌파 종목 선택."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        price = v(ind,t,'price',d)
        don_h = v(ind,t,'don_high',d)
        mom = v(ind,t,'mom63',d)
        if pd.isna(price) or pd.isna(don_h) or pd.isna(mom): continue
        if don_h > 0 and price >= don_h * 0.99:  # near 20d high
            cands.append((t, mom))
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks: return {'BIL': 1.0}, 'NoDon'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s08_multi_indicator(d, ind):
    """Multi-Indicator Composite: 5 indicators → composite z-score.
    RSI(norm) + MACD(sign) + BB%B + Stoch + MOM → 종합 점수."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    scores = []
    for t in off:
        rsi  = v(ind,t,'rsi14',d)
        mh   = v(ind,t,'macd_hist',d)
        pctb = v(ind,t,'bb_pctb',d)
        stk  = v(ind,t,'stoch_k',d)
        mom  = v(ind,t,'mom63',d)
        if any(pd.isna(x) for x in [rsi,mh,pctb,stk,mom]): continue

        # Normalize each to 0-1 range
        rsi_n  = rsi / 100.0
        macd_n = 1.0 if mh > 0 else 0.0
        bb_n   = max(0, min(1, pctb))
        stk_n  = stk / 100.0
        mom_n  = max(0, min(1, mom + 0.5))  # shift so 0 return → 0.5

        composite = 0.2*rsi_n + 0.25*macd_n + 0.15*bb_n + 0.15*stk_n + 0.25*mom_n
        scores.append((t, composite))

    scores.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in scores[:5]]
    if not picks: return {'BIL': 1.0}, 'NoMulti'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s09_atr_sizing(d, ind):
    """ATR Position Sizing: VT canary → top 5 momentum, ATR-based sizing.
    ATR이 작은(변동성 낮은) 종목에 더 많이 투자."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        mom = v(ind,t,'mom63',d)
        atr = v(ind,t,'atr_pct',d)
        if pd.notna(mom) and pd.notna(atr) and mom > 0 and atr > 0:
            cands.append((t, mom, atr))
    cands.sort(key=lambda x: x[1], reverse=True)
    top = cands[:5]
    if not top: return {'BIL': 1.0}, 'NoATR'

    # ATR-inverse sizing
    inv = {t: 1.0/atr for t,_,atr in top}
    tot = sum(inv.values())
    return {t: w/tot for t, w in inv.items()}, 'On'


def s10_willr_contrarian(d, ind):
    """Williams %R Contrarian: Buy when %R < -80 (oversold) in uptrend.
    윌리엄스 %R 역발상 매수."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        wr = v(ind,t,'willr14',d)
        sma = v(ind,t,'sma200',d)
        price = v(ind,t,'price',d)
        if pd.isna(wr) or pd.isna(sma) or pd.isna(price): continue
        if price > sma and wr < -70:  # oversold in uptrend
            cands.append((t, wr))  # more negative = more oversold
    cands.sort(key=lambda x: x[1])  # most oversold first
    picks = [t for t,_ in cands[:5]]
    if not picks:
        # Fallback: momentum
        rows = [(t, v(ind,t,'mom63',d)) for t in off
                if pd.notna(v(ind,t,'mom63',d)) and v(ind,t,'mom63',d) > 0]
        rows.sort(key=lambda x: x[1], reverse=True)
        picks = [t for t,_ in rows[:3]]
    if not picks: return {'BIL': 1.0}, 'NoWR'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s11_golden_cross(d, ind):
    """Golden/Death Cross: SMA50 vs SMA200 cross for each ETF.
    골든크로스 종목만 투자, 모멘텀으로 순위."""
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        sma50  = v(ind,t,'sma50',d)
        sma200 = v(ind,t,'sma200',d)
        mom    = v(ind,t,'mom63',d)
        if pd.isna(sma50) or pd.isna(sma200) or pd.isna(mom): continue
        if sma50 > sma200:  # golden cross
            cands.append((t, mom))
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks: return {'IEF': 1.0}, 'Death'
    return {t: 1.0/len(picks) for t in picks}, 'On'


def s12_rsi_macd_combo(d, ind):
    """RSI+MACD Combo: RSI(30-70) AND MACD hist > 0 → rank by momentum.
    두 지표 동시 확인으로 필터링 강화."""
    if not (has(ind,'VT',d) and v(ind,'VT','price',d) > v(ind,'VT','sma200',d)):
        return {'IEF': 1.0}, 'Off'

    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','VNQ','QUAL','MTUM','VGK']
    cands = []
    for t in off:
        rsi = v(ind,t,'rsi14',d)
        mh  = v(ind,t,'macd_hist',d)
        mom = v(ind,t,'mom63',d)
        if pd.isna(rsi) or pd.isna(mh) or pd.isna(mom): continue
        if 30 <= rsi <= 70 and mh > 0:
            cands.append((t, mom))
    cands.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in cands[:5]]
    if not picks: return {'BIL': 1.0}, 'NoCombo'
    return {t: 1.0/len(picks) for t in picks}, 'On'


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_bt(data, ind, strat_func, start, end, capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))

    cash = capital
    hold = {}
    hist = []
    rebals = 0
    prev_st = None

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u * (row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        tgt, st = strat_func(today, ind)
        flip = prev_st is not None and st != prev_st
        prev_st = st

        if today in month_ends or flip:
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
    'Cap Defend (기준)':   s00_cap_defend,
    'RSI Rotation':       s01_rsi_rotation,
    'MACD Momentum':      s02_macd_momentum,
    'BB Breakout':        s03_bb_breakout,
    'ADX Trend':          s04_adx_trend,
    'Stoch MeanRev':      s05_stoch_mean_rev,
    'CCI Momentum':       s06_cci_momentum,
    'Donchian Break':     s07_donchian_breakout,
    'Multi-Indicator':    s08_multi_indicator,
    'ATR Sizing':         s09_atr_sizing,
    'WillR Contrarian':   s10_willr_contrarian,
    'Golden Cross':       s11_golden_cross,
    'RSI+MACD Combo':     s12_rsi_macd_combo,
}


def main():
    data = load_etf_data()
    print("Pre-computing indicators (classic + RSI/MACD/BB/ADX/Stoch/CCI/Donchian/ATR/WillR)...")
    ind = precompute(data)

    total = len(STRATEGIES) * len(START_DATES)
    print(f"\nRunning {len(STRATEGIES)} strategies x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for name, func in STRATEGIES.items():
        for start in START_DATES:
            n += 1
            try:
                res = run_bt(data, ind, func, start, END_DATE)
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
        bm_spy = metrics(data['SPY'].loc[start:END_DATE].dropna())

        print(f"\n{'='*105}")
        print(f"  ETF V2 STRATEGIES — START: {start} ~ {END_DATE}")
        print(f"{'='*105}")
        print(f"{'#':>3} {'Strategy':<22} {'Final($)':>11} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'Reb':>5}")
        print(f"{'-'*105}")

        for i, (_, r) in enumerate(period.iterrows(), 1):
            mark = ' ***' if i <= 3 else ''
            print(f"{i:>3} {r['Strategy']:<22} {r['Final']:>11,.0f} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['WinRate']:>6.1%} {r['Rebals']:>5}{mark}")

        print(f"{'-'*105}")
        print(f"    {'SPY B&H':<22} {'':>11} {bm_spy['cagr']:>7.1%} {bm_spy['mdd']:>7.1%} {bm_spy['sharpe']:>8.2f}")

    # Stability ranking
    stab = df.groupby('Strategy').agg({
        'CAGR':'mean','MDD':'mean','Sharpe':'mean','Sortino':'mean','Rebals':'mean'
    }).reset_index()
    stab['Score'] = stab['Sharpe']*0.5 + (1+stab['MDD'])*2 + stab['CAGR']
    stab = stab.sort_values('Score', ascending=False)

    print(f"\n{'='*105}")
    print(f"  STABILITY RANKING — Average across all start dates (2012~2025)")
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
