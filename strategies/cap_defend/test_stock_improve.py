#!/usr/bin/env python3
"""
Stock ETF Strategy Comprehensive Improvement Test
===================================================
Tests ~90 configurations across 9 categories:
  A: Universe variations (offensive ETF set)
  B: Defense variations (defensive selection method)
  C: Health filters (individual ETF health gate)
  D: Tranche rebalancing (timing risk diversification)
  E: DD Exit (drawdown-based individual exit)
  F: Crash Breaker (SPY drop / VIX spike → cash)
  G: Canary variations (signal assets & SMA period)
  H: Selection method (momentum/sharpe/composite ranking)
  I: Weighting (EW / inverse-vol / rank-decay)

Usage:
  python3 test_stock_improve.py              # all categories
  python3 test_stock_improve.py --cat A B C  # specific categories
  python3 test_stock_improve.py --workers 12 # limit workers
"""

import os, sys, argparse, warnings
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '..', 'data')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'stock_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── Universe Constants ──────────────────────────────────────────
OFF_CURRENT = ('SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM')
OFF_SECTOR  = ('XLK','XLV','XLF','XLE','XLI','XLY','XLC','XLB','XLU','XLRE')
OFF_FACTOR  = ('VLUE','SIZE','USMV','DGRO')
OFF_INTL    = ('VWO','IEMG','SCZ','ACWX')
OFF_CORE    = ('SPY','QQQ','VT','EFA','EEM','GLD')
OFF_MEGA    = ('SPY','QQQ','VT')  # minimal

DEF_CURRENT  = ('IEF','BIL','BNDX','GLD','PDBC')
DEF_EXPANDED = ('IEF','BIL','BNDX','GLD','PDBC','TLT','SHY','AGG','TIP')
DEF_BOND_HEAVY = ('IEF','TLT','SHY','BIL','AGG','BNDX','TIP')

OFF_GROWTH  = ('VUG','IWF','SCHG')     # US large-cap growth
OFF_LEV2    = ('SSO','QLD')            # 2x leveraged (SPY, QQQ)
OFF_TECH    = ('XLK','SOXX','IGV')     # Tech/Semi/Software

# ─── Global Diversification Constants ─────────────────────────────
OFF_GLOB_CUR  = OFF_CURRENT + ('VGK','EWJ','ACWX','VWO')        # current + global (16)
OFF_REGION6   = ('SPY','VGK','EWJ','EEM','GLD','PDBC')           # 6: US/Europe/Japan/EM/Gold/Commodity
OFF_GLOB9     = ('SPY','QQQ','VGK','EWJ','EEM','VWO','ACWX','GLD','PDBC')  # 9: wide global
OFF_COUNTRY   = ('SPY','EWC','VGK','EWG','EWU','EWJ','EWA','EWY','EWT','EEM','EWZ')  # 11: country rotation
OFF_GLOB_BAL  = ('SPY','QQQ','VGK','EWJ','EWA','EEM','VWO','IQLT','IMTM','GLD','PDBC')  # 11: balanced
OFF_EXUS      = ('VGK','EWJ','EEM','VWO','ACWX','GLD')           # 6: no US equities
OFF_ALLWEATH  = ('SPY','VGK','EWJ','EEM','GLD','PDBC','VNQ','VNQI')  # 8: all-weather
DEF_GLOBAL    = ('IEF','BWX','BNDX','GLD','BIL')                 # + intl treasury
DEF_GLOB_WIDE = ('IEF','BWX','BNDX','GLD','BIL','SHY','EMB')    # + EM bonds

# ─── Country/Region ETF Constants ────────────────────────────────
# Individual country ETFs for targeted addition to R9
OFF_R9_CHN   = OFF_GLOB9 + ('FXI',)          # +China
OFF_R9_IND   = OFF_GLOB9 + ('INDA',)         # +India
OFF_R9_KOR   = OFF_GLOB9 + ('EWY',)          # +Korea
OFF_R9_TWN   = OFF_GLOB9 + ('EWT',)          # +Taiwan
OFF_R9_AUS   = OFF_GLOB9 + ('EWA',)          # +Australia
OFF_R9_BRZ   = OFF_GLOB9 + ('EWZ',)          # +Brazil
OFF_R9_ASIA  = OFF_GLOB9 + ('FXI','INDA','EWY','EWT')  # +4 Asia tigers
OFF_R9_BRIC  = OFF_GLOB9 + ('FXI','INDA','EWZ')        # +BRIC additions (R already has EEM)
OFF_R9_DEV   = OFF_GLOB9 + ('EWA','EWC','EWU','EWG')   # +developed: Aus/Can/UK/Ger
OFF_R9_ALL   = OFF_GLOB9 + ('FXI','INDA','EWY','EWT','EWA','EWZ','EWC','EWU','EWG')  # +9 countries (18 total)

CANARY_DEFAULT = ('VT','EEM')

# All tickers we might need
ALL_TICKERS = sorted(set(
    OFF_CURRENT + OFF_SECTOR + OFF_FACTOR + OFF_INTL + OFF_CORE + OFF_MEGA +
    OFF_GROWTH + OFF_LEV2 + OFF_TECH +
    OFF_COUNTRY + OFF_ALLWEATH +
    DEF_CURRENT + DEF_EXPANDED + DEF_BOND_HEAVY + DEF_GLOBAL + DEF_GLOB_WIDE +
    CANARY_DEFAULT + ('HYG','IEF','LQD','TLT','SHY','AGG','TIP','VIX','BND',
                       'VGK','VNQ','VNQI','BWX','EMB','INDA',
                       'FXI','EPI','EWS','EWU','EWG')
))


# ─── Params ──────────────────────────────────────────────────────
@dataclass
class SP:
    """Stock strategy parameters."""
    # Universe
    offensive: tuple = OFF_CURRENT
    defensive: tuple = DEF_CURRENT
    canary_assets: tuple = CANARY_DEFAULT

    # Canary
    canary_sma: int = 200
    canary_hyst: float = 0.0  # 0=no hysteresis (baseline)
    canary_extra: str = 'none'  # 'hyg_ief_50','hyg_ief_100','3asset_50','3asset_100'
    canary_band: float = 0.0   # 0=binary, >0=multi-step (e.g., 0.02=2% band → 50/50 in middle)
    canary_type: str = 'sma'   # 'sma' or 'ema'

    # Health filter for offensive
    health: str = 'none'  # 'sma100','sma150','sma200','mom21','mom42','mom63','mom126','mom21_63','mom63_vol'

    # Defense selection
    defense: str = 'top1'  # 'top1','top2','top3','sma_gate','fixed','health_gate'
    defense_sma: int = 100
    def_mom_period: int = 126  # defense momentum period (21, 63, 126, 252)
    def_use_sharpe: bool = False  # use Sharpe instead of momentum for defense

    # Selection
    select: str = 'mom3_sh3'  # 'momN_shN','momN','shN','compN','comp_sortN'
    n_mom: int = 3
    n_sh: int = 3
    mom_style: str = 'default'  # 'default'=50/30/20, '6m','eq','rh','lh','dual'
    sharpe_lookback: int = 126  # 63, 126, 252

    # Weighting
    weight: str = 'ew'  # 'ew','inv_vol','rank_decay'

    # Rebalancing
    tranche_days: tuple = (1,)  # (1,) = monthly day-1 only

    # DD Exit
    dd_lookback: int = 0   # 0=disabled
    dd_thresh: float = -0.15

    # Crash Breaker
    crash: str = 'none'  # 'spyN_Md','vixN_Md'
    crash_thresh: float = 0.0
    crash_cool: int = 0

    # Rebalancing triggers (coin-inspired)
    flip_rebal: bool = False       # PFD: rebal immediately on canary flip
    flip_delay: int = 0            # PFD delay: 0=immediate, 3=wait 3 days, 5=PFD5
    health_daily_exit: bool = False # Daily: sell ETF if health fails (Mom21<0 or SMA200 break)
    health_exit_type: str = 'mom21' # 'mom21','sma200','sma100'
    rank_buffer: int = 0           # Rank buffer: don't swap if still in top (N_picks + buffer)
    adaptive_fill: bool = False    # If healthy < n_picks, fill remainder with defense/cash

    # Partial allocation (breadth momentum from DAA paper)
    partial_alloc: bool = False    # Enable partial allocation
    # When True: each canary asset votes independently
    #   both ON → 100% offense, 1 ON → 50% offense + 50% defense, both OFF → 100% defense

    # Core-Satellite
    core_ratio: float = 0.0        # 0=disabled, 0.6=60% SPY core + 40% strategy satellite
    core_ticker: str = 'SPY'

    # Biweekly rebalancing
    biweekly: bool = False         # Rebalance every ~2 weeks instead of monthly

    # General
    tx_cost: float = 0.001
    start: str = '2017-01-01'
    end: str = '2026-12-31'
    capital: float = 10000.0

    # Internal (for tranche sub-runs)
    _anchor: int = 1
    _n_tranches: int = 1


# ─── Data Loading ────────────────────────────────────────────────
def load_prices(tickers, start='2014-01-01'):
    """Load adjusted close prices for all tickers.
    Priority: stock_cache (auto-adjusted) → main data Adj_Close → yfinance download."""
    prices = {}
    for t in tickers:
        series = None

        # 1. Stock cache (already adjusted via yfinance auto_adjust=True)
        f2 = os.path.join(CACHE_DIR, f'{t}.csv')
        if os.path.exists(f2):
            try:
                s = pd.read_csv(f2, index_col=0, parse_dates=True).squeeze()
                if isinstance(s, pd.Series) and len(s) > 100:
                    age = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(f2), unit='s')).days
                    if age < 7:
                        series = s
            except Exception:
                pass

        # 2. Main data dir (OHLCV format — use Adj_Close for adjusted prices)
        if series is None:
            f1 = os.path.join(DATA_DIR, f'{t}.csv')
            if os.path.exists(f1):
                try:
                    df = pd.read_csv(f1, index_col='Date', parse_dates=True)
                    if 'Adj_Close' in df.columns:
                        series = df['Adj_Close'].dropna()
                    elif 'Close' in df.columns:
                        series = df['Close'].dropna()
                except Exception:
                    pass

        # 3. yfinance download (auto_adjust=True → Close IS adjusted)
        if series is None and t != 'VIX':
            try:
                df = yf.download(t, start=start, progress=False, auto_adjust=True)
                if df is not None and len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        series = df['Close'][t]
                    else:
                        series = df['Close']
                    if isinstance(series, pd.Series):
                        series.to_csv(os.path.join(CACHE_DIR, f'{t}.csv'))
            except Exception as e:
                print(f"  ⚠️ {t}: {e}")

        # VIX: use Close (it's an index, not a stock)
        if series is None and t == 'VIX':
            f1 = os.path.join(DATA_DIR, f'{t}.csv')
            if os.path.exists(f1):
                try:
                    df = pd.read_csv(f1, index_col='Date', parse_dates=True)
                    series = df['Close'].dropna()
                except Exception:
                    pass

        if series is not None and len(series) > 50:
            prices[t] = series.astype(float)

    return prices


# ─── Precompute Indicators ───────────────────────────────────────
def precompute(prices):
    """Precompute all indicators for each ticker."""
    ind = {}
    for t, s in prices.items():
        df = pd.DataFrame({'price': s})
        df['ret'] = df['price'].pct_change()
        for w in (100, 150, 200, 250):
            df[f'sma{w}'] = df['price'].rolling(w).mean()
        # EMA for canary
        df['ema200'] = df['price'].ewm(span=200, adjust=False).mean()
        for n in (21, 42, 63, 126, 252):
            df[f'mom{n}'] = df['price'] / df['price'].shift(n) - 1
        # Weighted momentum variants
        df['wmom'] = 0.5 * df['mom63'] + 0.3 * df['mom126'] + 0.2 * df['mom252']  # default
        df['wmom_6m'] = df['mom126']
        df['wmom_eq'] = (df['mom63'] + df['mom126'] + df['mom252']) / 3
        df['wmom_rh'] = 0.7 * df['mom63'] + 0.2 * df['mom126'] + 0.1 * df['mom252']
        df['wmom_lh'] = 0.2 * df['mom63'] + 0.3 * df['mom126'] + 0.5 * df['mom252']
        df['wmom_dual'] = 0.5 * df['mom63'] + 0.5 * df['mom252']
        df['wmom_vl'] = 0.1 * df['mom63'] + 0.2 * df['mom126'] + 0.7 * df['mom252']  # very long
        df['wmom_12m'] = df['mom252']  # pure 12M
        # Sharpe & Sortino (multiple lookbacks)
        for lb in (63, 126, 252):
            rlb = df['ret'].rolling(lb)
            df[f'sharpe{lb}'] = rlb.mean() / rlb.std() * np.sqrt(252)
            neg = df['ret'].clip(upper=0)
            neg_std = neg.rolling(lb).std()
            df[f'sortino{lb}'] = rlb.mean() / neg_std * np.sqrt(252)
        # Vol (90d daily std)
        df['vol90'] = df['ret'].rolling(90).std()
        # Peak for DD calc
        for lb in (60, 90, 120, 252):
            df[f'peak{lb}'] = df['price'].rolling(lb).max()
        ind[t] = df
    return ind


# ─── Helpers ─────────────────────────────────────────────────────
def get_val(ind, ticker, date, col):
    """Safe indicator lookup."""
    df = ind.get(ticker)
    if df is None:
        return np.nan
    mask = df.index <= date
    if mask.sum() == 0:
        return np.nan
    return df.loc[mask, col].iloc[-1]

def get_price(ind, ticker, date):
    return get_val(ind, ticker, date, 'price')


# ─── Canary ──────────────────────────────────────────────────────
def resolve_canary(params, ind, date, prev_on):
    """Resolve canary signal → risk_on bool."""
    sma_col = f'ema{params.canary_sma}' if params.canary_type == 'ema' else f'sma{params.canary_sma}'

    # Base: all canary assets > SMA
    all_above = True
    for t in params.canary_assets:
        p = get_val(ind, t, date, 'price')
        sma = get_val(ind, t, date, sma_col)
        if np.isnan(p) or np.isnan(sma):
            return False
        if params.canary_hyst > 0 and prev_on is not None:
            if prev_on:
                if p < sma * (1 - params.canary_hyst):
                    all_above = False; break
            else:
                if p <= sma * (1 + params.canary_hyst):
                    all_above = False; break
        else:
            if p <= sma:
                all_above = False; break

    if not all_above:
        risk_on = False
    else:
        risk_on = True

    # Extra canary: credit spread or 3-asset canary
    if params.canary_extra != 'none' and risk_on:
        n = 50 if '50' in params.canary_extra else 100

        if params.canary_extra.startswith('3asset'):
            # 3-asset canary: HYG/IEF ratio AND LQD > SMA(n)
            # Must pass BOTH: credit spread + investment grade bond momentum
            hyg_p = get_val(ind, 'HYG', date, 'price')
            ief_p = get_val(ind, 'IEF', date, 'price')
            lqd_p = get_val(ind, 'LQD', date, 'price')
            lqd_sma = get_val(ind, 'LQD', date, 'sma100')

            # Check HYG/IEF ratio
            if not (np.isnan(hyg_p) or np.isnan(ief_p) or ief_p <= 0):
                ratio = hyg_p / ief_p
                hyg_df = ind.get('HYG')
                ief_df = ind.get('IEF')
                if hyg_df is not None and ief_df is not None:
                    common = hyg_df.index.intersection(ief_df.index)
                    common = common[common <= date]
                    if len(common) >= n:
                        h = hyg_df.loc[common[-n:], 'price']
                        e = ief_df.loc[common[-n:], 'price']
                        valid = e > 0
                        if valid.all():
                            ratio_sma = (h / e).mean()
                            if ratio < ratio_sma:
                                risk_on = False

            # Check LQD > SMA (investment-grade bond health)
            if risk_on and not (np.isnan(lqd_p) or np.isnan(lqd_sma)):
                if lqd_p < lqd_sma:
                    risk_on = False

        else:
            # hyg_ief: simple HYG/IEF credit spread ratio
            hyg_p = get_val(ind, 'HYG', date, 'price')
            ief_p = get_val(ind, 'IEF', date, 'price')
            if not (np.isnan(hyg_p) or np.isnan(ief_p) or ief_p <= 0):
                ratio = hyg_p / ief_p
                hyg_df = ind.get('HYG')
                ief_df = ind.get('IEF')
                if hyg_df is not None and ief_df is not None:
                    common = hyg_df.index.intersection(ief_df.index)
                    common = common[common <= date]
                    if len(common) >= n:
                        h = hyg_df.loc[common[-n:], 'price']
                        e = ief_df.loc[common[-n:], 'price']
                        valid = e > 0
                        if valid.all():
                            ratio_sma = (h / e).mean()
                            if ratio < ratio_sma:
                                risk_on = False

    return risk_on


def resolve_canary_breadth(params, ind, date, prev_on):
    """Resolve canary → breadth count (how many canary assets are above SMA).
    Returns (n_above, n_total) for partial allocation."""
    sma_col = f'sma{params.canary_sma}'
    n_above = 0
    n_total = len(params.canary_assets)

    for t in params.canary_assets:
        p = get_val(ind, t, date, 'price')
        sma = get_val(ind, t, date, sma_col)
        if np.isnan(p) or np.isnan(sma):
            continue
        above = False
        if params.canary_hyst > 0 and prev_on is not None:
            if prev_on:
                above = p >= sma * (1 - params.canary_hyst)
            else:
                above = p > sma * (1 + params.canary_hyst)
        else:
            above = p > sma
        if above:
            n_above += 1
    return n_above, n_total


# ─── Health Filter ───────────────────────────────────────────────
def filter_healthy(params, ind, date, tickers):
    """Filter offensive tickers by health criteria."""
    if params.health == 'none':
        return list(tickers)

    healthy = []
    for t in tickers:
        if params.health.startswith('sma'):
            period = int(params.health[3:])
            p = get_val(ind, t, date, 'price')
            sma = get_val(ind, t, date, f'sma{period}')
            if np.isnan(p) or np.isnan(sma):
                continue
            if p > sma:
                healthy.append(t)
        elif params.health == 'mom21':
            if get_val(ind, t, date, 'mom21') > 0:
                healthy.append(t)
        elif params.health == 'mom42':
            if get_val(ind, t, date, 'mom42') > 0:
                healthy.append(t)
        elif params.health == 'mom63':
            if get_val(ind, t, date, 'mom63') > 0:
                healthy.append(t)
        elif params.health == 'mom126':
            if get_val(ind, t, date, 'mom126') > 0:
                healthy.append(t)
        elif params.health == 'mom21_63':
            m21 = get_val(ind, t, date, 'mom21')
            m63 = get_val(ind, t, date, 'mom63')
            if not np.isnan(m21) and not np.isnan(m63) and m21 > 0 and m63 > 0:
                healthy.append(t)
        elif params.health == 'mom63_vol':
            m63 = get_val(ind, t, date, 'mom63')
            vol = get_val(ind, t, date, 'vol90')
            if not np.isnan(m63) and not np.isnan(vol) and m63 > 0 and vol < 0.025:
                healthy.append(t)
        else:
            healthy.append(t)
    return healthy


# ─── Offensive Selection ─────────────────────────────────────────
def select_offensive(params, ind, date, candidates):
    """Select and weight offensive ETFs."""
    if not candidates:
        return {}

    # Choose momentum and sharpe columns based on params
    wmom_col = 'wmom' if params.mom_style == 'default' else f'wmom_{params.mom_style}'
    sh_col = f'sharpe{params.sharpe_lookback}'
    sort_col = f'sortino{params.sharpe_lookback}'

    scores = []
    for t in candidates:
        wmom = get_val(ind, t, date, wmom_col)
        sh = get_val(ind, t, date, sh_col)
        sort = get_val(ind, t, date, sort_col)
        vol = get_val(ind, t, date, 'vol90')
        if np.isnan(wmom) or np.isnan(sh):
            continue
        scores.append({'t': t, 'wmom': wmom, 'sh': sh,
                       'sort': sort if not np.isnan(sort) else sh,
                       'vol': vol if not np.isnan(vol) else 0.01})
    if not scores:
        return {}

    df = pd.DataFrame(scores).set_index('t')

    sel = params.select
    if sel == 'mom3_sh3':
        top_m = df.nlargest(params.n_mom, 'wmom').index.tolist()
        top_s = df.nlargest(params.n_sh, 'sh').index.tolist()
        picks = list(dict.fromkeys(top_m + top_s))  # preserve order, dedupe
    elif sel == 'mom5_sh5':
        top_m = df.nlargest(5, 'wmom').index.tolist()
        top_s = df.nlargest(5, 'sh').index.tolist()
        picks = list(dict.fromkeys(top_m + top_s))
    elif sel.startswith('mom'):
        n = int(sel[3:])
        picks = df.nlargest(n, 'wmom').index.tolist()
    elif sel.startswith('sh'):
        n = int(sel[2:])
        picks = df.nlargest(n, 'sh').index.tolist()
    elif sel.startswith('comp_sort'):
        n = int(sel[9:])
        df['r_m'] = df['wmom'].rank(ascending=False)
        df['r_s'] = df['sh'].rank(ascending=False)
        df['r_sort'] = df['sort'].rank(ascending=False)
        df['score'] = df['r_m'] + df['r_s'] + df['r_sort']
        picks = df.nsmallest(n, 'score').index.tolist()
    elif sel.startswith('comp'):
        n = int(sel[4:])
        df['r_m'] = df['wmom'].rank(ascending=False)
        df['r_s'] = df['sh'].rank(ascending=False)
        df['score'] = df['r_m'] + df['r_s']
        picks = df.nsmallest(n, 'score').index.tolist()
    else:
        picks = df.nlargest(params.n_mom, 'wmom').index.tolist()

    if not picks:
        return {}

    # Weighting
    if params.weight == 'inv_vol':
        vols = {t: max(df.loc[t, 'vol'], 0.001) for t in picks if t in df.index}
        inv = {t: 1.0/v for t, v in vols.items()}
        s = sum(inv.values())
        return {t: w/s for t, w in inv.items()} if s > 0 else {t: 1.0/len(picks) for t in picks}
    elif params.weight == 'rank_decay':
        decay = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
        n = len(picks)
        w = decay[:n]
        s = sum(w)
        return {picks[i]: w[i]/s for i in range(n)}
    else:  # ew
        return {t: 1.0/len(picks) for t in picks}


# ─── Extended Selection (for rank buffer) ────────────────────────
def select_offensive_extended(params, ind, date, candidates, n):
    """Select top-N offensive ETFs (extended range for rank buffer)."""
    if not candidates:
        return {}
    scores = []
    for t in candidates:
        wmom = get_val(ind, t, date, 'wmom')
        sh = get_val(ind, t, date, 'sharpe126')
        if np.isnan(wmom) or np.isnan(sh):
            continue
        scores.append({'t': t, 'wmom': wmom, 'sh': sh})
    if not scores:
        return {}
    df = pd.DataFrame(scores).set_index('t')
    df['r_m'] = df['wmom'].rank(ascending=False)
    df['r_s'] = df['sh'].rank(ascending=False)
    df['score'] = df['r_m'] + df['r_s']
    picks = df.nsmallest(n, 'score').index.tolist()
    return {t: 1.0/len(picks) for t in picks}


# ─── Defensive Selection ─────────────────────────────────────────
def select_defensive(params, ind, date):
    """Select defensive assets."""
    score_col = f'sharpe{params.def_mom_period}' if params.def_use_sharpe else f'mom{params.def_mom_period}'
    results = []
    for t in params.defensive:
        m = get_val(ind, t, date, score_col)
        p = get_val(ind, t, date, 'price')
        if np.isnan(m):
            continue
        results.append((t, m, p))

    if not results:
        return {'Cash': 1.0}

    if params.defense == 'fixed':
        basket = {'IEF': 0.4, 'GLD': 0.3, 'BIL': 0.3}
        # only include available
        avail = {t: w for t, w in basket.items() if t in [r[0] for r in results] or t == 'BIL'}
        s = sum(avail.values())
        return {t: w/s for t, w in avail.items()} if s > 0 else {'Cash': 1.0}

    # Sort by 6M return
    results.sort(key=lambda x: x[1], reverse=True)

    if params.defense == 'top1':
        best = results[0]
        return {best[0]: 1.0} if best[1] > 0 else {'Cash': 1.0}

    elif params.defense == 'top2':
        picks = [r for r in results[:2] if r[1] > 0]
        if not picks:
            return {'Cash': 1.0}
        return {r[0]: 1.0/len(picks) for r in picks}

    elif params.defense == 'top3':
        picks = [r for r in results[:3] if r[1] > 0]
        if not picks:
            return {'Cash': 1.0}
        return {r[0]: 1.0/len(picks) for r in picks}

    elif params.defense == 'sma_gate':
        best = results[0]
        if best[1] <= 0:
            return {'Cash': 1.0}
        sma = get_val(ind, best[0], date, f'sma{params.defense_sma}')
        if np.isnan(sma) or best[2] <= sma:
            return {'Cash': 1.0}  # fails SMA gate → cash
        return {best[0]: 1.0}

    elif params.defense == 'health_gate':
        for r in results:
            if r[1] <= 0:
                continue
            m63 = get_val(ind, r[0], date, 'mom63')
            if not np.isnan(m63) and m63 > 0:
                return {r[0]: 1.0}
        return {'Cash': 1.0}

    return {'Cash': 1.0}


# ─── Crash Breaker ───────────────────────────────────────────────
def check_crash(params, ind, date):
    """Check if crash breaker triggers. Returns True if should go to cash."""
    if params.crash == 'none':
        return False
    if params.crash.startswith('spy'):
        spy_ret = get_val(ind, 'SPY', date, 'ret')
        if not np.isnan(spy_ret) and spy_ret <= -params.crash_thresh:
            return True
    elif params.crash.startswith('vix'):
        vix = get_val(ind, 'VIX', date, 'price')
        if not np.isnan(vix) and vix >= params.crash_thresh:
            return True
    return False


# ─── DD Exit ─────────────────────────────────────────────────────
def check_dd_exit(params, ind, date, holdings):
    """Check individual ETF drawdown exit. Returns list of tickers to sell."""
    if params.dd_lookback <= 0 or not holdings:
        return []
    exits = []
    peak_col = f'peak{params.dd_lookback}'
    for t in list(holdings.keys()):
        p = get_val(ind, t, date, 'price')
        pk = get_val(ind, t, date, peak_col)
        if np.isnan(p) or np.isnan(pk) or pk <= 0:
            continue
        dd = p / pk - 1
        if dd <= params.dd_thresh:
            exits.append(t)
    return exits


# ─── Backtest Engine ─────────────────────────────────────────────
def run_backtest(prices_dict, ind, params):
    """Run a single backtest with given parameters."""
    spy = ind.get('SPY')
    if spy is None:
        return None

    dates = spy.index[(spy.index >= params.start) & (spy.index <= params.end)]
    if len(dates) < 2:
        return None

    # Determine rebalance day for this tranche
    anchor_day = params._anchor

    holdings = {}  # {ticker: shares}
    cash = params.capital
    prev_month = None
    prev_risk_on = None
    mode = 'Init'
    history = []
    rebal_count = 0
    flip_count = 0
    crash_cooldown = 0

    for date in dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        # Portfolio value
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        # Daily: crash breaker check
        if crash_cooldown > 0:
            crash_cooldown -= 1
        elif check_crash(params, ind, date):
            # Sell everything to cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)
            holdings = {}
            crash_cooldown = params.crash_cool
            pv = cash

        # Daily: DD exit check
        if crash_cooldown <= 0:
            dd_exits = check_dd_exit(params, ind, date, holdings)
            for t in dd_exits:
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)

        # Determine if rebalance day
        is_rebal_day = False
        if is_first:
            is_rebal_day = True
        elif is_month_change:
            # Check if this is near the anchor day
            if date.day >= anchor_day:
                is_rebal_day = True
        elif prev_month == cur_month and not is_month_change:
            # Check if we missed anchor day (first trading day >= anchor)
            pass

        # More precise: track if anchor day was hit this month
        if not is_first and not is_rebal_day and cur_month == prev_month:
            pass  # not a rebal day

        # Simplified: rebalance on first trading day of month >= anchor_day
        if is_month_change and date.day >= anchor_day:
            is_rebal_day = True
        # Handle case where anchor_day > 1 and month just changed
        # We track a flag per month
        if not is_first and not is_rebal_day:
            # Check if this is the first day >= anchor in current month
            # by checking if previous day was in different month or day < anchor
            pass

        if crash_cooldown > 0:
            is_rebal_day = False  # no rebalancing during crash cooldown

        if is_rebal_day:
            # Recalc PV
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

            risk_on = resolve_canary(params, ind, date, prev_risk_on)
            if prev_risk_on is not None and prev_risk_on != risk_on:
                flip_count += 1

            if risk_on:
                candidates = filter_healthy(params, ind, date, params.offensive)
                weights = select_offensive(params, ind, date, candidates)
                if not weights:
                    weights = {'Cash': 1.0}
                mode = f'Off({len(weights)})'
            else:
                weights = select_defensive(params, ind, date)
                best_def = next((k for k in weights if k != 'Cash'), 'Cash')
                mode = f'Def({best_def})'

            # Sell all
            if holdings:
                sell_val = 0
                for t, shares in holdings.items():
                    p = get_price(ind, t, date)
                    if not np.isnan(p):
                        sell_val += shares * p
                cash = (sell_val + cash) * (1 - params.tx_cost)
                holdings = {}
            pv = cash

            # Buy new
            for t, w in weights.items():
                if t == 'Cash':
                    continue
                p = get_price(ind, t, date)
                if np.isnan(p) or p <= 0:
                    continue
                alloc = pv * w
                holdings[t] = alloc / p
                cash -= alloc

            prev_risk_on = risk_on
            rebal_count += 1

        # Record
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        history.append({'Date': date, 'Value': pv})
        prev_month = cur_month

    if not history:
        return None

    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    df.attrs['flip_count'] = flip_count
    return df


# ─── Improved Rebalance Day Tracking ────────────────────────────
# Override the simplified run_backtest with proper anchor-day tracking
def run_bt(prices_dict, ind, params):
    """Run backtest with proper anchor-day monthly rebalancing."""
    spy = ind.get('SPY')
    if spy is None:
        return None

    dates = spy.index[(spy.index >= params.start) & (spy.index <= params.end)]
    if len(dates) < 2:
        return None

    anchor = params._anchor
    holdings = {}
    cash = params.capital
    prev_month = None
    prev_risk_on = None
    history = []
    rebal_count = 0
    flip_count = 0
    crash_cooldown = 0
    rebalanced_this_month = False
    flip_pending_days = -1  # for PFD delay
    prev_trading_date = None  # for 1-day signal delay (look-ahead bias fix)
    biweekly_second_done = False  # track 2nd biweekly rebal within month

    for date in dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        if is_month_change:
            rebalanced_this_month = False
            biweekly_second_done = False

        # Portfolio value
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        # Daily: crash breaker
        crash_just_ended = False
        if crash_cooldown > 0:
            crash_cooldown -= 1
            if crash_cooldown == 0:
                if check_crash(params, ind, date):
                    crash_cooldown = params.crash_cool
                else:
                    crash_just_ended = True
        elif check_crash(params, ind, date):
            sold_any = False
            for t in list(holdings.keys()):
                if t in ('IEF','BIL','BNDX','GLD','PDBC','TLT','SHY','AGG','TIP','LQD'):
                    continue
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)
                    sold_any = True
            if sold_any or not holdings:
                crash_cooldown = params.crash_cool
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

        # Daily: DD exit
        dd_triggered = False
        if crash_cooldown <= 0:
            dd_exits = check_dd_exit(params, ind, date, holdings)
            if dd_exits:
                dd_triggered = True
            for t in dd_exits:
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)

        # Daily: health exit (sell individual ETFs that fail health check)
        if params.health_daily_exit and crash_cooldown <= 0 and holdings:
            for t in list(holdings.keys()):
                if t in params.defensive or t in ('Cash','IEF','BIL','BNDX','GLD','PDBC','TLT','SHY','AGG','TIP','LQD'):
                    continue  # skip defensive assets
                sell_it = False
                if params.health_exit_type == 'mom21':
                    m = get_val(ind, t, date, 'mom21')
                    sell_it = not np.isnan(m) and m < 0
                elif params.health_exit_type == 'sma200':
                    p_val = get_val(ind, t, date, 'price')
                    sma = get_val(ind, t, date, 'sma200')
                    sell_it = not (np.isnan(p_val) or np.isnan(sma)) and p_val < sma
                elif params.health_exit_type == 'sma100':
                    p_val = get_val(ind, t, date, 'price')
                    sma = get_val(ind, t, date, 'sma100')
                    sell_it = not (np.isnan(p_val) or np.isnan(sma)) and p_val < sma
                if sell_it:
                    p = get_price(ind, t, date)
                    shares = holdings.pop(t, 0)
                    if shares > 0 and not np.isnan(p):
                        cash += shares * p * (1 - params.tx_cost)
                        dd_triggered = True  # force rebal to redistribute

        # Daily: check canary flip for PFD trigger
        if params.flip_rebal and not is_first and crash_cooldown <= 0:
            daily_risk_on = resolve_canary(params, ind, date, prev_risk_on)
            if prev_risk_on is not None and daily_risk_on != prev_risk_on:
                if params.flip_delay <= 0:
                    flip_pending_days = 0  # immediate
                else:
                    flip_pending_days = params.flip_delay
            if flip_pending_days > 0:
                flip_pending_days -= 1
            elif flip_pending_days == 0:
                flip_pending_days = -1  # reset

        # Rebalance trigger
        is_rebal = False
        if is_first:
            is_rebal = True
        elif params.biweekly:
            # Biweekly: rebal on day >= anchor AND day >= anchor+14
            if not rebalanced_this_month and date.day >= anchor:
                is_rebal = True
            elif rebalanced_this_month and not biweekly_second_done and date.day >= anchor + 14:
                is_rebal = True
                biweekly_second_done = True
        elif not rebalanced_this_month and date.day >= anchor:
            is_rebal = True

        # PFD: canary flip trigger
        if params.flip_rebal and flip_pending_days == 0:
            is_rebal = True
            flip_pending_days = -1

        # Force re-entry after crash cooldown ends
        if crash_just_ended and not holdings:
            is_rebal = True
        # DD exit or health exit: force rebalance to redistribute
        if dd_triggered:
            is_rebal = True

        if crash_cooldown > 0:
            is_rebal = False

        if is_rebal:
            rebalanced_this_month = True
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

            # Use previous day's data for signals to avoid look-ahead bias
            sig_date = prev_trading_date if prev_trading_date is not None else date

            # Determine offense/defense split
            if params.canary_band > 0:
                # Multi-step canary: 3-step with band
                sma_col = f'ema{params.canary_sma}' if params.canary_type == 'ema' else f'sma{params.canary_sma}'
                all_above = True
                all_below = True
                for t_c in params.canary_assets:
                    p_c = get_val(ind, t_c, sig_date, 'price')
                    sma_c = get_val(ind, t_c, sig_date, sma_col)
                    if np.isnan(p_c) or np.isnan(sma_c):
                        all_above = False; continue
                    if p_c <= sma_c * (1 + params.canary_band):
                        all_above = False
                    if p_c >= sma_c * (1 - params.canary_band):
                        all_below = False
                if all_above:
                    off_frac = 1.0
                elif all_below:
                    off_frac = 0.0
                else:
                    off_frac = 0.5
                risk_on = off_frac > 0
                if prev_risk_on is not None and prev_risk_on != risk_on:
                    flip_count += 1
            elif params.partial_alloc:
                n_above, n_total = resolve_canary_breadth(params, ind, sig_date, prev_risk_on)
                off_frac = n_above / n_total if n_total > 0 else 0
                risk_on = off_frac > 0
                if prev_risk_on is not None and prev_risk_on != risk_on:
                    flip_count += 1
            else:
                risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)
                if prev_risk_on is not None and prev_risk_on != risk_on:
                    flip_count += 1
                off_frac = 1.0 if risk_on else 0.0

            if off_frac > 0:
                candidates = filter_healthy(params, ind, sig_date, params.offensive)
                off_weights = select_offensive(params, ind, sig_date, candidates)
                if not off_weights:
                    off_weights = {}
            else:
                off_weights = {}

            if off_frac < 1.0:
                def_weights = select_defensive(params, ind, sig_date)
            else:
                def_weights = {}

            # Combine offense + defense with partial allocation
            weights = {}
            for t, w in off_weights.items():
                weights[t] = w * off_frac
            def_frac = 1.0 - off_frac
            for t, w in def_weights.items():
                if t in weights:
                    weights[t] += w * def_frac
                else:
                    weights[t] = w * def_frac

            if not weights:
                weights = {'Cash': 1.0}

            # Rank buffer: keep existing holdings if still in top (N + buffer)
            if params.rank_buffer > 0 and risk_on and holdings:
                held_tickers = set(holdings.keys())
                new_tickers = set(weights.keys()) - {'Cash'}
                if held_tickers != new_tickers:
                    # Check if old holdings are still in extended top
                    all_candidates = filter_healthy(params, ind, sig_date, params.offensive)
                    if all_candidates:
                        ext_weights = select_offensive_extended(
                            params, ind, sig_date, all_candidates,
                            params.n_mom + params.rank_buffer)
                        ext_tickers = set(ext_weights.keys()) - {'Cash'}
                        # Keep old if all still in extended range
                        if held_tickers <= ext_tickers:
                            weights = {t: 1.0/len(held_tickers) for t in held_tickers}

            # Sell all
            if holdings:
                sell_val = 0
                for t, shares in holdings.items():
                    p = get_price(ind, t, date)
                    if not np.isnan(p):
                        sell_val += shares * p
                cash += sell_val * (1 - params.tx_cost)
                holdings = {}
            pv = cash

            # Buy (with buy-side transaction cost)
            buy_budget = pv / (1 + params.tx_cost) if params.tx_cost > 0 else pv
            for t, w in weights.items():
                if t == 'Cash':
                    continue
                p = get_price(ind, t, date)
                if np.isnan(p) or p <= 0:
                    continue
                alloc = buy_budget * w
                holdings[t] = alloc / p
                cash -= alloc * (1 + params.tx_cost)

            prev_risk_on = risk_on
            rebal_count += 1

        # Final PV
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        history.append({'Date': date, 'Value': pv})
        prev_trading_date = date
        prev_month = cur_month

    if not history:
        return None

    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    df.attrs['flip_count'] = flip_count
    return df


# ─── Metrics ─────────────────────────────────────────────────────
def metrics(df):
    """Compute performance metrics from equity curve."""
    if df is None or len(df) < 2:
        return None
    v = df['Value']
    days = (v.index[-1] - v.index[0]).days
    yrs = days / 365.25
    if yrs <= 0:
        return None
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1/yrs) - 1
    peak = v.cummax()
    mdd = (v / peak - 1).min()
    dr = v.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = dr.mean() / down.std() * np.sqrt(252) if len(down) > 1 and down.std() > 0 else sharpe
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    total_ret = v.iloc[-1] / v.iloc[0] - 1
    return {
        'CAGR': cagr, 'MDD': mdd, 'Sharpe': sharpe, 'Sortino': sortino,
        'Calmar': calmar, 'Final': v.iloc[-1],
        'Rebals': df.attrs.get('rebal_count', 0),
        'Flips': df.attrs.get('flip_count', 0),
        'TotalRet': total_ret,
        'Days': days,
    }


# ─── Multiprocessing ─────────────────────────────────────────────
_g_prices = None
_g_ind = None

def _init(prices, ind):
    global _g_prices, _g_ind
    _g_prices = prices
    _g_ind = ind

def _run_one(params):
    """Run single or tranche backtest."""
    if params._n_tranches <= 1:
        df = run_bt(_g_prices, _g_ind, params)
        if df is None:
            return None
        # Core-Satellite: blend with SPY B&H
        if params.core_ratio > 0:
            spy_df = _g_ind.get(params.core_ticker)
            if spy_df is not None:
                spy_prices = spy_df['price']
                common = df.index.intersection(spy_prices.index)
                if len(common) > 1:
                    sat_v = df.loc[common, 'Value']
                    spy_v = spy_prices.loc[common]
                    # Normalize both to start at capital
                    sat_norm = sat_v / sat_v.iloc[0] * params.capital
                    spy_norm = spy_v / spy_v.iloc[0] * params.capital
                    # Blend
                    cr = params.core_ratio
                    blended = spy_norm * cr + sat_norm * (1 - cr)
                    df = pd.DataFrame({'Value': blended})
                    df.attrs['rebal_count'] = df.attrs.get('rebal_count', 0)
                    df.attrs['flip_count'] = df.attrs.get('flip_count', 0)
        return metrics(df)
    else:
        # Run each tranche
        dfs = []
        for day in params.tranche_days:
            tp = SP(**{k: v for k, v in params.__dict__.items()})
            tp._anchor = day
            tp._n_tranches = 1
            tp.capital = params.capital / len(params.tranche_days)
            df = run_bt(_g_prices, _g_ind, tp)
            if df is not None:
                dfs.append(df)
        if not dfs:
            return None
        # Merge: sum values
        merged = dfs[0].copy()
        for df in dfs[1:]:
            common = merged.index.intersection(df.index)
            merged.loc[common, 'Value'] += df.loc[common, 'Value']
        total_rebals = sum(d.attrs.get('rebal_count', 0) for d in dfs)
        total_flips = max(d.attrs.get('flip_count', 0) for d in dfs)
        merged.attrs['rebal_count'] = total_rebals
        merged.attrs['flip_count'] = total_flips
        return metrics(merged)


# ─── Config Generators ───────────────────────────────────────────
def base(**kw):
    """Create SP with overrides."""
    return SP(**kw)

def gen_A():
    """A: Universe variations."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    B('A0:Baseline(12)',     offensive=OFF_CURRENT)
    B('A1:Core(6)',          offensive=OFF_CORE)
    B('A2:Mega(3)',          offensive=OFF_MEGA)
    B('A3:+Sector(22)',      offensive=OFF_CURRENT + OFF_SECTOR)
    B('A4:+Factor(16)',      offensive=OFF_CURRENT + OFF_FACTOR)
    B('A5:+Intl(16)',        offensive=OFF_CURRENT + OFF_INTL)
    B('A6:Cur+Sec+Fac(26)',  offensive=OFF_CURRENT + OFF_SECTOR + OFF_FACTOR)
    B('A7:All(30)',          offensive=OFF_CURRENT + OFF_SECTOR + OFF_FACTOR + OFF_INTL)
    B('A8:Sector only(10)',  offensive=OFF_SECTOR)
    B('A9:Factor only(4)',   offensive=OFF_FACTOR)
    B('A10:Core+Factor(10)', offensive=OFF_CORE + OFF_FACTOR)
    B('A11:NoGLD/PDBC(10)',  offensive=('SPY','QQQ','EFA','EEM','VT','VEA','QUAL','MTUM','IQLT','IMTM'))
    return cfgs

def gen_B():
    """B: Defense variations."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    B('B0:Top1(current)',    defense='top1')
    B('B1:Top2 EW',         defense='top2')
    B('B2:Top3 EW',         defense='top3')
    B('B3:SMA100 gate',     defense='sma_gate', defense_sma=100)
    B('B4:SMA150 gate',     defense='sma_gate', defense_sma=150)
    B('B5:Fixed basket',    defense='fixed')
    B('B6:Health gate',     defense='health_gate')
    B('B7:Exp Top1',        defense='top1', defensive=DEF_EXPANDED)
    B('B8:Exp Top2',        defense='top2', defensive=DEF_EXPANDED)
    B('B9:Exp SMA gate',    defense='sma_gate', defensive=DEF_EXPANDED, defense_sma=100)
    B('B10:BondH Top1',     defense='top1', defensive=DEF_BOND_HEAVY)
    B('B11:BondH Top2',     defense='top2', defensive=DEF_BOND_HEAVY)
    return cfgs

def gen_C():
    """C: Health filters."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    B('C0:None(baseline)',   health='none')
    B('C1:SMA100',           health='sma100')
    B('C2:SMA150',           health='sma150')
    B('C3:SMA200',           health='sma200')
    B('C4:Mom21>0',          health='mom21')
    B('C5:Mom63>0',          health='mom63')
    B('C6:Mom126>0',         health='mom126')
    B('C7:Mom21+63>0',       health='mom21_63')
    B('C8:Mom63+Vol<2.5%',   health='mom63_vol')
    return cfgs

def gen_D():
    """D: Tranche rebalancing."""
    cfgs = []
    cfgs.append(('D0:Monthly(1d)',  base(tranche_days=(1,), _n_tranches=1)))
    cfgs.append(('D1:2-tranche',    base(tranche_days=(1, 15), _n_tranches=2)))
    cfgs.append(('D2:3-tranche',    base(tranche_days=(1, 10, 20), _n_tranches=3)))
    cfgs.append(('D3:5-tranche',    base(tranche_days=(1, 5, 10, 15, 20), _n_tranches=5)))
    cfgs.append(('D4:Mid-month',    base(tranche_days=(15,), _n_tranches=1)))
    cfgs.append(('D5:Late-month',   base(tranche_days=(20,), _n_tranches=1)))
    return cfgs

def gen_E():
    """E: DD Exit."""
    cfgs = []
    cfgs.append(('E0:None', base()))
    for lb in (60, 90, 120, 252):
        for th in (-0.10, -0.12, -0.15, -0.18, -0.20):
            name = f'E:{lb}d/{th:.0%}'
            cfgs.append((name, base(dd_lookback=lb, dd_thresh=th)))
    return cfgs

def gen_F():
    """F: Crash Breaker."""
    cfgs = []
    cfgs.append(('F0:None', base()))
    for th, label in [(0.03,'3%'), (0.04,'4%'), (0.05,'5%')]:
        for cool in (3, 5):
            cfgs.append((f'F:SPY-{label}/{cool}d',
                          base(crash=f'spy', crash_thresh=th, crash_cool=cool)))
    for th, label in [(30,'30'), (35,'35'), (40,'40')]:
        for cool in (3, 5):
            cfgs.append((f'F:VIX>{label}/{cool}d',
                          base(crash='vix', crash_thresh=th, crash_cool=cool)))
    return cfgs

def gen_G():
    """G: Canary variations."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    B('G0:VT+EEM SMA200',  canary_sma=200)
    B('G1:SMA150',          canary_sma=150)
    B('G2:SMA250',          canary_sma=250)
    B('G3:Hyst 0.3%',       canary_sma=200, canary_hyst=0.003)
    B('G4:Hyst 0.5%',       canary_sma=200, canary_hyst=0.005)
    B('G5:Hyst 0.7%',       canary_sma=200, canary_hyst=0.007)
    B('G6:Hyst 1.0%',       canary_sma=200, canary_hyst=0.01)
    B('G7:Hyst 1.5%',       canary_sma=200, canary_hyst=0.015)
    B('G8:Hyst 2.0%',       canary_sma=200, canary_hyst=0.02)
    B('G9:Hyst 3.0%',       canary_sma=200, canary_hyst=0.03)
    return cfgs

def gen_H():
    """H: Selection methods."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    B('H0:Mom3+Sh3(base)',  select='mom3_sh3')
    B('H1:Mom5+Sh5',        select='mom5_sh5')
    B('H2:Mom3 only',       select='mom3')
    B('H3:Mom5 only',       select='mom5')
    B('H4:Sh3 only',        select='sh3')
    B('H5:Sh5 only',        select='sh5')
    B('H6:Composite3',      select='comp3')
    B('H7:Composite5',      select='comp5')
    B('H8:Comp+Sort3',      select='comp_sort3')
    B('H9:Comp+Sort5',      select='comp_sort5')
    return cfgs

def gen_I():
    """I: Weighting."""
    cfgs = []
    cfgs.append(('I0:EW(baseline)', base(weight='ew')))
    cfgs.append(('I1:Inv Vol',      base(weight='inv_vol')))
    cfgs.append(('I2:Rank Decay',   base(weight='rank_decay')))
    return cfgs

def gen_J():
    """J: Stack test — combine top improvements."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # Baseline
    B('J0:Baseline', )

    # Single improvements (reference)
    # J1:HYG/IEF — REMOVED (HYG data insufficient: only 342 days, 2024-08~2025-12)
    B('J2:Mom21 only',         health='mom21')
    B('J3:Fixed Def only',     defense='fixed')
    B('J4:Mega(3) only',       offensive=OFF_MEGA)
    B('J5:Hyst0.5% only',      canary_hyst=0.005)

    # 2-way combos (J6,J7,J9,J11 REMOVED — HYG)
    B('J8:Mom21+FixDef',       health='mom21', defense='fixed')
    B('J10:Mom21+Mega',        health='mom21', offensive=OFF_MEGA)
    B('J12:Mom21+Hyst0.5%',    health='mom21', canary_hyst=0.005)

    # 3-way combos (J13-J16 REMOVED — HYG)
    B('J17:Mom21+FixDef+Mega',  health='mom21', defense='fixed', offensive=OFF_MEGA)

    # J18-J24 REMOVED — all used HYG/3asset canary

    # Mom4 combos (best selection from K tests)
    B('J25:Mom4',              select='mom4')
    B('J26:Mom4+Mom21',        select='mom4', health='mom21')
    B('J27:Mom4+Mom21+Top2',   select='mom4', health='mom21', defense='top2')
    B('J28:Mom4+Mom21+Fix',    select='mom4', health='mom21', defense='fixed')
    B('J29:Mom4+Mom21+T2+PFD', select='mom4', health='mom21', defense='top2',
                                flip_rebal=True, flip_delay=0)
    B('J30:Mom4+Mom21+Fix+PFD', select='mom4', health='mom21', defense='fixed',
                                 flip_rebal=True, flip_delay=0)

    # Mom3 combos with PFD (existing best + PFD)
    B('J31:Mom21+Top2+PFD0',   health='mom21', defense='top2',
                                flip_rebal=True, flip_delay=0)

    # Final candidates: Mom21 + Hyst 0.5% + Defense combos
    B('J32:Mom21+H0.5+Top2',   health='mom21', canary_hyst=0.005, defense='top2')
    B('J33:Mom21+H0.5+Fix',    health='mom21', canary_hyst=0.005, defense='fixed')
    B('J34:Mom21+H0.5+Top3',   health='mom21', canary_hyst=0.005, defense='top3')
    B('J35:Mom4+M21+H0.5+T2',  select='mom4', health='mom21', canary_hyst=0.005, defense='top2')
    B('J36:Mom4+M21+H0.5+Fix', select='mom4', health='mom21', canary_hyst=0.005, defense='fixed')

    return cfgs


def gen_K():
    """K: Advanced Selection (AI-recommended)."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # Baseline reference
    B('K0:Mom3+Sh3(base)',    select='mom3_sh3')

    # Composite rank (Z-score normalized Mom+Sharpe) — Gemini #1 recommendation
    B('K1:Comp3',             select='comp3')
    B('K2:Comp4',             select='comp4')
    B('K3:Comp5',             select='comp5')
    B('K4:Comp+Sort3',        select='comp_sort3')

    # Pure momentum (Antonacci Dual Mom style — already have health as abs momentum)
    B('K5:Mom3 only',         select='mom3')
    B('K6:Mom4 only',         select='mom4')

    # Number of holdings sweep
    B('K7:Mom3+Sh3+InvVol',   select='mom3_sh3', weight='inv_vol')
    B('K8:Comp3+InvVol',      select='comp3', weight='inv_vol')
    B('K9:Comp4+InvVol',      select='comp4', weight='inv_vol')

    # Adaptive fill: if <N healthy, fill with defense
    B('K10:Mom3+Adaptive',    select='mom3', adaptive_fill=True)
    B('K11:Comp3+Adaptive',   select='comp3', adaptive_fill=True)

    # Rank buffer: don't swap if old picks still in top N+2
    B('K12:Mom3+Buf2',        select='mom3', rank_buffer=2)
    B('K13:Comp3+Buf2',       select='comp3', rank_buffer=2)
    B('K14:Mom3+Sh3+Buf2',    select='mom3_sh3', rank_buffer=2)

    return cfgs


def gen_L():
    """L: Rebalancing Triggers (coin-inspired)."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # Baseline: monthly only
    B('L0:Monthly(base)',     )

    # PFD: canary flip → immediate rebal
    B('L1:PFD0(immediate)',   flip_rebal=True, flip_delay=0)
    B('L2:PFD3(3d delay)',    flip_rebal=True, flip_delay=3)
    B('L3:PFD5(5d delay)',    flip_rebal=True, flip_delay=5)

    # Health daily exit: sell held ETF if health fails
    B('L4:HExit Mom21',       health_daily_exit=True, health_exit_type='mom21')
    B('L5:HExit SMA200',      health_daily_exit=True, health_exit_type='sma200')
    B('L6:HExit SMA100',      health_daily_exit=True, health_exit_type='sma100')

    # Combined: PFD + Health daily
    B('L7:PFD0+HExit Mom21',  flip_rebal=True, flip_delay=0,
                               health_daily_exit=True, health_exit_type='mom21')
    B('L8:PFD0+HExit SMA200', flip_rebal=True, flip_delay=0,
                               health_daily_exit=True, health_exit_type='sma200')

    # With Mom21 health filter + PFD
    B('L9:Mom21+PFD0',        health='mom21', flip_rebal=True, flip_delay=0)
    B('L10:Mom21+PFD3',       health='mom21', flip_rebal=True, flip_delay=3)
    B('L11:Mom21+PFD0+HExit', health='mom21', flip_rebal=True, flip_delay=0,
                               health_daily_exit=True, health_exit_type='sma200')

    # With Top2 defense + Mom21 + PFD
    B('L12:Mom21+Top2+PFD0',  health='mom21', defense='top2',
                               flip_rebal=True, flip_delay=0)
    B('L13:Mom21+Top2+PFD3',  health='mom21', defense='top2',
                               flip_rebal=True, flip_delay=3)

    return cfgs


def gen_M():
    """M: Structural changes (Partial Alloc, Core-Satellite, Biweekly)."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # J34 baseline for comparison
    B('M0:J34 Baseline',       health='mom21', canary_hyst=0.005, defense='top3')

    # 1. Partial Allocation (Breadth Momentum from DAA paper)
    B('M1:J34+Partial',        health='mom21', canary_hyst=0.005, defense='top3',
                                partial_alloc=True)
    B('M2:M21+T2+Partial',     health='mom21', defense='top2', partial_alloc=True)
    B('M3:M21+T3+Partial',     health='mom21', defense='top3', partial_alloc=True)
    B('M4:M21+Fix+Partial',    health='mom21', defense='fixed', partial_alloc=True)
    B('M5:Base+Partial',       partial_alloc=True)
    B('M6:H0.5+Partial',       canary_hyst=0.005, partial_alloc=True)

    # 2. Core-Satellite (SPY B&H core + strategy satellite)
    B('M7:Core50+J34',         health='mom21', canary_hyst=0.005, defense='top3',
                                core_ratio=0.5)
    B('M8:Core60+J34',         health='mom21', canary_hyst=0.005, defense='top3',
                                core_ratio=0.6)
    B('M9:Core70+J34',         health='mom21', canary_hyst=0.005, defense='top3',
                                core_ratio=0.7)
    B('M10:Core50+M21+T2',     health='mom21', defense='top2', core_ratio=0.5)
    B('M11:Core60+M21+T2',     health='mom21', defense='top2', core_ratio=0.6)
    B('M12:Core50+Partial',    health='mom21', canary_hyst=0.005, defense='top3',
                                partial_alloc=True, core_ratio=0.5)

    # 3. Biweekly Rebalancing
    B('M13:J34+Biweekly',      health='mom21', canary_hyst=0.005, defense='top3',
                                biweekly=True)
    B('M14:M21+T3+Biweekly',   health='mom21', defense='top3', biweekly=True)
    B('M15:Partial+Biweekly',  health='mom21', canary_hyst=0.005, defense='top3',
                                partial_alloc=True, biweekly=True)

    return cfgs


def gen_N():
    """N: Universe Expansion (Growth, Leveraged, Tech ETFs)."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # Base J34 config for reference
    _j34 = dict(health='mom21', canary_hyst=0.005, defense='top3')

    # 1. Add Growth ETFs
    B('N0:J34 Baseline',       **_j34)
    B('N1:+Growth(VUG,IWF,SCHG)', offensive=OFF_CURRENT + OFF_GROWTH, **_j34)
    B('N2:+VUG only',          offensive=OFF_CURRENT + ('VUG',), **_j34)

    # 2. Add Leveraged ETFs (2x)
    B('N3:+SSO(2xSPY)',        offensive=OFF_CURRENT + ('SSO',), **_j34)
    B('N4:+QLD(2xQQQ)',        offensive=OFF_CURRENT + ('QLD',), **_j34)
    B('N5:+SSO+QLD',           offensive=OFF_CURRENT + ('SSO','QLD'), **_j34)

    # 3. Add Tech ETFs
    B('N6:+XLK',               offensive=OFF_CURRENT + ('XLK',), **_j34)
    B('N7:+SOXX(Semi)',        offensive=OFF_CURRENT + ('SOXX',), **_j34)
    B('N8:+XLK+SOXX',          offensive=OFF_CURRENT + ('XLK','SOXX'), **_j34)

    # 4. Growth + Leveraged combos
    B('N9:+Growth+SSO',        offensive=OFF_CURRENT + OFF_GROWTH + ('SSO',), **_j34)
    B('N10:+Growth+QLD',       offensive=OFF_CURRENT + OFF_GROWTH + ('QLD',), **_j34)

    # 5. Partial allocation with expanded universe
    B('N11:+SSO+Partial',      offensive=OFF_CURRENT + ('SSO',), partial_alloc=True, **_j34)
    B('N12:+QLD+Partial',      offensive=OFF_CURRENT + ('QLD',), partial_alloc=True, **_j34)
    B('N13:+Growth+Partial',   offensive=OFF_CURRENT + OFF_GROWTH, partial_alloc=True, **_j34)

    # 6. Core-Satellite with expanded universe
    B('N14:Core50+SSO',        offensive=OFF_CURRENT + ('SSO',), core_ratio=0.5, **_j34)
    B('N15:Core50+QLD',        offensive=OFF_CURRENT + ('QLD',), core_ratio=0.5, **_j34)

    # 7. Canary asset changes
    B('N16:Canary SPY only',   canary_assets=('SPY',), **_j34)
    B('N17:Canary EEM only',   canary_assets=('EEM',), **_j34)
    B('N18:Canary VT only',    canary_assets=('VT',), **_j34)

    return cfgs


def gen_P():
    """P: Global Diversification — Universe, Canary, Defense for long-term global robustness."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    _j34 = dict(health='mom21', canary_hyst=0.005, defense='top3')

    # ── 0. Baseline ──
    B('P0:Baseline(J34)', **_j34)

    # ── 1. Universe: add global ETFs to current ──
    B('P1:+VGK+EWJ',        offensive=OFF_CURRENT + ('VGK','EWJ'), **_j34)
    B('P2:+VGK+EWJ+VWO',    offensive=OFF_CURRENT + ('VGK','EWJ','VWO'), **_j34)
    B('P3:+ACWX',            offensive=OFF_CURRENT + ('ACWX',), **_j34)
    B('P4:CurGlobal(16)',    offensive=OFF_GLOB_CUR, **_j34)

    # ── 2. Universe: regional rotation designs ──
    B('P5:Region6',          offensive=OFF_REGION6, **_j34)
    B('P6:Region9',          offensive=OFF_GLOB9, **_j34)
    B('P7:Country11',        offensive=OFF_COUNTRY, **_j34)
    B('P8:GlobBal11',        offensive=OFF_GLOB_BAL, **_j34)
    B('P9:AllWeather8',      offensive=OFF_ALLWEATH, **_j34)

    # ── 3. Universe: no/minimal US ──
    B('P10:ExUS(6)',         offensive=OFF_EXUS, **_j34)
    B('P11:ExUS+PDBC(7)',    offensive=OFF_EXUS + ('PDBC',), **_j34)
    B('P12:MinUS(VT+Glob)',  offensive=('VT','VGK','EWJ','EEM','VWO','GLD','PDBC'), **_j34)

    # ── 4. Canary: global alternatives ──
    B('P13:Can EEM',         canary_assets=('EEM',), **_j34)
    B('P14:Can ACWX',        canary_assets=('ACWX',), **_j34)
    B('P15:Can VWO',         canary_assets=('VWO',), **_j34)
    B('P16:Can VT+EEM+VGK',  canary_assets=('VT','EEM','VGK'), **_j34)
    B('P17:Can SPY+EFA+EEM', canary_assets=('SPY','EFA','EEM'), **_j34)
    B('P18:Can ACWX+EEM',    canary_assets=('ACWX','EEM'), **_j34)

    # ── 5. Defense: global bonds ──
    B('P19:Def Global',      defensive=DEF_GLOBAL, **_j34)
    B('P20:Def GlobWide',    defensive=DEF_GLOB_WIDE, **_j34)
    B('P21:Def IntlBond',    defensive=('BWX','BNDX','EMB','GLD','BIL'), **_j34)

    # ── 6. Best combos: global universe + canary + defense ──
    _gd = dict(defensive=DEF_GLOBAL)
    B('P22:CurGlob+EEM+GD',
      offensive=OFF_GLOB_CUR, canary_assets=('EEM',), **_gd, **_j34)
    B('P23:Region9+EEM+GD',
      offensive=OFF_GLOB9, canary_assets=('EEM',), **_gd, **_j34)
    B('P24:Country+EEM+GD',
      offensive=OFF_COUNTRY, canary_assets=('EEM',), **_gd, **_j34)
    B('P25:GlobBal+EEM+GD',
      offensive=OFF_GLOB_BAL, canary_assets=('EEM',), **_gd, **_j34)
    B('P26:ExUS+EEM+GD',
      offensive=OFF_EXUS, canary_assets=('EEM',), **_gd, **_j34)
    B('P27:CurGlob+ACWX+GD',
      offensive=OFF_GLOB_CUR, canary_assets=('ACWX',), **_gd, **_j34)
    B('P28:Region9+ACWX+GD',
      offensive=OFF_GLOB9, canary_assets=('ACWX',), **_gd, **_j34)
    B('P29:CurGlob+3Can+GD',
      offensive=OFF_GLOB_CUR, canary_assets=('VT','EEM','VGK'), **_gd, **_j34)

    # ── 7. Partial allocation with 3 canaries + global ──
    _gd_pa = dict(defensive=DEF_GLOBAL, partial_alloc=True)
    B('P30:CurGlob+3C+PA',
      offensive=OFF_GLOB_CUR, canary_assets=('VT','EEM','VGK'), **_gd_pa, **_j34)
    B('P31:Region9+3C+PA',
      offensive=OFF_GLOB9, canary_assets=('VT','EEM','VGK'), **_gd_pa, **_j34)
    B('P32:Country+3C+PA',
      offensive=OFF_COUNTRY, canary_assets=('VT','EEM','VGK'), **_gd_pa, **_j34)
    B('P33:GlobBal+3C+PA',
      offensive=OFF_GLOB_BAL, canary_assets=('VT','EEM','VGK'), **_gd_pa, **_j34)

    # ── 8. Selection method variations with global universe ──
    B('P34:CurGlob+Mom4',
      offensive=OFF_GLOB_CUR, select='mom4', **_j34)
    B('P35:CurGlob+Comp4',
      offensive=OFF_GLOB_CUR, select='comp4', **_j34)
    B('P36:Region9+Mom4',
      offensive=OFF_GLOB9, select='mom4', **_j34)
    B('P37:Country+Comp5',
      offensive=OFF_COUNTRY, select='comp5', **_j34)

    return cfgs


def gen_Q():
    """Q: Global Combo Refinement — best global universe × canary × defense."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    _h = dict(health='mom21')

    # ── Baselines for reference ──
    B('Q0:Baseline(J34)', canary_hyst=0.005, defense='top3', **_h)
    B('Q1:P6 Region9',    offensive=OFF_GLOB9, canary_hyst=0.005, defense='top3', **_h)
    B('Q2:P13 Can EEM',   canary_assets=('EEM',), canary_hyst=0.005, defense='top3', **_h)

    # ── Region9 × canary ──
    B('Q3:R9+EEM',         offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_hyst=0.005, defense='top3', **_h)
    B('Q4:R9+ACWX+EEM',   offensive=OFF_GLOB9, canary_assets=('ACWX','EEM'),
      canary_hyst=0.005, defense='top3', **_h)
    B('Q5:R9+ACWX',        offensive=OFF_GLOB9, canary_assets=('ACWX',),
      canary_hyst=0.005, defense='top3', **_h)

    # ── Region9 × defense ──
    B('Q6:R9+DefT2',      offensive=OFF_GLOB9, defense='top2', canary_hyst=0.005, **_h)
    B('Q7:R9+DefFix',     offensive=OFF_GLOB9, defense='fixed', canary_hyst=0.005, **_h)
    B('Q8:R9+DefGlobT3',  offensive=OFF_GLOB9, defensive=DEF_GLOBAL, defense='top3',
      canary_hyst=0.005, **_h)

    # ── Region9 + EEM canary × defense ──
    B('Q9:R9+EEM+DefT2',  offensive=OFF_GLOB9, canary_assets=('EEM',),
      defense='top2', canary_hyst=0.005, **_h)
    B('Q10:R9+EEM+DefFix', offensive=OFF_GLOB9, canary_assets=('EEM',),
      defense='fixed', canary_hyst=0.005, **_h)
    B('Q11:R9+EEM+GlobDef', offensive=OFF_GLOB9, canary_assets=('EEM',),
      defensive=DEF_GLOBAL, defense='top3', canary_hyst=0.005, **_h)

    # ── Partial alloc ──
    B('Q12:R9+ACWX+EEM+PA', offensive=OFF_GLOB9, canary_assets=('ACWX','EEM'),
      partial_alloc=True, canary_hyst=0.005, defense='top3', **_h)
    B('Q13:R9+EEM+PA',     offensive=OFF_GLOB9, canary_assets=('EEM',),
      partial_alloc=True, canary_hyst=0.005, defense='top3', **_h)

    # ── Selection method with Region9+EEM ──
    B('Q14:R9+EEM+Mom4',  offensive=OFF_GLOB9, canary_assets=('EEM',),
      select='mom4', canary_hyst=0.005, defense='top3', **_h)
    B('Q15:R9+EEM+Comp4',  offensive=OFF_GLOB9, canary_assets=('EEM',),
      select='comp4', canary_hyst=0.005, defense='top3', **_h)
    B('Q16:R9+EEM+Mom5',  offensive=OFF_GLOB9, canary_assets=('EEM',),
      select='mom5', canary_hyst=0.005, defense='top3', **_h)

    # ── Universe size variations with EEM canary ──
    B('Q17:R6+EEM',        offensive=OFF_REGION6, canary_assets=('EEM',),
      canary_hyst=0.005, defense='top3', **_h)
    B('Q18:Glob16+EEM',   offensive=OFF_GLOB_CUR, canary_assets=('EEM',),
      canary_hyst=0.005, defense='top3', **_h)
    B('Q19:GlobBal+EEM',  offensive=OFF_GLOB_BAL, canary_assets=('EEM',),
      canary_hyst=0.005, defense='top3', **_h)

    # ── Canary SMA period with Region9 ──
    B('Q20:R9+SMA150',    offensive=OFF_GLOB9, canary_sma=150, canary_hyst=0.005,
      defense='top3', **_h)
    B('Q21:R9+SMA250',    offensive=OFF_GLOB9, canary_sma=250, canary_hyst=0.005,
      defense='top3', **_h)
    B('Q22:R9+EEM+SMA150', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_sma=150, canary_hyst=0.005, defense='top3', **_h)
    B('Q23:R9+EEM+SMA250', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_sma=250, canary_hyst=0.005, defense='top3', **_h)

    # ── Hysteresis variations with Region9+EEM ──
    B('Q24:R9+EEM+NoHyst', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_hyst=0.0, defense='top3', **_h)
    B('Q25:R9+EEM+H0.3%', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_hyst=0.003, defense='top3', **_h)
    B('Q26:R9+EEM+H1.0%', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_hyst=0.01, defense='top3', **_h)

    return cfgs


def gen_R():
    """R: Selection & Weighting sweep for Region9+EEM (global strategy)."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    # Base: Region9 + EEM canary + Mom21 health + Hyst 0.5%
    _r = dict(offensive=OFF_GLOB9, canary_assets=('EEM',), canary_hyst=0.005, health='mom21')

    # ── 1. Selection method ──
    B('R0:Mom3+Sh3(base)', defense='top3', **_r)
    B('R1:Mom3',           select='mom3', defense='top3', **_r)
    B('R2:Mom4',           select='mom4', defense='top3', **_r)
    B('R3:Mom5',           select='mom5', defense='top3', **_r)
    B('R4:Sh3',            select='sh3', defense='top3', **_r)
    B('R5:Sh4',            select='sh4', defense='top3', **_r)
    B('R6:Sh5',            select='sh5', defense='top3', **_r)
    B('R7:Comp3',          select='comp3', defense='top3', **_r)
    B('R8:Comp4',          select='comp4', defense='top3', **_r)
    B('R9:Comp5',          select='comp5', defense='top3', **_r)
    B('R10:CompSort3',     select='comp_sort3', defense='top3', **_r)
    B('R11:CompSort4',     select='comp_sort4', defense='top3', **_r)
    B('R12:CompSort5',     select='comp_sort5', defense='top3', **_r)
    B('R13:Mom5+Sh5',      select='mom5_sh5', defense='top3', **_r)
    B('R14:Mom2',          select='mom2', defense='top3', **_r)
    B('R15:Sh2',           select='sh2', defense='top3', **_r)
    B('R16:Comp2',         select='comp2', defense='top3', **_r)

    # ── 2. Weighting ──
    B('R17:EW(base)',      weight='ew', defense='top3', **_r)
    B('R18:InvVol',        weight='inv_vol', defense='top3', **_r)
    B('R19:RankDecay',     weight='rank_decay', defense='top3', **_r)

    # ── 3. Best selection × weighting ──
    B('R20:Mom3+InvVol',   select='mom3', weight='inv_vol', defense='top3', **_r)
    B('R21:Sh3+InvVol',    select='sh3', weight='inv_vol', defense='top3', **_r)
    B('R22:Comp3+InvVol',  select='comp3', weight='inv_vol', defense='top3', **_r)
    B('R23:Mom3+RkDecay',  select='mom3', weight='rank_decay', defense='top3', **_r)
    B('R24:Sh3+RkDecay',   select='sh3', weight='rank_decay', defense='top3', **_r)

    # ── 4. Defense variations with best selections ──
    B('R25:Mom3+DefT2',    select='mom3', defense='top2', **_r)
    B('R26:Sh3+DefT2',     select='sh3', defense='top2', **_r)
    B('R27:Mom3+DefFix',   select='mom3', defense='fixed', **_r)
    B('R28:Sh3+DefFix',    select='sh3', defense='fixed', **_r)

    # ── 5. N_picks with fixed select style ──
    B('R29:Mom2+Sh2',      select='mom3_sh3', n_mom=2, n_sh=2, defense='top3', **_r)
    B('R30:Mom4+Sh4',      select='mom3_sh3', n_mom=4, n_sh=4, defense='top3', **_r)

    return cfgs


def gen_S():
    """S: R9+EEM Fine Tuning — multi-step canary, momentum weights, Sharpe lookback, etc."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    # Base config: Region9 + EEM canary + Mom21 health + Hyst 0.5% + Top3 defense
    _b = dict(offensive=OFF_GLOB9, canary_assets=('EEM',), canary_hyst=0.005,
              health='mom21', defense='top3')

    # ── 0. Baseline ──
    B('S0:R9+EEM Baseline', **_b)

    # ── 1. Multi-step canary (3-step with band) ──
    B('S1:Band 1%',   canary_band=0.01, **_b)
    B('S2:Band 2%',   canary_band=0.02, **_b)
    B('S3:Band 3%',   canary_band=0.03, **_b)
    B('S4:Band 5%',   canary_band=0.05, **_b)
    B('S5:Band 7%',   canary_band=0.07, **_b)

    # ── 2. Multi-anchor snapshot (tranche) ──
    B('S6:2-Tranche',  tranche_days=(1, 15), _n_tranches=2, **_b)
    B('S7:3-Tranche',  tranche_days=(1, 10, 19), _n_tranches=3, **_b)
    B('S8:MidMonth',   _anchor=15, **_b)

    # ── 3. Momentum weight style ──
    B('S9:Mom 100%6M',  mom_style='6m', **_b)
    B('S10:Mom eq',     mom_style='eq', **_b)
    B('S11:Mom 70/20/10', mom_style='rh', **_b)
    B('S12:Mom 20/30/50', mom_style='lh', **_b)
    B('S13:Mom dual',   mom_style='dual', **_b)

    # ── 4. Sharpe lookback ──
    B('S14:Sharpe 63d',  sharpe_lookback=63, **_b)
    B('S15:Sharpe 252d', sharpe_lookback=252, **_b)

    # ── 5. Defense momentum period ──
    B('S16:Def Mom21',   def_mom_period=21, **_b)
    B('S17:Def Mom63',   def_mom_period=63, **_b)
    B('S18:Def Mom252',  def_mom_period=252, **_b)
    B('S19:Def Sharpe',  def_use_sharpe=True, **_b)

    # ── 6. Health filter variations ──
    B('S20:Health Mom42', health='mom42', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_hyst=0.005, defense='top3')
    B('S21:Health Mom14', health='mom21', offensive=OFF_GLOB9, canary_assets=('EEM',),
      canary_hyst=0.005, defense='top3')  # mom14 not in precompute, use mom21 as ref

    # ── 7. EMA canary ──
    B('S22:EMA200',     canary_type='ema', **_b)
    B('S23:EMA+Band2%', canary_type='ema', canary_band=0.02, **_b)

    # ── 8. Best combos ──
    # Multi-step + 3-tranche
    B('S24:Band2%+3Tr',  canary_band=0.02, tranche_days=(1, 10, 19), _n_tranches=3, **_b)
    B('S25:Band3%+3Tr',  canary_band=0.03, tranche_days=(1, 10, 19), _n_tranches=3, **_b)
    # Multi-step + best mom style (will fill after seeing results)
    B('S26:Band2%+6M',   canary_band=0.02, mom_style='6m', **_b)
    B('S27:Band3%+Sh63', canary_band=0.03, sharpe_lookback=63, **_b)
    # 3-tranche + mom style
    B('S28:3Tr+6M',      tranche_days=(1, 10, 19), _n_tranches=3, mom_style='6m', **_b)
    B('S29:3Tr+Sh63',    tranche_days=(1, 10, 19), _n_tranches=3, sharpe_lookback=63, **_b)
    # Triple combo
    B('S30:Band2%+3Tr+6M', canary_band=0.02, tranche_days=(1, 10, 19), _n_tranches=3,
      mom_style='6m', **_b)
    B('S31:Band3%+3Tr+6M', canary_band=0.03, tranche_days=(1, 10, 19), _n_tranches=3,
      mom_style='6m', **_b)

    return cfgs


def gen_T():
    """T: Final Global Combos — combine best findings from S."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    _b = dict(offensive=OFF_GLOB9, canary_assets=('EEM',), canary_hyst=0.005,
              health='mom21', defense='top3')

    # ── Baselines ──
    B('T0:R0 Baseline', **_b)
    B('T1:S12 Mom20/30/50', mom_style='lh', **_b)

    # ── 1. Mom weight sweep (even more long-biased) ──
    B('T2:Mom 10/20/70',  mom_style='vl', **_b)
    B('T3:Mom 0/0/100',   mom_style='12m', **_b)

    # ── 2. S12(Mom lh) + S15(Sharpe 252d) ──
    B('T5:lh+Sh252',      mom_style='lh', sharpe_lookback=252, **_b)
    B('T6:dual+Sh252',    mom_style='dual', sharpe_lookback=252, **_b)
    B('T7:eq+Sh252',      mom_style='eq', sharpe_lookback=252, **_b)

    # ── 3. S12 + defense variations ──
    B('T8:lh+DefT2',      mom_style='lh', defense='top2', offensive=OFF_GLOB9,
      canary_assets=('EEM',), canary_hyst=0.005, health='mom21')
    B('T9:lh+DefFix',     mom_style='lh', defense='fixed', offensive=OFF_GLOB9,
      canary_assets=('EEM',), canary_hyst=0.005, health='mom21')

    # ── 4. S12 + weighting ──
    B('T10:lh+RankDecay', mom_style='lh', weight='rank_decay', **_b)

    # ── 5. S12 + canary SMA ──
    B('T11:lh+SMA150',    mom_style='lh', canary_sma=150, **_b)
    B('T12:lh+SMA250',    mom_style='lh', canary_sma=250, **_b)

    # ── 6. S12 with current (US-heavy) universe for comparison ──
    B('T13:lh+CurUniv',   mom_style='lh', canary_assets=('EEM',), canary_hyst=0.005,
      health='mom21', defense='top3')  # default offensive = OFF_CURRENT

    # ── 7. Even more long-biased momentum (custom) ──
    # We need to add wmom variants. For now use mom252 directly via select
    B('T14:Pure 12M',     select='mom3', mom_style='lh', **_b)  # lh + mom3 (effectively long-biased)
    B('T15:Sh3 only+lh',  select='sh3', mom_style='lh', **_b)

    # ── 8. S12 + S15 + defense combos (best of everything) ──
    B('T16:lh+Sh252+T2',  mom_style='lh', sharpe_lookback=252, defense='top2',
      offensive=OFF_GLOB9, canary_assets=('EEM',), canary_hyst=0.005, health='mom21')
    B('T17:lh+Sh252+RkD', mom_style='lh', sharpe_lookback=252, weight='rank_decay', **_b)

    # ── 9. Verify robustness with different canary assets ──
    B('T18:lh+Can VT+EEM', mom_style='lh', canary_assets=('VT','EEM'),
      offensive=OFF_GLOB9, canary_hyst=0.005, health='mom21', defense='top3')
    B('T19:lh+Can ACWX+EEM', mom_style='lh', canary_assets=('ACWX','EEM'),
      offensive=OFF_GLOB9, canary_hyst=0.005, health='mom21', defense='top3')

    return cfgs


def gen_U():
    """U: Universe expansion + fixed-N selection with long momentum."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    _c = dict(canary_assets=('EEM',), canary_hyst=0.005, health='mom21', defense='top3', mom_style='lh')

    # ── 0. Baselines ──
    B('U0:R9+EW(base)',      offensive=OFF_GLOB9, **_c)
    B('U1:R9+RkDecay',       offensive=OFF_GLOB9, weight='rank_decay', **_c)

    # ── 1. Fixed-N selection (no variable picks) ──
    B('U2:R9+Comp3',         offensive=OFF_GLOB9, select='comp3', **_c)
    B('U3:R9+Comp4',         offensive=OFF_GLOB9, select='comp4', **_c)
    B('U4:R9+Comp5',         offensive=OFF_GLOB9, select='comp5', **_c)
    B('U5:R9+Mom3',          offensive=OFF_GLOB9, select='mom3', **_c)
    B('U6:R9+Mom4',          offensive=OFF_GLOB9, select='mom4', **_c)
    # Fixed-N + RankDecay
    B('U7:R9+Comp3+RkD',     offensive=OFF_GLOB9, select='comp3', weight='rank_decay', **_c)
    B('U8:R9+Comp4+RkD',     offensive=OFF_GLOB9, select='comp4', weight='rank_decay', **_c)
    B('U9:R9+Mom3+RkD',      offensive=OFF_GLOB9, select='mom3', weight='rank_decay', **_c)
    B('U10:R9+Mom4+RkD',     offensive=OFF_GLOB9, select='mom4', weight='rank_decay', **_c)

    # ── 2. Expanded universe (R9 + additional global ETFs) with long momentum ──
    # R9 + existing universe ETFs
    _r9plus = OFF_GLOB9 + ('EFA','VEA','IQLT','IMTM')  # 13: + intl factors
    _r9all  = OFF_GLOB9 + ('EFA','VEA','IQLT','IMTM','QUAL','MTUM','VT')  # 16: + US factors + VT
    _r9cur  = OFF_GLOB_CUR  # 16: OFF_CURRENT + VGK/EWJ/ACWX/VWO
    _r9wide = OFF_GLOB9 + ('EFA','IQLT','IMTM','SCZ')  # 13: + intl small cap
    _r9big  = OFF_GLOB9 + ('EFA','VEA','IQLT','IMTM','QUAL','MTUM','VT','SCZ','EWA','EWY')  # 19

    B('U11:R9+IntlFac(13)',   offensive=_r9plus, **_c)
    B('U12:R9+AllFac(16)',    offensive=_r9all, **_c)
    B('U13:CurGlob(16)',      offensive=_r9cur, **_c)
    B('U14:R9+IntlDiv(13)',   offensive=_r9wide, **_c)
    B('U15:R9+Big(19)',       offensive=_r9big, **_c)

    # ── 3. Expanded universe + RankDecay ──
    B('U16:R9+IntlFac+RkD',  offensive=_r9plus, weight='rank_decay', **_c)
    B('U17:R9+AllFac+RkD',   offensive=_r9all, weight='rank_decay', **_c)
    B('U18:CurGlob+RkD',     offensive=_r9cur, weight='rank_decay', **_c)
    B('U19:R9+Big+RkD',      offensive=_r9big, weight='rank_decay', **_c)

    # ── 4. Expanded universe + fixed-N selection ──
    B('U20:IntlFac+Comp4',    offensive=_r9plus, select='comp4', **_c)
    B('U21:AllFac+Comp4',     offensive=_r9all, select='comp4', **_c)
    B('U22:IntlFac+Comp5',    offensive=_r9plus, select='comp5', **_c)
    B('U23:AllFac+Comp5',     offensive=_r9all, select='comp5', **_c)
    B('U24:Big+Comp5',        offensive=_r9big, select='comp5', **_c)

    # ── 5. Expanded + Comp + RankDecay (triple) ──
    B('U25:IntlFac+C4+RkD',  offensive=_r9plus, select='comp4', weight='rank_decay', **_c)
    B('U26:AllFac+C4+RkD',   offensive=_r9all, select='comp4', weight='rank_decay', **_c)
    B('U27:AllFac+C5+RkD',   offensive=_r9all, select='comp5', weight='rank_decay', **_c)
    B('U28:Big+C5+RkD',      offensive=_r9big, select='comp5', weight='rank_decay', **_c)

    return cfgs


def gen_V():
    """V: Country/Region ETF addition to R9 — targeted expansion test."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    _c = dict(canary_assets=('EEM',), canary_hyst=0.005, health='mom21',
              defense='top3', mom_style='lh', weight='rank_decay')

    # ── 0. Baselines ──
    B('V0:R9+RkD(base)',       offensive=OFF_GLOB9, **_c)

    # ── 1. Single country additions ──
    B('V1:R9+China(FXI)',      offensive=OFF_R9_CHN, **_c)
    B('V2:R9+India(INDA)',     offensive=OFF_R9_IND, **_c)
    B('V3:R9+Korea(EWY)',      offensive=OFF_R9_KOR, **_c)
    B('V4:R9+Taiwan(EWT)',     offensive=OFF_R9_TWN, **_c)
    B('V5:R9+Australia(EWA)',  offensive=OFF_R9_AUS, **_c)
    B('V6:R9+Brazil(EWZ)',     offensive=OFF_R9_BRZ, **_c)

    # ── 2. Multi-country groups ──
    B('V7:R9+Asia4(FXI/INDA/EWY/EWT)',  offensive=OFF_R9_ASIA, **_c)
    B('V8:R9+BRIC3(FXI/INDA/EWZ)',      offensive=OFF_R9_BRIC, **_c)
    B('V9:R9+Dev4(EWA/EWC/EWU/EWG)',    offensive=OFF_R9_DEV, **_c)
    B('V10:R9+All9(18total)',            offensive=OFF_R9_ALL, **_c)

    # ── 3. Single additions with EW (no RankDecay) ──
    _c_ew = dict(canary_assets=('EEM',), canary_hyst=0.005, health='mom21',
                 defense='top3', mom_style='lh')
    B('V11:R9+China+EW',      offensive=OFF_R9_CHN, **_c_ew)
    B('V12:R9+India+EW',      offensive=OFF_R9_IND, **_c_ew)
    B('V13:R9+Korea+EW',      offensive=OFF_R9_KOR, **_c_ew)
    B('V14:R9+Asia4+EW',      offensive=OFF_R9_ASIA, **_c_ew)

    # ── 4. Best country additions with Comp4 selection ──
    B('V15:R9+China+Comp4',   offensive=OFF_R9_CHN, select='comp4', **_c)
    B('V16:R9+India+Comp4',   offensive=OFF_R9_IND, select='comp4', **_c)
    B('V17:R9+Asia4+Comp4',   offensive=OFF_R9_ASIA, select='comp4', **_c)
    B('V18:R9+BRIC3+Comp4',   offensive=OFF_R9_BRIC, select='comp4', **_c)

    # ── 5. Larger picks with country expansion ──
    B('V19:R9+Asia4+Mom4',    offensive=OFF_R9_ASIA, select='mom4', **_c)
    B('V20:R9+All9+Mom4',     offensive=OFF_R9_ALL, select='mom4', **_c)
    B('V21:R9+All9+Comp5',    offensive=OFF_R9_ALL, select='comp5', **_c)
    B('V22:R9+All9+C5+RkD',   offensive=OFF_R9_ALL, select='comp5', **_c)

    return cfgs


def gen_W():
    """W: Country ETF + volatility-compensating selection/weighting."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # Common: EEM canary, long-heavy momentum, top3 defense
    _c = dict(canary_assets=('EEM',), canary_hyst=0.005, defense='top3', mom_style='lh')

    # ── 0. Baselines ──
    B('W0:R9+base',            offensive=OFF_GLOB9, health='mom21', weight='rank_decay', **_c)

    # ── 1. +China(FXI): volatility-adjusted selection ──
    _chn = dict(offensive=OFF_R9_CHN, **_c)
    B('W1:+CHN+Sh3',          select='sh3', health='mom21', **_chn)
    B('W2:+CHN+Sh4',          select='sh4', health='mom21', **_chn)
    B('W3:+CHN+CompSort3',    select='comp_sort3', health='mom21', **_chn)
    B('W4:+CHN+CompSort4',    select='comp_sort4', health='mom21', **_chn)
    B('W5:+CHN+InvVol',       health='mom21', weight='inv_vol', **_chn)
    B('W6:+CHN+RkD+Sh3',     select='sh3', health='mom21', weight='rank_decay', **_chn)
    B('W7:+CHN+Mom21_63',     health='mom21_63', weight='rank_decay', **_chn)
    B('W8:+CHN+Mom63Vol',     health='mom63_vol', weight='rank_decay', **_chn)
    B('W9:+CHN+Comp4+InvV',  select='comp4', health='mom21', weight='inv_vol', **_chn)

    # ── 2. +India(INDA): volatility-adjusted selection ──
    _ind = dict(offensive=OFF_R9_IND, **_c)
    B('W10:+IND+Sh3',         select='sh3', health='mom21', **_ind)
    B('W11:+IND+Sh4',         select='sh4', health='mom21', **_ind)
    B('W12:+IND+CompSort3',   select='comp_sort3', health='mom21', **_ind)
    B('W13:+IND+CompSort4',   select='comp_sort4', health='mom21', **_ind)
    B('W14:+IND+InvVol',      health='mom21', weight='inv_vol', **_ind)
    B('W15:+IND+RkD+Sh3',    select='sh3', health='mom21', weight='rank_decay', **_ind)
    B('W16:+IND+Mom21_63',    health='mom21_63', weight='rank_decay', **_ind)
    B('W17:+IND+Mom63Vol',    health='mom63_vol', weight='rank_decay', **_ind)
    B('W18:+IND+Comp4+InvV', select='comp4', health='mom21', weight='inv_vol', **_ind)

    # ── 3. +Korea(EWY): volatility-adjusted selection ──
    _kor = dict(offensive=OFF_R9_KOR, **_c)
    B('W19:+KOR+Sh3',         select='sh3', health='mom21', **_kor)
    B('W20:+KOR+InvVol',      health='mom21', weight='inv_vol', **_kor)
    B('W21:+KOR+Mom21_63',    health='mom21_63', weight='rank_decay', **_kor)
    B('W22:+KOR+CompSort4',   select='comp_sort4', health='mom21', **_kor)

    # ── 4. +Taiwan(EWT): volatility-adjusted selection ──
    _twn = dict(offensive=OFF_R9_TWN, **_c)
    B('W23:+TWN+Sh3',         select='sh3', health='mom21', **_twn)
    B('W24:+TWN+InvVol',      health='mom21', weight='inv_vol', **_twn)
    B('W25:+TWN+Mom21_63',    health='mom21_63', weight='rank_decay', **_twn)

    # ── 5. +Asia4: volatility-adjusted selection (핵심 테스트) ──
    _asia = dict(offensive=OFF_R9_ASIA, **_c)
    B('W26:+Asia4+Sh3',       select='sh3', health='mom21', **_asia)
    B('W27:+Asia4+Sh4',       select='sh4', health='mom21', **_asia)
    B('W28:+Asia4+CompSort3', select='comp_sort3', health='mom21', **_asia)
    B('W29:+Asia4+CompSort4', select='comp_sort4', health='mom21', **_asia)
    B('W30:+Asia4+InvVol',    health='mom21', weight='inv_vol', **_asia)
    B('W31:+Asia4+RkD+Sh3',  select='sh3', health='mom21', weight='rank_decay', **_asia)
    B('W32:+Asia4+Mom21_63',  health='mom21_63', weight='rank_decay', **_asia)
    B('W33:+Asia4+Mom63Vol',  health='mom63_vol', weight='rank_decay', **_asia)
    B('W34:+Asia4+C4+InvV',  select='comp4', health='mom21', weight='inv_vol', **_asia)
    B('W35:+Asia4+CS4+InvV', select='comp_sort4', health='mom21', weight='inv_vol', **_asia)
    B('W36:+Asia4+Mom21_63+InvV', health='mom21_63', weight='inv_vol', **_asia)

    # ── 6. +All9(18): volatility-adjusted (최대 유니버스) ──
    _all = dict(offensive=OFF_R9_ALL, **_c)
    B('W37:+All9+Sh4',        select='sh4', health='mom21', **_all)
    B('W38:+All9+InvVol',     health='mom21', weight='inv_vol', **_all)
    B('W39:+All9+CS4+InvV',  select='comp_sort4', health='mom21', weight='inv_vol', **_all)
    B('W40:+All9+Mom21_63+IV', health='mom21_63', weight='inv_vol', **_all)

    # ── 7. R9 with same vol-adjusted methods (공정 비교) ──
    _r9 = dict(offensive=OFF_GLOB9, **_c)
    B('W41:R9+Sh3',           select='sh3', health='mom21', **_r9)
    B('W42:R9+InvVol',        health='mom21', weight='inv_vol', **_r9)
    B('W43:R9+CompSort3',     select='comp_sort3', health='mom21', **_r9)
    B('W44:R9+Mom21_63+RkD',  health='mom21_63', weight='rank_decay', **_r9)

    return cfgs


def gen_X():
    """X: Final Validation — cost sensitivity, rebal-day robustness,
    leave-one-asset-out, defense overlap, signal delay, plateau check."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # Base params (no offensive/defensive — add explicitly to avoid conflicts)
    _b = dict(canary_assets=('EEM',), canary_hyst=0.005,
              health='mom21', defense='top3', mom_style='lh', weight='rank_decay')
    # Full best = _b + offensive=OFF_GLOB9 + defensive=DEF_GLOBAL
    def best(**kw):
        d = dict(offensive=OFF_GLOB9, defensive=DEF_GLOBAL, **_b)
        d.update(kw)
        return d

    # ── 0. Baseline ──
    B('X0:Baseline(0.1%)',     **best())

    # ── 1. TX Cost Sensitivity ──
    for tx, label in [(0.0005,'0.05%'), (0.0015,'0.15%'), (0.002,'0.2%'),
                      (0.003,'0.3%'), (0.005,'0.5%'), (0.01,'1.0%')]:
        B(f'X-TX:{label}',    **best(tx_cost=tx))

    # ── 2. Rebalance Day Robustness ──
    for day in [1, 3, 5, 8, 10, 12, 15, 18, 20]:
        B(f'X-Day{day:02d}',  **best(_anchor=day))

    # ── 3. Leave-One-Asset-Out (offensive) ──
    for etf in OFF_GLOB9:
        reduced = tuple(e for e in OFF_GLOB9 if e != etf)
        B(f'X-No{etf}',       **best(offensive=reduced))

    # ── 4. GLD/PDBC dual-role overlap ──
    _ng_o = tuple(e for e in OFF_GLOB9 if e != 'GLD')
    _np_o = tuple(e for e in OFF_GLOB9 if e != 'PDBC')
    _ng_d = tuple(e for e in DEF_GLOBAL if e != 'GLD')
    _np_d = tuple(e for e in DEF_GLOBAL if e != 'PDBC')
    B('X-NoGLD_Off',       **best(offensive=_ng_o))
    B('X-NoGLD_Def',       **best(defensive=_ng_d))
    B('X-NoGLD_Both',      **best(offensive=_ng_o, defensive=_ng_d))
    B('X-NoPDBC_Off',      **best(offensive=_np_o))
    B('X-NoPDBC_Def',      **best(defensive=_np_d))
    B('X-NoPDBC_Both',     **best(offensive=_np_o, defensive=_np_d))

    # ── 5. Defense pool variations ──
    B('X-Def:IEF+BIL',    **best(defensive=('IEF','BIL')))
    B('X-Def:IEF+GLD+BIL', **best(defensive=('IEF','GLD','BIL')))
    B('X-Def:Current5',    **best(defensive=DEF_CURRENT))
    B('X-Def:GlobWide7',   **best(defensive=DEF_GLOB_WIDE))

    # ── 6. Health filter robustness ──
    for h in ['none', 'mom42', 'mom63', 'mom126', 'sma200', 'mom21_63']:
        B(f'X-H:{h}',      **best(health=h))

    # ── 7. Canary SMA neighborhood (precomputed: 100,150,200,250) ──
    for sma in [100, 150, 200, 250]:
        B(f'X-Can:SMA{sma}', **best(canary_sma=sma))

    # ── 8. Canary hysteresis neighborhood ──
    for hyst in [0.0, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
        B(f'X-Hyst:{hyst:.1%}', **best(canary_hyst=hyst))

    # ── 9. Mom weight neighborhood ──
    for style, label in [('lh','20/30/50'), ('vl','10/20/70'), ('12m','0/0/100'),
                         ('dual','50/50/0'), ('eq','33/33/33'), ('default','50/30/20'),
                         ('rh','70/20/10'), ('6m','0/100/0')]:
        B(f'X-Mom:{label}',  **best(mom_style=style))

    # ── 10. Selection method robustness ──
    B('X-Sel:Mom3',        **best(select='mom3'))
    B('X-Sel:Sh3',         **best(select='sh3'))
    B('X-Sel:Comp3',       **best(select='comp3'))
    B('X-Sel:Comp4',       **best(select='comp4'))
    B('X-Sel:CompSort3',   **best(select='comp_sort3'))
    B('X-Sel:Mom2_Sh2',    **best(select='mom3_sh3', n_mom=2, n_sh=2))
    B('X-Sel:Mom4_Sh4',    **best(select='mom3_sh3', n_mom=4, n_sh=4))

    # ── 11. Weighting robustness ──
    B('X-Wt:EW',           **best(weight='ew'))
    B('X-Wt:InvVol',       **best(weight='inv_vol'))

    return cfgs


def gen_Z():
    """Z: Final strategy anchor-day averaging — R8+12M+EW vs alternatives."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))
    R8 = ('SPY','QQQ','VGK','EWJ','EEM','VWO','GLD','PDBC')
    DAYS = [1, 2, 3, 4, 5, 8, 10, 12, 15, 18, 20]

    # ── Final strategy: R8 + 12M + EW ──
    def final(**kw):
        d = dict(offensive=R8, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='12m', weight='ew')
        d.update(kw)
        return d

    for day in DAYS:
        B(f'Z-Final:D{day:02d}', **final(_anchor=day))

    # ── Comparison: R8+12M+RkD ──
    def final_rkd(**kw):
        d = dict(offensive=R8, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='12m', weight='rank_decay')
        d.update(kw)
        return d

    for day in DAYS:
        B(f'Z-RkD:D{day:02d}', **final_rkd(_anchor=day))

    # ── Comparison: R9+lh+RkD (previous best) ──
    def prev(**kw):
        d = dict(offensive=OFF_GLOB9, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='lh', weight='rank_decay')
        d.update(kw)
        return d

    for day in DAYS:
        B(f'Z-Prev:D{day:02d}', **prev(_anchor=day))

    # ── Comparison: R8+lh+EW ──
    def r8lh(**kw):
        d = dict(offensive=R8, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='lh', weight='ew')
        d.update(kw)
        return d

    for day in DAYS:
        B(f'Z-R8lhEW:D{day:02d}', **r8lh(_anchor=day))

    return cfgs


def gen_Y():
    """Y: Simplified candidates + anchor-day robustness + top-N day removal."""
    cfgs = []
    B = lambda name, **kw: cfgs.append((name, base(**kw)))

    # R8 = R9 minus ACWX
    R8 = ('SPY','QQQ','VGK','EWJ','EEM','VWO','GLD','PDBC')

    # ── Candidate definitions ──
    # Current: R9 + lh(20/30/50) + Mom3+Sh3 + RkD + DEF_CURRENT
    def curr(**kw):
        d = dict(offensive=OFF_GLOB9, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='lh', weight='rank_decay')
        d.update(kw)
        return d
    # Candidate A: drop ACWX (8 ETF), rest same
    def candA(**kw):
        d = dict(offensive=R8, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='lh', weight='rank_decay')
        d.update(kw)
        return d
    # Candidate B: drop ACWX + pure 12M
    def candB(**kw):
        d = dict(offensive=R8, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='12m', weight='rank_decay')
        d.update(kw)
        return d
    # Candidate C: R9 + pure 12M (keep ACWX but simplify mom)
    def candC(**kw):
        d = dict(offensive=OFF_GLOB9, defensive=DEF_CURRENT,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='mom21', defense='top3', mom_style='12m', weight='rank_decay')
        d.update(kw)
        return d

    # ── 0. Baselines (Day 1) ──
    B('Y0:Current(R9+lh)',    **curr())
    B('Y1:CandA(R8+lh)',      **candA())
    B('Y2:CandB(R8+12M)',     **candB())
    B('Y3:CandC(R9+12M)',     **candC())

    # ── 1. Anchor day sweep for all candidates ──
    for day in [1, 3, 5, 8, 10, 15, 20]:
        B(f'Y-Cur:D{day:02d}',  **curr(_anchor=day))
        B(f'Y-A:D{day:02d}',    **candA(_anchor=day))
        B(f'Y-B:D{day:02d}',    **candB(_anchor=day))
        B(f'Y-C:D{day:02d}',    **candC(_anchor=day))

    # ── 2. Multi-canary test (Gemini suggestion) ──
    # 2-of-3 breadth canary: SPY+EFA+EEM
    B('Y-MultiCan:SPY+EFA+EEM', **curr(canary_assets=('SPY','EFA','EEM')))
    B('Y-MultiCan:SPY+VGK+EEM', **curr(canary_assets=('SPY','VGK','EEM')))
    B('Y-MultiCan:VT+EEM',      **curr(canary_assets=('VT','EEM')))
    # Single canary alternatives
    B('Y-Can:VT',                **curr(canary_assets=('VT',)))
    B('Y-Can:SPY',               **curr(canary_assets=('SPY',)))

    # ── 3. EW variants (simpler weighting) ──
    B('Y-A:EW',                 **candA(weight='ew'))
    B('Y-B:EW',                 **candB(weight='ew'))

    # ── 4. TX cost stress for all candidates ──
    for tx in [0.001, 0.002, 0.003, 0.005]:
        B(f'Y-Cur:TX{tx:.1%}',  **curr(tx_cost=tx))
        B(f'Y-B:TX{tx:.1%}',    **candB(tx_cost=tx))

    return cfgs


GENERATORS = {
    'A': ('Universe', gen_A),
    'B': ('Defense', gen_B),
    'C': ('Health Filter', gen_C),
    'D': ('Tranche Rebal', gen_D),
    'E': ('DD Exit', gen_E),
    'F': ('Crash Breaker', gen_F),
    'G': ('Canary', gen_G),
    'H': ('Selection', gen_H),
    'I': ('Weighting', gen_I),
    'J': ('Stack Combos', gen_J),
    'K': ('Advanced Selection', gen_K),
    'L': ('Rebal Triggers', gen_L),
    'M': ('Structural Changes', gen_M),
    'N': ('Universe Expansion', gen_N),
    'P': ('Global Diversification', gen_P),
    'Q': ('Global Combo Refinement', gen_Q),
    'R': ('R9+EEM Selection Sweep', gen_R),
    'S': ('R9+EEM Fine Tuning', gen_S),
    'T': ('Final Global Combos', gen_T),
    'U': ('Universe+Selection Refine', gen_U),
    'V': ('Country ETF Addition', gen_V),
    'W': ('Country+VolAdjust', gen_W),
    'X': ('Final Validation', gen_X),
    'Y': ('Simplify+AnchorRobust', gen_Y),
    'Z': ('Final:AnchorAvg', gen_Z),
}


# ─── Display ─────────────────────────────────────────────────────
def print_table(title, results):
    """Print results table for a category."""
    if not results:
        return
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    hdr = f"  {'Name':<28} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'Flips':>6} {'Rebals':>7} {'Final($)':>10}"
    print(hdr)
    print(f"  {'─'*102}")

    best_sh = max((r[1]['Sharpe'] for r in results if r[1]), default=0)
    for name, m in results:
        if m is None:
            print(f"  {name:<28} {'FAILED':>7}")
            continue
        star = ' ★' if m['Sharpe'] == best_sh and len(results) > 1 else '  '
        print(f"  {name:<28} {m['CAGR']:>+6.1%} {m['MDD']:>6.1%} {m['Sharpe']:>7.3f}{star}"
              f"{m['Sortino']:>8.3f} {m['Calmar']:>7.2f} {m['Flips']:>6.0f} {m['Rebals']:>7.0f} ${m['Final']:>9,.0f}")


def print_top_n(all_results, n=25):
    """Print overall top N by Sharpe."""
    valid = [(name, m) for name, m in all_results if m is not None]
    valid.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n{'='*100}")
    print(f"  🏆 Overall Top {n} by Sharpe")
    print(f"{'='*100}")
    hdr = f"  {'#':>3} {'Name':<28} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'Final($)':>10}"
    print(hdr)
    print(f"  {'─'*102}")
    for i, (name, m) in enumerate(valid[:n], 1):
        print(f"  {i:>3} {name:<28} {m['CAGR']:>+6.1%} {m['MDD']:>6.1%} {m['Sharpe']:>7.3f}"
              f" {m['Sortino']:>8.3f} {m['Calmar']:>7.2f} ${m['Final']:>9,.0f}")


# ─── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat', nargs='*', default=list(GENERATORS.keys()),
                        help='Categories to test (A B C D E F G H I)')
    parser.add_argument('--workers', type=int, default=min(24, cpu_count()),
                        help='Number of parallel workers')
    parser.add_argument('--start', default='2017-01-01')
    parser.add_argument('--end', default='2026-12-31')
    args = parser.parse_args()

    print("=" * 100)
    print("  Stock ETF Strategy Comprehensive Improvement Test")
    print("=" * 100)

    # Collect all needed tickers
    cats = [c.upper() for c in args.cat]
    all_configs = []
    for cat in cats:
        if cat not in GENERATORS:
            print(f"  ⚠️ Unknown category: {cat}")
            continue
        title, gen_fn = GENERATORS[cat]
        cfgs = gen_fn()
        # Override start/end
        for name, p in cfgs:
            p.start = args.start
            p.end = args.end
        all_configs.extend([(cat, name, p) for name, p in cfgs])

    # Gather all tickers needed
    needed = set(ALL_TICKERS)
    for _, _, p in all_configs:
        needed.update(p.offensive)
        needed.update(p.defensive)
        needed.update(p.canary_assets)
    needed.add('SPY')
    needed.add('VIX')
    needed.add('HYG')
    needed.add('IEF')
    needed.add('LQD')

    print(f"\n📈 Loading {len(needed)} ETFs...")
    prices = load_prices(sorted(needed))
    print(f"  {len(prices)} loaded successfully")
    missing = needed - set(prices.keys())
    if missing:
        print(f"  ⚠️ Missing: {sorted(missing)}")

    print(f"\n🔧 Precomputing indicators...")
    ind = precompute(prices)
    print(f"  Done ({len(ind)} tickers)")

    # Run all configs with multiprocessing
    print(f"\n🚀 Running {len(all_configs)} configs with {args.workers} workers...")

    params_list = [p for _, _, p in all_configs]

    with Pool(args.workers, initializer=_init, initargs=(prices, ind)) as pool:
        results_list = pool.map(_run_one, params_list)

    # Group results by category
    cat_results = {}
    all_results = []
    for (cat, name, _), m in zip(all_configs, results_list):
        if cat not in cat_results:
            cat_results[cat] = []
        cat_results[cat].append((name, m))
        all_results.append((name, m))

    # Print per-category tables
    for cat in cats:
        if cat in cat_results:
            title, _ = GENERATORS[cat]
            print_table(f"[{cat}] {title}", cat_results[cat])

    # Benchmark: SPY & VT Buy & Hold
    for bm_ticker, bm_label in [('SPY', 'SPY B&H (US)'), ('VT', 'VT  B&H (Global)')]:
        bm_ind = ind.get(bm_ticker)
        if bm_ind is not None:
            s = bm_ind['price']
            s = s[(s.index >= args.start) & (s.index <= args.end)]
            if len(s) > 1:
                yrs = (s.index[-1] - s.index[0]).days / 365.25
                bm_cagr = (s.iloc[-1] / s.iloc[0]) ** (1/yrs) - 1
                pk = s.cummax()
                bm_mdd = (s / pk - 1).min()
                dr = s.pct_change().dropna()
                bm_sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
                print(f"\n  📊 Benchmark: {bm_label}  CAGR {bm_cagr:+.1%}  MDD {bm_mdd:.1%}  Sharpe {bm_sh:.3f}")

    # Overall top 25
    print_top_n(all_results, 25)

    # ─── Year-by-year validation for Top 5 ───
    valid = [(name, m) for name, m in all_results if m is not None]
    valid.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    top5_names = [name for name, _ in valid[:5]]
    top5_params = {name: p for cat, name, p in all_configs if name in top5_names}

    print(f"\n{'='*100}")
    print(f"  📅 Year-by-Year Validation (Top 5)")
    print(f"{'='*100}")

    # Run year-by-year for each top5 config
    for rank, name in enumerate(top5_names, 1):
        p = top5_params.get(name)
        if p is None:
            continue

        yearly = []
        for year in range(2017, 2027):
            yp = SP(**{k: v for k, v in p.__dict__.items()})
            yp.start = f'{year}-01-01'
            yp.end = f'{year}-12-31'
            yp._anchor = p._anchor
            yp._n_tranches = 1
            df = run_bt(prices, ind, yp)
            m = metrics(df) if df is not None else None
            yearly.append((year, m))

        print(f"\n  #{rank} {name}")
        print(f"  {'Year':>6} {'Return':>8} {'MDD':>8} {'Sharpe':>8}")
        print(f"  {'─'*40}")
        for year, m in yearly:
            if m is None:
                print(f"  {year:>6} {'N/A':>8}")
            elif m.get('Days', 365) < 300:
                # Partial year: show actual return, not annualized CAGR
                print(f"  {year:>6} {m['TotalRet']:>+7.1%}* {m['MDD']:>7.1%} {m['Sharpe']:>8.3f}")
            else:
                print(f"  {year:>6} {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Sharpe']:>8.3f}")
        print(f"  (* = YTD, not annualized)")

    # ─── Start-date sensitivity for Top 5 ───
    print(f"\n{'='*100}")
    print(f"  🔄 Start-Date Sensitivity (Top 5, start 2016~2020)")
    print(f"{'='*100}")
    print(f"  {'Name':<22} {'2016':>8} {'2017':>8} {'2018':>8} {'2019':>8} {'2020':>8} {'σ(S)':>7}")
    print(f"  {'─'*72}")

    for name in top5_names:
        p = top5_params.get(name)
        if p is None:
            continue
        sharpes = []
        row = f"  {name:<22}"
        for start_yr in (2016, 2017, 2018, 2019, 2020):
            sp = SP(**{k: v for k, v in p.__dict__.items()})
            sp.start = f'{start_yr}-01-01'
            sp.end = args.end
            sp._anchor = p._anchor
            sp._n_tranches = 1
            df = run_bt(prices, ind, sp)
            m = metrics(df) if df is not None else None
            if m:
                sharpes.append(m['Sharpe'])
                row += f" {m['Sharpe']:>8.3f}"
            else:
                row += f" {'N/A':>8}"
        sigma = np.std(sharpes) if len(sharpes) > 1 else 0
        row += f" {sigma:>7.3f}"
        print(row)

    print(f"\n  Total configs tested: {len(all_configs)}")


if __name__ == '__main__':
    main()
