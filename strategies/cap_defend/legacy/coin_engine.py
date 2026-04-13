#!/usr/bin/env python3
"""
Strategy Engine V14 — Unified backtest for Cap Defend coin strategy.

V14: DD Exit (60d/-25%), Blacklist (-15%/7d), Drift (10% half-TO)
FIX v2: Look-ahead bias (iloc→loc), R2 order, G4/G5 force rebal, W5 edge case
"""

import os, json, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
UNIVERSE_FILE = os.path.join(BASE_DIR, '..', '..', 'backup_20260125',
                             'historical_universe.json')

STABLECOINS = {'USDT','USDC','BUSD','DAI','UST','TUSD','PAX','GUSD',
               'FRAX','LUSD','MIM','USDN','FDUSD'}
EXCLUDE = STABLECOINS | {'WBTC','USD1','USDE'}

# ─── V18 방어자산 ────────────────────────────────────────────────
DEFENSE_TICKERS = ['PAXG-USD']                    # 백테스트/시그널 유니버스
DEFENSE_TICKER_MAP = {'PAXG-USD': 'XAUT'}         # 백테스트→실매매 매핑
DEFENSE_SMA = 120            # SMA(120) — 장기 추세 필터
DEFENSE_LOOKBACK = 126       # 6개월 (거래일)
DEFENSE_MAX_PICKS = 3
DEFENSE_CAP = 0.33


# ─── Params ─────────────────────────────────────────────────────────

@dataclass
class Params:
    canary: str = 'baseline'       # baseline, K1..K5
    health: str = 'baseline'       # baseline, H1..H5
    selection: str = 'baseline'    # baseline, S1..S15
    weighting: str = 'baseline'    # baseline, W1..W8
    rebalancing: str = 'baseline'  # baseline, R1..R5
    risk: str = 'baseline'         # baseline, G1..G5
    tx_cost: float = 0.004
    sma_period: int = 150
    # K8 vote system: list of SMA periods + momentum periods, threshold
    vote_smas: tuple = ()       # e.g. (80, 50) — BTC > SMA(n) conditions
    vote_moms: tuple = ()       # e.g. (21,) — mom(n) > 0 conditions
    vote_threshold: int = 2     # need this many True to be ON
    # K9 SMA crossover: short_sma > long_sma
    cross_short: int = 0        # e.g. 20 — SMA(20)
    cross_long: int = 0         # e.g. 80 — SMA(80)
    # K10 persistence: require N consecutive months ON/OFF
    persist_period: int = 0     # SMA period to check
    persist_months: int = 2     # consecutive months required
    # Health params
    health_sma: int = 30
    health_mom_short: int = 21
    health_mom_long: int = 90
    health_vol_window: int = 90
    vol_cap: float = 0.05
    n_picks: int = 5
    top_n: int = 50
    start_date: str = '2018-01-01'
    end_date: str = '2025-06-30'
    initial_capital: float = 10000.0
    post_flip_delay: int = 0    # Extra rebal N days after OFF→ON flip (0=disabled)
    yellow_threshold: int = 0   # Risk-On: if healthy < N, go 50% cash (0=disabled)
    canary_band: float = 0.0   # Hysteresis band ±% (0=disabled)
    canary_grace: int = 0      # Grace days before turning OFF (0=disabled)
    canary_consec: int = 0     # Consecutive days to confirm state change (0=disabled)
    # V14: DD Exit, Blacklist, Drift
    dd_exit_lookback: int = 0      # 0=disabled, 60=check 60-day peak
    dd_exit_threshold: float = -0.25  # exit if DD worse than this
    bl_threshold: float = 0.0      # 0=disabled, -0.15=blacklist on -15% daily drop
    bl_days: int = 7               # days to exclude after blacklist trigger
    drift_threshold: float = 0.0   # 0=disabled, 0.10=rebal on 10% half-TO

    @property
    def label(self):
        parts = []
        for layer in ('canary','health','selection','weighting','rebalancing','risk'):
            v = getattr(self, layer)
            if v != 'baseline':
                parts.append(v)
        return '+'.join(parts) if parts else 'BASELINE'


# ─── Data Loading ───────────────────────────────────────────────────

def load_universe():
    for path in [UNIVERSE_FILE, os.path.join(DATA_DIR, 'historical_universe.json')]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("historical_universe.json not found")

def load_price(ticker):
    fpath = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(fpath): return None
    df = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date').sort_index()
    for col in ['Open','High','Low','Close','Volume']:
        if col not in df.columns:
            df[col] = 0 if col == 'Volume' else df.get('Close', df.iloc[:,0])
    return df

def load_all_prices(tickers):
    prices = {}
    for t in tickers:
        df = load_price(t)
        if df is not None and len(df) > 30:
            prices[t] = df
    return prices

def filter_universe(universe_map, top_n=50):
    out = {}
    for mk, tickers in universe_map.items():
        clean = [t for t in tickers if t.replace('-USD','') not in EXCLUDE]
        out[mk] = clean[:top_n]
    return out

def get_universe_for_date(universe_map, date):
    mk = date.strftime('%Y-%m') + '-01'
    for k in sorted(universe_map.keys(), reverse=True):
        if k <= mk:
            return universe_map[k]
    return []

def load_data(top_n=50):
    um = load_universe()
    fm = filter_universe(um, top_n)
    tickers = set()
    for ts in fm.values():
        tickers.update(ts)
    tickers.update(['BTC-USD','ETH-USD'])
    prices = load_all_prices(tickers)
    return prices, fm


# ─── Indicators ─────────────────────────────────────────────────────

def _close_to(ticker, prices, date):
    """Get Close series up to date (inclusive). No look-ahead."""
    if ticker not in prices: return pd.Series(dtype=float)
    return prices[ticker]['Close'].loc[:date]

def calc_ret(s, d):
    if len(s) < d + 1: return 0
    return s.iloc[-1] / s.iloc[-d-1] - 1

def get_vol(s, d):
    if len(s) < d + 1: return 1.0
    return s.pct_change().iloc[-d:].std()

def get_sma(s, period):
    if len(s) < period: return np.nan
    return s.rolling(period).mean().iloc[-1]

def _calc_sharpe_score(s, d):
    """Sharpe ratio over trailing d days (annualized √252)."""
    if len(s) < d + 1: return 0
    ret = s.pct_change().iloc[-d:]
    return (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0


def _cap_rank_norm(ticker, ordered_tickers):
    """0=largest cap, 1=smallest cap. Universe order is market-cap order."""
    if not ordered_tickers:
        return 1.0
    denom = max(1, len(ordered_tickers) - 1)
    try:
        return ordered_tickers.index(ticker) / denom
    except ValueError:
        return 1.0


def _coin_selection_metrics(ticker, ordered_tickers, prices, date):
    close = _close_to(ticker, prices, date)
    mom30 = calc_ret(close, 30) if len(close) > 30 else -999
    vol90 = get_vol(close, 90)
    sharpe_like = mom30 / max(vol90, 1e-6)
    cap_norm = _cap_rank_norm(ticker, ordered_tickers)
    return {
        'ticker': ticker,
        'mom30': mom30,
        'vol90': vol90,
        'sharpe_like': sharpe_like,
        'cap_norm': cap_norm,
        'close': close,
    }

def get_price(ticker, prices, date):
    if ticker not in prices: return 0
    idx = prices[ticker].index.get_indexer([date], method='ffill')[0]
    return prices[ticker]['Close'].iloc[idx] if idx >= 0 else 0


# ─── Canary ─────────────────────────────────────────────────────────

def resolve_canary(prices, date, params, state):
    close = _close_to('BTC-USD', prices, date)
    min_len = 200 if params.canary == 'K2' else params.sma_period
    if len(close) < min_len: return False

    cur = close.iloc[-1]
    sma150 = get_sma(close, params.sma_period)
    prev = state.get('prev_canary', False)
    c = params.canary

    if c == 'none':
        return True

    if c == 'baseline':
        return cur > sma150

    elif c == 'K1':  # Hysteresis band
        return cur > sma150 * 0.98 if prev else cur > sma150 * 1.02

    elif c == 'K2':  # Asymmetric MA (entry SMA150, exit SMA200)
        sma200 = get_sma(close, 200)
        return cur > sma200 if prev else cur > sma150

    elif c == 'K3':  # Grace period (7-day OFF delay)
        raw = cur > sma150
        if raw:
            state['canary_off_days'] = 0
            return True
        state['canary_off_days'] = state.get('canary_off_days', 0) + 1
        return prev if state['canary_off_days'] <= 7 else False

    elif c == 'K4':  # Dual (BTC + ETH)
        ec = _close_to('ETH-USD', prices, date)
        if len(ec) < params.sma_period: return False
        return cur > sma150 and ec.iloc[-1] > get_sma(ec, params.sma_period)

    elif c == 'K5':  # 2-of-3 vote
        sma50 = get_sma(close, 50)
        mom21 = calc_ret(close, 21)
        return sum([cur > sma150, cur > sma50, mom21 > 0]) >= 2

    elif c == 'K6':  # graduated entry: 5-level (SMA 60/70/80/90/100)
        levels = [60, 70, 80, 90, 100]
        count = sum(1 for p in levels if cur > get_sma(close, p))
        state['canary_level'] = count
        prev_level = state.get('prev_canary_level', 0)
        state['canary_level_changed'] = (count != prev_level)
        return count > 0

    elif c == 'K7':  # graduated entry: 3-level (SMA 60/80/100)
        levels = [60, 80, 100]
        count = sum(1 for p in levels if cur > get_sma(close, p))
        state['canary_level'] = count
        prev_level = state.get('prev_canary_level', 0)
        state['canary_level_changed'] = (count != prev_level)
        return count > 0

    elif c == 'K8':  # custom vote system
        votes = []
        for p in params.vote_smas:
            sma_val = get_sma(close, p)
            if params.canary_band > 0:
                # Hysteresis: easier to stay ON, harder to turn ON
                if prev:
                    votes.append(cur > sma_val * (1 - params.canary_band / 100))
                else:
                    votes.append(cur > sma_val * (1 + params.canary_band / 100))
            else:
                votes.append(cur > sma_val)
        for p in params.vote_moms:
            mom = calc_ret(close, p)
            votes.append(mom > 0)
        raw = sum(votes) >= params.vote_threshold

        # Grace period: delay turning OFF by N days
        if params.canary_grace > 0:
            if raw:
                state['canary_grace_count'] = 0
                return True
            grace = state.get('canary_grace_count', 0) + 1
            state['canary_grace_count'] = grace
            if grace <= params.canary_grace and prev:
                return True  # stay ON during grace
            return False

        # Consecutive days: require N days to confirm state change
        if params.canary_consec > 0:
            if raw == prev:
                state['canary_change_count'] = 0
                return prev
            cnt = state.get('canary_change_count', 0) + 1
            state['canary_change_count'] = cnt
            if cnt >= params.canary_consec:
                state['canary_change_count'] = 0
                return raw  # confirmed
            return prev  # not yet confirmed

        return raw

    elif c == 'K9':  # SMA crossover: short SMA > long SMA
        sma_s = get_sma(close, params.cross_short)
        sma_l = get_sma(close, params.cross_long)
        if sma_s is None or sma_l is None:
            return False
        return float(sma_s) > float(sma_l)

    elif c == 'K10':  # persistence: require N consecutive days ON/OFF
        raw = cur > get_sma(close, params.persist_period)
        days_needed = params.persist_months * 21  # trading days
        on_consec = state.get('canary_on_consec', 0)
        off_consec = state.get('canary_off_consec', 0)
        if raw:
            on_consec += 1
            off_consec = 0
        else:
            off_consec += 1
            on_consec = 0
        state['canary_on_consec'] = on_consec
        state['canary_off_consec'] = off_consec
        if prev:  # currently ON
            return off_consec < days_needed  # stay ON until enough OFF days
        else:  # currently OFF
            return on_consec >= days_needed  # turn ON after enough ON days

    return cur > sma150


# ─── Health ─────────────────────────────────────────────────────────

def check_health_raw(ticker, prices, date, params):
    close = _close_to(ticker, prices, date)
    if len(close) < 90: return False

    cur = close.iloc[-1]
    sma30 = get_sma(close, 30)
    mom21 = calc_ret(close, 21)
    vol90 = get_vol(close, 90)
    h = params.health

    if h == 'none':
        return True

    if h in ('baseline', 'H3'):
        return cur > sma30 and mom21 > 0 and vol90 <= params.vol_cap

    elif h == 'H1':  # + mom90
        return (cur > sma30 and mom21 > 0
                and calc_ret(close, 90) > 0 and vol90 <= params.vol_cap)

    elif h == 'H2':  # dual MA
        sma90 = get_sma(close, 90)
        return cur > sma30 and cur > sma90 and vol90 <= params.vol_cap

    elif h == 'H4':  # relative strength vs BTC
        bc = _close_to('BTC-USD', prices, date)
        if len(bc) < 30: return False
        # Date-aligned ratio (Pandas auto-aligns on index)
        ratio = (close / bc).dropna()
        if len(ratio) < 30: return False
        base_ok = cur > sma30 and mom21 > 0 and vol90 <= params.vol_cap
        return base_ok and ratio.iloc[-1] > ratio.rolling(30).mean().iloc[-1]

    elif h == 'H5':  # vol acceleration block
        vol30 = get_vol(close, 30)
        return (cur > sma30 and mom21 > 0
                and vol90 <= params.vol_cap and vol30 <= vol90 * 1.5)

    elif h == 'HK':  # configurable health params
        h_mom_s = calc_ret(close, params.health_mom_short)
        h_vol = get_vol(close, params.health_vol_window)
        base = h_mom_s > 0 and h_vol <= params.vol_cap
        if params.health_sma > 0:
            base = base and cur > get_sma(close, params.health_sma)
        if params.health_mom_long > 0:
            return base and calc_ret(close, params.health_mom_long) > 0
        return base

    elif h == 'HL':  # minimal: mom_long + vol only (no SMA, no short mom)
        mom_l = calc_ret(close, params.health_mom_long)
        h_vol = get_vol(close, params.health_vol_window)
        return mom_l > 0 and h_vol <= params.vol_cap

    elif h == 'HM':  # one-fast: SMA + mom_long + vol (no short mom)
        h_sma = get_sma(close, params.health_sma)
        mom_l = calc_ret(close, params.health_mom_long)
        h_vol = get_vol(close, params.health_vol_window)
        return cur > h_sma and mom_l > 0 and h_vol <= params.vol_cap

    elif h == 'HN':  # smooth crossover: short SMA > long SMA + vol
        sma_s = get_sma(close, params.health_mom_short)  # reuse as short SMA
        sma_l = get_sma(close, params.health_sma)         # reuse as long SMA
        h_vol = get_vol(close, params.health_vol_window)
        if sma_s is None or sma_l is None:
            return False
        return float(sma_s) > float(sma_l) and h_vol <= params.vol_cap

    elif h == 'HO':  # risk-adjusted: mom/vol ratio threshold
        mom_l = calc_ret(close, params.health_mom_long)
        h_vol = get_vol(close, params.health_vol_window)
        if h_vol <= 0:
            return False
        return mom_l > 0 and (mom_l / h_vol) > 0.5 and h_vol <= params.vol_cap

    elif h == 'HP':  # stable winner: mom + vol + vol acceleration
        mom_l = calc_ret(close, params.health_mom_long)
        h_vol = get_vol(close, params.health_vol_window)
        vol30 = get_vol(close, 30)
        return (mom_l > 0 and h_vol <= params.vol_cap
                and vol30 <= h_vol * 1.35)

    elif h == 'HQ':  # SMA only + vol (simplest trend)
        h_sma = get_sma(close, params.health_sma)
        h_vol = get_vol(close, params.health_vol_window)
        return cur > h_sma and h_vol <= params.vol_cap

    elif h == 'HV':  # mom vote: Vol hard gate + mom lookback vote
        # health_mom_short = threshold (how many mom conditions must pass)
        # vote_moms = tuple of mom periods to vote on
        h_vol = get_vol(close, params.health_vol_window)
        if h_vol > params.vol_cap:
            return False
        votes = sum(1 for p in params.vote_moms if calc_ret(close, p) > 0)
        return votes >= params.health_mom_short

    return cur > sma30 and mom21 > 0 and vol90 <= params.vol_cap


def check_coin_dd_exit(ticker, prices, date, lookback, threshold):
    """Returns True if coin should be EXITED (DD from recent peak worse than threshold)."""
    close = _close_to(ticker, prices, date)
    if len(close) < lookback:
        return False
    recent = close.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0:
        return False
    dd = close.iloc[-1] / peak - 1
    return dd <= threshold


def get_healthy_coins(prices, universe, date, params, state):
    h = params.health

    # HU: union filter (pass if H1 OR H5), then standard top-N selection
    # HX: dual top-N selection, return union (5~10 coins)
    if h in ('HU', 'HX'):
        from copy import copy
        p1 = copy(params); p1.health = 'H1'
        p5 = copy(params); p5.health = 'H5'
        h1_coins = [t for t in universe if check_health_raw(t, prices, date, p1)]
        h5_coins = [t for t in universe if check_health_raw(t, prices, date, p5)]

        if h == 'HU':
            # Union of healthy pools, deduplicated, market cap order preserved
            seen = set()
            result = []
            for t in universe:
                if (t in h1_coins or t in h5_coins) and t not in seen:
                    result.append(t)
                    seen.add(t)
            return result
        else:  # HX: union of top-N picks from each
            n = params.n_picks
            picks1 = h1_coins[:n]
            picks5 = h5_coins[:n]
            seen = set()
            result = []
            for t in picks1 + picks5:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
            return result

    healthy = []
    streak = state.setdefault('health_fail_streak', {})
    for t in universe:
        raw = check_health_raw(t, prices, date, params)
        if h == 'H3':
            if raw:
                streak[t] = 0
                healthy.append(t)
            else:
                streak[t] = streak.get(t, 0) + 1
                if streak[t] < 3:
                    healthy.append(t)
        else:
            if raw:
                healthy.append(t)
    return healthy


# ─── Selection ──────────────────────────────────────────────────────

def select_coins(healthy, prices, date, params, state):
    n = params.n_picks
    s = params.selection
    if not healthy:
        return []

    # HX: healthy is already the pre-selected union, return all
    if params.health == 'HX':
        return healthy

    if s == 'baseline':
        return healthy[:n]

    elif s == 'SG':  # V18 Greedy Absorption: 시총 큰 코인이 Mom30 높으면 작은 코인 제거
        picks = list(healthy[:n])
        for i in range(len(picks) - 1, 0, -1):
            c_above = _close_to(picks[i-1], prices, date)
            c_below = _close_to(picks[i], prices, date)
            mom_above = calc_ret(c_above, 30) if len(c_above) > 30 else -999
            mom_below = calc_ret(c_below, 30) if len(c_below) > 30 else -999
            if mom_above >= mom_below:
                picks.pop(i)
        return picks

    elif s == 'S1':  # cap-constrained momentum
        pool = healthy[:15]
        scored = []
        for t in pool:
            c = _close_to(t, prices, date)
            scored.append((t, calc_ret(c, 21) if len(c) > 21 else -999))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t,_ in scored[:n]]

    elif s == 'S2':  # blend rank
        moms = []
        for t in healthy:
            c = _close_to(t, prices, date)
            moms.append((t, calc_ret(c, 21) if len(c) > 21 else -999))
        mom_sorted = sorted(moms, key=lambda x: x[1], reverse=True)
        mom_rank = {t: i for i,(t,_) in enumerate(mom_sorted)}
        cap_rank = {t: i for i,t in enumerate(healthy)}
        blend = [(t, 0.5*cap_rank[t] + 0.5*mom_rank.get(t, len(healthy)))
                 for t in healthy]
        blend.sort(key=lambda x: x[1])
        return [t for t,_ in blend[:n]]

    elif s == 'S3':  # 3+2 bucket
        core = healthy[:min(3, len(healthy))]
        rest = [t for t in healthy[3:] if t not in core]
        scored = []
        for t in rest:
            c = _close_to(t, prices, date)
            scored.append((t, calc_ret(c, 21) if len(c) > 21 else -999))
        scored.sort(key=lambda x: x[1], reverse=True)
        sats = [t for t,_ in scored[:max(0, n - len(core))]]
        return core + sats

    elif s == 'S4':  # core-satellite (BTC+ETH fixed)
        core = [t for t in ['BTC-USD','ETH-USD'] if t in healthy]
        rest = [t for t in healthy if t not in core]
        return (core + rest)[:n]

    elif s == 'S5':  # incumbent carry (+2 bonus)
        prev = set(state.get('prev_picks', []))
        ranked = [(t, i + (-2 if t in prev else 0)) for i,t in enumerate(healthy)]
        ranked.sort(key=lambda x: x[1])
        return [t for t,_ in ranked[:n]]

    elif s == 'S6':  # Sharpe(126)+Sharpe(252) score — 라이브 V12 방식
        scored = []
        for t in healthy:
            c = _close_to(t, prices, date)
            sh = _calc_sharpe_score(c, 126) + _calc_sharpe_score(c, 252)
            scored.append((t, sh))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t,_ in scored[:n]]

    elif s == 'S7':  # Cap-constrained Sharpe: Top15 중 Sharpe score Top5
        pool = healthy[:15]
        scored = []
        for t in pool:
            c = _close_to(t, prices, date)
            sh = _calc_sharpe_score(c, 126) + _calc_sharpe_score(c, 252)
            scored.append((t, sh))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t,_ in scored[:n]]

    elif s == 'S8':  # Sharpe + incumbent carry
        prev = set(state.get('prev_picks', []))
        scored = []
        for i, t in enumerate(healthy):
            c = _close_to(t, prices, date)
            sh = _calc_sharpe_score(c, 126) + _calc_sharpe_score(c, 252)
            # incumbent bonus: +1.0 to score (roughly equivalent to +2 rank)
            if t in prev:
                sh += 1.0
            scored.append((t, sh))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t,_ in scored[:n]]

    elif s == 'S9':  # hysteresis: keep incumbents unless new coin is 3+ ranks higher
        prev = set(state.get('prev_picks', []))
        # Start with market cap order
        top_n = healthy[:n]
        # For each incumbent in healthy but not in top_n, check if it should stay
        for t in list(prev):
            if t in healthy and t not in top_n:
                t_rank = healthy.index(t)
                # Find the lowest-ranked non-incumbent in current top_n
                worst_new = None
                worst_rank = -1
                for c in top_n:
                    if c not in prev:
                        cr = healthy.index(c)
                        if cr > worst_rank:
                            worst_rank = cr
                            worst_new = c
                # Keep incumbent if replacement isn't 3+ ranks better
                if worst_new and t_rank - worst_rank < 3:
                    top_n = [worst_new if x == worst_new else x for x in top_n]
                    top_n[top_n.index(worst_new)] = t  # actually swap
        # Re-sort to maintain consistency
        order = {t: healthy.index(t) if t in healthy else 999 for t in top_n}
        top_n.sort(key=lambda x: order[x])
        return top_n[:n]

    elif s == 'S10':  # min 2-month hold: keep incumbents if healthy
        prev = set(state.get('prev_picks', []))
        # Keep all healthy incumbents
        keep = [t for t in healthy if t in prev][:n]
        # Fill remaining slots from market cap order
        remaining = n - len(keep)
        if remaining > 0:
            fill = [t for t in healthy if t not in keep][:remaining]
            keep.extend(fill)
        return keep[:n]

    elif s == 'S11':  # flexible cap bonus: Mom30 × f(market cap)
        scored = []
        for t in healthy:
            m = _coin_selection_metrics(t, healthy, prices, date)
            cap_mult = 1.30 - 0.30 * m['cap_norm']  # large cap +30%, tail +0%
            scored.append((t, m['mom30'] * cap_mult))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t,_ in scored[:n]]

    elif s == 'S12':  # blended rank: Mom30 rank + market-cap rank
        metrics = [_coin_selection_metrics(t, healthy, prices, date) for t in healthy]
        mom_sorted = sorted(metrics, key=lambda x: x['mom30'], reverse=True)
        mom_rank = {m['ticker']: i for i, m in enumerate(mom_sorted)}
        scored = []
        for cap_rank, t in enumerate(healthy):
            score = 0.70 * mom_rank[t] + 0.30 * cap_rank
            scored.append((t, score))
        scored.sort(key=lambda x: x[1])
        return [t for t,_ in scored[:n]]

    elif s == 'S13':  # Sharpe-like rank + market-cap rank blend
        metrics = [_coin_selection_metrics(t, healthy, prices, date) for t in healthy]
        sh_sorted = sorted(metrics, key=lambda x: x['sharpe_like'], reverse=True)
        sh_rank = {m['ticker']: i for i, m in enumerate(sh_sorted)}
        scored = []
        for cap_rank, t in enumerate(healthy):
            score = 0.65 * sh_rank[t] + 0.35 * cap_rank
            scored.append((t, score))
        scored.sort(key=lambda x: x[1])
        return [t for t,_ in scored[:n]]

    elif s == 'S14':  # full-pool Sharpe-like sort; weighting decides cap tilt
        scored = []
        for t in healthy:
            m = _coin_selection_metrics(t, healthy, prices, date)
            scored.append((t, m['sharpe_like']))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t,_ in scored[:n]]

    elif s == 'S15':  # challenger replace: small caps must beat incumbents clearly
        metrics = {t: _coin_selection_metrics(t, healthy, prices, date) for t in healthy}
        picks = healthy[:n]
        if not picks:
            return []
        margin = 1.15
        for challenger in healthy[n:]:
            weakest = min(picks, key=lambda t: metrics[t]['sharpe_like'])
            c_score = metrics[challenger]['sharpe_like']
            w_score = metrics[weakest]['sharpe_like']
            if c_score > max(w_score * margin, w_score + 0.50):
                picks[picks.index(weakest)] = challenger
        picks = list(dict.fromkeys(picks))
        picks.sort(key=lambda t: (
            -metrics[t]['sharpe_like'],
            metrics[t]['cap_norm'],
        ))
        return picks[:n]

    return healthy[:n]


# ─── Weighting ──────────────────────────────────────────────────────

def compute_weights(picks, prices, date, params, state):
    if not picks:
        return {'CASH': 1.0}
    n = len(picks)
    w = params.weighting

    if w == 'baseline':
        return {t: 1.0/n for t in picks}

    elif w == 'WC':  # EW with cash fill: always 5 slots, empty slots → cash
        max_n = params.n_picks  # typically 5
        coin_pct = 1.0 / max_n  # always 20% per slot
        wts = {t: coin_pct for t in picks}
        cash_pct = (max_n - n) * coin_pct
        if cash_pct > 0.001:
            wts['CASH'] = cash_pct
        return wts

    elif w == 'WG':  # V18 Greedy: EW + Cap 33%
        cap = 1.0 / 3  # 33%
        coin_pct = min(1.0 / n, cap)
        wts = {t: coin_pct for t in picks}
        cash_pct = 1.0 - sum(wts.values())
        if cash_pct > 0.001:
            wts['CASH'] = cash_pct
        return wts

    elif w == 'W1':  # rank decay
        decay = [0.30, 0.25, 0.20, 0.15, 0.10]
        wts = {picks[i]: decay[i] if i < len(decay) else decay[-1]
               for i in range(n)}
        s = sum(wts.values())
        return {t: v/s for t,v in wts.items()}

    elif w == 'W2':  # shrunk inv-vol (70% inv-vol + 30% equal)
        ivs = []
        for t in picks:
            c = _close_to(t, prices, date)
            v = get_vol(c, 90)
            ivs.append(1/v if v > 0 else 1.0)
        total = sum(ivs)
        eq = 1.0 / n
        wts = {}
        for i,t in enumerate(picks):
            iv_w = ivs[i] / total if total > 0 else eq
            wts[t] = 0.7 * iv_w + 0.3 * eq
        s = sum(wts.values())
        return {t: v/s for t,v in wts.items()}

    elif w == 'W6':  # pure inverse volatility (100%)
        ivs = []
        for t in picks:
            c = _close_to(t, prices, date)
            v = get_vol(c, 90)
            ivs.append(1/v if v > 0 else 1.0)
        total = sum(ivs)
        if total <= 0:
            return {t: 1.0/n for t in picks}
        return {t: (ivs[i]/total) for i,t in enumerate(picks)}

    elif w == 'W7':  # market-cap proportional (supply × price proxy)
        supply_path = os.path.join(DATA_DIR, 'circulating_supply.json')
        try:
            with open(supply_path) as f:
                supply_map = json.load(f)
        except FileNotFoundError:
            return {t: 1.0/n for t in picks}
        mcaps = []
        for t in picks:
            s = supply_map.get(t, 0)
            if s > 0:
                c = _close_to(t, prices, date)
                p = c.iloc[-1] if len(c) > 0 else 0
                mcaps.append(s * p)
            else:
                mcaps.append(0)
        total = sum(mcaps)
        if total <= 0:
            return {t: 1.0/n for t in picks}
        return {t: (mcaps[i]/total) for i,t in enumerate(picks)}

    elif w == 'W8':  # score pick first, then tilt weights toward larger caps
        ordered_universe = state.get('current_universe', picks)
        cap_bonus = []
        for t in picks:
            cap_norm = _cap_rank_norm(t, ordered_universe)
            cap_bonus.append(1.20 - 0.20 * cap_norm)  # top cap gets modest extra weight
        total = sum(cap_bonus)
        return {t: cap_bonus[i] / total for i, t in enumerate(picks)}

    elif w == 'W3':  # momentum tilt
        moms = []
        for t in picks:
            c = _close_to(t, prices, date)
            moms.append((t, calc_ret(c, 21) if len(c) > 21 else 0))
        moms.sort(key=lambda x: x[1], reverse=True)
        base = 1.0 / n
        wts = {}
        for rank,(t,_) in enumerate(moms):
            if rank < 2:
                wts[t] = base + 0.05
            elif rank >= n - 2 and n > 2:
                wts[t] = max(0.01, base - 0.05)
            else:
                wts[t] = base
        s = sum(wts.values())
        return {t: v/s for t,v in wts.items()}

    elif w == 'W4':  # breadth-scaled
        hc = state.get('healthy_count', n)
        exposure = min(1.0, hc / params.n_picks)
        cw = exposure / n
        cash_w = 1.0 - exposure
        wts = {t: cw for t in picks}
        if cash_w > 0.001:
            wts['CASH'] = cash_w
        return wts

    elif w == 'W5':  # BTC fill
        base_w = 1.0 / params.n_picks
        wts = {t: base_w for t in picks}
        remaining = params.n_picks - n
        if remaining > 0:
            wts['BTC-USD'] = wts.get('BTC-USD', 0) + remaining * base_w
        s = sum(wts.values())
        return {t: v/s for t,v in wts.items()}

    return {t: 1.0/n for t in picks}


# ─── Risk Overlay ───────────────────────────────────────────────────

def apply_risk(weights, prices, date, params, state):
    g = params.risk
    state['risk_force_rebal'] = False
    if g == 'baseline':
        return weights
    if weights.get('CASH', 0) >= 0.999:
        return weights

    scale = 1.0

    if g == 'G1':  # soft DD overlay
        hwm = state.get('high_watermark', params.initial_capital)
        pv = state.get('current_port_val', params.initial_capital)
        if hwm > 0 and (pv / hwm - 1) < -0.20:
            scale = 0.5
            state['risk_force_rebal'] = True

    elif g == 'G2':  # vol target (80% annual)
        vals = state.get('recent_port_vals', [])
        if len(vals) >= 30:
            rets = pd.Series(vals).pct_change().dropna()
            ann_vol = rets.std() * np.sqrt(365)
            if ann_vol > 0.80:
                scale = max(0.2, min(1.0, 0.80 / ann_vol))

    elif g == 'G3':  # breadth ladder
        hc = state.get('healthy_count', 5)
        if hc == 0:
            state['risk_force_rebal'] = True
            return {'CASH': 1.0}
        elif hc <= 2:
            scale = 0.5
            state['risk_force_rebal'] = True

    elif g == 'G4':  # rank floor — sell coins outside universe
        uni = set(state.get('current_universe', []))
        new_w = {}
        changed = False
        for t,v in weights.items():
            if t == 'CASH' or t in uni:
                new_w[t] = v
            else:
                changed = True
        if changed:
            state['risk_force_rebal'] = True
        if not any(t != 'CASH' for t in new_w):
            return {'CASH': 1.0}
        s = sum(new_w.values())
        return {t: v/s for t,v in new_w.items()} if s > 0 else {'CASH': 1.0}

    elif g == 'G5':  # crash breaker: BTC -10% → 3d cash
        bc = _close_to('BTC-USD', prices, date)
        if len(bc) >= 2 and (bc.iloc[-1] / bc.iloc[-2] - 1) < -0.10:
            state['crash_cooldown'] = 4  # +1 so today counts
        if state.get('crash_cooldown', 0) > 0:
            state['crash_cooldown'] -= 1
            state['risk_force_rebal'] = True
            return {'CASH': 1.0}

    elif g.startswith('G5'):  # G5 variants: G5t{thresh}c{cool}
        # Parse: G5t8c3 = threshold -8%, cooldown 3 days
        # G5t10c3 = original G5
        import re
        m = re.match(r'G5t(\d+)c(\d+)', g)
        if m:
            thresh = -int(m.group(1)) / 100.0  # e.g. -0.08
            cool = int(m.group(2))              # e.g. 3
        else:
            thresh, cool = -0.10, 3
        bc = _close_to('BTC-USD', prices, date)
        if len(bc) >= 2 and (bc.iloc[-1] / bc.iloc[-2] - 1) < thresh:
            state['crash_cooldown'] = cool + 1
        if state.get('crash_cooldown', 0) > 0:
            state['crash_cooldown'] -= 1
            state['risk_force_rebal'] = True
            return {'CASH': 1.0}

    elif g == 'G6':  # per-coin crash: exclude crashed coins, keep rest
        coin_cd = state.get('coin_cooldowns', {})
        # Check each coin for -10% daily drop
        for t in list(weights.keys()):
            if t == 'CASH':
                continue
            c = _close_to(t, prices, date)
            if len(c) >= 2 and (c.iloc[-1] / c.iloc[-2] - 1) < -0.10:
                coin_cd[t] = 4
        state['coin_cooldowns'] = coin_cd
        # Remove coins in cooldown
        new_w = {}
        changed = False
        for t, v in weights.items():
            if t == 'CASH':
                new_w[t] = v
            elif coin_cd.get(t, 0) > 0:
                coin_cd[t] -= 1
                changed = True
            else:
                new_w[t] = v
        if changed:
            state['risk_force_rebal'] = True
        if not any(t != 'CASH' for t in new_w):
            return {'CASH': 1.0}
        s = sum(new_w.values())
        return {t: v/s for t, v in new_w.items()} if s > 0 else {'CASH': 1.0}

    elif g == 'G7':  # G5 + G6 combo: BTC crash → all cash, coin crash → exclude coin
        # BTC-level crash breaker
        bc = _close_to('BTC-USD', prices, date)
        if len(bc) >= 2 and (bc.iloc[-1] / bc.iloc[-2] - 1) < -0.10:
            state['crash_cooldown'] = 4
        if state.get('crash_cooldown', 0) > 0:
            state['crash_cooldown'] -= 1
            state['risk_force_rebal'] = True
            return {'CASH': 1.0}
        # Per-coin crash
        coin_cd = state.get('coin_cooldowns', {})
        for t in list(weights.keys()):
            if t == 'CASH':
                continue
            c = _close_to(t, prices, date)
            if len(c) >= 2 and (c.iloc[-1] / c.iloc[-2] - 1) < -0.10:
                coin_cd[t] = 4
        state['coin_cooldowns'] = coin_cd
        new_w = {}
        changed = False
        for t, v in weights.items():
            if t == 'CASH':
                new_w[t] = v
            elif coin_cd.get(t, 0) > 0:
                coin_cd[t] -= 1
                changed = True
            else:
                new_w[t] = v
        if changed:
            state['risk_force_rebal'] = True
        if not any(t != 'CASH' for t in new_w):
            return {'CASH': 1.0}
        s = sum(new_w.values())
        return {t: v/s for t, v in new_w.items()} if s > 0 else {'CASH': 1.0}

    if scale < 1.0:
        scaled = {}
        cash_add = 0
        for t,v in weights.items():
            if t == 'CASH':
                scaled[t] = v
            else:
                scaled[t] = v * scale
                cash_add += v * (1 - scale)
        scaled['CASH'] = scaled.get('CASH', 0) + cash_add
        weights = scaled

    # Yellow mode: partial defense on breadth collapse
    if params.yellow_threshold > 0:
        hc = state.get('healthy_count', params.n_picks)
        if hc < params.yellow_threshold and weights.get('CASH', 0) < 0.999:
            yw = {}
            cash_add = 0
            for t, v in weights.items():
                if t == 'CASH':
                    yw[t] = v
                else:
                    yw[t] = v * 0.5
                    cash_add += v * 0.5
            yw['CASH'] = yw.get('CASH', 0) + cash_add
            weights = yw
            state['risk_force_rebal'] = True

    return weights


# ─── Rebalancing Decision ──────────────────────────────────────────

def should_rebalance(weights, holdings, cash, prices, date, params, state):
    if state.get('is_first_day'):
        return True
    if state.get('canary_flipped'):
        return True
    if state.get('risk_force_rebal'):
        return True

    r = params.rebalancing
    imc = state.get('is_month_change', False)

    if r == 'baseline' or r == 'R1':
        return imc

    elif r == 'R2':  # catastrophic exit
        if imc:
            return True
        ms = state.get('month_start_value', params.initial_capital)
        pv = state.get('current_port_val', params.initial_capital)
        if ms > 0 and (pv / ms - 1) < -0.15 and len(holdings) > 0:
            state['catastrophic_triggered'] = True
            return True
        return False

    elif r == 'R3':  # ultra-high TO only
        if imc:
            pv = state.get('current_port_val', 0)
            if pv <= 0:
                return True
            cur_w = {}
            for t, units in holdings.items():
                p = get_price(t, prices, date)
                cur_w[t] = (units * p) / pv
            tgt_w = {t: v for t,v in weights.items() if t != 'CASH'}
            all_t = set(list(cur_w.keys()) + list(tgt_w.keys()))
            to = sum(abs(cur_w.get(t,0) - tgt_w.get(t,0)) for t in all_t) / 2
            return to > 0.50
        return False

    elif r == 'R4':  # banded weight (±5pp)
        if imc:
            return True
        pv = state.get('current_port_val', 0)
        if pv <= 0:
            return False
        for t, tw in weights.items():
            if t == 'CASH': continue
            aw = (holdings.get(t, 0) * get_price(t, prices, date)) / pv
            if abs(aw - tw) > 0.05:
                return True
        return False

    elif r == 'R5':  # anchor day (15th)
        if date.day >= 15:
            key = date.strftime('%Y-%m') + '_a'
            if not state.get(key, False):
                state[key] = True
                return True
        return False

    elif r == 'R6':  # trailing stop from 60d high (-20%)
        if imc:
            return True
        pv = state.get('current_port_val', params.initial_capital)
        rpv = state.get('recent_port_vals', [pv])
        hwm = max(rpv[-60:]) if rpv else pv
        if hwm > 0 and (pv / hwm - 1) < -0.20 and len(holdings) > 0:
            state['catastrophic_triggered'] = True
            return True
        return False

    elif r == 'R7':  # R2 variant: MTD -10% (tighter threshold)
        if imc:
            return True
        ms = state.get('month_start_value', params.initial_capital)
        pv = state.get('current_port_val', params.initial_capital)
        if ms > 0 and (pv / ms - 1) < -0.10 and len(holdings) > 0:
            state['catastrophic_triggered'] = True
            return True
        return False

    elif r == 'R8':  # R2 variant: MTD -20% (looser threshold)
        if imc:
            return True
        ms = state.get('month_start_value', params.initial_capital)
        pv = state.get('current_port_val', params.initial_capital)
        if ms > 0 and (pv / ms - 1) < -0.20 and len(holdings) > 0:
            state['catastrophic_triggered'] = True
            return True
        return False

    elif r == 'RA':  # weekly (every Monday)
        return date.weekday() == 0  # Monday

    elif r == 'RB':  # bi-weekly (every other Monday)
        week_num = date.isocalendar()[1]
        return date.weekday() == 0 and week_num % 2 == 0

    elif r == 'RC':  # mid-month (15th)
        key = date.strftime('%Y-%m') + '_mid'
        if date.day >= 15 and not state.get(key, False):
            state[key] = True
            return True
        return False

    elif r == 'RD':  # bi-monthly (every 2 months)
        return imc and date.month % 2 == 1

    elif r == 'RE':  # quarterly
        return imc and date.month in (1, 4, 7, 10)

    elif r == 'RF':  # daily
        return True

    elif r.startswith('RH'):  # health-triggered rebalancing
        # Parse: RH=1+,  RH2=2+,  RH3=3+
        # Optional anchor: RH2a10 = 2+ unhealthy, monthly anchor on day 10
        import re
        m = re.match(r'RH(\d*)(a(\d+))?$', r)
        threshold = int(m.group(1)) if m and m.group(1) else 1
        anchor_day = int(m.group(3)) if m and m.group(3) else 0

        # Monthly trigger: either imc (day 1) or anchor-day based
        if anchor_day > 0:
            key = date.strftime('%Y-%m') + f'_rh{anchor_day}'
            monthly_trigger = (date.day >= anchor_day and not state.get(key, False))
            if monthly_trigger:
                state[key] = True
        else:
            monthly_trigger = imc

        if monthly_trigger:
            return True

        # Health trigger: check if enough held coins are unhealthy
        healthy_set = state.get('current_healthy_set', set())
        n_bad = sum(1 for t in holdings if t not in healthy_set and holdings[t] > 0)
        return n_bad >= threshold

    elif r.startswith('RX'):  # RX5, RX10, RX15, RX20, RX25 — day-of-month anchor
        anchor_day = int(r[2:])
        key = date.strftime('%Y-%m') + f'_d{anchor_day}'
        if date.day >= anchor_day and not state.get(key, False):
            state[key] = True
            return True
        return False

    elif r == 'R9':  # trailing stop from 30d high (-15%)
        if imc:
            return True
        pv = state.get('current_port_val', params.initial_capital)
        rpv = state.get('recent_port_vals', [pv])
        hwm = max(rpv[-30:]) if rpv else pv
        if hwm > 0 and (pv / hwm - 1) < -0.15 and len(holdings) > 0:
            state['catastrophic_triggered'] = True
            return True
        return False

    return imc


# ─── Execute Rebalance ─────────────────────────────────────────────

def execute_rebalance(holdings, cash, weights, prices, date, tx_cost):
    current_values = {}
    port_val = cash
    for t, units in holdings.items():
        p = get_price(t, prices, date)
        v = units * p
        current_values[t] = v
        port_val += v

    if weights.get('CASH', 0) >= 0.999:
        sell_total = sum(current_values.values())
        return {}, cash + sell_total * (1 - tx_cost)

    target_values = {t: port_val * w for t,w in weights.items() if t != 'CASH'}
    new_holdings = {}
    new_cash = cash
    all_tickers = set(list(current_values.keys()) + list(target_values.keys()))

    # Sells first
    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        p = get_price(t, prices, date)
        if p <= 0:
            if t in holdings:
                new_holdings[t] = holdings[t]
            continue
        if tgt_val >= cur_val:
            if cur_val > 0:
                new_holdings[t] = holdings[t]
        else:
            sell_amount = cur_val - tgt_val
            new_cash += sell_amount * (1 - tx_cost)
            if tgt_val > 0:
                new_holdings[t] = tgt_val / p

    # Then buys
    buys = {}
    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        if tgt_val > cur_val:
            buys[t] = tgt_val - cur_val

    total_buy = sum(buys.values())
    if total_buy > 0:
        scale = min(1.0, new_cash / total_buy)
        for t, buy_val in buys.items():
            p = get_price(t, prices, date)
            if p <= 0: continue
            actual = buy_val * scale
            bought = actual * (1 - tx_cost)
            new_cash -= actual
            new_holdings[t] = new_holdings.get(t, 0) + bought / p

    return new_holdings, new_cash


# ─── Main Backtest ──────────────────────────────────────────────────

def run_backtest(prices, universe_map, params=None, return_state=False):
    if params is None:
        params = Params()

    btc = prices.get('BTC-USD')
    if btc is None:
        return _empty_result(params)

    all_dates = btc.index[(btc.index >= params.start_date) &
                          (btc.index <= params.end_date)]
    if len(all_dates) == 0:
        return _empty_result(params)

    holdings = {}
    cash = params.initial_capital
    state = {
        'prev_canary': False,
        'canary_off_days': 0,
        'health_fail_streak': {},
        'prev_picks': [],
        'scaled_months': 2,
        'month_start_value': params.initial_capital,
        'high_watermark': params.initial_capital,
        'crash_cooldown': 0,
        'coin_cooldowns': {},
        'recent_port_vals': [],
        'prev_month': None,
        'catastrophic_triggered': False,
        'risk_force_rebal': False,
        'canary_on_date': None,
        'post_flip_refreshed': False,
        'blacklist': {},
        'dd_exit_count': 0,
    }

    portfolio_values = []
    rebal_count = 0

    for i, date in enumerate(all_dates):
        cur_month = date.strftime('%Y-%m')
        imc = (state['prev_month'] is not None and cur_month != state['prev_month'])

        pv = _port_val(holdings, cash, prices, date)
        state['current_port_val'] = pv
        state['high_watermark'] = max(state['high_watermark'], pv)
        state['recent_port_vals'].append(pv)
        if len(state['recent_port_vals']) > 60:
            state['recent_port_vals'] = state['recent_port_vals'][-60:]

        if imc:
            state['month_start_value'] = pv
            state['catastrophic_triggered'] = False
            if state['prev_canary']:
                state['scaled_months'] = state.get('scaled_months', 2) + 1

        # V14: Blacklist — daily update
        if params.bl_threshold < 0:
            bl = state['blacklist']
            for t in list(bl.keys()):
                bl[t] -= 1
                if bl[t] <= 0:
                    del bl[t]
            for t in get_universe_for_date(universe_map, date):
                if t not in bl:
                    c = _close_to(t, prices, date)
                    if len(c) >= 2 and (c.iloc[-1] / c.iloc[-2] - 1) <= params.bl_threshold:
                        bl[t] = params.bl_days

        # V14: DD Exit — daily check on held coins
        if params.dd_exit_lookback > 0 and holdings:
            dd_exits = [t for t in list(holdings.keys())
                        if check_coin_dd_exit(t, prices, date,
                                              params.dd_exit_lookback,
                                              params.dd_exit_threshold)]
            if dd_exits:
                for t in dd_exits:
                    p = get_price(t, prices, date)
                    units = holdings.pop(t, 0)
                    if units > 0:
                        cash += units * p * (1 - params.tx_cost)
                state['dd_exit_count'] += len(dd_exits)
                pv = _port_val(holdings, cash, prices, date)
                state['current_port_val'] = pv

        # 1. Canary
        canary_on = resolve_canary(prices, date, params, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['scaled_months'] = 0
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        # K6: level change also counts as canary flip (triggers rebal)
        if params.canary in ('K6', 'K7') and state.get('canary_level_changed'):
            canary_flipped = True

        state['is_first_day'] = (i == 0)
        state['is_month_change'] = imc
        state['canary_flipped'] = canary_flipped
        state['canary_on'] = canary_on

        # 2. Universe & Health
        universe = get_universe_for_date(universe_map, date)
        # V14: Filter blacklisted coins from universe
        if params.bl_threshold < 0:
            universe = [t for t in universe if t not in state['blacklist']]
        state['current_universe'] = universe

        if canary_on:
            healthy = get_healthy_coins(prices, universe, date, params, state)
            state['healthy_count'] = len(healthy)
            state['current_healthy_set'] = set(healthy)
        else:
            healthy = []
            state['healthy_count'] = 0
            state['current_healthy_set'] = set()

        # 3. Selection
        picks = (select_coins(healthy, prices, date, params, state)
                 if canary_on and healthy else [])

        # W5 special: fill with BTC even when no healthy coins
        if not picks and canary_on and params.weighting == 'W5':
            picks = []  # compute_weights W5 handles empty picks below

        # 4. Weighting
        if params.canary in ('K6', 'K7') and canary_on and picks:
            # Graduated entry: level determines allocation & coin count
            level = state.get('canary_level', 0)
            if params.canary == 'K6':
                # K6: 5-level, each level = 1 coin at 20%
                n_coins = min(level, len(picks))
                alloc = level * 0.20
            else:
                # K7: 3-level → 33/66/100% with 2/3/5 coins
                n_coins_map = {1: 2, 2: 3, 3: 5}
                alloc_map = {1: 0.333, 2: 0.666, 3: 1.0}
                n_coins = min(n_coins_map.get(level, 0), len(picks))
                alloc = alloc_map.get(level, 0)
            if n_coins > 0:
                grad_picks = picks[:n_coins]
                coin_pct = alloc / n_coins
                weights = {t: coin_pct for t in grad_picks}
                cash_pct = 1.0 - alloc
                if cash_pct > 0.001:
                    weights['CASH'] = cash_pct
            else:
                weights = {'CASH': 1.0}
        elif picks:
            weights = compute_weights(picks, prices, date, params, state)
        elif canary_on and params.weighting == 'W5':
            # W5: all slots → BTC
            weights = {'BTC-USD': 1.0}
        else:
            weights = {'CASH': 1.0}

        # R1: Scaled reentry — first month after OFF→ON at 50%
        if params.rebalancing == 'R1' and canary_on and state.get('scaled_months', 2) == 0:
            scaled = {}
            ca = 0
            for t,v in weights.items():
                if t == 'CASH':
                    scaled[t] = v
                else:
                    scaled[t] = v * 0.5
                    ca += v * 0.5
            scaled['CASH'] = scaled.get('CASH', 0) + ca
            weights = scaled

        # 5. Risk overlay (may set risk_force_rebal)
        weights = apply_risk(weights, prices, date, params, state)

        # 6. Rebalance decision
        do_rebal = should_rebalance(weights, holdings, cash, prices, date, params, state)

        # V14: Drift trigger — half-turnover exceeds threshold
        if not do_rebal and params.drift_threshold > 0 and canary_on and holdings:
            pv_now = state.get('current_port_val', 0)
            if pv_now > 0:
                cur_w = {}
                for t, units in holdings.items():
                    p = get_price(t, prices, date)
                    cur_w[t] = (units * p) / pv_now
                tgt_w = {t: v for t, v in weights.items() if t != 'CASH'}
                all_t = set(list(cur_w.keys()) + list(tgt_w.keys()))
                half_to = sum(abs(cur_w.get(t, 0) - tgt_w.get(t, 0))
                              for t in all_t) / 2
                if half_to > params.drift_threshold:
                    do_rebal = True

        # Post-flip refresh: extra rebal N days after OFF→ON
        if not do_rebal and params.post_flip_delay > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= params.post_flip_delay:
                    state['post_flip_refreshed'] = True
                    do_rebal = True

        # Catastrophic exit: Override weights (R2, R6-R9)
        if params.rebalancing in ('R2','R6','R7','R8','R9') and state.get('catastrophic_triggered'):
            weights = {'CASH': 1.0}
            picks = []

        if do_rebal:
            holdings, cash = execute_rebalance(holdings, cash, weights, prices,
                                               date, params.tx_cost)
            rebal_count += 1
            state['prev_picks'] = picks[:]

        pv = _port_val(holdings, cash, prices, date)
        portfolio_values.append({'Date': date, 'Value': pv})
        state['prev_canary'] = canary_on
        if params.canary in ('K6', 'K7'):
            state['prev_canary_level'] = state.get('canary_level', 0)
        state['prev_month'] = cur_month

    if not portfolio_values:
        return _empty_result(params)

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    ym = calc_yearly_metrics(pvdf)

    result = {
        'metrics': m,
        'yearly': ym,
        'rebal_count': rebal_count,
        'dd_exit_count': state.get('dd_exit_count', 0),
        'label': params.label,
        'params': params,
        'pv': pvdf,
    }
    if return_state:
        result['final_holdings'] = holdings
        result['final_cash'] = cash
        result['final_state'] = state
        result['last_date'] = all_dates[-1]
    return result


def _port_val(holdings, cash, prices, date):
    v = cash
    for t, units in holdings.items():
        v += units * get_price(t, prices, date)
    return v

def _empty_result(params):
    return {'metrics': {'CAGR':0,'MDD':0,'Sharpe':0,'Sortino':0,'Final':0},
            'yearly': {}, 'rebal_count': 0, 'label': params.label,
            'params': params, 'pv': pd.DataFrame()}


# ─── Metrics ────────────────────────────────────────────────────────

def calc_metrics(pv):
    if len(pv) < 2:
        return {'CAGR':0,'MDD':0,'Sharpe':0,'Sortino':0,'Final':0}
    values = pv['Value']
    days = (pv.index[-1] - pv.index[0]).days
    years = days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    peak = values.cummax()
    mdd = (values / peak - 1).min()
    dr = values.pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(365) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = ((dr.mean() / down.std()) * np.sqrt(365)
               if len(down) > 1 and down.std() > 0 else sharpe)
    return {'CAGR': cagr, 'MDD': mdd, 'Sharpe': sharpe,
            'Sortino': sortino, 'Final': values.iloc[-1]}

def calc_yearly_metrics(pv):
    out = {}
    for y in range(pv.index[0].year, pv.index[-1].year + 1):
        mask = pv.index.year == y
        if mask.sum() < 10: continue
        out[y] = calc_metrics(pv[mask])
    return out


# ─── Multiprocessing ────────────────────────────────────────────────

_g_prices = None
_g_universe = None

def init_pool(prices, universe_map):
    global _g_prices, _g_universe
    _g_prices = prices
    _g_universe = universe_map

def run_single(params):
    return run_backtest(_g_prices, _g_universe, params)


# ─── Main ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    result = run_backtest(prices, universe)
    m = result['metrics']
    print(f"\nBASELINE: Sharpe={m['Sharpe']:.3f}  MDD={m['MDD']:.1%}"
          f"  CAGR={m['CAGR']:+.1%}  Final=${m['Final']:,.0f}"
          f"  Rebals={result['rebal_count']}")
